# filename: main.py
# -*- coding: utf-8 -*-
import os
import io
import asyncio
import time
import tempfile
import datetime
import logging
import warnings
import subprocess
from time import sleep

import requests
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from pydub import AudioSegment
import speech_recognition as sr
import edge_tts
import nest_asyncio

# 🛠️ התיקון: הפעלת nest_asyncio כדי לאפשר ל-asyncio.run לרוץ בתוך סביבת Flask/Gunicorn
try:
    nest_asyncio.apply()
except Exception as e:
    logging.info(f"⚠️ Could not apply nest_asyncio: {e}")

# === הגדרות בסיס ===
USERNAME = "0733181201"
PASSWORD = "6714453"
TOKEN = f"{USERNAME}:{PASSWORD}"
YEMOT_DOWNLOAD_URL = "https://www.call2all.co.il/ym/api/DownloadFile"
YEMOT_UPLOAD_URL = "https://www.call2all.co.il/ym/api/UploadFile"

CSV_PATH = "stock_data.csv"
FFMPEG_BIN = "ffmpeg"  # ודא זמין ב-PATH. אחרת כתוב נתיב מלא

# === Flask + לוגים ===
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")

# === טעינת טבלת ניירות ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ לא נמצא הקובץ {CSV_PATH}")
# comment="#" ידלג על כותרות/כותרות ביניים
stock_df = pd.read_csv(CSV_PATH, comment="#", dtype=str).fillna("")
# נורמליזציה קלה לשימוש
stock_df["name_norm"] = stock_df["name"].str.strip().str.lower()
stock_df["display_name_norm"] = stock_df["display_name"].str.strip().str.lower()
stock_df["symbol"] = stock_df["symbol"].str.strip()

# =====================================================
# === זיהוי דיבור =====================================
# =====================================================

def add_silence(input_path: str) -> AudioSegment:
    """הוספת שנייה שקט בתחילה ובסוף (עוזר ל-ASR)"""
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence

def recognize_speech(audio_segment: AudioSegment) -> str:
    """זיהוי דיבור בעברית"""
    rec = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            audio_segment.export(tmp.name, format="wav")
            with sr.AudioFile(tmp.name) as source:
                data = rec.record(source)
            text = rec.recognize_google(data, language="he-IL")
            logging.info(f"✅ זוהה דיבור: {text}")
            return text.strip()
    except sr.UnknownValueError:
        logging.info("❌ לא זוהה דיבור ברור.")
        return ""
    except Exception as e:
        logging.info(f"❌ שגיאה בזיהוי: {e}")
        return ""

def transcribe_audio(filename: str) -> str:
    try:
        processed = add_silence(filename)
        return recognize_speech(processed)
    except Exception as e:
        logging.info(f"❌ שגיאה בתמלול: {e}")
        return ""

# =====================================================
# === עזרי שוק ========================================
# =====================================================

def _as_float(x):
    """המרה זהירה ל-float"""
    try:
        if isinstance(x, (float, int)):
            return float(x)
        if hasattr(x, "values"):
            return float(x.values[0])
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return 0.0

def _yf_download_with_retries(ticker, start, end, max_retries=3, wait_sec=5):
    """ריטריים חכמים ל-yfinance לרבות RateLimit"""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            # yfinance לפעמים לא זורק חריגה, רק מחזיר empty
            if data is not None and not data.empty:
                return data
            # אם ריק – נזרוק חריגה כדי להיכנס למנגנון הריטריים
            raise RuntimeError("EmptyDataFromYahoo")
        except Exception as e:
            last_exc = e
            msg = str(e)
            logging.info(f"⚠️ נסיון {attempt}/{max_retries} נכשל: {msg}")
            if attempt < max_retries:
                sleep(wait_sec)
    # נכשל סופית
    raise last_exc if last_exc else RuntimeError("UnknownYFinanceError")

def calculate_dca_return(ticker, start_date_str, start_amount, monthly_amount, throb_days):
    """חישוב תשואה לפי הפקדות מדורגות (DCA) עם ריטריים לנתונים"""
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%d-%m-%Y").date()
        end_date = datetime.date.today()

        data = _yf_download_with_retries(ticker, start=start_date, end=end_date)
        # אם עדיין ריק – נחזיר שגיאה
        if data is None or data.empty:
            return {"error": "לא נמצאו נתוני שוק עבור הנייר"}

        total_units = 0.0
        total_invested = 0.0
        deposits = []

        first_price = _as_float(data["Close"].iloc[0])
        current_price = _as_float(data["Close"].iloc[-1])

        # הפקדה ראשונה
        if start_amount > 0:
            total_units += start_amount / max(first_price, 1e-9)
            total_invested += start_amount
            deposits.append((start_date, start_amount, first_price))

        # הפקדות מחזוריות
        if monthly_amount > 0 and throb_days > 0:
            next_date = start_date + datetime.timedelta(days=throb_days)
            while next_date <= end_date:
                # מציאת יום מסחר קרוב
                closest_idx = min(data.index, key=lambda d: abs(d.date() - next_date))
                price = _as_float(data.loc[closest_idx]["Close"])
                total_units += monthly_amount / max(price, 1e-9)
                total_invested += monthly_amount
                deposits.append((next_date, monthly_amount, price))
                next_date += datetime.timedelta(days=throb_days)

        current_value = total_units * current_price
        profit = current_value - total_invested
        percent = (profit / total_invested) * 100 if total_invested > 0 else 0

        logging.info("📊 --- סיכום טרייד ---")
        logging.info(f"נייר ערך: {ticker}")
        logging.info(f"מחיר התחלתי: {first_price:.2f}$ | נוכחי: {current_price:.2f}$")
        logging.info(f"השקעה כוללת: {total_invested:.2f}$ | שווי נוכחי: {current_value:.2f}$")
        logging.info(f"רווח: {profit:.2f}$ ({percent:.2f}%)")
        logging.info("----------------------------")

        return {
            "ticker": ticker,
            "start_date": start_date.strftime("%d-%m-%Y"),
            "end_date": end_date.strftime("%d-%m-%Y"),
            "first_price": round(first_price, 2),
            "current_price": round(current_price, 2),
            "total_invested": round(total_invested, 2),
            "current_value": round(current_value, 2),
            "profit": round(profit, 2),
            "percent": round(percent, 2),
            "deposits_count": len(deposits),
        }
    except Exception as e:
        return {"error": str(e)}

# =====================================================
# === חיפוש טיקר וטקסט קולי ==========================
# =====================================================

def find_ticker(recognized_text: str):
    """חיפוש סימבול לפי טקסט מזוהה (בעברית/תצוגה/אנגלית)"""
    if not recognized_text:
        return None, None
    txt = recognized_text.strip().lower()
    # ניסיון התאמה לפי עמודות טקסט
    for _, row in stock_df.iterrows():
        name = (row.get("name_norm") or "").strip()
        dsp = (row.get("display_name_norm") or "").strip()
        sym = (row.get("symbol") or "").strip()
        if not sym:
            continue
        if name and name in txt:
            return sym, row.get("display_name") or row.get("name") or sym
        if dsp and dsp in txt:
            return sym, row.get("display_name") or row.get("name") or sym
        # גם אם המשתמש אמר את הסימבול עצמו
        if sym.lower() in txt:
            return sym, row.get("display_name") or row.get("name") or sym
    return None, None

def build_success_tts_text(display_name_he, start_date, start_amount, monthly_amount,
                           first_price, current_price, total_invested, current_value, profit, percent):
    """
    בונה טקסט קולי קריא וברור בעברית, כולל ההסתייגות שבחרת.
    הערכים נשארים מספריים (לא הופכים למילים) כדי לצאת טבעי בסינתזה.
    """
    # עיבוד מינוחי כסף פשוטים (ש"ח/דולר)
    start_amount_nis = int(round(start_amount))
    monthly_amount_nis = int(round(monthly_amount))
    total_invested_nis = int(round(total_invested))
    current_value_nis = int(round(current_value))
    profit_nis = int(round(profit))

    text = (
        f"להלן התוצאה. נייר הערך שבחרת הוא {display_name_he}. "
        f"התחלת להשקיע בתאריך {start_date.replace('-', ' ')}. "
        f"עם סכום ראשוני של {start_amount_nis} שקלים. "
        f"והוספת בכל חודש {monthly_amount_nis} שקלים נוספים. "
        f"מחיר הנייר ביום ההפקדה עמד על {first_price} דולרים. "
        f"המחיר כעת עומד על {current_price} דולרים. "
        f"סך הכול הפקדת {total_invested_nis} שקלים. "
        f"והשווי הנוכחי של ההשקעה שלך הוא {current_value_nis} שקלים. "
        f"הרווח הכולל שלך עומד על {profit_nis} שקלים. "
        f"שהם תשואה של {round(percent, 2)} אחוזים. "
        "לתשומת לב, הנתונים נשלפו ממקורות עדכניים, אך ייתכנו הבדלים קלים לעומת הנתונים הרשמיים. "
        "שימוש במידע הוא באחריות המשתמש בלבד."
    )
    return text

def build_error_tts_text(err_msg: str):
    return (
        f"שגיאה. {err_msg}. "
        "לתשומת לבך, ייתכנו פערים קלים או עיכוב בעדכון הנתונים. "
        "אנא נסה שוב מאוחר יותר."
    )

# =====================================================
# === Edge TTS + FFmpeg + העלאה ליֶמוֹט ================
# =====================================================

async def _edge_tts_synthesize(text: str, out_mp3_path: str,
                               voice: str = "he-IL-AvriNeural", rate: str = "+0%"):
    """סינתזה ל-MP3 עם Edge TTS"""
    tts = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await tts.save(out_mp3_path)

def mp3_to_wav_pcm8k_mono(in_mp3: str, out_wav: str):
    """המרה ל-WAV 8kHz מונו PCM (נפוץ ומתאים למערכות IVR)"""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", in_mp3,
        "-ac", "1",          # מונו
        "-ar", "8000",       # 8kHz
        "-acodec", "pcm_s16le",
        out_wav
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {res.stderr.decode(errors='ignore')}")

def upload_to_yemot(local_file_path: str, remote_path_with_filename: str):
    """
    העלאת קובץ ל-Yemot:
    remote_path_with_filename לדוגמה:
    ivr2:/100/5/Phone/0531234567/result_1699999999.wav
    """
    with open(local_file_path, "rb") as f:
        files = {"file": (os.path.basename(remote_path_with_filename), f, "audio/wav")}
        data = {"token": TOKEN, "path": remote_path_with_filename}
        r = requests.post(YEMOT_UPLOAD_URL, data=data, files=files, timeout=60)
        r.raise_for_status()
        return r.text

def make_and_upload_tts(text: str, api_phone: str):
    """יוצר קובץ TTS, ממיר ומעלה ל-Yemot. מחזיר נתיב יעד מלא."""
    base_dir = f"ivr2:/100/5/Phone/{api_phone.strip()}/"
    filename = f"result_{int(time.time())}.wav"
    remote_full_path = base_dir + filename

    with tempfile.TemporaryDirectory() as td:
        mp3_path = os.path.join(td, "tts.mp3")
        wav_path = os.path.join(td, "tts.wav")
        # סינתזה: הפעלת הקורוטינה
        asyncio.run(_edge_tts_synthesize(text, mp3_path))
        # המרה
        mp3_to_wav_pcm8k_mono(mp3_path, wav_path)
        # העלאה
        upload_to_yemot(wav_path, remote_full_path)

    return remote_full_path

# =====================================================
# === נקודת קצה ראשית =================================
# =====================================================

@app.route("/ivr", methods=["GET"])
def process_investment():
    logging.info("\n" + "=" * 60)
    logging.info(f"📞 בקשה התקבלה ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    # פרמטרים מה-API של ימות
    api_phone = request.args.get("ApiPhone", "").strip()
    stock_name_path = request.args.get("stock_name")
    start_date = request.args.get("Starting_date") or request.args.get("Startig_date")
    start_amount = float(request.args.get("Starting_amount", 0))
    monthly_amount = float(request.args.get("Monthly_amount", 0))
    throb = int(request.args.get("throb", 30))

    # בדיקות בסיס
    if not stock_name_path or not start_date or start_amount <= 0:
        err = "חסרים פרמטרים נדרשים"
        logging.info(f"❌ {err}")
        if api_phone:
            try:
                t = make_and_upload_tts(build_error_tts_text(err), api_phone)
                return jsonify({"error": err, "audio": t, "next_ext": "100/5"}), 400
            except Exception:
                pass
        return jsonify({"error": err}), 400

    # הורדת ההקלטה מימות
    try:
        yemot_path = f"ivr2:/{stock_name_path.lstrip('/')}"
        params = {"token": TOKEN, "path": yemot_path}
        r = requests.get(YEMOT_DOWNLOAD_URL, params=params, timeout=30)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name
    except Exception as e:
        err = f"שגיאה בהורדת הקלטה: {e}"
        logging.info(f"❌ {err}")
        if api_phone:
            try:
                t = make_and_upload_tts(build_error_tts_text(err), api_phone)
                return jsonify({"error": err, "audio": t, "next_ext": "100/5"}), 500
            except Exception:
                pass
        return jsonify({"error": err}), 500

    # זיהוי דיבור -> מציאת טיקר
    try:
        recognized_text = transcribe_audio(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not recognized_text:
        err = "לא זוהה דיבור ברור"
        logging.info(f"❌ {err}")
        if api_phone:
            try:
                t = make_and_upload_tts(build_error_tts_text(err), api_phone)
                return jsonify({"error": err, "audio": t, "next_ext": "100/5"}), 200
            except Exception:
                pass
        return jsonify({"error": err}), 200

    ticker, display_name_he = find_ticker(recognized_text)
    if not ticker:
        err = f"לא נמצא נייר ערך מתאים לטקסט '{recognized_text}'"
        logging.info(f"❌ {err}")
        if api_phone:
            try:
                t = make_and_upload_tts(build_error_tts_text(err), api_phone)
                return jsonify({"error": err, "audio": t, "next_ext": "100/5"}), 200
            except Exception:
                pass
        return jsonify({"error": err}), 200

    # חישוב תשואה (עם ריטריים פנימי)
    result = calculate_dca_return(ticker, start_date, start_amount, monthly_amount, throb)

    # אם שגיאה — נקריא אותה
    if "error" in result:
        err = result["error"]
        logging.info(f"✅ תוצאה JSON: {result}")
        if api_phone:
            try:
                t = make_and_upload_tts(build_error_tts_text(err), api_phone)
                return jsonify({"error": err, "audio": t, "next_ext": "100/5"}), 200
            except Exception as e:
                return jsonify({"error": f"{err}; ושגיאת הקלטה: {e}"}), 200
        return jsonify(result), 200

    # יצירת טקסט קולי מוצלח
    tts_text = build_success_tts_text(
        display_name_he=display_name_he or ticker,
        start_date=result["start_date"],
        start_amount=start_amount,
        monthly_amount=monthly_amount,
        first_price=result["first_price"],
        current_price=result["current_price"],
        total_invested=result["total_invested"],
        current_value=result["current_value"],
        profit=result["profit"],
        percent=result["percent"],
    )

    audio_remote_path = None
    if api_phone:
        try:
            audio_remote_path = make_and_upload_tts(tts_text, api_phone)
        except Exception as e:
            logging.info(f"⚠️ כשל ביצירת/העלאת TTS: {e}")

    # החזרה ללקוח (כולל שלוחה הבאה)
    out = {
        "result": result,
        "recognized_text": recognized_text,
        "ticker": ticker,
        "display_name": display_name_he or ticker,
        "audio": audio_remote_path,   # לדוגמה: ivr2:/100/5/Phone/<ApiPhone>/result_xxx.wav
        "next_ext": "100/5"           # תוכל להשתמש כדי לבצע ניתוב בשלוחה
    }
    logging.info(f"✅ תוצאה JSON: {out}")
    logging.info("=" * 60 + "\n")
    return jsonify(out), 200


if __name__ == "__main__":
    # Render וכד' נוהגים להאזין ל-0.0.0.0
    app.run(host="0.0.0.0", port=5000)
