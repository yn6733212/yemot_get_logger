# filename: main.py
# -*- coding: utf-8 -*-
import requests
import os
import tempfile
import datetime
import yfinance as yf
from flask import Flask, request, jsonify
from pydub import AudioSegment
import speech_recognition as sr
import logging
import warnings

# --- הגדרות בסיס ---
USERNAME = "0733181201"
PASSWORD = "6714453"
TOKEN = f"{USERNAME}:{PASSWORD}"
YEMOT_DOWNLOAD_URL = "https://www.call2all.co.il/ym/api/DownloadFile"

app = Flask(__name__)

# --- לוגים נקיים ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")

# =====================================================
# === פונקציות זיהוי דיבור ============================
# =====================================================

def add_silence(input_path: str) -> AudioSegment:
    """הוספת שנייה שקט בתחילת וסוף הקובץ"""
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence


def recognize_speech(audio_segment: AudioSegment) -> str:
    """זיהוי דיבור בעברית"""
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")
            with sr.AudioFile(temp_wav.name) as source:
                data = recognizer.record(source)
            text = recognizer.recognize_google(data, language="he-IL")
            logging.info(f"✅ זוהה דיבור: {text}")
            return text
    except sr.UnknownValueError:
        logging.info("❌ לא זוהה דיבור ברור.")
        return ""
    except Exception as e:
        logging.info(f"❌ שגיאה בזיהוי: {e}")
        return ""


def transcribe_audio(filename: str) -> str:
    """עטיפת התהליך"""
    try:
        processed_audio = add_silence(filename)
        return recognize_speech(processed_audio)
    except Exception as e:
        logging.info(f"❌ שגיאה בתמלול: {e}")
        return ""


# =====================================================
# === פונקציית עזר להמרת ערכים ל-float ===============
# =====================================================

def _as_float(x):
    """המרת סוגים שונים ל-float"""
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


# =====================================================
# === פונקציית חישוב תשואה מדויקת ====================
# =====================================================

def calculate_dca_return(ticker, start_date, start_amount, monthly_amount, throb_days):
    """חישוב תשואה לפי הפקדות מדורגות"""
    try:
        start_date = datetime.datetime.strptime(start_date, "%d-%m-%Y").date()
        end_date = datetime.date.today()

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return {"error": "לא נמצאו נתוני שוק עבור הנייר"}

        total_units = 0.0
        total_invested = 0.0
        deposits = []

        first_price = _as_float(data["Close"].iloc[0])
        current_price = _as_float(data["Close"].iloc[-1])

        # הפקדה ראשונה
        total_units += start_amount / first_price
        total_invested += start_amount
        deposits.append((start_date, start_amount, first_price))

        # הפקדות חוזרות (אם יש)
        if monthly_amount > 0:
            next_date = start_date + datetime.timedelta(days=throb_days)
            while next_date <= end_date:
                closest_date = min(data.index, key=lambda d: abs(d.date() - next_date))
                price = _as_float(data.loc[closest_date]["Close"])
                total_units += monthly_amount / price
                total_invested += monthly_amount
                deposits.append((next_date, monthly_amount, price))
                next_date += datetime.timedelta(days=throb_days)

        current_value = total_units * current_price
        profit = current_value - total_invested
        percent = (profit / total_invested) * 100 if total_invested > 0 else 0

        # 🧾 --- לוגים בעברית ---
        logging.info("📊 --- סיכום טרייד ---")
        logging.info(f"נייר ערך: {ticker}")
        logging.info(f"מחיר התחלתי בתאריך {start_date.strftime('%d-%m-%Y')}: {first_price:.2f}$")
        logging.info(f"מחיר נוכחי: {current_price:.2f}$")
        logging.info(f"סכום כולל שהושקע: {total_invested:.2f}$")
        logging.info(f"שווי נוכחי כולל: {current_value:.2f}$")
        logging.info(f"סה״כ רווח: {profit:.2f}$ ({percent:.2f}%)")
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
            "deposits_count": len(deposits)
        }

    except Exception as e:
        return {"error": str(e)}


# =====================================================
# === נקודת קצה ראשית ================================
# =====================================================

@app.route("/ivr", methods=["GET"])
def process_investment():
    logging.info("\n" + "=" * 60)
    logging.info(f"📞 בקשה התקבלה ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    stock_name = request.args.get("stock_name")
    start_date = request.args.get("Starting_date") or request.args.get("Startig_date")
    start_amount = float(request.args.get("Starting_amount", 0))
    monthly_amount = float(request.args.get("Monthly_amount", 0))
    throb = int(request.args.get("throb", 30))

    if not stock_name or not start_date or not start_amount:
        return jsonify({"error": "חסרים פרמטרים נדרשים"}), 400

    logging.info(f"⬇️ מוריד הקלטה מימות: {stock_name}")
    path_on_yemot = f"ivr2:/{stock_name.lstrip('/')}"
    params = {"token": TOKEN, "path": path_on_yemot}
    response = requests.get(YEMOT_DOWNLOAD_URL, params=params, timeout=30)
    response.raise_for_status()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.write(response.content)
    temp_wav.close()

    recognized_text = transcribe_audio(temp_wav.name)
    os.remove(temp_wav.name)

    if not recognized_text:
        return jsonify({"error": "לא זוהה דיבור ברור"})

    mapping = {
        "ביטקוין": "BTC-USD",
        "טסלה": "TSLA",
        "אס אנד פי": "SPY",
        "תל אביב": "TA35.TA"
    }

    ticker = None
    for key, value in mapping.items():
        if key in recognized_text:
            ticker = value
            break

    if not ticker:
        return jsonify({"error": f"לא נמצא טיקר תואם למילה '{recognized_text}'"})

    result = calculate_dca_return(ticker, start_date, start_amount, monthly_amount, throb)
    logging.info(f"✅ תוצאה JSON: {result}")
    logging.info("=" * 60 + "\n")

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
