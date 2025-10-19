# filename: main.py
# -*- coding: utf-8 -*-
import requests
import os
import io
import tempfile
import datetime
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from pydub import AudioSegment
import speech_recognition as sr
import subprocess
import logging
import warnings

# --- הגדרות בסיס ---
USERNAME = "0733181201"
PASSWORD = "6714453"
TOKEN = f"{USERNAME}:{PASSWORD}"
YEMOT_DOWNLOAD_URL = "https://www.call2all.co.il/ym/api/DownloadFile"
FFMPEG_EXECUTABLE = "ffmpeg"

app = Flask(__name__)

# --- לוגים נקיים ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")


# =====================================================
# === פונקציות זיהוי דיבור ============================
# =====================================================

def add_silence(input_path: str) -> AudioSegment:
    """הוספת שנייה שקט בתחילת וסוף הקובץ (מייצב את הזיהוי)"""
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence


def recognize_speech(audio_segment: AudioSegment) -> str:
    """זיהוי דיבור בעברית באמצעות Google Speech Recognition"""
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
# === פונקציית חישוב תשואה מדויקת ====================
# =====================================================

def calculate_dca_return(ticker, start_date, start_amount, monthly_amount, throb_days):
    """חישוב תשואה מדויקת לפי הפקדות מדורגות (DCA)"""
    try:
        start_date = datetime.datetime.strptime(start_date, "%d-%m-%Y").date()
        end_date = datetime.date.today()

        # הורדת נתונים היסטוריים
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return {"error": "לא נמצאו נתוני שוק עבור הנייר"}

        total_units = 0.0
        total_invested = 0.0
        deposits = []

        current_price = data["Close"].iloc[-1]

        # הפקדה ראשונית
        first_price = data["Close"].iloc[0]
        total_units += start_amount / first_price
        total_invested += start_amount
        deposits.append((start_date, start_amount, first_price))

        # הפקדות חוזרות
        next_date = start_date + datetime.timedelta(days=throb_days)
        while next_date <= end_date:
            closest_date = min(data.index, key=lambda d: abs(d.date() - next_date))
            price = data.loc[closest_date]["Close"]
            total_units += monthly_amount / price
            total_invested += monthly_amount
            deposits.append((next_date, monthly_amount, price))
            next_date += datetime.timedelta(days=throb_days)

        current_value = total_units * current_price
        profit = current_value - total_invested
        percent = (profit / total_invested) * 100

        return {
            "ticker": ticker,
            "start_date": start_date.strftime("%d-%m-%Y"),
            "end_date": end_date.strftime("%d-%m-%Y"),
            "total_invested": round(total_invested, 2),
            "current_value": round(current_value, 2),
            "profit": round(profit, 2),
            "percent": round(percent, 2),
            "current_price": round(current_price, 2),
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

    # --- שליפת פרמטרים ---
    stock_name = request.args.get("stock_name")
    start_date = request.args.get("Starting_date") or request.args.get("Startig_date")
    start_amount = float(request.args.get("Starting_amount", 0))
    monthly_amount = float(request.args.get("Monthly_amount", 0))
    throb = int(request.args.get("throb", 30))

    if not stock_name or not start_date or not start_amount:
        return jsonify({"error": "חסרים פרמטרים נדרשים"}), 400

    # --- הורדת הקלטה מימות ---
    logging.info(f"⬇️ מוריד הקלטה מימות: {stock_name}")
    path_on_yemot = f"ivr2:/{stock_name.lstrip('/')}"
    params = {"token": TOKEN, "path": path_on_yemot}
    response = requests.get(YEMOT_DOWNLOAD_URL, params=params, timeout=30)
    response.raise_for_status()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.write(response.content)
    temp_wav.close()

    # --- זיהוי דיבור ---
    recognized_text = transcribe_audio(temp_wav.name)
    os.remove(temp_wav.name)

    if not recognized_text:
        return jsonify({"error": "לא זוהה דיבור ברור"})

    # === כאן נשתמש בעתיד בקובץ CSV ===
    # כרגע ניקח דוגמה קטנה זמנית:
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

    # --- חישוב תשואה ---
    result = calculate_dca_return(ticker, start_date, start_amount, monthly_amount, throb)
    logging.info(f"✅ תוצאה: {result}")
    logging.info("=" * 60 + "\n")

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
