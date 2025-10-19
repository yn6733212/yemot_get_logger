# filename: yemot_get_logger.py
from flask import Flask, request, jsonify
import datetime
import json

app = Flask(__name__)

@app.route("/ivr", methods=["GET"])
def ivr_receiver():
    # כל הפרמטרים שנשלחו ע"י ימות (דרך ה-URL)
    params = request.args.to_dict(flat=False)  # flat=False שומר רשימות אם יש כפילויות

    # לוג יפה למסוף
    print("\n" + "="*60)
    print(f"📞 בקשה התקבלה ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("="*60 + "\n", flush=True)

    # מחזירים גם תשובה JSON (סתם לנוחות בדפדפן)
    return jsonify({
        "status": "ok",
        "received_params": params
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
