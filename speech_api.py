
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)
CORS(app, origins=["https://"])

# Load trained model
artifact = joblib.load(Path(__file__).parent / "speech_model.joblib")

model = artifact["model"]
feature_names = artifact["feature_names"]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model": "ParkinSense Speech Random Forest",
        "features_required": len(feature_names)
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        missing = [
            feature for feature in feature_names
            if feature not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        values = {}

        for feature in feature_names:
            value = float(data[feature])

            if not pd.notna(value):
                return jsonify({
                    "error": f"Invalid value for {feature}"
                }), 400

            values[feature] = value

        X = pd.DataFrame([values], columns=feature_names)

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][
            list(model.classes_).index(1)
        ])

        return jsonify({
            "success": True,
            "prediction": prediction,
            "class": (
                "Parkinson's-related pattern"
                if prediction == 1
                else "Healthy-control-like pattern"
            ),
            "model_score": round(probability, 4),
            "notice": (
                "Research output only. Not a medical diagnosis."
            )
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Feature values must be numeric."
        }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
