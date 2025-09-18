from flask import Flask, request, jsonify
import joblib
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows requests from frontend

# Load the saved model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return "✅ Phishing Detector API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        X = vectorizer.transform([text])
        prediction = int(model.predict(X)[0])

        # Optional: confidence score
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X).max())

        return jsonify({
            "text": text,
            "prediction": prediction,   # 1 = phishing, 0 = safe
            "probability": probability
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
