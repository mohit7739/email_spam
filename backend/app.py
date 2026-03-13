from flask import Flask, request, jsonify
from flask_cors import CORS
from spam_processor import SpamClassifier

app = Flask(__name__)
CORS(app)

# Load model once at startup
try:
    classifier = SpamClassifier.load()
except FileNotFoundError as e:
    print(f"\n Model not found: {e}\n")
    classifier = None


def model_ready():
    if classifier is None:
        return jsonify({"error": "Model not loaded. Run python train.py first."}), 503
    return None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": classifier is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    err = model_ready()
    if err:
        return err

    data = request.get_json(silent=True)
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' field in request body."}), 400

    message = data['message']
    if not isinstance(message, str):
        return jsonify({"error": "'message' must be a string."}), 400

    result = classifier.predict(message)
    return jsonify(result), 200


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    err = model_ready()
    if err:
        return err

    data = request.get_json(silent=True)
    if not data or 'messages' not in data:
        return jsonify({"error": "Missing 'messages' field in request body."}), 400

    messages = data['messages']
    if not isinstance(messages, list):
        return jsonify({"error": "'messages' must be a JSON array of strings."}), 400
    if len(messages) == 0:
        return jsonify({"error": "'messages' array is empty."}), 400
    if len(messages) > 100:
        return jsonify({"error": "Batch limit is 100 messages per request."}), 400

    for i, msg in enumerate(messages):
        if not isinstance(msg, str):
            return jsonify({"error": f"Item at index {i} is not a string."}), 400

    results = classifier.predict_batch(messages)
    return jsonify({"count": len(results), "results": results}), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found.",
        "available_routes": ["GET /health", "POST /predict", "POST /predict/batch"]
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "HTTP method not allowed for this endpoint."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error.", "details": str(e)}), 500


if __name__ == '__main__':
    print("\n Spam Classifier API starting...")
    print("   GET  http://localhost:5000/health")
    print("   POST http://localhost:5000/predict")
    print("   POST http://localhost:5000/predict/batch")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)