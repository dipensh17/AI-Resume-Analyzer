"""Flask server exposing an embeddings-based recommend endpoint.

Run: `python scripts/embeddings_server.py`
Listens on port 5001 by default.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from embeddings_recommender import EmbeddingsRecommender
import os

app = Flask(__name__)
# Enable CORS for all routes/origins and ensure preflight/options are handled.
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


@app.after_request
def add_cors_headers(response):
    # Ensure all responses (including errors) carry CORS headers so browsers
    # don't block responses when preflighting or when server errors occur.
    response.headers.setdefault('Access-Control-Allow-Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

# Load model once
MODEL_NAME = os.environ.get("EMB_MODEL", "all-MiniLM-L6-v2")
recommender = None


def get_recommender():
    global recommender
    if recommender is None:
        recommender = EmbeddingsRecommender(model_name=MODEL_NAME)
    return recommender


@app.route("/recommend", methods=["POST", "OPTIONS"])
def recommend():
    # Handle preflight
    if request.method == "OPTIONS":
        return (jsonify({}), 200)
    data = request.get_json() or {}
    text = data.get("text", "")
    n = int(data.get("n", 5))
    rec = get_recommender().recommend(text, n=n)
    # Convert numpy floats to python floats
    out = [
        {
            "title": r["title"],
            # convert similarity (0..1) to integer percentage (e.g. 0.3534 -> 35)
            "score": int(round(float(r["score"]) * 100)),
            "explanation": r.get("explanation", ""),
        }
        for r in rec
    ]
    return jsonify({"recommendations": out})


if __name__ == "__main__":
    print(f"Starting embeddings recommend server on http://127.0.0.1:5001 using model {MODEL_NAME}")
    app.run(port=5001, debug=True)
