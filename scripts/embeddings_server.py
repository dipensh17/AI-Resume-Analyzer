"""Flask server exposing an embeddings-based recommend endpoint.

Run: `python scripts/embeddings_server.py`
Listens on port 5001 by default.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from embeddings_recommender import EmbeddingsRecommender
import os

app = Flask(__name__)
CORS(app)

# Load model once
MODEL_NAME = os.environ.get("EMB_MODEL", "all-MiniLM-L6-v2")
recommender = None


def get_recommender():
    global recommender
    if recommender is None:
        recommender = EmbeddingsRecommender(model_name=MODEL_NAME)
    return recommender


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json() or {}
    text = data.get("text", "")
    n = int(data.get("n", 5))
    rec = get_recommender().recommend(text, n=n)
    # Convert numpy floats to python floats
    out = [
        {"title": r["title"], "score": float(r["score"]), "explanation": r.get("explanation", "")}
        for r in rec
    ]
    return jsonify({"recommendations": out})


if __name__ == "__main__":
    print(f"Starting embeddings recommend server on http://127.0.0.1:5001 using model {MODEL_NAME}")
    app.run(port=5001, debug=True)
