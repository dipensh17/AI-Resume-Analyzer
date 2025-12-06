"""Small Flask server exposing /recommend for the front-end to call.

Run: `python scripts/recommend_server.py`

This server accepts POST /recommend with JSON {"text": "..."}
and returns JSON {"recommendations": [...]}.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from recommend import recommend_jobs

app = Flask(__name__)
CORS(app)


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json() or {}
    text = data.get("text", "")
    n = int(data.get("n", 5))
    recs = recommend_jobs(text, n=n)
    return jsonify({"recommendations": recs})


if __name__ == "__main__":
    print("Starting recommend server on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
