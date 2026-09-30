import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types
from chatbot_config import MODEL_NAME, SYSTEM_PROMPT

load_dotenv()
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
)

key = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=key) if key and key != "your_gemini_api_key_here" else None

def make_prompt(message, history):
    lines = []
    for item in history[-12:]:
        role = "Student" if item.get("role") == "user" else "Tutor"
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append(f"Student: {message.strip()}")
    lines.append("Tutor:")
    return "\n".join(lines)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Message is required."}), 400
    if not isinstance(history, list):
        history = []

    if client is None:
        return jsonify({"error": "GEMINI_API_KEY is not configured on the deployment."}), 503

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=make_prompt(message, history),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            return jsonify({"error": "The model returned an empty response."}), 502
        return jsonify({"answer": answer})
    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({"error": "Unable to generate a response right now."}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
