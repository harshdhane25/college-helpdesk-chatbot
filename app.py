from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os
import logging
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, static_folder=None)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read from environment variable (set in Vercel dashboard)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("MODEL", "openrouter/free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """
You are CollegeAssist, an AI-powered College Helpdesk Assistant.
ROLE:
You are a professional, friendly, patient, accurate and student-focused college helpdesk assistant.
PURPOSE:
Your purpose is to help students understand academic, administrative and general college-related information.
TARGET USERS:
Students, applicants and other users seeking college helpdesk information.
TONE:
- Friendly
- Professional
- Clear
- Respectful
- Concise
- Easy for students to understand
CORE INSTRUCTIONS:
1. Answer only questions related to the college helpdesk scope.
2. Never invent college rules, fees, deadlines, office timings, policies, names or contact information.
3. If information is unavailable, clearly state that you do not have verified information.
4. When necessary, ask a concise clarification question before answering.
5. Give structured answers using headings, bullets or numbered steps when useful.
6. Do not unnecessarily repeat the student's question.
7. Do not provide hidden chain-of-thought.
8. For calculations, perform the calculation carefully and provide the final result with a concise verification.
9. If the student provides multiple fee components, identify each component, apply scholarships/concessions correctly, and calculate the final payable amount.
10. Clearly distinguish between known information, assumptions and information that needs confirmation.
11. Never claim to have accessed college databases, student records, portals or official systems unless such access actually exists.
12. Do not request sensitive information such as passwords, OTPs, banking PINs or authentication credentials.
13. If the query is outside the chatbot's scope, politely explain the limitation and suggest the appropriate college department or official channel.
14. If a question is ambiguous, ask for the missing information instead of guessing.
15. Keep answers practical and student-friendly.
REASONING POLICY:
For multi-step questions:
- Identify the required values.
- Perform the necessary reasoning/calculation internally.
- Verify the result.
- Return only the final answer and a concise explanation.
- Never reveal private chain-of-thought.
OUT-OF-SCOPE POLICY:
Questions unrelated to college helpdesk services should receive a polite refusal/redirect.
RESPONSE QUALITY:
Before responding, internally check:
- Is the question within scope?
- Do I have enough information?
- Am I inventing anything?
- Is clarification required?
- Is the calculation correct?
- Is the response clear and useful?
If information is missing, do not fabricate it.
ADDITIONAL GUIDELINES:
- If asked for exact college-specific data (e.g., fees for a particular course) and you do not have verified data, say so explicitly and direct the user to the appropriate office or portal.
- For fee calculations, use Indian Rupees (₹) formatting when appropriate.
- Always remind users to confirm critical details with the college office.
"""

def validate_api_key():
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("sk-or-v1-PASTE") or "PASTE_YOUR" in OPENROUTER_API_KEY:
        return False, "OpenRouter API key is missing. Set OPENROUTER_API_KEY in Vercel Environment Variables."
    return True, ""

def call_openrouter(messages, temperature=0.3, max_tokens=800):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://college-helpdesk-chatbot-rouge.vercel.app",  # will be updated after deploy
        "X-Title": "College Helpdesk Chatbot",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_info = error_data.get("error", {})
                error_message = error_info.get("message", "Unknown error from OpenRouter")
            except Exception:
                error_message = response.text
            logger.error("OpenRouter error: %s", error_message)
            if response.status_code == 401:
                return None, "Invalid OpenRouter API key."
            if response.status_code == 429:
                return None, "AI service is busy or rate-limited. Please try again shortly."
            if response.status_code >= 500:
                return None, "The AI service is temporarily unavailable. Please try again later."
            return None, f"OpenRouter error: {error_message}"

        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            reply = data["choices"][0].get("message", {}).get("content")
            if reply:
                return reply.strip(), None
            return None, "The AI returned an empty response."
        if "error" in data:
            return None, data["error"].get("message", "Unknown AI service error")
        return None, "Unexpected response from AI service."
    except requests.exceptions.Timeout:
        return None, "The AI service took too long to respond. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the AI service. Check your internet connection."
    except Exception as e:
        logger.exception("Unexpected error calling OpenRouter")
        return None, "Unexpected error occurred."

@app.route("/api/chat", methods=["POST"])
def chat():
    key_ok, key_error = validate_api_key()
    if not key_ok:
        return jsonify({"success": False, "error": key_error}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid request: no JSON provided"}), 400

    message = data.get("message", "").strip()
    if not message:
        return jsonify({"success": False, "error": "Empty message"}), 400

    conversation = data.get("conversation", [])
    if not isinstance(conversation, list):
        conversation = []
    conversation = [msg for msg in conversation if isinstance(msg, dict) and "role" in msg and "content" in msg]
    conversation = conversation[-10:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation)
    messages.append({"role": "user", "content": message})

    reply, error = call_openrouter(messages)
    if error:
        logger.error(f"OpenRouter call failed: {error}")
        return jsonify({"success": False, "error": error}), 503

    return jsonify({"success": True, "reply": reply})

# Health check (optional but useful)
@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})
    
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

# Vercel serverless handler
# This is the entry point Vercel uses
def handler(request):
    return app(request)
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)