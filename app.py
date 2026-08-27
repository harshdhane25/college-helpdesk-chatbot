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

# Multiple models are defined directly in code.
# The chatbot will try them in this order if one fails.
MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "inclusionai/ling-3.0-flash:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

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
    if (
        not OPENROUTER_API_KEY
        or OPENROUTER_API_KEY.startswith("sk-or-v1-PASTE")
        or "PASTE_YOUR" in OPENROUTER_API_KEY
    ):
        return False, "OpenRouter API key is missing. Set OPENROUTER_API_KEY in Vercel Environment Variables."

    return True, ""


import time


def call_openrouter(messages, temperature=0.3, max_tokens=800, retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://college-helpdesk-chatbot-rouge.vercel.app",
        "X-Title": "College Helpdesk Chatbot",
    }

    # Store the last error so that if every model fails,
    # we can return a useful error message.
    last_error = None

    # Try every model in the MODELS list.
    for model_index, model in enumerate(MODELS, start=1):

        logger.info(
            f"Trying model {model_index}/{len(MODELS)}: {model}"
        )

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(retries):
            try:
                response = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    if "choices" in data and len(data["choices"]) > 0:
                        reply = data["choices"][0].get("message", {}).get("content")

                        if reply:
                            logger.info(
                                f"Successfully received response from model: {model}"
                            )
                            return reply.strip(), None

                        last_error = f"{model}: The AI returned an empty response."

                        logger.warning(
                            f"Model {model} returned an empty response. Trying next model."
                        )

                        # Empty response -> try next model
                        break

                    if "error" in data:
                        error_message = data["error"].get(
                            "message",
                            "Unknown AI service error"
                        )

                        last_error = f"{model}: {error_message}"

                        logger.warning(
                            f"Model {model} returned an error: {error_message}"
                        )

                        # Try next model
                        break

                    last_error = f"{model}: Unexpected response from AI service."

                    logger.warning(
                        f"Unexpected response from model {model}. Trying next model."
                    )

                    # Try next model
                    break

                # Handle rate limits and server errors with retry
                if response.status_code in (429, 500, 502, 503, 504):

                    wait = 2 ** attempt

                    logger.warning(
                        f"Model {model} attempt {attempt + 1}/{retries} "
                        f"failed with {response.status_code}. "
                        f"Retrying in {wait}s..."
                    )

                    last_error = (
                        f"{model}: OpenRouter returned status "
                        f"{response.status_code}"
                    )

                    time.sleep(wait)
                    continue

                # Other errors (401, 400, etc.) – don't retry
                try:
                    error_data = response.json()
                    error_message = error_data.get(
                        "error",
                        {}
                    ).get(
                        "message",
                        "Unknown error"
                    )
                except Exception:
                    error_message = response.text

                logger.error(
                    f"OpenRouter error from model {model}: {error_message}"
                )

                last_error = f"{model}: {error_message}"

                # Invalid API key affects every model,
                # so don't waste time trying other models.
                if response.status_code == 401:
                    return None, "Invalid OpenRouter API key."

                # Try the next model for other errors.
                break

            except requests.exceptions.Timeout:

                last_error = (
                    f"{model}: The AI service timed out."
                )

                if attempt < retries - 1:

                    wait = 2 ** attempt

                    logger.warning(
                        f"Timeout on model {model}, "
                        f"attempt {attempt + 1}/{retries}. "
                        f"Retrying in {wait}s..."
                    )

                    time.sleep(wait)
                    continue

                logger.warning(
                    f"Model {model} timed out after {retries} attempts. "
                    f"Trying next model."
                )

                # Try next model
                break

            except requests.exceptions.ConnectionError:

                last_error = (
                    f"{model}: Could not connect to the AI service."
                )

                logger.warning(
                    f"Connection error with model {model}. "
                    f"Trying next model."
                )

                # Try next model
                break

            except Exception as e:

                logger.exception(
                    f"Unexpected error calling OpenRouter with model {model}"
                )

                last_error = (
                    f"{model}: Unexpected error occurred."
                )

                # Try next model
                break

    # All models failed
    logger.error(
        f"All {len(MODELS)} models failed. Last error: {last_error}"
    )

    return None, (
        "The AI service is currently unavailable. "
        "All available models were unable to respond. "
        "Please try again later."
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    key_ok, key_error = validate_api_key()

    if not key_ok:
        return jsonify({
            "success": False,
            "error": key_error
        }), 500

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "error": "Invalid request: no JSON provided"
        }), 400

    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "success": False,
            "error": "Empty message"
        }), 400

    conversation = data.get("conversation", [])

    if not isinstance(conversation, list):
        conversation = []

    conversation = [
        msg
        for msg in conversation
        if isinstance(msg, dict)
        and "role" in msg
        and "content" in msg
    ]

    conversation = conversation[-10:]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(conversation)

    messages.append({
        "role": "user",
        "content": message
    })

    reply, error = call_openrouter(messages)

    if error:
        logger.error(
            f"OpenRouter call failed: {error}"
        )

        return jsonify({
            "success": False,
            "error": error
        }), 503

    return jsonify({
        "success": True,
        "reply": reply
    })


# Health check (optional but useful)
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok"
    })


@app.route('/')
def index():
    return send_from_directory(
        'public',
        'index.html'
    )


# Vercel serverless handler
# This is the entry point Vercel uses
def handler(request):
    return app(request)


if __name__ == "__main__":
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
