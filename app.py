import os
import speech_recognition as sr
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if it's loaded correctly
if not GEMINI_API_KEY:
    raise ValueError("⚠️ ERROR: GEMINI_API_KEY is missing!")

# Initialize Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
best_model = None

for available_model in genai.list_models():
    model_name = available_model.name.lower()
    
    if "gemini" in model_name and "flash" not in model_name and "vision" not in model_name:
        best_model = available_model.name  # Store the best match
        best_model = available_model.name.replace("models/", "")
        break

# Initialize model dynamically
model = genai.GenerativeModel(best_model)
print(f"✅ Using model: {best_model}")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Welcome to the AI Debate Coach Backend 🎤</h1>"

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    """Converts speech to text using Google Speech Recognition."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files["file"]
    file_path = "input.wav"
    audio_file.save(file_path)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return jsonify({"transcription": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError:
        return jsonify({"error": "Speech Recognition API unavailable"}), 500

@app.route("/evaluate-argument", methods=["POST"])
def evaluate_argument():
    """Uses AI to analyze rationality, provide feedback, and improve the argument."""
    data = request.get_json()
    if not data or "text" not in data or "topic" not in data:
        return jsonify({"error": "No text or topic provided"}), 400

    topic = data["topic"]
    argument = data["text"]

    # AI prompt for evaluation
    prompt = f"""
    You are an AI debate coach. The topic of the debate is: "{topic}".

    **1️⃣ Evaluate the argument based on the following criteria:**  
    - **Logical Structure:** Is the argument well-organized? Does it follow a clear progression? If it's already structured well, state that no improvements are necessary.  
    - **Clarity & Coherence:** Is the argument clear and easy to understand? Are there ambiguous or vague points? If it's already clear, explicitly mention that.  
    - **Supporting Evidence:** Does the argument provide strong evidence? If it lacks evidence, suggest improvements. If it's well-supported, state that it's sufficient.  
    - **Potential Counterarguments:**  
      - Identify specific counterarguments that an opposing debater might use.  
      - Provide at least **one concrete example of a counterpoint** phrased as a debate challenge (e.g., “If we allow X, then what stops Y?”).  

    **2️⃣ Assess the rationality of the argument:**  
    - Provide a **rationality score** from **0 (highly emotional) to 1 (highly rational)**.  
    - Explain why the argument was scored that way.  

    **3️⃣ Generate an improved version of the argument** that:  
    - Incorporates the feedback above.  
    - Fixes weaknesses while keeping the argument’s core ideas.  
    - Uses better structure, clarity, and stronger reasoning if necessary.  

    **User's Argument:**  
    {argument}

    **Format your response as follows:**  
    ---
    **Rationality Score:** X.X  
    **Reasoning for Score:** (explanation)  

    **Feedback:**  
    - **Logical Structure:** (comment)  
    - **Clarity & Coherence:** (comment)  
    - **Supporting Evidence:** (comment)  
    - **Potential Counterarguments:**  
      - (general explanation of weaknesses in counterarguments)  
      - **Example Counterpoint:** *"If we allow X, then what stops Y?"*  

    **Improved Argument:**  
    (Provide the improved version of the argument)
    """
    try:
        response = model.generate_content(prompt)

        # Handle blocked responses gracefully
        if not response.parts or not response.text:
            return jsonify({
                "error": "⚠️ AI could not generate a response due to content restrictions. Please rephrase your argument."
            }), 400

        ai_output = response.text.strip()

        # Extract rationality score
        rationality_score_match = re.search(r"\*\*Rationality Score:\*\* (\d+\.\d+|\d+)", ai_output)
        rationality_score = float(rationality_score_match.group(1)) if rationality_score_match else 0.5

        # Extract reason for score
        reasoning_match = re.search(r"\*\*Reasoning for Score:\*\*\s*(.*?)(?=\*\*Feedback:\*\*|\Z)", ai_output, re.DOTALL)
        reason_for_score = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."

        # Extract feedback
        feedback_match = re.search(r"\*\*Feedback:\*\*\s*(.*?)(?=\*\*Improved Argument:\*\*|\Z)", ai_output, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided."

        # Extract improved argument
        improved_argument_match = re.search(r"\*\*Improved Argument:\*\*\s*(.*)", ai_output, re.DOTALL)
        improved_argument = improved_argument_match.group(1).strip() if improved_argument_match else "No improved argument provided."

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "rationality_score": rationality_score,
        "reason_for_score": reason_for_score,
        "feedback": feedback,
        "improved_argument": improved_argument,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)