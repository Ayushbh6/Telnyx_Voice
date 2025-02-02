import os
import uuid
import json
from flask import Flask, request, Response as FlaskResponse, send_from_directory
import openai

# Configure your OpenAI API key (set via environment variable for security)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# A simple in-memory store for conversation context keyed by CallSid
conversations = {}

# Function to call OpenAI Chat API (multi-turn conversation)
def chat_with_openai(conversation_history):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=conversation_history,
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

# Function to convert text to speech via OpenAI TTS API.
# (This example assumes the API returns raw binary audio data.)
def text_to_speech(text):
    # Call the OpenAI audio TTS endpoint (example using model "tts-1" and voice "alloy")
    response = openai.Audio.speech.create(
        model="tts-1",
        input=text,
        voice="alloy"
    )
    # Generate a unique filename and save the binary audio content.
    filename = f"response_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("static", filename)
    os.makedirs("static", exist_ok=True)
    # The actual method to retrieve the audio may differ; here we assume response is raw binary.
    with open(filepath, "wb") as f:
        f.write(response)  
    # Construct a public URL for the saved file.
    public_url = request.host_url.rstrip("/") + "/static/" + filename
    return public_url

@app.route("/webhook", methods=["POST"])
def webhook():
    # Parse the incoming Telnyx webhook.
    # Telnyx sends various parameters; we assume the transcription text is passed in "TranscriptionText"
    data = request.form if request.form else request.json or {}
    call_sid = data.get("CallSid", "unknown")
    transcription = data.get("TranscriptionText", "").strip()
    
    # Initialize conversation history if needed.
    if call_sid not in conversations:
        # Start with a system prompt defining the assistant’s role.
        conversations[call_sid] = [{"role": "system", "content": "You are an AI voice assistant."}]
    
    # If caller’s speech was transcribed, add it as a user message.
    if transcription:
        conversations[call_sid].append({"role": "user", "content": transcription})
    
    # Call OpenAI’s Chat API to get the assistant’s response.
    ai_response = chat_with_openai(conversations[call_sid])
    conversations[call_sid].append({"role": "assistant", "content": ai_response})
    
    # Use OpenAI’s TTS API to convert the assistant response into speech.
    audio_url = text_to_speech(ai_response)
    
    # Return a TeXML document that plays the generated audio and then records the next utterance.
    texml_response = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{audio_url}</Play>
  <Record action="https://telnyxvoice-production.up.railway.app/webhook" playBeep="true" finishOnKey="#" />
</Response>'''
    
    # Return the TeXML response with the proper content type.
    return FlaskResponse(texml_response, mimetype="application/xml")

# Endpoint to serve static audio files.
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    # Run the Flask server (use a production server in a real deployment)
    app.run(host="0.0.0.0", port=5000)
