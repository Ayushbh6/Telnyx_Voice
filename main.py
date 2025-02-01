import os
import json
from openai import OpenAI
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Telnyx TeXML Server is running!"}

@app.post("/inbound")
async def incoming_call(request: Request):
    print("Incoming call received")
    headers = request.headers
    
    texml_path = os.path.join(os.path.dirname(__file__), 'texml.xml')
    
    try:
        with open(texml_path, 'r') as file:
            texml_response = file.read()
        texml_response = texml_response.replace("{host}", headers.get("host"))
        print(f"TeXML Response: {texml_response}")
    except FileNotFoundError:
        print(f"File not found at: {texml_path}")
        return PlainTextResponse("TeXML file not found", status_code=500)
    
    return PlainTextResponse(texml_response, media_type="text/xml")

@app.post("/process-speech")
async def process_speech(request: Request):
    # Get the speech transcription from Telnyx
    data = await request.json()
    transcription = data.get('speech', {}).get('text', '')
    print(f"Received transcription: {transcription}")
    
    if not transcription:
        return generate_texml_response("I couldn't hear anything. Could you please try again?")
    
    try:
        # Process with OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and friendly AI assistant. Keep responses concise and natural."},
                {"role": "user", "content": transcription}
            ],
            max_tokens=150
        )
        
        # Get AI response
        ai_response = response.choices[0].message.content
        print(f"AI Response: {ai_response}")
        
        # Generate TeXML response with text-to-speech
        return generate_texml_response(ai_response)
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return generate_texml_response("I'm sorry, I encountered an error. Please try again.")

def generate_texml_response(text):
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say>{text}</Say>
        <Gather input="speech" transcribe="true" language="en-US" speechEndThreshold="2000" speechTimeout="auto"/>
        <Redirect>process-speech</Redirect>
    </Response>
    """
    return PlainTextResponse(texml, media_type="text/xml")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)