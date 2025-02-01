import os
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import openai

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    logger.error('Missing OpenAI API key. Please set it in the .env file.')
    exit(1)

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Constants
SYSTEM_MESSAGE = (
    'You are a helpful and bubbly AI assistant who loves to chat about anything '
    'the user is interested about and is prepared to offer them facts. '
    'Keep responses concise and conversational.'
)

app = FastAPI()

async def get_ai_response(user_input: str) -> str:
    """Get response from OpenAI."""
    try:
        response = await openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return "I apologize, but I'm having trouble processing your request right now."

@app.post("/transcribe")
async def handle_transcription(request: Request):
    """Handle the transcribed speech and return TeXML with AI response."""
    try:
        # Get the transcribed text from Telnyx's Gather
        form_data = await request.form()
        transcription = form_data.get('speech_result', '')
        logger.info(f"Received transcription: {transcription}")

        if not transcription:
            response_text = "I didn't catch that. Could you please try again?"
        else:
            # Get AI response
            response_text = await get_ai_response(transcription)
            logger.info(f"AI Response: {response_text}")

        # Create TeXML response with Say and Gather
        texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="female">{response_text}</Say>
    <Gather input="speech" timeout="5" speechTimeout="2" action="/transcribe" method="POST">
        <Say voice="female">What else would you like to know?</Say>
    </Gather>
</Response>"""
        
        return PlainTextResponse(content=texml, media_type="text/xml")

    except Exception as e:
        logger.error(f"Error in transcribe handler: {e}")
        error_texml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="female">I'm sorry, but I encountered an error. Please try again later.</Say>
    <Hangup/>
</Response>"""
        return PlainTextResponse(content=error_texml, media_type="text/xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)