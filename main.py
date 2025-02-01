import os
import asyncio
import json
import base64
import logging
from typing import Dict, Any

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    logger.error('Missing OpenAI API key. Please set it in the .env file.')
    exit(1)

# Constants
SYSTEM_MESSAGE = (
    'You are a helpful and bubbly AI assistant who loves to chat about anything '
    'the user is interested about and is prepared to offer them facts.'
)
VOICE = 'alloy'
PORT = int(os.getenv('PORT', 8080))

class RealtimeSession:
    def __init__(self, websocket: WebSocket):
        self.client_ws = websocket
        self.openai_ws = None
        self.audio_buffer = b''
        # Used to accumulate transcript deltas from OpenAI
        self.current_ai_transcript = ""
        # Maintain conversation history with messages of the form:
        # { "role": "user"|"assistant", "content": "..." }
        self.conversation_history = []
        
        # Session configuration (this is sent on connection/reconnection)
        self.session_config = {
            "modalities": ["audio", "text"],
            "instructions": SYSTEM_MESSAGE,
            "voice": VOICE,
            "input_audio_format": "g711_ulaw",  # Telnyx uses g711_ulaw
            "output_audio_format": "g711_ulaw",
            "turn_detection": {"type": "server_vad"},
            "temperature": 0.7
        }

    async def connect_to_openai(self):
        """Establish connection to OpenAI's Realtime API and configure the session, including conversation history."""
        try:
            self.openai_ws = await websockets.connect(
                'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            logger.info("Connected to OpenAI Realtime API")
            
            # Merge conversation history into the session config if available.
            session_payload = self.session_config.copy()
            if self.conversation_history:
                session_payload["history"] = self.conversation_history
            
            # Update session configuration with instructions and context
            await self.send_openai_event({
                "type": "session.update",
                "session": session_payload
            })
            
            # Start a new conversation turn.
            await self.send_openai_event({"type": "response.create"})
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            raise

    async def send_openai_event(self, event: Dict[str, Any]):
        """Send event to OpenAI; if the connection is closed then reconnect automatically."""
        if not self.openai_ws or self.openai_ws.closed:
            logger.info("OpenAI websocket is closed, reconnecting...")
            await self.connect_to_openai()
        if self.openai_ws:
            await self.openai_ws.send(json.dumps(event))
            logger.debug(f"Sent event to OpenAI: {event.get('type')}")

    async def handle_openai_response(self):
        """Handle and forward OpenAI responses, as well as update conversation history."""
        try:
            async for message in self.openai_ws:
                response = json.loads(message)
                event_type = response.get("type")
                
                if event_type.startswith("response.audio_transcript.delta"):
                    # Aggregate transcript deltas.
                    delta_text = response.get("delta", "")
                    self.current_ai_transcript += delta_text
                    await self.client_ws.send_json({
                        "event": "transcript_delta",
                        "text": delta_text
                    })
                elif event_type == "response.audio_transcript.done":
                    # When transcript is complete, add it to conversation history.
                    final_transcript = self.current_ai_transcript
                    self.current_ai_transcript = ""  # Reset for next turn.
                    if final_transcript.strip():
                        self.conversation_history.append({"role": "assistant", "content": final_transcript})
                    await self.client_ws.send_json({
                        "event": "transcript_done",
                        "text": final_transcript
                    })
                elif event_type == "response.audio.delta":
                    # Forward audio chunks directly to Telnyx client.
                    if response.get("delta"):
                        await self.client_ws.send_json({
                            "event": "media",
                            "media": {
                                "payload": response["delta"]
                            }
                        })
                elif event_type == "response.done":
                    # Turn completed — notify the client so that a new turn can start.
                    await self.client_ws.send_json({"event": "turn_done"})
                    logger.info("Turn completed, ready for next interaction")
                    # Don't break here - let the connection close naturally and reconnect
                elif event_type == "error":
                    logger.error(f"OpenAI error: {response}")
                else:
                    logger.debug(f"Received OpenAI event: {event_type}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("OpenAI connection closed normally, ready for next turn")
        except Exception as e:
            logger.error(f"Error handling OpenAI response: {e}")
            # Don't raise here - let the main loop handle reconnection

    async def handle_client_message(self, message: Dict[str, Any]):
        """Process incoming messages from the Telnyx client."""
        event_type = message.get("event")
        
        if event_type == "media":
            # Forward audio data to OpenAI.
            await self.send_openai_event(message)
        elif event_type == "start":
            logger.info(f"Stream started: {message.get('stream_id')}")
            # You may want to set up per-turn state here. For example, you could clear
            # any pending user transcript, if implementing client-side ASR.
        elif event_type == "user.text":
            # Optionally handle text from the user (if provided) and add to conversation history.
            # This is useful if the client sends a text message alongside or instead of audio.
            user_text = message.get("text", "")
            if user_text.strip():
                self.conversation_history.append({"role": "user", "content": user_text})
            await self.send_openai_event(message)
        else:
            logger.debug(f"Received client event: {event_type}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Telnyx Media Stream Server is running!"}

@app.post("/inbound")
async def incoming_call(request: Request):
    logger.info("Incoming call received")
    headers = request.headers
    
    texml_path = os.path.join(os.path.dirname(__file__), 'texml.xml')
    
    try:
        with open(texml_path, 'r') as file:
            texml_response = file.read()
        texml_response = texml_response.replace("{host}", headers.get("host"))
        logger.debug(f"TeXML Response: {texml_response}")
    except FileNotFoundError:
        logger.error(f"File not found at: {texml_path}")
        return PlainTextResponse("TeXML file not found", status_code=500)
    
    return PlainTextResponse(texml_response, media_type="text/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    session = RealtimeSession(websocket)
    openai_response_task = None

    try:
        while True:
            # If there's no active OpenAI connection or it's closed, reconnect
            if not session.openai_ws or session.openai_ws.closed:
                logger.info("Establishing/re-establishing OpenAI connection...")
                await session.connect_to_openai()
                if openai_response_task:
                    openai_response_task.cancel()
                openai_response_task = asyncio.create_task(session.handle_openai_response())
            
            # Wait for a message from the Telnyx client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await session.handle_client_message(message)
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing client message: {e}")
                # Don't break here - continue listening for more messages
                continue
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if openai_response_task:
            openai_response_task.cancel()
        if session.openai_ws and not session.openai_ws.closed:
            await session.openai_ws.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)