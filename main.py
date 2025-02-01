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
SYSTEM_MESSAGE = 'You are a helpful and bubbly AI assistant who loves to chat about anything the user is interested about and is prepared to offer them facts.'
VOICE = 'alloy'
PORT = int(os.getenv('PORT', 8080))

class RealtimeSession:
    def __init__(self, websocket: WebSocket):
        self.client_ws = websocket
        self.openai_ws = None
        self.audio_buffer = b''
        self.is_active = True
        
        # Session configuration
        self.session_config = {
            "modalities": ["audio", "text"],
            "instructions": SYSTEM_MESSAGE,
            "voice": VOICE,
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": {"type": "server_vad"},
            "temperature": 0.7
        }

    async def connect_to_openai(self):
        """Establish connection to OpenAI's Realtime API"""
        try:
            self.openai_ws = await websockets.connect(
                'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                },
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=60    # Wait up to 60 seconds for pong
            )
            logger.info("Connected to OpenAI Realtime API")
            
            # Configure session
            await self.send_openai_event({
                "type": "session.update",
                "session": self.session_config
            })
            
            # Initialize conversation
            await self.send_openai_event({"type": "response.create"})
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            raise

    async def send_openai_event(self, event: Dict[str, Any]):
        """Send event to OpenAI"""
        if self.openai_ws and not self.openai_ws.closed:
            await self.openai_ws.send(json.dumps(event))
            logger.debug(f"Sent event to OpenAI: {event['type']}")
        else:
            logger.warning("OpenAI WebSocket is closed, attempting to reconnect...")
            await self.connect_to_openai()
            await self.openai_ws.send(json.dumps(event))

    async def heartbeat(self):
        """Keep the connections alive with periodic messages"""
        while self.is_active:
            try:
                if self.openai_ws and not self.openai_ws.closed:
                    await self.send_openai_event({"type": "ping"})
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)  # Wait before retry

    async def handle_openai_response(self):
        """Handle messages from OpenAI"""
        while self.is_active:
            try:
                async for message in self.openai_ws:
                    response = json.loads(message)
                    event_type = response.get("type")
                    
                    if event_type == "response.audio.delta":
                        if response.get("delta"):
                            await self.client_ws.send_json({
                                "event": "media",
                                "media": {
                                    "payload": response["delta"]
                                }
                            })
                    elif event_type == "response.done":
                        logger.debug("Response complete, ready for next interaction")
                    elif event_type == "error":
                        logger.error(f"OpenAI error: {response}")
                    else:
                        logger.debug(f"Received OpenAI event: {event_type}")
                        
            except websockets.ConnectionClosed:
                logger.info("OpenAI connection closed, attempting to reconnect...")
                await asyncio.sleep(1)  # Wait before reconnecting
                await self.connect_to_openai()
            except Exception as e:
                logger.error(f"Error handling OpenAI response: {e}")
                await asyncio.sleep(1)
                if self.openai_ws and not self.openai_ws.closed:
                    await self.openai_ws.close()
                await self.connect_to_openai()

    async def handle_client_message(self, message: Dict[str, Any]):
        """Handle messages from Telnyx client"""
        try:
            event_type = message.get("event")
            
            if event_type == "media":
                # Forward audio data to OpenAI
                if self.openai_ws and not self.openai_ws.closed:
                    await self.send_openai_event(message)
                    # Create new response after receiving user audio
                    await self.send_openai_event({"type": "response.create"})
                else:
                    logger.warning("OpenAI connection lost, reconnecting...")
                    await self.connect_to_openai()
            elif event_type == "start":
                logger.info(f"Stream started: {message.get('stream_id')}")
            else:
                logger.debug(f"Received client event: {event_type}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    async def cleanup(self):
        """Clean up resources"""
        self.is_active = False
        if self.openai_ws and not self.openai_ws.closed:
            await self.openai_ws.close()

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
    
    try:
        # Connect to OpenAI
        await session.connect_to_openai()
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(session.heartbeat())
        
        # Handle messages concurrently
        async def receive_client_messages():
            while session.is_active:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await session.handle_client_message(message)
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error receiving client message: {e}")
                    await asyncio.sleep(1)  # Wait before retry

        # Run all handlers concurrently
        await asyncio.gather(
            session.handle_openai_response(),
            receive_client_messages(),
            return_exceptions=True
        )
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        heartbeat_task.cancel()
        await session.cleanup()
        logger.info("Session cleaned up")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)