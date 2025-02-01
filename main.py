import os
import asyncio
import json
import time
import base64

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print('Missing OpenAI API key. Please set it in the .env file.')
    exit(1)

# Constants
SYSTEM_MESSAGE = 'You are a helpful and bubbly AI assistant who loves to chat about anything the user is interested about and is prepared to offer them facts.'
VOICE = 'alloy'
PORT = int(os.getenv('PORT', 8080))  # Allow dynamic port assignment

# List of Event Types to log to the console. See OpenAI Realtime API Documentation.
LOG_EVENT_TYPES = [
    'response.content.done',
    'rate_limits.updated',
    'response.done',
    'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started',
    'session.created'
]

app = FastAPI()

# Add middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
async def root():
    return {"message": "Telnyx Media Stream Server is running!"}

# Route for telnyx to handle incoming and outgoing calls

@app.post("/inbound")
async def incoming_call(request: Request):
    print("Incoming call received")
    headers = request.headers
    
    # Construct the correct relative path to the texml.xml file
    texml_path = os.path.join(os.path.dirname(__file__), 'texml.xml')
    
    try:
        with open(texml_path, 'r') as file:
            texml_response = file.read()
        texml_response = texml_response.replace("{host}", headers.get("host"))
        print(f"TeXML Response: {texml_response}")  # Log the TeXML response
    except FileNotFoundError:
        print(f"File not found at: {texml_path}")
        return PlainTextResponse("TeXML file not found", status_code=500)
    
    return PlainTextResponse(texml_response, media_type="text/xml")

# WebSocket route for media-stream
@app.websocket("/media-stream")
async def media_stream(fastapi_ws: WebSocket):
    await fastapi_ws.accept()
    print("Client connected")

    # Create an asyncio.Queue to hold media data
    media_queue = asyncio.Queue()

    async def handle_openai_message(openai_ws, message, client_ws):
        try:
            response = json.loads(message)
            if response.get("type") in LOG_EVENT_TYPES:
                print(f"Received event: {response['type']}", response)
            if response.get("type") == "session.updated":
                print("Session updated successfully:", response)
            if response.get("type") == "response.audio.delta" and response.get("delta"):
                print(f"Received audio delta from OpenAI: {len(response['delta'])} bytes")
                # Convert PCM16 to g711_ulaw for Telnyx
                audio_delta = {
                    "event": "media",
                    "media": {
                        "payload": base64.b64encode(response["delta"].encode()).decode(),
                        "timestamp": int(time.time() * 1000),
                        "encoding": "L16", # Keep as L16 as per current logic, can convert to g711 if needed.
                        "channels": 1,
                        "rate": 16000
                    }
                }
                print(f"Sending audio to Telnyx: {len(response['delta'])} bytes")
                await client_ws.send_json(audio_delta)
        except Exception as e:
            print("Error processing OpenAI message:", e, "Raw message:", message)

    async def handle_openai_connection(client_ws, media_q):
        uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        try:
            async with websockets.connect(uri, extra_headers=headers) as openai_ws:
                print("OpenAI WebSocket connected")
                session_update = {
                    "type": "session.update",
                    "session": {
                        "turn_detection": {"type": "server_vad"},
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "voice": VOICE,
                        "instructions": SYSTEM_MESSAGE,
                        "modalities": ["text", "audio"],
                        "temperature": 0.8,
                    }
                }
                print("Sending session update:", json.dumps(session_update))
                await openai_ws.send(json.dumps(session_update))

                async def receive_from_openai():
                    async for message in openai_ws:
                        await handle_openai_message(openai_ws, message, client_ws)

                async def send_to_openai():
                    while True:
                        media = await media_q.get()
                        if media is None:
                            print("No more media to send. Closing send_to_openai task.")
                            break
                        print(f"Sending media to OpenAI: {len(media)} bytes")
                        await openai_ws.send(media)

                receiver_task = asyncio.create_task(receive_from_openai())
                sender_task = asyncio.create_task(send_to_openai())

                done, pending = await asyncio.wait(
                    [receiver_task, sender_task],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                for task in done:
                    if task.exception():
                        raise task.exception()

        except Exception as e:
            print(f"Error in OpenAI WebSocket connection: {e}")

    openai_task = asyncio.create_task(handle_openai_connection(fastapi_ws, media_queue))

    try:
        while True:
            data = await fastapi_ws.receive_text()
            message = json.loads(data)
            event_type = message.get("event")
            if event_type == "media":
                # Extract the media payload and enqueue it
                media_payload = message.get("media", {}).get("payload")
                if media_payload:
                    try:
                        # Decode the media payload from base64
                        media_bytes = base64.b64decode(media_payload)
                        await media_queue.put(media_bytes)
                        print(f"Enqueued media payload: {len(media_bytes)} bytes")
                    except Exception as e:
                        print("Failed to decode media payload:", e)
                else:
                    print("No media payload found in message.")
            elif event_type == "start":
                stream_sid = message["stream_id"]
                print(f"Incoming stream has started: {stream_sid}")
            else:
                print(f"Received non-media event: {event_type}")
    except WebSocketDisconnect:
        print("Telnyx client disconnected.")
    except Exception as e:
        print(f"Error in media_stream: {e}")
    finally:
        await media_queue.put(None)  # Signal the send_to_openai task to exit
        openai_task.cancel()
        print("OpenAI task cancelled.")

# Start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)