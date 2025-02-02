import os
import asyncio
import json
import audioop
import logging
import wave  # Added missing import
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from openai import OpenAI
import webrtcvad
from elevenlabs.client import ElevenLabs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration - PUT YOUR ACTUAL SYSTEM MESSAGE HERE
SYSTEM_MESSAGE = """You are a helpful assistant."""
SAMPLE_RATE = 8000
FRAME_DURATION = 30  # ms
VAD_AGGRESSIVENESS = 3
SILENCE_THRESHOLD = 600  # ms of silence to consider speech ended
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # ElevenLabs voice ID
PORT = int(os.getenv('PORT', 8080))

# Initialize clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceActivityDetector:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frame_size = int(SAMPLE_RATE * FRAME_DURATION / 1000) * 2
        self.silence_frames_needed = int(SILENCE_THRESHOLD / FRAME_DURATION)
        self.reset()
        
    def reset(self):
        self.audio_buffer = bytearray()
        self.silence_frames_count = 0
        self.speech_detected = False

    def process_frame(self, frame: bytes) -> bool:
        if len(frame) < self.frame_size:
            return False
            
        if self.vad.is_speech(frame, SAMPLE_RATE):
            self.audio_buffer.extend(frame)
            self.silence_frames_count = 0
            self.speech_detected = True
            return False
            
        if self.speech_detected:
            self.silence_frames_count += 1
            if self.silence_frames_count >= self.silence_frames_needed:
                return True  # Speech ended
                
        return False

def ulaw_to_pcm(ulaw_data: bytes) -> bytes:
    return audioop.ulaw2lin(ulaw_data, 2)

def pcm_to_ulaw(pcm_data: bytes) -> bytes:
    return audioop.lin2ulaw(pcm_data, 2)

async def process_conversation_turn(audio_buffer: bytes, websocket: WebSocket):
    logger.info("Starting conversation turn processing")
    
    try:
        # Convert to WAV format for Whisper
        with BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_buffer)
            wav_buffer.seek(0)

            # Transcribe with Whisper
            transcript = openai_client.audio.transcriptions.create(
                file=("audio.wav", wav_buffer.read(), "audio/wav"),
                model="whisper-1"
            )
            logger.info(f"Transcription: {transcript.text}")

        # Generate response
        chat_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": transcript.text}
            ],
            temperature=0.7,
            max_tokens=150
        )
        response_text = chat_response.choices[0].message.content
        logger.info(f"Generated response: {response_text}")

        # Generate TTS audio (streaming)
        tts_response = elevenlabs_client.generate(
            text=response_text,
            voice=VOICE_ID,
            model="eleven_turbo_v2_5",
            stream=True
        )

        # Stream audio back in chunks
        async for chunk in tts_response:
            if chunk:
                pcm_audio = ulaw_to_pcm(chunk)
                ulaw_chunk = pcm_to_ulaw(pcm_audio)
                await websocket.send_json({
                    "event": "media",
                    "media": {"payload": ulaw_chunk}
                })

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
    finally:
        logger.info("Conversation turn completed")

@app.get("/")
async def root():
    return {"message": "AI by DNA Call Center"}

@app.post("/inbound")
async def incoming_call(request: Request):
    host = request.headers.get('host', 'telnyxvoice-production.up.railway.app')
    texml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Start>
            <Stream url="wss://{host}/media-stream"/>
        </Start>
    </Response>"""
    return PlainTextResponse(texml_response, media_type="text/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    vad = VoiceActivityDetector()
    processing_lock = asyncio.Lock()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("event") == "media":
                pcm_data = ulaw_to_pcm(message["media"]["payload"])
                
                # Process audio frames
                while len(pcm_data) >= vad.frame_size:
                    frame = pcm_data[:vad.frame_size]
                    pcm_data = pcm_data[vad.frame_size:]
                    
                    if vad.process_frame(frame):
                        if vad.audio_buffer and not processing_lock.locked():
                            async with processing_lock:
                                audio_buffer = bytes(vad.audio_buffer)
                                vad.reset()
                                await process_conversation_turn(audio_buffer, websocket)
            
            elif message.get("event") == "stop":
                logger.info("Call ended")
                break

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("Media stream handler closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)