import os
import asyncio
import json
import logging
import wave
from io import BytesIO
from contextlib import asynccontextmanager
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from openai import OpenAI
import webrtcvad
from elevenlabs.client import ElevenLabs
import audioop  # Added to support audio format conversions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
SYSTEM_MESSAGE = """You are the personal call centre agent for AI by DNA..."""
SAMPLE_RATE = 8000
FRAME_DURATION = 30
VAD_AGGRESSIVENESS = 3
SILENCE_THRESHOLD = 600
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
PORT = int(os.getenv('PORT', 8080))

# Initialize clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("Starting up...")
    try:
        # Warm up APIs
        openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))
        elevenlabs_client.generate(text="ping", voice=VOICE_ID, stream=True)
    except Exception as e:
        logger.error(f"Startup warmup failed: {e}")
    
    yield
    
    # Shutdown code
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

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

    def process_frame(self, frame: bytes):
        if len(frame) < self.frame_size:
            return False
            
        if self.vad.is_speech(frame, SAMPLE_RATE):
            self.audio_buffer.extend(frame)
            self.silence_frames_count = 0
            self.speech_detected = True
            return False
            
        if self.speech_detected:
            self.silence_frames_count += 1
            return self.silence_frames_count >= self.silence_frames_needed
        return False

def convert_audio(input_data: bytes, in_format: str, out_format: str) -> bytes:
    """Convert between audio formats using raw PCM as intermediate"""
    if in_format == out_format:
        return input_data
        
    # Convert to PCM first
    if in_format == 'ulaw':
        pcm = audioop.ulaw2lin(input_data, 2)
    else:
        pcm = input_data

    # Convert to target format
    if out_format == 'ulaw':
        return audioop.lin2ulaw(pcm, 2)
    return pcm

async def process_conversation_turn(audio_buffer: bytes, websocket: WebSocket):
    try:
        # Convert Telnyx μ-law to PCM
        pcm_audio = convert_audio(audio_buffer, 'ulaw', 'pcm')
        
        # Create WAV file
        with BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm_audio)
            wav_buffer.seek(0)
            
            # Transcribe with Whisper
            transcript = openai_client.audio.transcriptions.create(
                file=("audio.wav", wav_buffer.read(), "audio/wav"),
                model="whisper-1"
            )
            logger.info(f"Transcription: {transcript.text}")

        # Generate response
        chat_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": transcript.text}
            ],
            temperature=0.7,
            max_tokens=150
        )
        response_text = chat_response.choices[0].message.content
        
        # Generate TTS
        tts_response = elevenlabs_client.generate(
            text=response_text,
            voice=VOICE_ID,
            model="eleven_turbo_v2",
            stream=True
        )

        # Stream audio back
        async for chunk in tts_response:
            if chunk:
                ulaw_audio = convert_audio(chunk, 'pcm', 'ulaw')
                await websocket.send_json({
                    "event": "media",
                    "media": {
                        "payload": ulaw_audio.hex(),
                        "type": "audio/wav"  # Telnyx requires this
                    }
                })

    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "AI by DNA Call Center"}

@app.post("/inbound")
async def incoming_call(request: Request):
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Answer>
            <Ringback>tone:ring</Ringback>
        </Answer>
        <Connect>
            <Stream url="wss://telnyxvoice-production.up.railway.app/media-stream"
                   format="wav"
                   track="both"/>
        </Connect>
    </Response>"""
    return PlainTextResponse(texml, media_type="text/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    
    vad = VoiceActivityDetector()
    active = True

    try:
        while active:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg.get("event") == "media":
                raw_audio = bytes.fromhex(msg["media"]["payload"])
                
                # Process audio frames
                remaining = raw_audio
                while len(remaining) >= vad.frame_size:
                    frame = remaining[:vad.frame_size]
                    remaining = remaining[vad.frame_size:]
                    
                    if vad.process_frame(frame):
                        if vad.audio_buffer:
                            await process_conversation_turn(bytes(vad.audio_buffer), websocket)
                            vad.reset()
                            
            elif msg.get("event") == "stop":
                active = False

    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Closing connection")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)