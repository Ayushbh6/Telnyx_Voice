import os
import asyncio
import json
import tempfile
import base64
import logging
import time
import subprocess
import io
import wave
import struct
import array
from typing import List, Dict, Any, Optional, Tuple

import websockets
import webrtcvad
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from elevenlabs.client import ElevenLabs
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("telnyx-voice-bot")

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

if not OPENAI_API_KEY:
    logger.error('Missing OpenAI API key. Please set it in the .env file.')
    exit(1)

if not ELEVENLABS_API_KEY:
    logger.error('Missing ElevenLabs API key. Please set it in the .env file.')
    exit(1)

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Initialize WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # Aggressiveness level (0-3), 3 is most aggressive

# Company information (Replace with your full AI by DNA text)
AI_by_DNA = """We empower organizations with Agentic AI
AI by DNA is an Artificial Intelligence transformation
agency that supports ambitious organizations to scale
their AI and Data capabilities, augmenting efficiency,
performance and growth.
We are your trusted partner to guide your AI transformation. We are neutral and will
compose the best solution to your needs. If it does not exist, we will build it for you.
Conversational Agent
Pharmaceutical Care
“AI by DNA has revolutionized our pictograms' way of communicating information to
patients, by transforming it to a natural language conversation experience for them!!!”
Sophia Demagou (Piktocare | GET2WORK)
Conversational Agents
Conversations are the new markets. AI-driven Conversational agents provide personalized,
real-time and in-depth interaction.
Unveil new AI by Chat opportunities: AI by Chat With reactively support and anytime recommend your product & services . you can proactively promote,
Gain and retain
customers.
Unlock the power of AI by Phone: Human-like AI Voice Assistants are radically changing
the way businesses use their phone lines. Improve productivity, efficiency and customer
satisfaction.
Scale your potential with AI by Clone: Such a video-based agent, showcasing a human-like
avatar, converse with people inside a specific knowledge context, via audio and in multiple
languages. Transform your customers' experience.
Knowledge Assistants
In the fast-paced environment of today, quick access to accurate information and reliable
action taking are key to enhance efficiency.
Unique Internal Data Retrieval Focus: AI by DNA assistants focus on internal information
retrieval, tapping seamlessly into complex sources, to generate your own knowledge based
search engines.
This means secure. procedures etc.
streamlined information and work flows, i.e. more accurate, efficient and
Improve efficiency on regulatory & compliance, inventory management, operating
Decision Engines
Data Processing & Predictive Analytics for Insightful Solutions: drives innovation, AI by DNA insights from complex data sets. efficiency.
In an era where data
data analysis capabilities empower you to derive actionable
Enhance decision-making, operational and commercial
Real-time data driven decision making: AI by DNA insights that help real-time resource allocation tools set delivers evidence-based
optimization, comprehensive revenue management,
dynamic pricing.
"Looking to harness the power of advanced language models with a foundation in data-
driven insights? Need a custom agent with predictive analytics, retrieval capabilities,
and seamless integration with vector stores and other tools? We are here to design,
develop, and deploy tailor-made solutions that meet your specific business objectives.
Let us turn your data and context into actionable intelligence."
George Kotzamanis, Co-Founder | Chief Operating Officer
Get in touch with "AI by DNA" today.
“We live at a time of massive tech disruption in almost all areas of work and life. We were
getting prepared for this, we are working on it, but now is the time to focus on what we might
accomplish for you.” - Kostas Varsamos, Co-Founder | CEO
Offices: Greece (Athens) | Germany (Frankfurt) - Email: contact@aibydna.com
"""


# Constants
SYSTEM_MESSAGE = f"""
You are the personal call centre agent for AI by DNA. Here is complete information about AI by DNA:
{AI_by_DNA}

CRITICAL RESPONSE RULES:
1. MUST keep responses to 2-3 short sentences maximum
2. NEVER explain or give background information
3. Answer directly and briefly
4. If asked about services, mention only ONE relevant service
5. End response immediately after answering the core question

Use English as default language. You are also proficient in Greek.
"""

# Voice settings - Fill in with your preferred voice ID
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', "IvVXvLXxiX8iXKrLlER8")
ELEVENLABS_MODEL_ID = os.getenv('ELEVENLABS_MODEL_ID', "eleven_flash_v2_5")

# Audio settings
WHISPER_MODEL = os.getenv('WHISPER_MODEL', "whisper-1")
CHAT_MODEL = os.getenv('CHAT_MODEL', "gpt-4o")
PORT = int(os.getenv('PORT', 8080))

# VAD settings
FRAME_DURATION_MS = 30  # 30ms frames for WebRTC VAD
SILENCE_THRESHOLD_MS = 800  # Consider silence after 800ms
SPEECH_THRESHOLD_MS = 100  # Consider speech after 100ms
SAMPLE_RATE = 8000  # 8kHz for ulaw audio from Telnyx

# Call state class
class CallState:
    def __init__(self):
        self.conversation_history = []
        self.is_bot_speaking = False
        self.is_processing = False
        self.should_interrupt = False
        self.audio_buffer = bytearray()
        self.vad_buffer = []
        self.last_voice_activity = 0
        self.is_speech_active = False
        self.speech_chunks = []
        self.active_tasks = set()
        
    def reset_speech_detection(self):
        self.speech_chunks = []
        self.is_speech_active = False
        self.last_voice_activity = time.time()
        
    def add_conversation_item(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        # Limit history to last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

# Voice Activity Detection utilities
def ulaw_to_pcm(ulaw_data):
    """Convert G.711 ulaw data to PCM."""
    pcm_data = array.array('h')
    
    for byte in ulaw_data:
        # Flip all bits
        byte = ~byte & 0xFF
        
        # Extract sign bit
        sign = 1 if (byte & 0x80) else -1
        
        # Extract and adjust mantissa
        mantissa = (((byte & 0x0F) << 3) + 0x84) << ((byte & 0x70) >> 4)
        
        # Adjust for sign bit set
        if sign == -1:
            mantissa = -mantissa
            
        pcm_data.append(mantissa)
    
    return pcm_data

def prepare_audio_for_vad(ulaw_data, sample_rate=8000):
    """Convert and prepare audio data for WebRTC VAD."""
    # Convert ulaw to PCM
    pcm_data = ulaw_to_pcm(ulaw_data)
    
    # Convert to 16-bit PCM
    pcm_bytes = pcm_data.tobytes()
    
    # Pack into wave frames
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 2 bytes for 16-bit PCM
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        
        wav_buffer.seek(0)
        with wave.open(wav_buffer, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
    
    return frames

def chunk_audio_for_vad(audio_data, frame_duration_ms=30, sample_rate=8000):
    """Split audio into chunks suitable for WebRTC VAD."""
    bytes_per_sample = 2  # 16-bit PCM is 2 bytes per sample
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    bytes_per_frame = samples_per_frame * bytes_per_sample
    
    frames = []
    for i in range(0, len(audio_data), bytes_per_frame):
        frame = audio_data[i:i+bytes_per_frame]
        # Ensure we have a full frame
        if len(frame) == bytes_per_frame:
            frames.append(frame)
    
    return frames

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

# Health check endpoint for monitoring
@app.get("/health")
async def health_check():
    # Check if OpenAI API is responsive
    try:
        openai_client.models.list()
        openai_status = "ok"
    except Exception as e:
        openai_status = f"error: {str(e)}"
    
    # Check if ElevenLabs API is responsive
    try:
        elevenlabs_client.voices.list()
        elevenlabs_status = "ok"
    except Exception as e:
        elevenlabs_status = f"error: {str(e)}"
    
    return {
        "status": "running",
        "openai": openai_status,
        "elevenlabs": elevenlabs_status,
        "timestamp": time.time()
    }

# Route for Telnyx to handle incoming calls
@app.post("/inbound")
async def incoming_call(request: Request):
    logger.info("Incoming call received")
    headers = request.headers
    
    # Construct the correct relative path to the texml.xml file
    texml_path = os.path.join(os.path.dirname(__file__), 'texml.xml')
    
    try:
        with open(texml_path, 'r') as file:
            texml_response = file.read()
        
        # Use the host from the request headers or a fallback URL
        host = headers.get("host") or os.getenv("APP_URL", "telnyxvoice-production.up.railway.app")
        texml_response = texml_response.replace("{host}", host)
        logger.info(f"TeXML Response generated with host: {host}")
    except FileNotFoundError:
        logger.error(f"File not found at: {texml_path}")
        return PlainTextResponse("TeXML file not found", status_code=500)
    
    return PlainTextResponse(texml_response, media_type="text/xml")

# 1. Speech-to-Text (STT) function
async def transcribe_audio(audio_data):
    """Convert audio to text using OpenAI Whisper"""
    try:
        start_time = time.time()
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        # Transcribe audio using OpenAI Whisper
        with open(temp_audio_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file
            )
        
        # Clean up temporary file
        os.unlink(temp_audio_path)
        
        duration = time.time() - start_time
        logger.info(f"Transcription completed in {duration:.2f}s: {transcription.text}")
        
        return transcription.text
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return ""

# 2. Text-to-Text (TTT) function
async def process_text(text, conversation_history):
    """Process text using OpenAI chat completion"""
    try:
        start_time = time.time()
        
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        
        # Add conversation history
        for message in conversation_history:
            messages.append(message)
        
        # Add user's new message
        messages.append({"role": "user", "content": text})
        
        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=150,  # Keep responses short
            temperature=0.7
        )
        
        response_text = completion.choices[0].message.content
        
        duration = time.time() - start_time
        logger.info(f"AI response generated in {duration:.2f}s: {response_text}")
        
        return response_text
    except Exception as e:
        logger.error(f"Error in text processing: {e}")
        return "I apologize, but I couldn't process your request. Could you please try again?"

# 3. Text-to-Speech (TTS) function with audio format conversion
async def synthesize_speech(text):
    """Convert text to speech using ElevenLabs and convert to ulaw format"""
    try:
        start_time = time.time()
        
        # Generate MP3 audio with ElevenLabs
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            output_format="mp3_44100_128",  # We'll convert this to ulaw
        )
        
        # Handle the generator case
        if hasattr(audio, '__iter__') or hasattr(audio, '__next__'):
            audio_bytes = b''
            for chunk in audio:
                audio_bytes += chunk
            audio = audio_bytes
        
        # Save MP3 to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio)
            temp_audio_path = temp_audio.name
        
        # Convert to WAV (intermediate step)
        wav_path = temp_audio_path.replace(".mp3", ".wav")
        
        # Load MP3 using pydub
        audio_segment = AudioSegment.from_mp3(temp_audio_path)
        
        # Convert to mono if stereo
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Convert to 8kHz sample rate for ulaw
        audio_segment = audio_segment.set_frame_rate(8000)
        
        # Export as WAV
        audio_segment.export(wav_path, format="wav")
        
        # Convert WAV to ulaw using ffmpeg
        ulaw_path = wav_path.replace(".wav", ".ulaw")
        try:
            subprocess.run([
                "ffmpeg", "-y", 
                "-i", wav_path, 
                "-ar", "8000", 
                "-ac", "1", 
                "-acodec", "pcm_mulaw", 
                ulaw_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode('utf-8')}")
            # Fallback to pydub method if ffmpeg fails
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="wav", parameters=["-acodec", "pcm_mulaw"])
            buffer.seek(0)
            with wave.open(buffer, 'rb') as wav:
                ulaw_data = wav.readframes(wav.getnframes())
                with open(ulaw_path, "wb") as ulaw_file:
                    ulaw_file.write(ulaw_data)
        
        # Read the ulaw file
        with open(ulaw_path, "rb") as ulaw_file:
            ulaw_data = ulaw_file.read()
        
        # Clean up temporary files
        for path in [temp_audio_path, wav_path, ulaw_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {path}: {e}")
        
        # Encode as base64
        encoded_audio = base64.b64encode(ulaw_data).decode('utf-8')
        
        duration = time.time() - start_time
        logger.info(f"Speech synthesis and conversion completed in {duration:.2f}s")
        
        return encoded_audio
    except Exception as e:
        logger.error(f"Error in speech synthesis: {e}", exc_info=True)
        # Fallback method in case of error
        try:
            # Try direct pydub method
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio))
            audio_segment = audio_segment.set_channels(1).set_frame_rate(8000)
            
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="wav", parameters=["-acodec", "pcm_mulaw"])
            buffer.seek(0)
            
            with wave.open(buffer, 'rb') as wav:
                ulaw_data = wav.readframes(wav.getnframes())
            
            encoded_audio = base64.b64encode(ulaw_data).decode('utf-8')
            logger.info("Used fallback audio conversion method")
            return encoded_audio
        except Exception as nested_e:
            logger.error(f"Fallback method also failed: {nested_e}")
            return None

# Process speech function
async def process_speech(call_state, websocket, content):
    # Mark as processing to prevent multiple parallel processing
    if call_state.is_processing:
        logger.info("Already processing speech, skipping")
        return
    
    call_state.is_processing = True
    
    try:
        # 1. STT - Transcribe speech to text
        text = await transcribe_audio(content)
        
        if not text.strip():
            logger.info("No speech detected in the audio")
            call_state.is_processing = False
            return
        
        # Add to conversation history
        call_state.add_conversation_item("user", text)
        
        # 2. TTT - Process with AI
        response_text = await process_text(text, call_state.conversation_history)
        
        # Add AI response to conversation history
        call_state.add_conversation_item("assistant", response_text)
        
        # 3. TTS - Convert response to audio
        if call_state.should_interrupt:
            logger.info("Processing interrupted by new speech, skipping response")
            call_state.should_interrupt = False
            call_state.is_processing = False
            return
        
        # Mark bot as speaking
        call_state.is_bot_speaking = True
        
        audio_response = await synthesize_speech(response_text)
        
        if audio_response and not call_state.should_interrupt:
            # Send audio back to caller
            audio_message = {
                "event": "media",
                "media": {
                    "payload": audio_response
                }
            }
            await websocket.send_json(audio_message)
        
        # Reset speaking status
        call_state.is_bot_speaking = False
        
    except Exception as e:
        logger.error(f"Error processing speech: {e}")
    finally:
        # Always reset processing state
        call_state.is_processing = False

# Voice activity detection coroutine
async def detect_voice_activity(call_state, audio_chunk):
    try:
        # Decode base64 audio chunk
        audio_data = base64.b64decode(audio_chunk)
        
        # Add to audio buffer
        call_state.audio_buffer.extend(audio_data)
        
        # Prepare audio for VAD
        pcm_audio = prepare_audio_for_vad(audio_data)
        frames = chunk_audio_for_vad(pcm_audio, FRAME_DURATION_MS, SAMPLE_RATE)
        
        is_speech = False
        for frame in frames:
            try:
                # Check if frame contains speech
                if len(frame) == int(SAMPLE_RATE * FRAME_DURATION_MS / 1000 * 2):  # 2 bytes per sample
                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    call_state.vad_buffer.append(is_speech)
                    
                    # Keep only the last 20 frames (about 600ms)
                    if len(call_state.vad_buffer) > 20:
                        call_state.vad_buffer.pop(0)
            except Exception as e:
                logger.error(f"VAD error: {e}")
        
        # Calculate speech ratio in the buffer
        speech_ratio = sum(call_state.vad_buffer) / max(1, len(call_state.vad_buffer))
        
        current_time = time.time()
        
        # Detect speech start
        if not call_state.is_speech_active and speech_ratio > 0.5:
            call_state.is_speech_active = True
            call_state.last_voice_activity = current_time
            logger.info("Speech detected")
            
            # If bot is speaking, trigger interruption
            if call_state.is_bot_speaking:
                call_state.should_interrupt = True
                logger.info("User interruption detected")
        
        # If speech is active, add audio chunk to collection
        if call_state.is_speech_active:
            call_state.speech_chunks.append(audio_data)
            
            # Reset activity timer if speech is detected
            if speech_ratio > 0.3:
                call_state.last_voice_activity = current_time
        
        # Detect end of speech (silence for SILENCE_THRESHOLD_MS)
        if call_state.is_speech_active and (current_time - call_state.last_voice_activity) * 1000 > SILENCE_THRESHOLD_MS:
            logger.info("End of speech detected")
            
            # Process the collected speech if we have enough data
            if len(call_state.speech_chunks) > 5:  # Minimum number of chunks to process
                # Combine speech chunks
                speech_content = b''.join(call_state.speech_chunks)
                
                # Reset speech detection
                call_state.reset_speech_detection()
                
                # Process the speech in a separate task
                return speech_content
            else:
                # Not enough speech data, reset
                call_state.reset_speech_detection()
                logger.info("Speech too short, ignoring")
        
        return None
    except Exception as e:
        logger.error(f"Error in voice activity detection: {e}")
        return None

# WebSocket route for media-stream
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket, background_tasks: BackgroundTasks):
    await websocket.accept()
    logger.info("Client connected")
    
    # Initialize call state
    call_state = CallState()
    
    try:
        # Send welcome message
        welcome_text = "Hello, this is AI by DNA. How can I assist you today?"
        
        # Add initial message to conversation history
        call_state.add_conversation_item("assistant", welcome_text)
        
        # Synthesize welcome speech
        welcome_audio = await synthesize_speech(welcome_text)
        
        if welcome_audio:
            call_state.is_bot_speaking = True
            audio_message = {
                "event": "media",
                "media": {
                    "payload": welcome_audio
                }
            }
            await websocket.send_json(audio_message)
            call_state.is_bot_speaking = False
        
        # Process incoming messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            event_type = message.get("event")
            
            if event_type == "media":
                # Process audio chunk for voice activity detection
                audio_payload = message["media"]["payload"]
                speech_content = await detect_voice_activity(call_state, audio_payload)
                
                if speech_content and not call_state.is_processing:
                    # Process detected speech in background task
                    task = asyncio.create_task(
                        process_speech(call_state, websocket, speech_content)
                    )
                    call_state.active_tasks.add(task)
                    task.add_done_callback(lambda t: call_state.active_tasks.remove(t))
            
            elif event_type == "start":
                stream_id = message.get("stream_id")
                logger.info(f"Incoming stream started: {stream_id}")
            
            elif event_type == "stop":
                logger.info("Stream stopped")
                break
            
            elif event_type == "connected":
                logger.info("Received non-media event: connected")
            
            else:
                logger.info(f"Received non-media event: {event_type}")
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cancel any active tasks
        for task in call_state.active_tasks:
            try:
                task.cancel()
            except Exception as e:
                logger.error(f"Error cancelling task: {e}")

# Start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)