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
vad.set_mode(2)  # Aggressiveness level (0-3), 3 is most aggressive

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
"AI by DNA has revolutionized our pictograms' way of communicating information to
patients, by transforming it to a natural language conversation experience for them!!!
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
"We live at a time of massive tech disruption in almost all areas of work and life. We were
getting prepared for this, we are working on it, but now is the time to focus on what we might
accomplish for you." - Kostas Varsamos, Co-Founder | CEO
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
FRAME_DURATION_MS = 20  # 20ms frames for WebRTC VAD (changed from 30ms)
SILENCE_THRESHOLD_MS = 500  # Consider silence after 500ms
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
        
        # Convert ulaw audio to PCM for better transcription
        pcm_data = ulaw_to_pcm_simplified(audio_data)
        
        # Create a proper WAV file with headers
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            
            # Write WAV header and PCM data
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)  # 8kHz
                wav_file.writeframes(pcm_data.tobytes())
        
        # Log the audio file properties for debugging
        logger.info(f"Created WAV file for transcription: {temp_path}")
        analyze_audio_file(temp_path)
        
        # Transcribe audio using OpenAI Whisper
        with open(temp_path, "rb") as audio_file:
            try:
                transcription = openai_client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=audio_file,
                    language="en"  # Specify language for better results
                )
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                duration = time.time() - start_time
                logger.info(f"Transcription completed in {duration:.2f}s: {transcription.text}")
                
                return transcription.text
            except Exception as whisper_error:
                logger.error(f"Whisper transcription error: {whisper_error}")
                
                # If Whisper fails, try with a different format
                logger.info("Trying alternative audio format for transcription")
                
                # Convert to higher quality audio format
                try:
                    # Use ffmpeg to convert to a higher quality format
                    mp3_path = temp_path.replace(".wav", ".mp3")
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-i", temp_path,
                        "-ar", "16000",  # Upsample to 16kHz
                        "-ac", "1",
                        mp3_path
                    ], check=True, capture_output=True)
                    
                    # Try transcription again with the converted file
                    with open(mp3_path, "rb") as mp3_file:
                        transcription = openai_client.audio.transcriptions.create(
                            model=WHISPER_MODEL,
                            file=mp3_file,
                            language="en"
                        )
                    
                    # Clean up temporary files
                    for path in [temp_path, mp3_path]:
                        if os.path.exists(path):
                            os.unlink(path)
                    
                    duration = time.time() - start_time
                    logger.info(f"Alternative transcription completed in {duration:.2f}s: {transcription.text}")
                    
                    return transcription.text
                except Exception as alt_error:
                    logger.error(f"Alternative transcription also failed: {alt_error}")
                    
                    # Clean up any remaining temporary files
                    for path in [temp_path, temp_path.replace(".wav", ".mp3")]:
                        if os.path.exists(path):
                            try:
                                os.unlink(path)
                            except:
                                pass
                    
                    return ""
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}", exc_info=True)
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
        
        # Handle the generator or bytes response
        audio_bytes = b''
        if hasattr(audio, '__iter__') and not isinstance(audio, bytes):
            # If it's a generator, collect all chunks
            for chunk in audio:
                if chunk:  # Make sure chunk is not None
                    audio_bytes += chunk
        elif isinstance(audio, bytes):
            # If it's already bytes, use it directly
            audio_bytes = audio
        else:
            # If it's something else, try to convert it
            audio_bytes = bytes(audio)
        
        if not audio_bytes:
            raise ValueError("No audio data received from ElevenLabs")
        
        # Save MP3 to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        # Load MP3 using pydub
        audio_segment = AudioSegment.from_mp3(temp_audio_path)
        
        # Convert to mono if stereo
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Convert to 8kHz sample rate for ulaw
        audio_segment = audio_segment.set_frame_rate(8000)
        
        # Method 1: Try using ffmpeg with wav format first, then convert to ulaw
        ulaw_data = None
        
        try:
            # Export as temporary WAV file
            wav_path = temp_audio_path.replace(".mp3", ".wav")
            audio_segment.export(wav_path, format="wav")
            
            # Use ffmpeg to convert WAV to raw u-law format (no .ulaw extension)
            raw_path = temp_audio_path.replace(".mp3", ".raw")
            
            subprocess.run([
                "ffmpeg", "-y",
                "-i", wav_path,
                "-ar", "8000",
                "-ac", "1",
                "-acodec", "pcm_mulaw",
                "-f", "mulaw",  # Force mulaw format
                raw_path
            ], check=True, capture_output=True)
            
            # Read the raw u-law data
            with open(raw_path, "rb") as raw_file:
                ulaw_data = raw_file.read()
                
            # Clean up temporary files
            for path in [temp_audio_path, wav_path, raw_path]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {path}: {e}")
                        
        except Exception as e:
            logger.warning(f"First conversion method failed: {e}")
            
            # Method 2: Use pydub's internal export methods as fallback
            try:
                # Clean up any temporary files from failed attempts
                for path in [temp_audio_path, temp_audio_path.replace(".mp3", ".wav"), temp_audio_path.replace(".mp3", ".raw")]:
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                        except Exception:
                            pass
                
                # Create a new audio segment with the right format
                processed_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                processed_segment = processed_segment.set_channels(1).set_frame_rate(8000)
                
                # Export directly to WAV with u-law encoding
                wav_buffer = io.BytesIO()
                processed_segment.export(
                    wav_buffer,
                    format="wav",
                    parameters=["-acodec", "pcm_mulaw"]
                )
                wav_buffer.seek(0)
                
                # Extract the raw PCM data from the WAV file
                with wave.open(wav_buffer, "rb") as wav_file:
                    ulaw_data = wav_file.readframes(wav_file.getnframes())
                
                logger.info("Used fallback conversion method successfully")
                
            except Exception as fallback_error:
                logger.error(f"Fallback conversion also failed: {fallback_error}")
                raise
        
        if not ulaw_data or len(ulaw_data) == 0:
            raise ValueError("No audio data after conversion")
            
        # Encode as base64
        encoded_audio = base64.b64encode(ulaw_data).decode('utf-8')
        
        duration = time.time() - start_time
        logger.info(f"Speech synthesis and conversion completed in {duration:.2f}s")
        
        return encoded_audio
    except Exception as e:
        logger.error(f"Error in speech synthesis: {e}", exc_info=True)
        
        # Method 3: Last resort fallback - directly convert using wave module
        try:
            logger.info("Attempting last resort conversion method")
            
            # Create a new audio segment with the right format
            processed_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            processed_segment = processed_segment.set_channels(1).set_frame_rate(8000)
            
            # Convert to PCM first
            pcm_buffer = io.BytesIO()
            processed_segment.export(pcm_buffer, format="wav")
            pcm_buffer.seek(0)
            
            # Read PCM data
            with wave.open(pcm_buffer, "rb") as wav_file:
                pcm_data = wav_file.readframes(wav_file.getnframes())
                
            # Manual conversion to ulaw
            ulaw_data = convert_pcm_to_ulaw(pcm_data)
            encoded_audio = base64.b64encode(ulaw_data).decode('utf-8')
            
            logger.info("Last resort conversion successful")
            return encoded_audio
            
        except Exception as last_error:
            logger.error(f"All conversion methods failed: {last_error}")
            return None

def convert_pcm_to_ulaw(pcm_data):
    """Manually convert PCM data to u-law format"""
    # Convert PCM bytes to 16-bit signed integers
    samples = []
    for i in range(0, len(pcm_data), 2):
        if i + 1 < len(pcm_data):
            sample = struct.unpack('<h', pcm_data[i:i+2])[0]
            samples.append(sample)
    
    # Convert to u-law
    ulaw_data = bytearray()
    for sample in samples:
        # Clip to 16 bits
        sample = max(-32768, min(32767, sample))
        
        # Apply u-law encoding algorithm
        if sample < 0:
            sign = 0x80
            sample = -sample
        else:
            sign = 0x00
        
        # Add bias to avoid taking log of zero
        sample = sample + 132
        
        # Convert to logarithmic scale and quantize
        if sample > 32767:
            sample = 32767
        
        # Determine the segment and mantissa
        segment = 7
        for i in range(7):
            if sample <= 16383 >> i:
                segment = i
                break
        
        # Combine the sign, segment, and mantissa
        mantissa = (sample >> (segment + 3)) & 0x0F
        value = ~(sign | (segment << 4) | mantissa) & 0xFF
        
        ulaw_data.append(value)
    
    return bytes(ulaw_data)

def analyze_audio_file(file_path):
    """Analyze an audio file and log its properties for debugging"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return
            
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        # Detect file type based on extension
        if file_path.endswith('.mp3'):
            try:
                audio = AudioSegment.from_mp3(file_path)
                logger.info(f"MP3 properties: channels={audio.channels}, sample_rate={audio.frame_rate}, "
                           f"sample_width={audio.sample_width}, duration={len(audio)/1000}s")
            except Exception as e:
                logger.error(f"Failed to analyze MP3: {e}")
        
        elif file_path.endswith('.wav'):
            try:
                with wave.open(file_path, 'rb') as wav:
                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    frame_rate = wav.getframerate()
                    n_frames = wav.getnframes()
                    duration = n_frames / frame_rate
                    
                    logger.info(f"WAV properties: channels={channels}, sample_rate={frame_rate}, "
                               f"sample_width={sample_width}, frames={n_frames}, duration={duration}s")
            except Exception as e:
                logger.error(f"Failed to analyze WAV: {e}")
        
        # Try using ffprobe for any audio file
        try:
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "stream=codec_name,channels,sample_rate,bit_rate", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout:
                logger.info(f"FFprobe analysis: {result.stdout.strip()}")
            else:
                logger.warning(f"FFprobe analysis failed: {result.stderr}")
        except Exception as e:
            logger.error(f"FFprobe error: {e}")
    
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")

# Process speech function
async def process_speech(call_state, websocket, content):
    # Mark as processing to prevent multiple parallel processing
    if call_state.is_processing:
        logger.info("Already processing speech, skipping")
        return
    
    call_state.is_processing = True
    logger.info(f"Starting speech processing pipeline with {len(content)} bytes of audio data")
    
    try:
        # 1. STT - Transcribe speech to text
        logger.info("Step 1: Transcribing speech to text...")
        text = await transcribe_audio(content)
        
        if not text.strip():
            logger.info("No speech detected in the audio or transcription failed")
            call_state.is_processing = False
            return
        
        logger.info(f"Transcription result: '{text}'")
        
        # Add to conversation history
        call_state.add_conversation_item("user", text)
        
        # 2. TTT - Process with AI
        logger.info("Step 2: Processing text with AI...")
        response_text = await process_text(text, call_state.conversation_history)
        
        # Add AI response to conversation history
        call_state.add_conversation_item("assistant", response_text)
        logger.info(f"AI response: '{response_text}'")
        
        # 3. TTS - Convert response to audio
        if call_state.should_interrupt:
            logger.info("Processing interrupted by new speech, skipping response")
            call_state.should_interrupt = False
            call_state.is_processing = False
            return
        
        # Mark bot as speaking
        call_state.is_bot_speaking = True
        logger.info("Step 3: Converting response to speech...")
        
        audio_response = await synthesize_speech(response_text)
        
        if audio_response and not call_state.should_interrupt:
            # Send audio back to caller
            logger.info("Sending audio response to caller...")
            audio_message = {
                "event": "media",
                "media": {
                    "payload": audio_response
                }
            }
            await websocket.send_json(audio_message)
            logger.info("Audio response sent successfully")
        else:
            if not audio_response:
                logger.error("Failed to synthesize speech")
            if call_state.should_interrupt:
                logger.info("Response interrupted by new speech")
        
        # Reset speaking status
        call_state.is_bot_speaking = False
        logger.info("Speech processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing speech: {e}", exc_info=True)
    finally:
        # Always reset processing state
        call_state.is_processing = False

def ulaw_to_pcm_simplified(ulaw_data):
    """Convert G.711 ulaw data to PCM using a simplified lookup table approach."""
    # Standard u-law to linear PCM lookup table
    ulaw_to_linear = [
        -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
        -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
        -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
        -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
        -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
        -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
        -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
        -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
        -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
        -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
        -876, -844, -812, -780, -748, -716, -684, -652,
        -620, -588, -556, -524, -492, -460, -428, -396,
        -372, -356, -340, -324, -308, -292, -276, -260,
        -244, -228, -212, -196, -180, -164, -148, -132,
        -120, -112, -104, -96, -88, -80, -72, -64,
        -56, -48, -40, -32, -24, -16, -8, 0,
        32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
        23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
        15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
        11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
        7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
        5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
        3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
        2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
        1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
        1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
        876, 844, 812, 780, 748, 716, 684, 652,
        620, 588, 556, 524, 492, 460, 428, 396,
        372, 356, 340, 324, 308, 292, 276, 260,
        244, 228, 212, 196, 180, 164, 148, 132,
        120, 112, 104, 96, 88, 80, 72, 64,
        56, 48, 40, 32, 24, 16, 8, 0
    ]
    
    pcm_data = array.array('h')
    
    for byte in ulaw_data:
        # u-law data is bitwise inverted (one's complement) when transmitted
        index = ~byte & 0xFF
        pcm_data.append(ulaw_to_linear[index])
    
    return pcm_data

def prepare_audio_for_vad_simplified(ulaw_data):
    """Simplified conversion from ulaw to PCM for VAD."""
    # Convert ulaw to PCM using the lookup table
    pcm_data = ulaw_to_pcm_simplified(ulaw_data)
    
    # Convert to bytes
    pcm_bytes = pcm_data.tobytes()
    
    return pcm_bytes

def chunk_audio_for_vad_simplified(audio_data):
    """Simplified chunking for WebRTC VAD with correct frame sizes."""
    # Calculate frame size based on FRAME_DURATION_MS and SAMPLE_RATE
    # For 20ms at 8kHz with 2 bytes per sample: 8000 * 0.02 * 2 = 320 bytes
    bytes_per_frame = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000 * 2)
    
    frames = []
    for i in range(0, len(audio_data), bytes_per_frame):
        frame = audio_data[i:i+bytes_per_frame]
        
        # Only use complete frames
        if len(frame) == bytes_per_frame:
            frames.append(frame)
    
    return frames

# Simple amplitude-based VAD as a fallback
def simple_amplitude_vad(audio_data, threshold=300):
    """Simple amplitude-based voice activity detection with lower threshold."""
    # Convert bytes to 16-bit PCM samples
    samples = []
    for i in range(0, len(audio_data), 2):
        if i + 1 < len(audio_data):
            sample = struct.unpack('<h', audio_data[i:i+2])[0]
            samples.append(abs(sample))
    
    # No samples, no speech
    if not samples:
        return False
    
    # Calculate average amplitude
    avg_amplitude = sum(samples) / len(samples)
    
    # Determine if it's speech based on amplitude
    is_speech = avg_amplitude > threshold
    
    logger.info(f"Simple VAD: avg amplitude = {avg_amplitude:.1f}, threshold = {threshold}, is_speech = {is_speech}")
    
    return is_speech

async def detect_voice_activity_simplified(call_state, audio_chunk):
    """Simplified voice activity detection function."""
    try:
        # Decode base64 audio chunk
        audio_data = base64.b64decode(audio_chunk)
        logger.info(f"Decoded audio data: {len(audio_data)} bytes")
        
        # Add to audio buffer
        call_state.audio_buffer.extend(audio_data)
        
        # Limit buffer size to prevent memory issues
        MAX_BUFFER_SIZE = 1024 * 1024  # 1MB max
        if len(call_state.audio_buffer) > MAX_BUFFER_SIZE:
            call_state.audio_buffer = call_state.audio_buffer[-MAX_BUFFER_SIZE:]
        
        # Convert ulaw to PCM
        pcm_audio = prepare_audio_for_vad_simplified(audio_data)
        logger.info(f"PCM bytes size: {len(pcm_audio)}")
        
        # Chunk the audio data into frames
        frames = chunk_audio_for_vad_simplified(pcm_audio)
        logger.info(f"Generated {len(frames)} frames from {len(pcm_audio)} bytes of audio data")
        
        # Initialize speech detection variables
        is_speech = False
        speech_frames = 0
        total_frames = len(frames)
        
        # Process each frame with WebRTC VAD
        if total_frames > 0:
            for frame in frames:
                try:
                    frame_is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    if frame_is_speech:
                        speech_frames += 1
                    call_state.vad_buffer.append(frame_is_speech)
                    is_speech = is_speech or frame_is_speech
                    
                    # Keep only the last 20 frames
                    if len(call_state.vad_buffer) > 20:
                        call_state.vad_buffer.pop(0)
                except Exception as e:
                    logger.error(f"VAD error on frame: {e}")
                    continue
        else:
            # If no frames, use simple amplitude-based VAD
            logger.info("No frames generated, using simple amplitude-based VAD")
            is_speech = simple_amplitude_vad(pcm_audio)
            call_state.vad_buffer.append(is_speech)
            if len(call_state.vad_buffer) > 20:
                call_state.vad_buffer.pop(0)
            
            if is_speech:
                speech_frames = 1
                total_frames = 1
        
        # Log frame analysis
        if total_frames > 0:
            logger.info(f"Frame analysis: {speech_frames}/{total_frames} frames detected as speech ({speech_frames/total_frames*100:.1f}%)")
        
        # Calculate speech ratio in the buffer
        speech_ratio = sum(call_state.vad_buffer) / max(1, len(call_state.vad_buffer))
        logger.info(f"Speech ratio: {speech_ratio:.3f}, is_speech_active: {call_state.is_speech_active}, buffer size: {len(call_state.vad_buffer)}")
        
        current_time = time.time()
        
        # Detect speech start - lower threshold for detection
        if not call_state.is_speech_active and speech_ratio > 0.15:
            call_state.is_speech_active = True
            call_state.last_voice_activity = current_time
            logger.info(f"Speech detected (ratio: {speech_ratio:.2f})")
            
            # If bot is speaking, trigger interruption
            if call_state.is_bot_speaking:
                call_state.should_interrupt = True
                logger.info("User interruption detected")
                
        # If speech is active, add audio chunk to collection
        if call_state.is_speech_active:
            call_state.speech_chunks.append(audio_data)
            
            # Reset activity timer if speech is detected
            if speech_ratio > 0.1:
                call_state.last_voice_activity = current_time
        
        # Detect end of speech (silence for SILENCE_THRESHOLD_MS)
        silence_duration = (current_time - call_state.last_voice_activity) * 1000
        if call_state.is_speech_active and silence_duration > SILENCE_THRESHOLD_MS:
            logger.info(f"End of speech detected after {silence_duration:.0f}ms of silence")
            
            # Process the collected speech if we have enough data
            if len(call_state.speech_chunks) > 2:
                # Combine speech chunks
                speech_content = b''.join(call_state.speech_chunks)
                logger.info(f"Collected {len(speech_content)} bytes of speech data from {len(call_state.speech_chunks)} chunks")
                
                # Reset speech detection
                call_state.reset_speech_detection()
                
                # Return the speech content for processing
                return speech_content
            else:
                # Not enough speech data, reset
                call_state.reset_speech_detection()
                logger.info("Speech too short, ignoring")
        
        # Check if we've been collecting speech for too long (force end after 10 seconds)
        if call_state.is_speech_active and len(call_state.speech_chunks) > 0:
            speech_duration = (current_time - call_state.last_voice_activity + silence_duration/1000) * 1000
            if speech_duration > 10000:  # 10 seconds max
                logger.info(f"Forcing end of speech after {speech_duration:.0f}ms (max duration reached)")
                
                # Combine speech chunks
                speech_content = b''.join(call_state.speech_chunks)
                logger.info(f"Collected {len(speech_content)} bytes of speech data from {len(call_state.speech_chunks)} chunks")
                
                # Reset speech detection
                call_state.reset_speech_detection()
                
                # Return the speech content for processing
                return speech_content
        
        return None
    except Exception as e:
        logger.error(f"Error in voice activity detection: {e}", exc_info=True)
        return None

# WebSocket route for media-stream
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket, background_tasks: BackgroundTasks):
    await websocket.accept()
    logger.info("Client connected to WebSocket")
    
    # Initialize call state
    call_state = CallState()
    
    try:
        # Send welcome message
        welcome_text = "Hello, this is AI by DNA. How can I assist you today?"
        logger.info(f"Sending welcome message: {welcome_text}")
        
        # Add initial message to conversation history
        call_state.add_conversation_item("assistant", welcome_text)
        
        try:
            # Synthesize welcome speech
            welcome_audio = await synthesize_speech(welcome_text)
            
            if welcome_audio:
                logger.info("Welcome audio synthesized successfully")
                call_state.is_bot_speaking = True
                audio_message = {
                    "event": "media",
                    "media": {
                        "payload": welcome_audio
                    }
                }
                await websocket.send_json(audio_message)
                logger.info("Welcome audio sent to client")
                call_state.is_bot_speaking = False
            else:
                logger.error("Failed to synthesize welcome audio")
        except Exception as welcome_error:
            logger.error(f"Error sending welcome message: {welcome_error}")
        
        # Process incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                event_type = message.get("event")
                
                if event_type == "media":
                    # Process audio chunk for voice activity detection
                    audio_payload = message["media"]["payload"]
                    speech_content = await detect_voice_activity_simplified(call_state, audio_payload)
                    
                    if speech_content and not call_state.is_processing:
                        # Process detected speech in background task
                        logger.info(f"Starting speech processing task ({len(speech_content)} bytes)")
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
                    
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error: {json_error} - Data: {data[:100]}...")
                continue
            except Exception as msg_error:
                logger.error(f"Error processing message: {msg_error}")
                continue
    
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Cancel any active tasks
        for task in call_state.active_tasks:
            try:
                if not task.done():
                    task.cancel()
                    logger.info("Cancelled active task")
            except Exception as e:
                logger.error(f"Error cancelling task: {e}")
        
        logger.info("WebSocket connection closed")
        
        
# Start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)