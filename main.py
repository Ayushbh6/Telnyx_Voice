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
        self.audio_buffer = bytearray()
        self.speech_chunks = []
        self.is_speech_active = False
        self.last_voice_activity = 0
        self.active_tasks = set()
        self.listening_mode = True  # New flag to control when we're actively listening
        self.last_listening_log = 0  # Track when we last logged listening status
        
    def reset_speech_detection(self):
        self.speech_chunks = []
        self.is_speech_active = False
        self.last_voice_activity = time.time()
        
    def add_conversation_item(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        # Limit history to last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
            
    def start_listening(self):
        """Enable listening mode to collect user speech"""
        logger.info("🎤 STARTING LISTENING MODE - Ready for user input")
        self.listening_mode = True
        self.reset_speech_detection()
        self.last_listening_log = time.time()
        
    def stop_listening(self):
        """Disable listening mode while AI is processing/speaking"""
        logger.info("🔇 STOPPING LISTENING MODE - Processing or speaking")
        self.listening_mode = False
        self.reset_speech_detection()
        
    def log_listening_status(self):
        """Periodically log that we're still listening"""
        current_time = time.time()
        # Log every 5 seconds that we're still listening
        if self.listening_mode and (current_time - self.last_listening_log) > 5:
            logger.info("👂 Still listening for user input...")
            self.last_listening_log = current_time

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
        logger.info(f"Starting transcription of {len(audio_data)} bytes of audio data")
        
        # Convert ulaw audio to PCM for better transcription
        pcm_data = ulaw_to_pcm_simplified(audio_data)
        logger.info(f"Converted to PCM: {len(pcm_data)} samples")
        
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
        
        # Try to upsample to 16kHz first (Whisper works better with 16kHz)
        mp3_path = temp_path.replace(".wav", ".mp3")
        try:
            logger.info("Upsampling audio to 16kHz for better transcription")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_path,
                "-ar", "16000",  # Upsample to 16kHz
                "-ac", "1",      # Ensure mono
                mp3_path
            ], check=True, capture_output=True)
            
            # Verify the MP3 file was created
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                logger.info(f"Successfully created upsampled MP3: {mp3_path} ({os.path.getsize(mp3_path)} bytes)")
                
                # Try transcription with the upsampled file
                with open(mp3_path, "rb") as mp3_file:
                    logger.info("Sending upsampled audio to Whisper API")
                    transcription = openai_client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=mp3_file,
                        language="en"  # Specify language for better results
                    )
                
                # Clean up temporary files
                for path in [temp_path, mp3_path]:
                    if os.path.exists(path):
                        os.unlink(path)
                
                duration = time.time() - start_time
                logger.info(f"Transcription completed in {duration:.2f}s: {transcription.text}")
                
                return transcription.text
            else:
                logger.warning("Upsampling failed or produced empty file, falling back to original WAV")
        except Exception as upsample_error:
            logger.error(f"Error upsampling audio: {upsample_error}")
            logger.info("Falling back to original WAV file")
        
        # Fallback: Transcribe using the original WAV file
        try:
            with open(temp_path, "rb") as audio_file:
                logger.info("Sending original WAV to Whisper API")
                transcription = openai_client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=audio_file,
                    language="en"  # Specify language for better results
                )
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            duration = time.time() - start_time
            logger.info(f"Transcription completed in {duration:.2f}s: {transcription.text}")
            
            return transcription.text
        except Exception as whisper_error:
            logger.error(f"Whisper transcription error with WAV: {whisper_error}")
            
            # Last resort: Try with a different format
            try:
                # Convert to FLAC format
                flac_path = temp_path.replace(".wav", ".flac")
                logger.info("Trying FLAC format as last resort")
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", temp_path,
                    "-ar", "16000",  # Upsample to 16kHz
                    "-ac", "1",
                    flac_path
                ], check=True, capture_output=True)
                
                # Try transcription with FLAC
                with open(flac_path, "rb") as flac_file:
                    logger.info("Sending FLAC to Whisper API")
                    transcription = openai_client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=flac_file,
                        language="en"
                    )
                
                # Clean up temporary files
                for path in [temp_path, flac_path]:
                    if os.path.exists(path):
                        os.unlink(path)
                
                duration = time.time() - start_time
                logger.info(f"FLAC transcription completed in {duration:.2f}s: {transcription.text}")
                
                return transcription.text
            except Exception as flac_error:
                logger.error(f"FLAC transcription also failed: {flac_error}")
                
                # Clean up any remaining temporary files
                for path in [temp_path, mp3_path, flac_path]:
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
    """Analyze audio file properties for debugging purposes"""
    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        logger.info(f"Audio file size: {file_size} bytes")
        
        # Get audio properties using wave module for WAV files
        if file_path.endswith('.wav'):
            with wave.open(file_path, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / framerate
                
                logger.info(f"WAV properties: {channels} channels, {sample_width*8}-bit, "
                           f"{framerate} Hz, {n_frames} frames, {duration:.2f} seconds")
        
        # Try to get more detailed info using ffprobe
        try:
            result = subprocess.run([
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration,bit_rate:stream=codec_name,codec_type,sample_rate,channels", 
                "-of", "json", 
                file_path
            ], capture_output=True, text=True, check=True)
            
            info = json.loads(result.stdout)
            logger.info(f"FFprobe analysis: {json.dumps(info, indent=2)}")
        except Exception as e:
            logger.warning(f"FFprobe analysis failed: {e}")
            
    except Exception as e:
        logger.error(f"Error analyzing audio file: {e}")

# Process speech function
async def process_speech(call_state, websocket, content):
    """Process speech with strict turn-taking"""
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
            # Add a delay before re-enabling listening mode
            await asyncio.sleep(1)
            call_state.start_listening()  # Re-enable listening mode
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
        # Mark bot as speaking
        call_state.is_bot_speaking = True
        logger.info("Step 3: Converting response to speech...")
        
        audio_response = await synthesize_speech(response_text)
        
        if audio_response:
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
            logger.error("Failed to synthesize speech")
        
        # Reset speaking status
        call_state.is_bot_speaking = False
        logger.info("Speech processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing speech: {e}", exc_info=True)
    finally:
        # Always reset processing state and re-enable listening after a delay
        call_state.is_processing = False
        
        # Add a 2-second delay before re-enabling listening mode
        # This helps prevent the bot from picking up its own speech or echoes
        logger.info("Waiting 2 seconds before re-enabling listening mode...")
        await asyncio.sleep(2)
        
        call_state.start_listening()  # Re-enable listening mode

def ulaw_to_pcm_simplified(ulaw_data):
    """Convert G.711 ulaw data to PCM using a simplified approach."""
    # Create a numpy array to hold the PCM data
    pcm_data = np.zeros(len(ulaw_data), dtype=np.int16)
    
    for i, byte in enumerate(ulaw_data):
        # Flip all bits (bitwise NOT)
        byte = ~byte & 0xFF
        
        # Extract sign bit: 1 for positive, -1 for negative
        sign = 1 if (byte & 0x80) else -1
        
        # Extract position bits and calculate position
        position = ((byte & 0x70) >> 4)
        
        # Extract segment bits
        segment = (byte & 0x0F)
        
        # Calculate linear PCM value
        value = segment << (position + 1)
        
        # Add bias
        value += (1 << position)
        
        # Apply sign
        value = value * sign
        
        # Store in array
        pcm_data[i] = value
    
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
def simple_amplitude_vad(audio_data, threshold=500):
    """Simple amplitude-based voice activity detection with higher threshold."""
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
    
    # Calculate peak amplitude (95th percentile to avoid outliers)
    sorted_samples = sorted(samples)
    peak_idx = min(int(len(sorted_samples) * 0.95), len(sorted_samples) - 1)
    peak_amplitude = sorted_samples[peak_idx] if sorted_samples else 0
    
    # Determine if it's speech based on amplitude
    is_speech = avg_amplitude > threshold and peak_amplitude > threshold * 2
    
    logger.info(f"Amplitude VAD: avg={avg_amplitude:.1f}, peak={peak_amplitude:.1f}, threshold={threshold}, is_speech={is_speech}")
    
    return is_speech

async def detect_voice_activity_simplified(audio_chunk_base64, call_state):
    """Simplified voice activity detection with strict turn-taking"""
    try:
        # Skip processing if not in listening mode
        if not call_state.listening_mode:
            return None
            
        # Skip processing if bot is speaking or already processing
        if call_state.is_bot_speaking or call_state.is_processing:
            return None
            
        # Decode base64 audio chunk
        audio_chunk = base64.b64decode(audio_chunk_base64)
        
        # Add to audio buffer
        call_state.audio_buffer.extend(audio_chunk)
        
        # Limit buffer size to prevent memory issues (max 1MB)
        MAX_BUFFER_SIZE = 1024 * 1024  # 1MB
        if len(call_state.audio_buffer) > MAX_BUFFER_SIZE:
            logger.warning(f"Audio buffer exceeded {MAX_BUFFER_SIZE} bytes, truncating")
            call_state.audio_buffer = call_state.audio_buffer[-MAX_BUFFER_SIZE:]
        
        # Convert ulaw to PCM for VAD
        pcm_data = ulaw_to_pcm_simplified(audio_chunk)
        audio_for_vad = pcm_data.tobytes()
        
        # Chunk audio for VAD
        frames = chunk_audio_for_vad_simplified(audio_for_vad)
        
        # Count frames with speech
        speech_frames = 0
        total_frames = len(frames)
        
        for frame in frames:
            try:
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                if is_speech:
                    speech_frames += 1
            except Exception as e:
                logger.error(f"VAD error on frame: {e}")
        
        # Calculate speech ratio
        speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
        
        # Also check amplitude-based VAD as a secondary confirmation
        is_amplitude_speech = simple_amplitude_vad(audio_for_vad, threshold=500)  # Increased threshold
        
        # Detect speech with a higher threshold (30% of frames contain speech)
        # AND require amplitude confirmation for more robust detection
        is_speech = speech_ratio >= 0.30 and is_amplitude_speech
        
        # Update speech detection state
        current_time = time.time()
        
        # If we detect speech and we're not already in speech mode
        if is_speech and not call_state.is_speech_active:
            logger.info(f"Speech detected (ratio: {speech_ratio:.2%}, amplitude confirmed: {is_amplitude_speech})")
            call_state.is_speech_active = True
            call_state.last_voice_activity = current_time
            call_state.speech_chunks.append(audio_chunk)
            return None
        
        # If we're in speech mode
        elif call_state.is_speech_active:
            # Add the chunk to our collection
            call_state.speech_chunks.append(audio_chunk)
            
            # Update last activity time if we detect speech
            if is_speech:
                call_state.last_voice_activity = current_time
            
            # Calculate time since last voice activity
            time_since_last_activity = current_time - call_state.last_voice_activity
            
            # Check if we've collected enough speech (at least 40 chunks, ~4 seconds)
            # Increased from 30 to 40 to ensure we have enough speech
            enough_speech = len(call_state.speech_chunks) >= 40
            
            # Check if we've been silent for the threshold duration
            silence_detected = (not is_speech and 
                               time_since_last_activity * 1000 >= SILENCE_THRESHOLD_MS)
            
            # Check if we've exceeded maximum speech duration (10 seconds)
            max_duration_exceeded = len(call_state.speech_chunks) >= 100  # ~10 seconds
            
            # Process speech if we've detected silence after speech or exceeded max duration
            if silence_detected or max_duration_exceeded or enough_speech:
                # Log the reason for processing
                if silence_detected:
                    logger.info(f"End of speech detected after {time_since_last_activity:.2f}s of silence")
                elif max_duration_exceeded:
                    logger.info("Maximum speech duration exceeded (10s), processing speech")
                elif enough_speech:
                    logger.info(f"Collected {len(call_state.speech_chunks)} chunks (~{len(call_state.speech_chunks)/10:.1f}s), processing speech")
                
                # Only process if we have collected a minimum amount of speech chunks
                if len(call_state.speech_chunks) >= 20:  # At least 2 seconds of speech
                    # Combine all collected audio chunks
                    all_speech = b''.join(call_state.speech_chunks)
                    logger.info(f"Processing {len(all_speech)} bytes of speech data")
                    
                    # Stop listening while we process this speech
                    call_state.stop_listening()
                    
                    return all_speech
                else:
                    # Not enough speech to process, likely a false positive
                    logger.info(f"Discarding {len(call_state.speech_chunks)} chunks as likely false positive")
                    call_state.reset_speech_detection()
            
            return None
        
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
        
        # Temporarily disable listening while sending welcome message
        call_state.stop_listening()
        
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
        finally:
            # Wait 3 seconds before re-enabling listening to avoid picking up echoes
            logger.info("Waiting 3 seconds before enabling listening mode...")
            await asyncio.sleep(3)
            
            # Re-enable listening after welcome message
            call_state.start_listening()
            logger.info("👂 Waiting for user to speak...")
        
        # Process incoming messages
        while True:
            try:
                # Set a timeout for receiving messages to allow for periodic status updates
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    
                    message = json.loads(data)
                    event_type = message.get("event")
                    
                    if event_type == "media":
                        # Log listening status periodically
                        call_state.log_listening_status()
                        
                        # Process audio chunk for voice activity detection
                        audio_payload = message["media"]["payload"]
                        speech_content = await detect_voice_activity_simplified(audio_payload, call_state)
                        
                        if speech_content:
                            # Process detected speech directly (not in background)
                            logger.info(f"Starting speech processing ({len(speech_content)} bytes)")
                            await process_speech(call_state, websocket, speech_content)
                    
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
                
                except asyncio.TimeoutError:
                    # This is expected - use this opportunity to log status
                    call_state.log_listening_status()
                    continue
                    
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error: {json_error} - Data: {data[:100]}...")
                continue
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
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

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and available in the PATH"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("ffmpeg is installed and available")
            return True
        else:
            logger.error("ffmpeg check failed with non-zero return code")
            return False
    except Exception as e:
        logger.error(f"ffmpeg is not installed or not in PATH: {e}")
        return False

def check_openai_api_key():
    """Check if the OpenAI API key is valid by making a test request"""
    try:
        # Create a temporary client to test the API key
        test_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Make a simple test request
        response = test_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test request."}],
            max_tokens=5
        )
        
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            logger.info("OpenAI API key is valid")
            return True
        else:
            logger.error("OpenAI API key test failed: Unexpected response format")
            return False
    except Exception as e:
        logger.error(f"OpenAI API key test failed: {e}")
        return False

def check_environment_variables():
    """Check if all required environment variables are set"""
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
        "ELEVENLABS_VOICE_ID": os.getenv("ELEVENLABS_VOICE_ID"),
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file or environment")
        return False
    
    logger.info("All required environment variables are set")
    return True

# Main function
async def main():
    """Main function to run the voice bot"""
    # Check environment variables
    check_environment_variables()
    
    # Check if ffmpeg is installed
    check_ffmpeg_installed()
    
    # Check if OpenAI API key is valid
    check_openai_api_key()
    
    # Start the FastAPI server
    logger.info(f"Starting server on port {PORT}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

# Entry point
if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())