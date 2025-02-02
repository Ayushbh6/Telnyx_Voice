import os
import asyncio
import json
import traceback

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
patients, by transforming it to a natural language conversation experience for them!!!""
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
You are the personla call centre agent for AI by DNA.
Here is complete information about AI by DNA:
{AI_by_DNA}

User task is to engage actively with the client and help them to understand the services offered by AI by DNA.
Be friendly and helpful.
You are proficient in English and Greek.
Use English as the default language.

IMPORTANT: Keep your answers to maximum 5 lines of text.
"""
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
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    try:
        async with websockets.connect(
                'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
        ) as openai_ws:
            async def send_session_update():
                session_update = {
                    "type": "session.update",
                    "session": {
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 600,
                            "create_response": True
                        },
                        "response_interruption": {
                            "type": "auto"
                        },
                        "input_audio_format": "g711_ulaw",
                        "output_audio_format": "g711_ulaw",
                        "voice": VOICE,
                        "instructions": SYSTEM_MESSAGE,
                        "modalities": ["text", "audio"],
                        "temperature": 0.8
                    }
                }
                print("Sending session update:", json.dumps(session_update))
                await openai_ws.send(json.dumps(session_update))
                await openai_ws.send(json.dumps({"type": "response.create"}))

            await asyncio.sleep(0.25)
            await send_session_update()
            
            async def receive_openai_messages():
                async for message in openai_ws:
                    try:
                        response = json.loads(message)
                        if response.get("type") in LOG_EVENT_TYPES:
                            print(f"Received OpenAI event: {response['type']}", response)
                        if response.get("type") == "session.updated":
                            print("Session updated successfully:", response)
                        if response.get("type") == "response.audio.delta" and response.get("delta"):
                            print("Sending audio delta to Telnyx")
                            audio_delta = {
                                "event": "media",
                                "media": {
                                    "payload": response["delta"]
                                }
                            }
                            await websocket.send_json(audio_delta)
                    except Exception as e:
                        print("Error processing OpenAI message:", e, "Raw message:", message)

            async def receive_telnyx_messages():
                while True:
                    try:
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        event_type = message.get("event")
                        
                        print(f"Received Telnyx event: {event_type}")
                        
                        if event_type == "media":
                            if not openai_ws.open:
                                print("Warning: OpenAI WebSocket is closed")
                                continue
                                
                            print("Received media payload, sending to OpenAI")
                            audio_event = {
                                "type": "input_audio_buffer.append",
                                "audio": message["media"]["payload"]
                            }
                            await openai_ws.send(json.dumps(audio_event))
                            
                        elif event_type == "start":
                            stream_sid = message["stream_id"]
                            print(f"Incoming stream started: {stream_sid}")
                            
                        elif event_type == "stop":
                            print("Stream stopped")
                            
                        else:
                            print(f"Received non-media event from Telnyx: {event_type}")
                            
                    except websockets.exceptions.ConnectionClosed:
                        print("Telnyx WebSocket connection closed")
                        break
                    except Exception as e:
                        print(f"Error in receive_telnyx_messages: {str(e)}")
                        print(f"Raw data: {data if 'data' in locals() else 'No data'}")

            try:
                await asyncio.gather(
                    receive_openai_messages(),
                    receive_telnyx_messages()
                )
            except Exception as e:
                print(f"Error in main loop: {str(e)}")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        traceback.print_exc()

# Start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)