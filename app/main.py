"""
AI English Speaking Tutor Backend - Single-file FastAPI app
Contains:
- FastAPI app
- SQLite database with SQLAlchemy (User, Conversation, Message models)
- Whisper transcription (OpenAI/whisper package)
- Flan-T5 reply generation (transformers pipeline)
- gTTS TTS generation
- Static file serving for generated audio
- Config via environment variables for RENDER URL / BASE URL

Save this file as `main.py` and also create a `requirements.txt` (listed below).

Requirements (put in requirements.txt):
fastapi
uvicorn
gtts
pydub
transformers
torch
whisper
python-multipart
sqlalchemy
alembic
pydantic
python-dotenv
aiofiles

Notes:
- For deployment on Render, set the environment variable BASE_URL to your Render service URL (eg: https://your-backend.onrender.com)
- This single-file approach is convenient for testing; for production, split into modules.

"""

import os
import uuid
import shutil
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# ML / TTS imports
from gtts import gTTS
import whisper
from transformers import pipeline

# Environment / config
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_tutor.db")
AUDIO_DIR = "static/replies"
TEMP_DIR = "temp_files"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Database setup ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversations = relationship("Conversation", back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(200), default="Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    user = relationship("User", back_populates="conversations")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    sender = Column(String(20))  # 'user' or 'ai'
    text = Column(Text)
    audio_path = Column(String(400), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")


Base.metadata.create_all(bind=engine)


# --- Pydantic Schemas ---
class UserCreate(BaseModel):
    username: str


class MessageOut(BaseModel):
    id: int
    sender: str
    text: str
    audio_url: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    messages: List[MessageOut]

    class Config:
        orm_mode = True


# --- FastAPI app ---
app = FastAPI(title="AI English Speaking Tutor Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory so generated audios are downloadable
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Load ML models (global so loaded once) ---
print("Loading Whisper model... this can take a while on first run")
whisper_model = whisper.load_model("base")

print("Loading Flan-T5 pipeline... this can take memory")
reply_model = pipeline("text2text-generation", model="google/flan-t5-base")


# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Utility functions ---
def save_temp_file(upload: UploadFile) -> str:
    name = f"{uuid.uuid4().hex}_{upload.filename}"
    path = os.path.join(TEMP_DIR, name)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def transcribe_file(path: str) -> str:
    # whisper returns a dict with 'text'
    result = whisper_model.transcribe(path)
    return result.get("text", "").strip()


def generate_reply(user_text: str) -> str:
    prompt = f"You are a friendly, patient English speaking tutor. Reply naturally and concisely to the user message. Keep tone encouraging.\nUser: {user_text}\nTutor:"
    out = reply_model(prompt, max_length=150, temperature=0.7)
    if isinstance(out, list) and len(out) > 0:
        return out[0].get("generated_text", "").strip()
    return "I'm sorry, I couldn't form a reply. Could you say that again?"


def tts_save(text: str) -> str:
    filename = f"{uuid.uuid4().hex}.mp3"
    path = os.path.join(AUDIO_DIR, filename)
    tts = gTTS(text)
    tts.save(path)
    return path


def build_audio_url(path: str) -> str:
    # path is like static/replies/xxxxx.mp3
    rel = path.replace("\\", "/")
    if rel.startswith("static/"):
        return f"{BASE_URL}/{rel}"
    return f"{BASE_URL}/{rel}"


# --- Endpoints ---
@app.post("/users", response_model=dict)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    u = User(username=user.username)
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"id": u.id, "username": u.username}


@app.get("/users", response_model=List[dict])
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "username": u.username} for u in users]


@app.post("/conversations", response_model=dict)
def create_conversation(user_id: int = Form(...), title: str = Form("Conversation"), db: Session = Depends(get_db)):
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    conv = Conversation(user_id=user_id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"id": conv.id, "title": conv.title}


@app.get("/conversations/{conv_id}", response_model=ConversationOut)
def get_conversation(conv_id: int, db: Session = Depends(get_db)):
    conv = db.query(Conversation).get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.post("/speak")
async def speak(conversation_id: int = Form(...), audio: UploadFile = None, text_input: str = Form(None), db: Session = Depends(get_db)):
    """
    Main endpoint. Accepts either an uploaded audio file or a plain text_input.
    Steps:
      - If audio: save temp -> transcribe
      - Generate AI reply
      - Save reply TTS audio to static/replies
      - Store user message and AI message in database
      - Return JSON with user_text, ai_reply, audio_url, message ids
    """
    # validate conversation
    conv = db.query(Conversation).get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 1) Get user_text either from uploaded audio or text input
    user_text = ""
    temp_path = None
    try:
        if audio is not None:
            temp_path = save_temp_file(audio)
            user_text = transcribe_file(temp_path)
        elif text_input:
            user_text = text_input.strip()
        else:
            raise HTTPException(status_code=400, detail="Provide audio file or text_input")

        # Store user message
        user_msg = Message(conversation_id=conv.id, sender="user", text=user_text)
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)

        # 2) Generate AI reply
        ai_reply = generate_reply(user_text)

        # 3) Convert reply -> speech
        audio_path = tts_save(ai_reply)
        audio_url = build_audio_url(audio_path)

        # 4) Store AI message
        ai_msg = Message(conversation_id=conv.id, sender="ai", text=ai_reply, audio_path=audio_path)
        db.add(ai_msg)
        db.commit()
        db.refresh(ai_msg)

        return JSONResponse({
            "conversation_id": conv.id,
            "user_message_id": user_msg.id,
            "ai_message_id": ai_msg.id,
            "user_text": user_text,
            "ai_reply": ai_reply,
            "audio_url": audio_url,
        })
    finally:
        # cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/message/{message_id}", response_model=MessageOut)
def get_message(message_id: int, db: Session = Depends(get_db)):
    msg = db.query(Message).get(message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    audio_url = None
    if msg.audio_path:
        audio_url = build_audio_url(msg.audio_path)
    return {
        "id": msg.id,
        "sender": msg.sender,
        "text": msg.text,
        "audio_url": audio_url,
        "created_at": msg.created_at,
    }


@app.get("/history/{user_id}", response_model=List[ConversationOut])
def user_history(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.conversations


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/")
def root():
    return {"message": "AI English Tutor Backend running. Use /docs for API docs."}


# Quick helper for running locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
