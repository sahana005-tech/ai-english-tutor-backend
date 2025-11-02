# main.py
import os
import io
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from gtts import gTTS
import openai
import aiofiles
import sqlite3

# ---------- Load config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", "8000"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
TTS_LANG = os.getenv("TTS_LANG", "en")

openai.api_key = OPENAI_API_KEY

app = FastAPI(title="AI English Tutor Backend (OpenAI Version)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace * with your frontend URL when ready
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
TTS_DIR = "./tts_cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)

# ---------- Database ----------
def init_db():
    conn = sqlite3.connect("./app.db", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        role TEXT,
        content TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    return conn

db_conn = init_db()

def save_message(role: str, content: str):
    try:
        cur = db_conn.cursor()
        cur.execute("INSERT INTO conversations (id, role, content) VALUES (?, ?, ?)", (str(uuid.uuid4()), role, content))
        db_conn.commit()
    except Exception as e:
        print("DB error:", e)

# ---------- Pydantic Models ----------
class ChatRequest(BaseModel):
    prompt: str

class TTSRequest(BaseModel):
    text: str
    lang: str = TTS_LANG

# ---------- Routes ----------

@app.get("/")
def root():
    return {"status": "ok", "message": "AI English Tutor Backend (OpenAI) running successfully!"}

# 🎙️ 1. Speech to Text (Whisper)
@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    try:
        temp_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
        async with aiofiles.open(temp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        with open(temp_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        text = transcript.text
        save_message("user_audio_transcript", text)
        return {"transcript": text, "provider": "openai", "model": "whisper-1"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

# 💬 2. AI Chat (GPT-4)
@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    try:
        save_message("user", req.prompt)

        completion = openai.chat.completions.create(
            model="gpt-4o-mini",  # fast, cost-effective version of GPT-4
            messages=[
                {"role": "system", "content": "You are an English conversation tutor. Correct grammar, reply naturally."},
                {"role": "user", "content": req.prompt}
            ]
        )

        reply = completion.choices[0].message.content
        save_message("assistant", reply)
        return {"reply": reply, "provider": "openai", "model": "gpt-4o-mini"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# 🔊 3. Text to Speech (gTTS)
@app.post("/api/tts")
async def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    try:
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(TTS_DIR, filename)
        tts = gTTS(text=req.text, lang=req.lang)
        tts.save(filepath)
        save_message("tts_generated", req.text)
        return FileResponse(path=filepath, filename="speech.mp3", media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.get("/api/conversations")
def get_conversations():
    cur = db_conn.cursor()
    cur.execute("SELECT id, role, content, created_at FROM conversations ORDER BY created_at DESC LIMIT 30")
    rows = cur.fetchall()
    return {"count": len(rows), "items": [{"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]}

# ---------- Run locally ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
