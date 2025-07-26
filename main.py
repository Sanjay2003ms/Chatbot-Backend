from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime
import os
import uvicorn
import sqlite3
import openai

app = FastAPI(title="Custom Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "chatbot.db"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY == "" or len(GROQ_API_KEY) < 6:
    raise RuntimeError("Could not get the groq api key. \n Check enviroment variables")


try:
    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
except Exception as e:
    raise RuntimeError("Could not create openai client " + str(e))



def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_email TEXT,
            title TEXT,
            start_time DATETIME,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            human TEXT,
            ai TEXT,
            timestamp DATETIME,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )''')
        conn.commit()

init_db()

class ChatMessage(BaseModel):
    human: str
    ai: str
    timestamp: datetime

class SendMessageRequest(BaseModel):
    message: str
    session_id: str
    model: str = "llama3-70b-8192"
    persona: str = "Default"
    memory_length: int = 5
    user_email: str

class SendMessageResponse(BaseModel):
    response: str
    session_id: str
    message_count: int

class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    user_email: str

class SessionResponse(BaseModel):
    session_id: str
    chat_history: List[ChatMessage]
    message_count: int
    start_time: Optional[datetime]

class ClearSessionRequest(BaseModel):
    session_id: str

class UserSession(BaseModel):
    session_id: str
    title: str
    start_time: Optional[datetime]
    message_count: int 

class UserSessionsResponse(BaseModel):
    sessions: List[UserSession]

def get_custom_prompt(persona: str) -> str:
    personas = {
        'Default': (
            "You are a friendly and helpful AI assistant, providing clear, concise, and accurate responses. "
            "Focus on being approachable and ensuring the user feels understood and supported."
        ),
        'Expert': (
            "You are a highly knowledgeable and authoritative expert across various fields. "
            "Offer in-depth, precise, and technical explanations, citing examples or relevant research when necessary. "
            "Avoid jargon when possible, but feel free to introduce advanced concepts where appropriate."
        ),
        'Creative': (
            "You are an imaginative and inventive AI with a flair for creative problem-solving and thinking outside the box. "
            "Use metaphors, vivid descriptions, and unconventional ideas to inspire and captivate the user. "
            "Feel free to suggest unique approaches or surprising solutions to problems."
        )
    }
    return personas.get(persona, personas['Default'])



@app.api_route("/", methods=["GET", "HEAD"])
async def home():
    return {"message": "CHATBOT BACKEND"}

@app.api_route("/healthz", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "ok"}


@app.post("/api/chat/send", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    try:
        session_id = request.session_id
        user_email = request.user_email

        # ✅ Store session if needed
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO users (email) VALUES (?)", (user_email,))
            c.execute("SELECT session_id FROM sessions WHERE session_id = ? AND user_email = ?", (session_id, user_email))
            if not c.fetchone():
                c.execute("INSERT INTO sessions (session_id, user_email, title, start_time) VALUES (?, ?, ?, ?)",
                         (session_id, user_email, f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}", datetime.now()))
            conn.commit()

        # ✅ Build conversation history (memory-like behavior)
        messages = []
        system_prompt = get_custom_prompt(request.persona)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT human, ai FROM messages WHERE session_id = ? ORDER BY timestamp", (session_id,))
            history = c.fetchall()
            for human, ai in history[-request.memory_length:]:
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": ai})

        # ✅ Add current message
        messages.append({"role": "user", "content": request.message})

        # ✅ Call Groq API using OpenAI client
        chat_response = client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=0.7
        )

        reply = chat_response.choices[0].message.content

        # ✅ Store message in DB
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO messages (session_id, human, ai, timestamp) VALUES (?, ?, ?, ?)",
                     (session_id, request.message, reply, datetime.now()))
            c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
            message_count = c.fetchone()[0]
            conn.commit()

        # ✅ Return response
        return SendMessageResponse(
            response=reply,
            session_id=session_id,
            message_count=message_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/session", response_model=SessionResponse)
async def get_or_create_session(request: SessionRequest):
    session_id = request.session_id or str(uuid.uuid4())
    user_email = request.user_email

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users (email) VALUES (?)", (user_email,))
        c.execute("SELECT session_id FROM sessions WHERE session_id = ? AND user_email = ?", (session_id, user_email))
        if not c.fetchone():
            c.execute("INSERT INTO sessions (session_id, user_email, title, start_time) VALUES (?, ?, ?, ?)",
                     (session_id, user_email, f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}", datetime.now()))
        c.execute("SELECT human, ai, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp", (session_id,))
        chat_history = [ChatMessage(human=row[0], ai=row[1], timestamp=row[2]) for row in c.fetchall()]
        c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        message_count = c.fetchone()[0]
        c.execute("SELECT start_time FROM sessions WHERE session_id = ?", (session_id,))
        start_time = c.fetchone()[0]
        conn.commit()

    return SessionResponse(
        session_id=session_id,
        chat_history=chat_history,
        message_count=message_count,
        start_time=start_time
    )

@app.post("/api/chat/clear")
async def clear_session(request: ClearSessionRequest):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE session_id = ?", (request.session_id,))
        c.execute("DELETE FROM sessions WHERE session_id = ?", (request.session_id,))
        conn.commit()
    return {"message": "Session cleared successfully"}

@app.delete("/api/chat/memory/{session_id}")
async def clear_memory_only(session_id: str):
    return {"message": "Memory cleared successfully"}

@app.get("/api/models")
async def get_available_models():
    return {
        "models": [
            "llama3-70b-8192",
            "gemma2-9b-it",
            "mixtral-8x7b-32768"
        ]
    }

@app.get("/api/personas")
async def get_available_personas():
    return {
        "personas": [
            "Default",
            "Expert",
            "Creative"
        ]
    }

@app.get("/api/user/sessions/{user_email}", response_model=UserSessionsResponse)
async def get_user_sessions(user_email: str):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT s.session_id, s.title, s.start_time, COUNT(m.id)
            FROM sessions s
            LEFT JOIN messages m ON s.session_id = m.session_id
            WHERE s.user_email = ?
            GROUP BY s.session_id
            ORDER BY s.start_time DESC
        """, (user_email,))
        sessions = [UserSession(session_id=row[0], title=row[1], start_time=row[2], message_count=row[3]) for row in c.fetchall()]
    return UserSessionsResponse(sessions=sessions)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
