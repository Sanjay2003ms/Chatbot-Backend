from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime
import os
import sqlite3

app = FastAPI(title="Custom Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "chatbot.db"

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

def get_custom_prompt(persona: str):
    personas = {
        'Default': "You are a helpful assistant. Human: {input} AI:",
        'Expert': "You are an expert. Human: {input} Expert:",
        'Creative': "You are a creative thinker. Human: {input} Creative AI:"
    }
    return personas.get(persona, personas['Default'])

def custom_chatbot_response(prompt: str, history: List[tuple], user_input: str) -> str:
    # A placeholder AI logic (replace with your actual LLM call)
    return f"You said: {user_input}"

@app.get("/")
async def home():
    return {"message": "CHATBOT BACKEND"}

@app.post("/api/chat/send", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    try:
        session_id = request.session_id
        user_email = request.user_email

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO users (email) VALUES (?)", (user_email,))
            c.execute("SELECT session_id FROM sessions WHERE session_id = ? AND user_email = ?", (session_id, user_email))
            if not c.fetchone():
                c.execute("INSERT INTO sessions (session_id, user_email, title, start_time) VALUES (?, ?, ?, ?)",
                         (session_id, user_email, f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}", datetime.now()))
            conn.commit()

        history = []
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT human, ai FROM messages WHERE session_id = ? ORDER BY timestamp", (session_id,))
            history = c.fetchall()

        prompt_template = get_custom_prompt(request.persona)
        ai_response = custom_chatbot_response(prompt_template, history, request.message)

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO messages (session_id, human, ai, timestamp) VALUES (?, ?, ?, ?)",
                     (session_id, request.message, ai_response, datetime.now()))
            c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
            message_count = c.fetchone()[0]
            conn.commit()

        return SendMessageResponse(
            response=ai_response,
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
