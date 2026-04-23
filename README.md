# GenAI Memory Workflow

A production-oriented GenAI system with multi-layer memory (L1–L4), session-based conversations, RAG support, tool execution, and a modern frontend dashboard.

## Features (Current)

* Multi-session chat support
* User memory across sessions (L3)
* Session memory (L2)
* Retrieval-Augmented Generation (RAG)
* Tool execution (calculator, etc.)
* Modern frontend UI (React + Vite)

## Project Structure

```
.
├── app/                  # Backend (FastAPI, orchestrator, memory, tools)
├── frontend/
│   └── cognitive-canvas/ # Frontend UI (React + Vite)
├── docs/                 # Design documents (memory, context, scaling)
└── README.md
```

## Running the Project

### Backend

```bash
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend/cognitive-canvas
npm install
npm run dev
```

Frontend runs on:

```
http://localhost:8080/
```

## API

Main endpoint:

```
POST /chat
```

Example request:

```json
{
  "user_id": "u1",
  "session_id": "s1",
  "message": "Hello",
  "document_ids": []
}
```

## Note

This README is intentionally kept basic for now and will be improved after completing the project.

---
