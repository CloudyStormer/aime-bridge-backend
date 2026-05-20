# AIME Bridge Backend

Backend service for the AIME Bridge frontend. It provides chat history and AI reply endpoints compatible with the current React client.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The frontend should set:

```bash
REACT_APP_API_BASE_URL=http://localhost:8000
```

## API

- `GET /health` basic health check
- `GET /ai/status` AI runtime status
- `GET /api/chat/history` returns `{ "messages": ChatMessage[] }`
- `POST /api/chat/message` accepts JSON or multipart form data and returns `{ "message": ChatMessage }`
- `GET /api/training/history` returns training chat history
- `POST /api/training/message` accepts training examples or rules and returns an assistant confirmation
- `GET /api/chat/review?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD` returns a frontend-ready summary for the selected local-date range
- `POST /api/conversation-review` returns detailed stats, summary, and source messages for a datetime range

## AI providers

DeepSeek is the default real AI provider for AIME chat. Runtime secrets live in `.env`, which is intentionally ignored by git.

To enable real AI replies, copy `.env.example` to `.env`, then set:

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Verify real mode with:

```bash
curl http://127.0.0.1:8000/ai/status
```

The response should include `"provider": "deepseek"`, `"mode": "real"`, and `"api_key_configured": true`.

To use another OpenAI-compatible provider:

```bash
LLM_PROVIDER=openai_compatible
LLM_API_KEY=your-key
LLM_MODEL=your-model
LLM_BASE_URL=https://your-compatible-endpoint/v1
```
