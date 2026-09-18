# ChestAI Gateway (Node.js / Express)

A Node.js API gateway that sits in front of the existing FastAPI ML backend for
[ChestAI](https://github.com/Vinodhini-03/NIH-ChestXray14-MultiLabel-CNN-RAG).
The gateway owns everything that isn't model inference: authentication,
authorization, input validation, and rate limiting — then proxies validated,
authenticated requests through to FastAPI.

```
React (Vite)  →  Node/Express Gateway  →  FastAPI (ResNet-18, Grad-CAM, RAG, Groq)
                  auth · validation ·
                  rate limiting · CORS
```

## Why a gateway instead of rewriting the ML backend in Node

Rewriting model inference (PyTorch, Grad-CAM) in Node isn't worth it — Python's
ML ecosystem is the right tool there. What FastAPI *doesn't* currently do is
authenticate users, check who's allowed to call what, or validate uploads
before they reach the model. A small Node/Express gateway is a standard way to
add that layer without touching the ML code, and it's the pattern many real
products use (BFF / API gateway in front of internal services).

## What it does

- **Auth** — `/auth/register` and `/auth/login` issue short-lived JWTs
  (bcrypt-hashed passwords, timing-safe login comparison).
- **Permissions** — `requireRole()` middleware gates routes by role
  (`clinician`, `admin`), ready to extend as the app grows.
- **Input validation** — uploaded files are checked for MIME type, size limit,
  and magic bytes (so a renamed file with a spoofed `Content-Type` is
  rejected) before anything reaches FastAPI.
- **Rate limiting** — a general limiter on all routes, plus a tighter limit on
  `/predict` specifically, since each call triggers a full model pass.
- **Security headers & CORS** — `helmet()` defaults, and CORS locked to the
  configured React app origin(s) only.
- **Streaming proxy** — `/chat/stream` pipes FastAPI's Server-Sent Events
  straight through to the React chat UI. Integrating this required two
  frontend changes: adding a login/register screen (none existed before,
  since FastAPI had no auth), and fixing a field-name mismatch between this
  route's original validation (`message`/`history`) and what FastAPI's
  `ChatRequest` actually expects (`question`/`probs`) — the gateway's
  `/chat/stream` handler now validates and forwards `question`/`probs` to
  match.

## Setup

```bash
cd gateway
npm install
cp .env.example .env   # then set a real JWT_SECRET
npm start               # or: npm run dev (auto-restarts on file changes)
```

Requires the FastAPI backend running (see the main repo's `backend/` setup)
and reachable at `FASTAPI_BASE_URL`.

A demo account is seeded on startup for local testing:
`demo@chestai.dev` / `ChangeMe123!` — replace the in-memory user store with a
real database before deploying anywhere real users will hit it.

## Endpoints

| Method | Path             | Auth required | Description                          |
|--------|------------------|----------------|---------------------------------------|
| GET    | `/health`        | No             | Liveness check                        |
| POST   | `/auth/register` | No             | Create an account, returns a JWT      |
| POST   | `/auth/login`    | No             | Returns a JWT                         |
| POST   | `/predict`       | Yes            | Upload an X-ray → proxies to FastAPI  |
| POST   | `/chat/stream`   | Yes            | Streams the RAG/LLM chat response     |

## Integrating into the main repo

Drop this folder in as `gateway/` alongside the existing `backend/` and
`frontend/` folders, and point the React app's API base URL at the gateway
(`http://localhost:4000`) instead of FastAPI directly. Two things the
frontend also needs, beyond just changing the base URL:

- A login/register screen — FastAPI had no auth before, so nothing in the
  React app collected credentials or held a token
- Every `fetch` call to `/predict` and `/chat/stream` needs an
  `Authorization: Bearer <token>` header added

See the main repo's README for how this was done in ChestAI's `App.jsx`.

## What's simplified for now (and worth naming if asked)

- Users are stored in memory, not a database — fine for a demo, not for
  production.
- No refresh-token flow; JWTs simply expire after `JWT_EXPIRES_IN`.
- No automated tests yet beyond manual smoke testing — `npm test` is wired up
  as a placeholder for adding them.
