# Real-Time AI Meeting Assistant

## Overview
This is a FastAPI-based application with WebSocket support that transcribes live audio from your microphone, summarizes meeting content, and extracts action items using GPU-accelerated speech recognition and pluggable LLM providers (local vLLM, xAI Grok, and more to come).

## Description
Real-Time AI Meeting Assistant
The Real-Time AI Meeting Assistant is an innovative FastAPI-based application designed to transform meeting productivity by transcribing live audio, generating concise summaries, and extracting actionable items in real time. Powered by the faster-whisper GPU pipeline for speech-to-text and a modular LLM layer that can target local vLLM deployments, xAI Grok, or other providers, this tool leverages cutting-edge AI to streamline collaboration and task management. Ideal for professionals, teams, and remote workers, it offers a seamless experience with text-to-speech feedback, live captions, and a dynamic web interface.
Key Features

- Binary WebSocket streaming for low-latency transcription with partial captions.
- GPU-accelerated faster-whisper ASR, optional pyannote diarization, and modular LLM provider routing.
- Workflow hooks for Slack, JSONL logging, and a roadmap toward agenda ingestion and ticketing automation.

Technical Highlights
Developed and tuned for a Lenovo ThinkPad P16 Gen 2 mobile workstation running Windows 11 Pro 64 with the following reference configuration:

- 13th Gen Intel® Core™ i9-13950HX vPro® (24 cores, up to 5.5 GHz P-cores)
- NVIDIA® RTX™ 5000 Ada Generation Laptop GPU with 16 GB GDDR6
- 128 GB DDR5-4000 MHz (4 × 32 GB)
- 4 TB PCIe Gen4 NVMe storage
- 16" WQUXGA OLED touch display (3840 × 2400, HDR 500, 100% DCI-P3)
- 1080p IR camera array with privacy shutter, fingerprint reader, 230 W PSU

The application runs best with CUDA-enabled PyTorch, faster-whisper GPU wheels, and optional Hugging Face diarization models that leverage the RTX GPU.
Getting Started
Clone the repository, set up your environment with the provided requirements, and configure local/cloud LLM providers before you begin. The app is ready for local deployment, with detailed setup instructions in `docs/Setup Guide for Real Time.md`. Whether for personal use or team collaboration, this assistant adapts to your meeting needs, making it a versatile tool for the modern workplace.
Future Potential
Planned enhancements include multi-provider auto-selection, multi-user session management, and richer meeting intelligence (agenda ingestion, Jira/email automation, neural TTS). Contributions are welcome to expand its capabilities and reach.

## Features
- Streams short audio windows from the browser over binary WebSockets for low-latency transcription.
- Transcribes speech incrementally with faster-whisper on GPU and emits partial transcript updates to the UI.
- Performs optional speaker diarization via pyannote.audio with Hugging Face authentication.
- Summarizes content via a provider-agnostic LLM layer (local vLLM by default, with optional Grok/OpenAI/Azure OpenAI/Claude/Gemini adapters).
- Provides optional text-to-speech playback and exposes workflow hooks (Slack, JSONL logs) for downstream automation.
- Designed for modular extension into additional modalities (slides, screen capture) and integrations (calendar, ticketing, knowledge base).
- Queue-backed inference orchestration keeps audio capture responsive while GPU-heavy ASR/LLM work executes on background workers.
- Built-in Prometheus metrics (`/metrics`) surface queue depth, latency, and job outcomes for observability dashboards.

## Installation
1. Clone the repository: `git clone https://github.com/yourusername/realtime-ai-assistant.git`
2. Navigate: `cd realtime-ai-assistant`
3. Create venv: `python -m venv .venv`
4. Activate: `source .venv/bin/activate` (Linux/macOS) or `.venv\\Scripts\\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Populate `.env` with the providers and integrations you plan to use:
   - `VLLM_BASE_URL`, `VLLM_MODEL_ID`, `VLLM_API_KEY` for your local vLLM deployment.
   - `XAI_API_KEY` for Grok fallback (optional).
   - `OPENAI_API_KEY`, `OPENAI_MODEL_ID`, `OPENAI_BASE_URL` for OpenAI (optional).
   - `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` for Azure OpenAI (optional).
   - `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL_ID` for Claude (optional).
   - `GOOGLE_GEMINI_API_KEY`, `GOOGLE_GEMINI_MODEL_ID` for Gemini (optional).
   - `HF_TOKEN` for pyannote diarization (optional); `NEMO_DIARIZATION_ENABLED=true` to experiment with future NeMo integration.
   - `MEETING_SLACK_WEBHOOK_URL`, `MEETING_LOG_PATH`, additional integration secrets as needed.
   - `LLM_PROVIDER_ORDER`, `LLM_MAX_RETRIES`, `LLM_BACKOFF_SECONDS` to control provider prioritisation and retry policy.
   - `INFERENCE_WORKERS`, `INFERENCE_BATCH_SIZE`, `ORCHESTRATOR_BACKEND` to tune queue behaviour; `AUDIO_ENCRYPTION_KEY` (Fernet key) to enable in-memory audio encryption.

## Usage
- Start your preferred LLM backend (e.g., `python -m vllm.entrypoints.openai.api_server --model meta-llama-3-8b-instruct`).
- Launch the FastAPI server: `uvicorn app:app --reload`.
- Open the browser to http://localhost:8000.
- Click **Record & Analyze** to capture a short utterance. Partial transcripts will appear in real time; when processing completes, the summary, actions, and workflow notifications fire automatically.
- Optionally scrape `http://localhost:8000/metrics` with Prometheus to monitor queue depth, latency, and job health.

### Docker Deployment (LAN-ready)
1. Install Docker, Docker Compose, the latest NVIDIA driver, and `nvidia-container-toolkit` on Ubuntu LTS.
2. Copy `.env.docker` to a secure location and populate secrets (LLM keys, encryption key).
3. Generate TLS certs inside `docker/certs/` (self-signed or ACME) and replace the default basic-auth hash in `docker/config/traefik.yml`.
4. Build and launch the stack:
   ```bash
   docker compose --profile model up --build -d
   ```
5. Share `https://assistant.lan/` (or your host IP) with trusted family members; credentials default to `family / welcome123` until you change them.
6. Monitor with Prometheus (`http://<host>:9090`) and Grafana (`http://<host>:3000`) or scrape `/metrics` directly.
See [docs/lan-access.md](docs/lan-access.md) for full details, certificate steps, and SSH tunnel instructions.

## Recent Updates
- Binary WebSocket transport with faster-whisper streaming ASR.
- Async LLM integration via vLLM (OpenAI-compatible) with Grok fallback and multi-provider roadmap.
- Partial transcript/status messaging in the WebSocket protocol and UI.
- Workflow integration hooks for Slack and JSONL logging.
- Queue-based inference orchestrator with model preloading and optional encrypted audio buffers.
- Added provider adapters for OpenAI, Azure OpenAI, Anthropic Claude, and Google Gemini with retry/backoff logic.
- Prometheus metrics endpoint for queue depth, latency, and job outcomes.
- Expanded documentation: streaming transport, async inference, AI build-out instructions.

## Contributing
- Fork and create a feature branch.
- Commit and open a pull request.

## License
[MIT] - Consult legal.

## Additional Setup
- [docs/Setup Guide for Real Time.md](docs/Setup%20Guide%20for%20Real%20Time.md)
- [docs/streaming-transport-upgrade.md](docs/streaming-transport-upgrade.md)
- [docs/async-inference-service.md](docs/async-inference-service.md)
- [docs/ai-build-instructions.md](docs/ai-build-instructions.md) *(new: guidance for automated agents)*
- Prometheus metrics available at `http://localhost:8000/metrics` (scrape with Prometheus/Grafana).

## High-Impact Upgrade Roadmap
- **GPU-first pipeline**: Preload ASR/diarization models, evaluate NeMo diarization, and explore whisper-large-v3 streaming support with bf16 on RTX 5000.
- **Structured orchestration**: Extend the new queue/batching layer to external backplanes (FastStream/Redis Streams) for parallel processing and scalability.
- **Multi-provider LLM layer**: DONE for Grok/OpenAI/Azure/Claude/Gemini; next add caching, per-provider rate limiting, and user-selectable policies.
- **Assistant intelligence**: Agenda ingestion (Outlook/Google), Jira/email automation, persistent vector memory (Chroma/LanceDB), and multi-modal capture (slides OCR, screen analysis).
- **Experience & UX**: React/TypeScript client with live confidence heat maps, export flows (Notion/Confluence/Teams), offline Electron wrapper, hotkeys/voice commands.
- **Reliability & Ops**: Expand observability (tracing), harden secrets (pydantic-settings, Vault/SOPS), add integration tests with recorded audio, chaos testing, zero-trust privacy modes.

## Security Notes
- Store LLM and integration API keys using a secrets manager (Windows Credential Manager, Azure Key Vault, SOPS, etc.) and inject them via environment variables rather than committing `.env` files.
- Rotate credentials regularly and prefer least-privilege tokens. The application never logs secret values, but review logging configuration before deploying to shared systems.
- Enable `AUDIO_ENCRYPTION_KEY` to encrypt queued PCM data in memory when tighter confidentiality controls are required.

### Security Hardening (Feb 2026)

- Bind address changed from `0.0.0.0` to `127.0.0.1`. The server no longer listens on all network interfaces, preventing other devices on the local network from reaching the WebSocket and `/metrics` endpoints. Use a reverse proxy (Traefik, nginx) for intentional LAN or public exposure.
