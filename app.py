from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketState
import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pyttsx3
import torch
import torchaudio
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from integrations.workflows import run_post_meeting_workflows
from services.asr_service import ASRService
from services.audio_buffer import SecureAudioBuffer
from services.llm_service import LLMService
from services.orchestrator import InferenceJob, InferenceOrchestrator
from observability import (
    ASR_LATENCY,
    DIARIZATION_LATENCY,
    INFERENCE_JOB_DURATION,
    INFERENCE_JOB_FAILURES,
    INFERENCE_JOBS_TOTAL,
    LLM_LATENCY,
    update_queue_depth,
)

# Load environment variables
load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("meeting-assistant")

# --- Global Constants & Initialization ---
HF_TOKEN = os.getenv("HF_TOKEN")  # For pyannote.audio
if not HF_TOKEN:
    LOGGER.info("HF_TOKEN not provided; speaker diarization disabled")
MODEL_RATE = int(os.getenv("MODEL_SAMPLE_RATE", "16000"))
ASR_MODEL_ID = os.getenv("ASR_MODEL_ID", "medium")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE")
INFERENCE_WORKERS = int(os.getenv("INFERENCE_WORKERS", "1"))
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "1"))
ORCHESTRATOR_BACKEND = os.getenv("ORCHESTRATOR_BACKEND", "in-memory")
AUDIO_ENCRYPTION_KEY = os.getenv("AUDIO_ENCRYPTION_KEY")
NEMO_DIARIZATION_ENABLED = os.getenv("NEMO_DIARIZATION_ENABLED", "false").lower() == "true"

app = FastAPI(docs_url=None, redoc_url=None)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Cache expensive resources ---
tts_engine: Optional[pyttsx3.Engine] = None
tts_lock: Optional[asyncio.Lock] = None
diarization_pipeline = None
asr_service: Optional[ASRService] = None
llm_service: Optional[LLMService] = None
orchestrator: Optional[InferenceOrchestrator] = None

def get_tts_engine():
    global tts_engine
    if tts_engine is None:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 180)
    return tts_engine


def get_asr_service() -> ASRService:
    global asr_service
    if asr_service is None:
        asr_service = ASRService(
            model_size=ASR_MODEL_ID,
            language=ASR_LANGUAGE,
            compute_type=ASR_COMPUTE_TYPE,
        )
    return asr_service


def get_llm_service() -> LLMService:
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service

def get_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is None and HF_TOKEN:
        try:
            LOGGER.info("Loading speaker diarization pipeline...")
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
            if torch.cuda.is_available():
                diarization_pipeline.to(torch.device("cuda"))
            LOGGER.info("Speaker diarization pipeline loaded successfully.")
        except Exception as e:
            LOGGER.exception("Failed to load diarization pipeline: %s", e)
            LOGGER.warning("Speaker diarization will be disabled.")
            diarization_pipeline = None
    return diarization_pipeline

# --- Core Functions ---
async def process_audio_session(
    audio_data: bytes, sample_rate: int, websocket: WebSocket
) -> Dict[str, object]:
    """Process a captured audio buffer end-to-end."""

    if not audio_data:
        return {"error": "No audio was captured. Please try recording again."}

    pcm = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        return {"error": "No speech detected. Please speak clearly and ensure your microphone is working."}

    waveform = torch.from_numpy(pcm).unsqueeze(0)
    if sample_rate != MODEL_RATE:
        LOGGER.debug("Resampling audio from %sHz to %sHz", sample_rate, MODEL_RATE)
        resampler = torchaudio.transforms.Resample(sample_rate, MODEL_RATE)
        waveform = resampler(waveform)

    audio_for_model = waveform.squeeze(0).numpy()

    asr = get_asr_service()
    segments: List[Dict[str, object]] = []
    transcript_parts: List[str] = []

    asr_start = time.perf_counter()
    try:
        async for segment in asr.stream_transcription(audio_for_model, MODEL_RATE):
            segments.append(segment)
            transcript_parts.append(segment["text"])
            await websocket.send_json({"type": "partial_transcript", "segment": segment})
    except RuntimeError as exc:
        LOGGER.exception("ASR failed: %s", exc)
        ASR_LATENCY.observe(time.perf_counter() - asr_start)
        return {"error": f"ASR error: {exc}"}
    else:
        ASR_LATENCY.observe(time.perf_counter() - asr_start)

    transcribed_text = " ".join(transcript_parts).strip()

    diarized_text = await build_diarized_transcript(segments, waveform)
    full_transcript = diarized_text or transcribed_text

    if not full_transcript:
        return {"error": "No speech detected. Please speak clearly and ensure your microphone is working."}

    await websocket.send_json({"type": "status", "message": "Summarizing meeting"})

    llm_start = time.perf_counter()
    llm_response = await query_llm(full_transcript)
    LLM_LATENCY.observe(time.perf_counter() - llm_start)
    if not llm_response:
        return {"error": "Failed to get LLM response."}

    summary = llm_response.get("summary", "")
    actions = llm_response.get("actions", [])

    await run_post_meeting_workflows(summary, actions)

    return {
        "transcription": full_transcript,
        "summary": summary,
        "actions": actions,
    }


async def build_diarized_transcript(
    segments: List[Dict[str, object]], waveform: torch.Tensor
) -> Optional[str]:
    if not HF_TOKEN or not segments:
        return None

    if NEMO_DIARIZATION_ENABLED:
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore  # pragma: no cover
        except ImportError:
            LOGGER.warning(
                "Nemo diarization enabled but nemo-toolkit is not installed; falling back to pyannote."
            )
        else:  # pragma: no cover - placeholder until NeMo integration is implemented
            LOGGER.info(
                "Nemo diarization toggle enabled. Future versions will route through NeMo; using pyannote for now."
            )

    pipeline = get_diarization_pipeline()
    if pipeline is None:
        return None

    diarization_start = time.perf_counter()
    try:
        diarization = await asyncio.to_thread(
            pipeline,
            {"waveform": waveform, "sample_rate": MODEL_RATE},
        )
    except Exception as exc:  # pragma: no cover - runtime logging only
        LOGGER.exception("Diarization error: %s", exc)
        return None
    finally:
        DIARIZATION_LATENCY.observe(time.perf_counter() - diarization_start)

    diarized_segments = []
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= start < turn.end or turn.start < end <= turn.end or (
                start <= turn.start and end >= turn.end
            ):
                speakers.append(speaker)
        speaker_label = speakers[0] if speakers else "Speaker"
        diarized_segments.append(f"{speaker_label}: {text}")

    return " ".join(diarized_segments)


async def query_llm(text: str) -> Optional[Dict[str, object]]:
    """Sends transcription to the configured LLM providers."""

    llm = get_llm_service()
    try:
        return await llm.summarize_meeting(text)
    except Exception as exc:  # pragma: no cover - runtime logging only
        LOGGER.exception("LLM summarization failed: %s", exc)
        return None


async def run_inference_job(job: InferenceJob) -> None:
    websocket = job.websocket
    if websocket.client_state != WebSocketState.CONNECTED:
        LOGGER.warning("WebSocket disconnected before processing job %s", job.job_id)
        return

    job_start = time.perf_counter()
    job_failed = False
    try:
        await websocket.send_json({"type": "status", "message": "Processing audio"})
        result = await process_audio_session(job.audio_data, job.sample_rate, websocket)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "final", **result})

        if result.get("error"):
            job_failed = True
        elif result.get("summary"):
            await speak_summary(str(result["summary"]))
    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected during job %s", job.job_id)
        job_failed = True
    except Exception as exc:  # pragma: no cover - runtime logging only
        LOGGER.exception("Processing error for job %s: %s", job.job_id, exc)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "final", "error": "Processing error."})
        job_failed = True
    finally:
        if job_failed:
            INFERENCE_JOB_FAILURES.inc()
        INFERENCE_JOBS_TOTAL.inc()
        INFERENCE_JOB_DURATION.observe(time.perf_counter() - job_start)


async def speak_summary(summary: str) -> None:
    if not summary:
        return
    engine = get_tts_engine()
    if engine is None:
        return
    global tts_lock
    if tts_lock is None:
        tts_lock = asyncio.Lock()
    async with tts_lock:
        await asyncio.to_thread(_speak_text, engine, summary)


def _speak_text(engine: pyttsx3.Engine, text: str) -> None:
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as exc:  # pragma: no cover - runtime logging only
        LOGGER.warning("Text-to-speech failed: %s", exc)


async def preload_services() -> None:
    LOGGER.info("Preloading inference services")
    await asyncio.to_thread(get_asr_service)
    await asyncio.to_thread(get_diarization_pipeline)
    get_llm_service()


@app.on_event("startup")
async def startup_event() -> None:
    global orchestrator, tts_lock
    tts_lock = asyncio.Lock()
    orchestrator = InferenceOrchestrator(
        run_inference_job,
        INFERENCE_WORKERS,
        backend_name=ORCHESTRATOR_BACKEND,
        batch_size=INFERENCE_BATCH_SIZE,
    )
    await orchestrator.start()
    update_queue_depth(0, ORCHESTRATOR_BACKEND)
    await preload_services()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = SecureAudioBuffer(AUDIO_ENCRYPTION_KEY)
    current_sample_rate = MODEL_RATE
    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")

            if message_type == "websocket.receive":
                binary_data = message.get("bytes")
                text_data = message.get("text")

                if binary_data is not None:
                    audio_buffer.append(binary_data)
                    continue

                if text_data is None:
                    continue

                try:
                    payload = json.loads(text_data)
                except json.JSONDecodeError:
                    LOGGER.warning("Received non-JSON text message: %s", text_data)
                    continue

                msg_type = payload.get("type")
                if msg_type == "start":
                    audio_buffer.reset()
                    current_sample_rate = int(payload.get("sampleRate", MODEL_RATE))
                    LOGGER.info("Starting new audio capture at %s Hz", current_sample_rate)
                    await websocket.send_json({"type": "status", "message": "Recording"})
                elif msg_type == "stop":
                    try:
                        audio_bytes = audio_buffer.to_bytes()
                    except Exception as exc:
                        LOGGER.exception("Failed to decrypt audio buffer: %s", exc)
                        await websocket.send_json({"type": "final", "error": "Audio decryption failed."})
                        audio_buffer.reset()
                        continue

                    LOGGER.info(
                        "Enqueuing audio buffer (%s bytes) for processing",
                        len(audio_bytes),
                    )
                    if not audio_bytes:
                        await websocket.send_json({"type": "final", "error": "No audio captured."})
                        audio_buffer.reset()
                        continue
                    await websocket.send_json({"type": "status", "message": "Queued for processing"})

                    if orchestrator is None:
                        LOGGER.error("Inference orchestrator not initialised")
                        await websocket.send_json({"type": "final", "error": "Inference service unavailable."})
                        audio_buffer.reset()
                        continue

                    await orchestrator.enqueue(
                        websocket=websocket,
                        audio_data=audio_bytes,
                        sample_rate=current_sample_rate,
                    )
                    audio_buffer.reset()
                else:
                    LOGGER.warning("Received unknown control message type: %s", msg_type)
            elif message_type == "websocket.disconnect":
                LOGGER.info("WebSocket disconnect received")
                break
    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected")

@app.get("/", response_class=HTMLResponse)
async def get():
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if orchestrator is not None:
        await orchestrator.stop()
        update_queue_depth(0, ORCHESTRATOR_BACKEND)
    if llm_service is not None:
        await llm_service.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
