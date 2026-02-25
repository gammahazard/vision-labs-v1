"""
routes/video_pipeline.py — Video production pipeline API endpoints.

PURPOSE:
    Orchestrates AI video generation by chaining:
    1. Script generation (Ollama) — breaks user prompt into scenes
    2. Scene image generation (ComfyUI) — generates visuals per scene
    3. Animation (ComfyUI + AnimateDiff) — animates key scenes
    4. Narration (Piper TTS) — text-to-speech per scene
    5. Assembly (ffmpeg) — stitches everything into a final MP4

ENDPOINTS:
    POST /api/video/script         — Generate a script from a prompt
    POST /api/video/generate       — Start full pipeline (background job)
    GET  /api/video/status/{job_id} — Poll pipeline progress
    POST /api/video/cancel/{job_id} — Cancel running pipeline
    GET  /api/video/list           — List completed videos
    GET  /api/video/download/{fn}  — Serve a completed video
"""

import os
import json
import uuid
import asyncio
import subprocess
import logging
from pathlib import Path
from datetime import datetime

import httpx
import redis as _redis
from fastapi import APIRouter
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from streams import GPU_PAUSE_KEY

logger = logging.getLogger("dashboard.video")
router = APIRouter()

COMFYUI_HOST = os.getenv("COMFYUI_HOST", "http://comfyui:8188")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
PIPER_HOST = os.getenv("PIPER_HOST", "http://piper:10200")
VIDEOS_DIR = Path("/data/videos")
WORK_DIR = Path("/tmp/video-pipeline")

# Active pipeline jobs: job_id -> job state
_jobs: dict[str, dict] = {}
_MAX_COMPLETED_JOBS = 20  # keep last N completed jobs, evict the rest

# Lazy Redis client for GPU pause flag
_pause_redis = None
GPU_PAUSE_TTL = 600  # 10 min safety — longer for video pipelines


def _get_pause_redis():
    global _pause_redis
    if _pause_redis is None:
        _pause_redis = _redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )
    return _pause_redis


def _set_gpu_pause():
    try:
        _get_pause_redis().set(GPU_PAUSE_KEY, "1", ex=GPU_PAUSE_TTL)
        logger.info("GPU pause flag SET (video pipeline)")
    except Exception as e:
        logger.warning(f"Failed to set GPU pause flag: {e}")


def _clear_gpu_pause():
    try:
        _get_pause_redis().delete(GPU_PAUSE_KEY)
        logger.info("GPU pause flag CLEARED (video pipeline)")
    except Exception as e:
        logger.warning(f"Failed to clear GPU pause flag: {e}")



# ---------------------------------------------------------------------------
# Script Generation — Ollama breaks a prompt into scenes
# ---------------------------------------------------------------------------
SCRIPT_SYSTEM_PROMPT = """You are a video script writer. Given a concept, create a structured video script.

For each scene, provide:
- scene_number: sequential integer starting at 1
- image_prompt: detailed Stable Diffusion prompt for the visual (include style, lighting, colors, composition). Always maintain character appearance consistency by repeating character descriptions.
- narration: text to be spoken aloud during this scene (1-3 sentences, natural speech)
- duration_seconds: how long this scene lasts (3-10 seconds)
- characters: list of character names that appear in this scene (empty list if none)

IMPORTANT:
- Define main characters at the start with DETAILED appearance descriptions (hair color, style, clothing, build, skin tone, distinguishing features).
- REPEAT the full character description in EVERY scene's image_prompt to maintain visual consistency.
- Include a "characters" array in each scene listing which characters appear by name.
- Keep narration natural — it should sound like a documentary, story, or explainer.
- Total scene durations should sum to approximately the target video length.

Respond ONLY with a valid JSON object with this structure:
{
  "title": "Video Title",
  "characters": [
    {"name": "Character Name", "description": "Full appearance description for SD prompts"}
  ],
  "scenes": [
    {
      "scene_number": 1,
      "image_prompt": "detailed SD prompt with character descriptions...",
      "narration": "spoken text for this scene",
      "duration_seconds": 5,
      "characters": ["Character Name"]
    }
  ]
}"""


@router.post("/api/video/script")
async def generate_script(request_body: dict):
    """Generate a video script from a text prompt using Ollama."""
    prompt = request_body.get("prompt", "")
    target_minutes = int(request_body.get("target_minutes", 5))
    model = request_body.get("model", "dolphin-llama3:8b")

    if not prompt:
        return {"error": "Prompt is required"}

    user_prompt = f"""Create a video script for the following concept:

"{prompt}"

Target video length: approximately {target_minutes} minutes ({target_minutes * 60} seconds).
Aim for {target_minutes * 12} to {target_minutes * 15} scenes.

Respond ONLY with the JSON object, no other text.
"""

    max_retries = 2
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            # Use /no_think suffix to suppress thinking tokens on qwen3
            req_model = model
            if "qwen3" in model.lower() and "/no_think" not in model:
                req_model = model  # thinking is fine, we strip it below

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": req_model,
                        "prompt": user_prompt,
                        "system": SCRIPT_SYSTEM_PROMPT,
                        "stream": False,
                        "options": {"temperature": 0.8, "num_predict": 16384},
                    },
                    timeout=600,
                )

            if resp.status_code != 200:
                last_error = f"Ollama error: {resp.text}"
                continue

            result = resp.json()
            response_text = result.get("response", "")

            logger.info(f"Raw Ollama response (first 500 chars): {response_text[:500]}")

            # Strip qwen3 <think>...</think> reasoning tags
            import re
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

            # Strip markdown code fences if present
            response_text = re.sub(r"^```(?:json)?\s*\n?", "", response_text)
            response_text = re.sub(r"\n?```\s*$", "", response_text)

            # Find JSON object in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]

            logger.info(f"Cleaned response (first 300 chars): {response_text[:300]}")

            # Parse the JSON response
            script = json.loads(response_text)

            # Validate structure
            if "scenes" not in script or not script["scenes"]:
                last_error = f"Invalid script format — missing or empty scenes (attempt {attempt + 1})"
                logger.warning(f"{last_error}. Response: {response_text[:200]}")
                if attempt < max_retries:
                    logger.info(f"Retrying script generation (attempt {attempt + 2})...")
                    continue
                return {"error": last_error}

            total_duration = sum(s.get("duration_seconds", 5) for s in script["scenes"])

            return {
                "script": script,
                "scene_count": len(script["scenes"]),
                "total_duration_seconds": total_duration,
                "estimated_minutes": round(total_duration / 60, 1),
            }

        except json.JSONDecodeError as e:
            last_error = f"Failed to parse script JSON: {e}"
            logger.warning(f"{last_error} (attempt {attempt + 1})")
            if attempt < max_retries:
                continue
        except httpx.ConnectError:
            return {"error": "Ollama is not running"}
        except (httpx.ReadTimeout, httpx.TimeoutException):
            return {"error": "Ollama timed out — try a shorter video or faster model"}
        except Exception as e:
            err_msg = str(e) or repr(e) or "Unknown error"
            logger.warning(f"Script generation error: {err_msg}")
            return {"error": err_msg}

    return {"error": last_error or "Script generation failed after retries"}


# ---------------------------------------------------------------------------
# Full Pipeline — Background job that chains all steps
# ---------------------------------------------------------------------------
@router.post("/api/video/generate")
async def start_pipeline(request_body: dict):
    """Start the full video production pipeline as a background job."""
    script = request_body.get("script")
    if not script or "scenes" not in script:
        return {"error": "Valid script with scenes is required"}

    # Image generation settings
    gen_settings = request_body.get("settings", {})

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "id": job_id,
        "status": "starting",
        "phase": "init",
        "progress": 0,
        "total_scenes": len(script["scenes"]),
        "completed_scenes": 0,
        "current_scene": 0,
        "message": "Initializing pipeline...",
        "cancelled": False,
        "started_at": datetime.now().isoformat(),
        "output_file": None,
        "errors": [],
    }

    # Launch background task
    asyncio.create_task(_run_pipeline(job_id, script, gen_settings))

    return {"job_id": job_id, "status": "started"}


async def _run_pipeline(job_id: str, script: dict, settings: dict):
    """Background task: runs the full video pipeline."""
    job = _jobs[job_id]
    scenes = script["scenes"]
    title = script.get("title", "Untitled")

    # Create work directory
    work = WORK_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)
    images_dir = work / "images"
    images_dir.mkdir(exist_ok=True)
    audio_dir = work / "audio"
    audio_dir.mkdir(exist_ok=True)
    clips_dir = work / "clips"
    clips_dir.mkdir(exist_ok=True)

    try:
        # Pause GPU detectors for the entire video pipeline
        _set_gpu_pause()

        # === PHASE 1: Generate animated clips ===
        job["phase"] = "animation"
        job["status"] = "generating_animation"
        job["message"] = "Freeing VRAM for animation..."

        # Unload Ollama models so ComfyUI gets full GPU memory for animation
        try:
            async with httpx.AsyncClient() as client:
                models_resp = await client.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
                if models_resp.status_code == 200:
                    for m in models_resp.json().get("models", []):
                        model_name = m.get("name", "")
                        if model_name:
                            await client.post(
                                f"{OLLAMA_HOST}/api/generate",
                                json={"model": model_name, "keep_alive": 0},
                                timeout=10,
                            )
                            logger.info(f"Video pipeline: unloaded Ollama model {model_name}")
        except Exception as e:
            logger.warning(f"Video pipeline: failed to unload Ollama: {e}")

        job["message"] = "Generating animated clips..."

        raw_model = settings.get("model", "")
        # Accept any model with a file extension; reject empty or placeholder values
        if raw_model and "." in raw_model and not raw_model.startswith("Loading"):
            model = raw_model
        else:
            model = "zillah.safetensors"
            logger.info(f"Invalid model '{raw_model}', using default: {model}")

        lora = settings.get("lora", "")
        lora_strength = float(settings.get("lora_strength", 0.8))
        width = int(settings.get("width", 512))
        height = int(settings.get("height", 288))  # 16:9, VRAM-safe for AnimateDiff+SDXL
        steps = int(settings.get("steps", 20))
        cfg = float(settings.get("cfg", 7.0))
        smooth = settings.get("smooth", False)

        # Phase F: Shared seed for style consistency across all scenes
        shared_seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using shared seed {shared_seed} for style consistency")

        # Pre-load character library for matching
        char_library: dict[str, dict] = {}  # name_lower -> {"name": str, "images": list, "description": str}
        char_dir = Path("/data/characters")
        if char_dir.exists():
            for cdir in char_dir.iterdir():
                meta_path = cdir / "meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        name = meta.get("name", "")
                        images = meta.get("images", [])
                        if name and images:
                            char_library[name.lower()] = {
                                "name": name,
                                "images": images,  # Phase A: use ALL images
                                "description": meta.get("description", ""),
                            }
                    except Exception:
                        pass

        for i, scene in enumerate(scenes):
            if job["cancelled"]:
                job["status"] = "cancelled"
                job["message"] = "Pipeline cancelled by user"
                return

            job["current_scene"] = i + 1
            job["message"] = f"Animating scene {i + 1}/{len(scenes)}..."
            job["progress"] = int((i / len(scenes)) * 33)  # Animation = 0-33%

            clip_path = clips_dir / f"scene_{i + 1:04d}.mp4"

            # Phase B: Match characters — prefer per-scene characters array, fallback to prompt search
            scene_chars: dict[str, list[str]] = {}
            scene_char_names = scene.get("characters", [])
            if scene_char_names:
                # Use explicit character list from script
                for char_name in scene_char_names:
                    key = char_name.lower()
                    if key in char_library:
                        scene_chars[char_library[key]["name"]] = char_library[key]["images"]
                        logger.info(f"Scene {i+1}: matched character '{char_name}' ({len(char_library[key]['images'])} refs)")
            else:
                # Fallback: substring match in image_prompt
                prompt_lower = scene.get("image_prompt", "").lower()
                for key, char_data in char_library.items():
                    if key in prompt_lower:
                        scene_chars[char_data["name"]] = char_data["images"]
                        logger.info(f"Scene {i+1}: matched character '{char_data['name']}' via prompt")

            try:
                clip_data = await _generate_scene_clip(
                    prompt=scene.get("image_prompt", ""),
                    negative="ugly, blurry, low quality, deformed, watermark, text",
                    model=model, lora=lora, lora_strength=lora_strength,
                    width=width, height=height, steps=steps, cfg=cfg,
                    num_frames=16,
                    duration_seconds=scene.get("duration_seconds", 5),
                    source_image=scene.get("source_image", "") if scene.get("source_image", "") else "",
                    character_refs=scene_chars if scene_chars else None,
                    seed=shared_seed,
                )
                if clip_data:
                    clip_path.write_bytes(clip_data)
                    job["completed_scenes"] = i + 1
                else:
                    job["errors"].append(f"Scene {i + 1}: no clip returned")
            except Exception as e:
                job["errors"].append(f"Scene {i + 1}: {e}")
                logger.warning(f"Scene {i + 1} clip failed: {e}")

        # === PHASE 1.5: Frame interpolation (optional smooth motion) ===
        if smooth:
            job["message"] = "Smoothing clip motion..."
            for clip_file in sorted(clips_dir.glob("scene_*.mp4")):
                smoothed = clips_dir / f"smooth_{clip_file.name}"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(clip_file),
                    "-vf", "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(smoothed),
                ]
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.wait()
                    if smoothed.exists() and smoothed.stat().st_size > 0:
                        smoothed.replace(clip_file)  # Overwrite original
                        logger.info(f"Smoothed {clip_file.name} to 24fps")
                    else:
                        logger.warning(f"Smooth failed for {clip_file.name}, keeping original")
                except Exception as e:
                    logger.warning(f"Interpolation error for {clip_file.name}: {e}")

        # === PHASE 2: Generate narration audio ===
        job["phase"] = "narration"
        job["status"] = "generating_narration"
        job["message"] = "Generating narration audio..."

        for i, scene in enumerate(scenes):
            if job["cancelled"]:
                job["status"] = "cancelled"
                job["message"] = "Pipeline cancelled by user"
                return

            narration = scene.get("narration", "")
            if not narration:
                continue

            job["message"] = f"Narrating scene {i + 1}/{len(scenes)}..."
            job["progress"] = 33 + int((i / len(scenes)) * 33)  # Narration = 33-66%

            audio_path = audio_dir / f"scene_{i + 1:04d}.wav"

            try:
                audio_data = await _generate_narration(narration)
                if audio_data:
                    audio_path.write_bytes(audio_data)
            except Exception as e:
                job["errors"].append(f"Scene {i + 1} narration: {e}")
                logger.warning(f"Scene {i + 1} narration failed: {e}")

        # === PHASE 3: Assembly with ffmpeg ===
        if job["cancelled"]:
            job["status"] = "cancelled"
            job["message"] = "Pipeline cancelled by user"
            return

        job["phase"] = "assembly"
        job["status"] = "assembling"
        job["message"] = "Assembling final video..."
        job["progress"] = 66

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)[:50]
        output_filename = f"{ts}_{safe_title}.mp4"
        output_path = work / output_filename

        try:
            await _assemble_video(scenes, clips_dir, audio_dir, output_path)
            job["progress"] = 90

            # Phase E: Optional background music mixing
            music_dir = Path("/data/music")
            if music_dir.exists():
                music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
                if music_files:
                    music_track = random.choice(music_files)
                    mixed_path = work / "mixed_final.mp4"
                    job["message"] = f"Mixing background music ({music_track.name})..."
                    mix_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(output_path),
                        "-stream_loop", "-1", "-i", str(music_track),
                        "-filter_complex",
                        "[1:a]volume=0.12[bg]; [0:a][bg]amix=inputs=2:duration=first[aout]",
                        "-map", "0:v", "-map", "[aout]",
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "128k",
                        "-shortest",
                        str(mixed_path),
                    ]
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            *mix_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await proc.wait()
                        if mixed_path.exists() and mixed_path.stat().st_size > 0:
                            mixed_path.replace(output_path)
                            logger.info(f"Mixed background music: {music_track.name}")
                        else:
                            logger.warning("Music mixing failed, keeping original audio")
                    except Exception as e:
                        logger.warning(f"Music mixing error: {e}")

            # Copy to QNAP videos folder
            VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            final_path = VIDEOS_DIR / output_filename
            import shutil
            shutil.copy2(str(output_path), str(final_path))

            # Save script as metadata
            meta_path = VIDEOS_DIR / f"{ts}_{safe_title}_script.json"
            meta_path.write_text(json.dumps(script, indent=2))

            job["output_file"] = output_filename
            job["status"] = "complete"
            job["progress"] = 100
            job["message"] = f"Video complete: {output_filename}"
            job["completed_at"] = datetime.now().isoformat()
            logger.info(f"Video pipeline complete: {output_filename}")

            # Evict old completed jobs to prevent memory leak
            _evict_completed_jobs()

        except Exception as e:
            job["status"] = "error"
            job["message"] = f"Assembly failed: {e}"
            job["errors"].append(f"Assembly: {e}")
            logger.error(f"Video assembly failed: {e}")

    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Pipeline error: {e}"
        job["completed_at"] = datetime.now().isoformat()
        logger.error(f"Video pipeline error: {e}")
        _evict_completed_jobs()
    finally:
        # Always resume GPU detectors when pipeline finishes
        _clear_gpu_pause()


def _evict_completed_jobs():
    """Keep only the last N completed/errored/cancelled jobs in memory."""
    terminal = [jid for jid, j in _jobs.items() if j.get("status") in ("complete", "error", "cancelled")]
    if len(terminal) > _MAX_COMPLETED_JOBS:
        # Sort by completion time, evict oldest
        terminal.sort(key=lambda jid: _jobs[jid].get("completed_at", ""))
        for jid in terminal[:-_MAX_COMPLETED_JOBS]:
            _jobs.pop(jid, None)
        logger.info(f"Evicted {len(terminal) - _MAX_COMPLETED_JOBS} old pipeline jobs")


# ---------------------------------------------------------------------------
# Helper: Build AnimateDiff workflow for ComfyUI
# ---------------------------------------------------------------------------
import random

def _build_animatediff_workflow(
    prompt: str,
    negative_prompt: str = "",
    model: str = "",
    width: int = 1024,
    height: int = 576,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    num_frames: int = 16,
    fps: int = 8,
    lora: str = "",
    lora_strength: float = 0.8,
    motion_model: str = "mm_sdxl_v10_beta.ckpt",
    source_image: str = "",
    character_refs: dict[str, str] | None = None,
) -> dict:
    """Build a ComfyUI API workflow for AnimateDiff animated clip generation."""
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)

    model_ref = ["4", 0]  # CheckpointLoader output
    clip_ref = ["4", 1]

    workflow = {
        # Checkpoint loader
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": model or "zillah.safetensors",
            },
        },
        # AnimateDiff loader — applies motion model to the checkpoint
        "20": {
            "class_type": "ADE_AnimateDiffLoaderWithContext",
            "inputs": {
                "model": model_ref,
                "model_name": motion_model,
                "beta_schedule": "sqrt_linear (AnimateDiff)",
                "context_options": ["21", 0],
            },
        },
        # Context options — how many frames to generate
        "21": {
            "class_type": "ADE_AnimateDiffUniformContextOptions",
            "inputs": {
                "context_length": num_frames,
                "context_stride": 1,
                "context_overlap": 4,
                "context_schedule": "uniform",
                "closed_loop": False,
            },
        },
    }

    # LoRA loader (optional)
    if lora:
        workflow["10"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
                "model": ["4", 0],
                "clip": ["4", 1],
            },
        }
        # AnimateDiff gets model from LoRA, CLIP from LoRA
        workflow["20"]["inputs"]["model"] = ["10", 0]
        clip_ref = ["10", 1]

    # Positive prompt
    workflow["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": prompt,
            "clip": clip_ref,
        },
    }

    # Negative prompt
    workflow["7"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": negative_prompt or "ugly, blurry, low quality, deformed, watermark, text, static",
            "clip": clip_ref,
        },
    }

    # Latent source: either from uploaded image or empty latent
    if source_image:
        # Image-to-Video: LoadImage → VAEEncode → RepeatLatentBatch
        workflow["40"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": source_image,
            },
        }
        workflow["41"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["40", 0],
                "vae": ["4", 2],
            },
        }
        workflow["42"] = {
            "class_type": "RepeatLatentBatch",
            "inputs": {
                "samples": ["41", 0],
                "amount": num_frames,
            },
        }
        latent_ref = ["42", 0]
        denoise = 0.75  # Keep most of the source image but add motion
    else:
        # Text-to-Video: EmptyLatentImage
        workflow["5"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": num_frames,
            },
        }
        latent_ref = ["5", 0]
        denoise = 1.0

    # --- IPAdapter face conditioning (character consistency) ---
    # Chain: AnimateDiff model → IPAdapter(s) → KSampler
    final_model_ref = ["20", 0]  # Start with AnimateDiff model output

    if character_refs:
        for i, (char_name, ref_images) in enumerate(character_refs.items()):
            # Phase A: support list of images (use first for IPAdapter, rest for future batch encode)
            if isinstance(ref_images, list):
                ref_image = ref_images[0]  # Primary reference
                extra_refs = ref_images[1:4]  # Up to 3 extra refs
            else:
                ref_image = ref_images
                extra_refs = []
            logger.info(f"Injecting IPAdapter for character '{char_name}' with {1 + len(extra_refs)} ref image(s)")

            # Load CLIP Vision encoder (shared, but one per character for simplicity)
            workflow[f"5{i}0"] = {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
            }

            # Load IP-Adapter model
            workflow[f"5{i}1"] = {
                "class_type": "IPAdapterModelLoader",
                "inputs": {"ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"},
            }

            # Load character reference image (primary)
            workflow[f"5{i}2"] = {
                "class_type": "LoadImage",
                "inputs": {"image": ref_image},
            }

            # Apply IP-Adapter — chains model from previous stage
            workflow[f"5{i}3"] = {
                "class_type": "IPAdapterApply",
                "inputs": {
                    "ipadapter": [f"5{i}1", 0],
                    "clip_vision": [f"5{i}0", 0],
                    "image": [f"5{i}2", 0],
                    "model": final_model_ref,
                    "weight": 0.7,
                    "noise": 0.0,
                    "weight_type": "linear",
                    "start_at": 0.0,
                    "end_at": 1.0,
                },
            }
            final_model_ref = [f"5{i}3", 0]  # Chain to next

    # KSampler — uses final model (AnimateDiff → IPAdapter chain)
    workflow["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": denoise,
            "model": final_model_ref,
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": latent_ref,
        },
    }

    # VAE Decode
    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2],
        },
    }

    # VideoHelperSuite — save as MP4
    workflow["30"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["8", 0],
            "frame_rate": fps,
            "loop_count": 0,
            "filename_prefix": "visionlabs_anim",
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True,
        },
    }

    # Also save as images for fallback
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "visionlabs_anim",
            "images": ["8", 0],
        },
    }

    return workflow


# ---------------------------------------------------------------------------
# Helper: Generate an animated scene clip via ComfyUI + AnimateDiff
# ---------------------------------------------------------------------------
async def _generate_scene_clip(
    prompt: str, negative: str, model: str, lora: str,
    lora_strength: float, width: int, height: int, steps: int, cfg: float,
    num_frames: int = 16, duration_seconds: int = 5,
    source_image: str = "",
    character_refs: dict[str, list[str]] | None = None,
    seed: int = -1,
) -> bytes | None:
    """Queue an AnimateDiff clip and wait for result. Returns MP4 bytes."""
    # Validate source_image exists in ComfyUI before building workflow
    if source_image:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{COMFYUI_HOST}/view",
                    params={"filename": source_image, "type": "input"},
                    timeout=5,
                )
                if resp.status_code != 200:
                    logger.warning(f"Source image '{source_image}' not found in ComfyUI, using txt2vid")
                    source_image = ""
        except Exception:
            logger.warning(f"Could not validate source image '{source_image}', using txt2vid")
            source_image = ""

    workflow = _build_animatediff_workflow(
        prompt=prompt, negative_prompt=negative, model=model,
        width=width, height=height, steps=steps, cfg=cfg,
        seed=seed, num_frames=num_frames,
        fps=max(4, round(num_frames / max(1, duration_seconds))),
        lora=lora, lora_strength=lora_strength,
        source_image=source_image,
        character_refs=character_refs,
    )

    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{COMFYUI_HOST}/prompt", json=payload, timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(f"ComfyUI queue failed: {resp.text}")

    prompt_id = resp.json().get("prompt_id", "")

    # Poll for completion (up to 30 min)
    for attempt in range(1800):
        await asyncio.sleep(1)

        async with httpx.AsyncClient() as client:
            hist_resp = await client.get(
                f"{COMFYUI_HOST}/history/{prompt_id}", timeout=10
            )

        if hist_resp.status_code != 200:
            continue

        data = hist_resp.json()
        if prompt_id not in data:
            continue

        history = data[prompt_id]
        if history.get("status", {}).get("status_str") == "error":
            raise RuntimeError("ComfyUI AnimateDiff generation failed")

        outputs = history.get("outputs", {})
        for node_output in outputs.values():
            # Check for video output first (VHS_VideoCombine)
            if "gifs" in node_output:
                vid = node_output["gifs"][0]
                async with httpx.AsyncClient() as vid_client:
                    vid_resp = await vid_client.get(
                        f"{COMFYUI_HOST}/view",
                        params={
                            "filename": vid["filename"],
                            "subfolder": vid.get("subfolder", ""),
                            "type": vid.get("type", "output"),
                        },
                        timeout=60,
                    )
                if vid_resp.status_code == 200:
                    return vid_resp.content

            # Fallback: collect images and assemble with ffmpeg
            if "images" in node_output:
                frames = node_output["images"]
                if len(frames) > 1:
                    return await _frames_to_mp4(frames, duration_seconds)

    raise RuntimeError("Scene clip generation timed out")


async def _frames_to_mp4(frames: list, duration_seconds: int) -> bytes:
    """Download individual frames from ComfyUI and assemble into MP4."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for i, frame in enumerate(frames):
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{COMFYUI_HOST}/view",
                    params={
                        "filename": frame["filename"],
                        "subfolder": frame.get("subfolder", ""),
                        "type": frame.get("type", "output"),
                    },
                    timeout=30,
                )
            if resp.status_code == 200:
                (tmpdir_path / f"frame_{i:04d}.png").write_bytes(resp.content)

        fps = max(2, len(frames) // max(1, duration_seconds))
        out_path = tmpdir_path / "clip.mp4"

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir_path / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-t", str(duration_seconds),
            str(out_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()

        if out_path.exists():
            return out_path.read_bytes()

    raise RuntimeError("Failed to assemble frames into clip")


# ---------------------------------------------------------------------------
# Helper: Generate narration audio via Piper TTS (Wyoming protocol)
# ---------------------------------------------------------------------------
async def _generate_narration(text: str) -> bytes | None:
    """Send text to Piper TTS via Wyoming protocol and return raw PCM audio bytes.

    Uses the proper ``wyoming`` library types (Synthesize, AudioChunk, AudioStop)
    instead of raw Event construction.
    """
    try:
        from wyoming.tts import Synthesize
        from wyoming.audio import AudioChunk, AudioStop
        from wyoming.event import async_write_event, async_read_event

        piper_host = PIPER_HOST.replace("http://", "").split(":")[0]
        piper_port = int(PIPER_HOST.split(":")[-1])

        reader, writer = await asyncio.open_connection(piper_host, piper_port)

        # Send synthesize request using the proper Wyoming type
        synth = Synthesize(text=text)
        await async_write_event(synth.event(), writer)

        # Read response events
        audio_chunks = []
        try:
            while True:
                event = await asyncio.wait_for(
                    async_read_event(reader), timeout=60
                )
                if event is None:
                    break
                if AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    if chunk.audio:
                        audio_chunks.append(chunk.audio)
                elif AudioStop.is_type(event.type):
                    break
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            pass

        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        if audio_chunks:
            return b"".join(audio_chunks)

        logger.warning("Piper TTS returned no audio chunks")
        return None

    except ConnectionRefusedError:
        logger.warning("Piper TTS not reachable — narration skipped")
        return None
    except Exception as e:
        logger.warning(f"Piper TTS error: {e}")
        return None


# ---------------------------------------------------------------------------
# Helper: Assemble final video with ffmpeg
# ---------------------------------------------------------------------------
async def _assemble_video(
    scenes: list, clips_dir: Path, audio_dir: Path,
    output_path: Path,
):
    """Combine animated scene clips + narration audio into final MP4."""
    final_clips = []

    for i, scene in enumerate(scenes):
        clip_path = clips_dir / f"scene_{i + 1:04d}.mp4"
        audio_path = audio_dir / f"scene_{i + 1:04d}.wav"
        final_clip = clips_dir / f"final_{i + 1:04d}.mp4"
        duration = scene.get("duration_seconds", 5)

        if not clip_path.exists():
            continue

        if audio_path.exists():
            # Merge clip + narration, pad/trim to target duration
            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", str(clip_path),
                "-i", str(audio_path),
                "-c:v", "libx264", "-c:a", "aac", "-b:a", "128k",
                "-t", str(duration),
                "-shortest",
                "-pix_fmt", "yuv420p",
                str(final_clip),
            ]
        else:
            # Loop clip to fill target duration, no audio
            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", str(clip_path),
                "-c:v", "libx264",
                "-t", str(duration),
                "-an", "-pix_fmt", "yuv420p",
                str(final_clip),
            ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()

        if final_clip.exists():
            final_clips.append(final_clip)

    if not final_clips:
        raise RuntimeError("No clips generated")

    if len(final_clips) == 1:
        # Single clip — just copy to output
        import shutil
        shutil.copy2(str(final_clips[0]), str(output_path))
        logger.info(f"Single clip copied to {output_path}")
        return

    # Phase D: Crossfade transitions between scenes
    xfade_duration = 0.5  # seconds overlap
    try:
        current = final_clips[0]
        for idx in range(1, len(final_clips)):
            next_clip = final_clips[idx]
            xfade_out = clips_dir / f"xfade_{idx:04d}.mp4"

            # Probe duration of current clip
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(current),
            ]
            probe = await asyncio.create_subprocess_exec(
                *probe_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            probe_out, _ = await probe.communicate()
            try:
                cur_duration = float(probe_out.decode().strip())
            except (ValueError, AttributeError):
                cur_duration = 5.0

            offset = max(0, cur_duration - xfade_duration)

            # Try audio+video crossfade first
            cmd_av = [
                "ffmpeg", "-y",
                "-i", str(current),
                "-i", str(next_clip),
                "-filter_complex",
                f"[0:v][1:v]xfade=transition=fade:duration={xfade_duration}:offset={offset}[v];"
                f"[0:a][1:a]acrossfade=d={xfade_duration}[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                str(xfade_out),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd_av,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()

            if proc.returncode != 0 or not xfade_out.exists():
                # Fallback: video-only xfade
                cmd_v = [
                    "ffmpeg", "-y",
                    "-i", str(current),
                    "-i", str(next_clip),
                    "-filter_complex",
                    f"[0:v][1:v]xfade=transition=fade:duration={xfade_duration}:offset={offset}[v]",
                    "-map", "[v]",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-an",
                    str(xfade_out),
                ]
                proc = await asyncio.create_subprocess_exec(
                    *cmd_v,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.wait()

            if xfade_out.exists() and xfade_out.stat().st_size > 0:
                current = xfade_out
            else:
                logger.warning(f"xfade failed at clip {idx}, falling back to concat")
                raise RuntimeError("xfade failed")

        import shutil
        shutil.copy2(str(current), str(output_path))
        logger.info(f"Assembled {len(final_clips)} clips with crossfade transitions")
        return

    except Exception as e:
        logger.warning(f"xfade assembly failed ({e}), falling back to concat")

    # Fallback: simple concat (original behavior)
    concat_path = clips_dir / "concat.txt"
    with open(concat_path, "w") as f:
        for clip in final_clips:
            f.write(f"file '{clip}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_path),
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {stderr.decode()[:500]}")

    logger.info(f"Assembled {len(final_clips)} clips (concat fallback)")


# ---------------------------------------------------------------------------
# Status + Cancel + List + Download
# ---------------------------------------------------------------------------
@router.get("/api/video/status/{job_id}")
async def get_pipeline_status(job_id: str):
    """Get the current status of a video pipeline job."""
    job = _jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return job


@router.post("/api/video/cancel/{job_id}")
async def cancel_pipeline(job_id: str):
    """Cancel a running video pipeline job and interrupt ComfyUI."""
    job = _jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    job["cancelled"] = True

    # Also interrupt ComfyUI's current render
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{COMFYUI_HOST}/interrupt", timeout=5)
    except Exception:
        pass

    # Resume GPU detectors
    _clear_gpu_pause()

    return {"status": "cancelling"}


@router.get("/api/video/list")
async def list_videos():
    """List completed videos."""
    if not VIDEOS_DIR.exists():
        return {"videos": []}

    videos = []
    for f in sorted(VIDEOS_DIR.glob("*.mp4"), reverse=True):
        try:
            stat = f.stat()
            videos.append({
                "filename": f.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "modified": stat.st_mtime,
            })
        except OSError:
            continue

    return {"videos": videos, "total": len(videos)}


@router.get("/api/video/download/{filename}")
async def download_video(filename: str):
    """Serve a completed video file."""
    safe_name = Path(filename).name
    if safe_name != filename:
        return {"error": "Invalid filename"}

    video_path = VIDEOS_DIR / safe_name
    if not video_path.exists():
        return {"error": "Video not found"}

    return FileResponse(str(video_path), media_type="video/mp4")


@router.delete("/api/video/{filename}")
async def delete_video(filename: str):
    """Delete a completed video file."""
    safe_name = Path(filename).name
    if safe_name != filename:
        return JSONResponse(status_code=400, content={"error": "Invalid filename"})

    video_path = VIDEOS_DIR / safe_name
    if not video_path.exists():
        return JSONResponse(status_code=404, content={"error": "Video not found"})

    try:
        video_path.unlink()
        logger.info(f"Deleted video: {safe_name}")
        return {"ok": True, "deleted": safe_name}
    except OSError as e:
        logger.error(f"Failed to delete video {safe_name}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to delete: {str(e)}"})


# ---------------------------------------------------------------------------
# Scene Image Upload — Upload reference images for i2v animation
# ---------------------------------------------------------------------------
IMAGES_DIR = Path("/data/images")


@router.post("/api/video/scene-image")
async def upload_scene_image(file: UploadFile = File(...)):
    """Upload an image to ComfyUI's input folder for i2v scene animation."""
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "File must be an image"}

    # Read file contents
    contents = await file.read()

    # Upload to ComfyUI's input folder via API
    import io
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{COMFYUI_HOST}/upload/image",
            files={"image": (file.filename, io.BytesIO(contents), file.content_type)},
            data={"overwrite": "true"},
            timeout=30,
        )

    if resp.status_code != 200:
        return {"error": f"ComfyUI upload failed: {resp.text}"}

    result = resp.json()
    return {
        "filename": result.get("name", file.filename),
        "subfolder": result.get("subfolder", ""),
    }


@router.get("/api/video/gallery-images")
async def list_gallery_images():
    """List generated images from gallery for scene reference picking."""
    images = []
    if IMAGES_DIR.exists():
        for f in sorted(IMAGES_DIR.iterdir(), reverse=True):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                images.append({
                    "filename": f.name,
                    "url": f"/api/gallery/images/{f.name}",
                })
            if len(images) >= 50:  # Cap at 50 most recent
                break
    return {"images": images}


# ---------------------------------------------------------------------------
# Character Management — CRUD for character reference images
# ---------------------------------------------------------------------------
CHARACTERS_DIR = Path("/data/characters")


@router.get("/api/video/characters")
async def list_characters():
    """List all characters with their reference images."""
    characters = []
    if CHARACTERS_DIR.exists():
        for cdir in sorted(CHARACTERS_DIR.iterdir()):
            meta_path = cdir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    meta["image_count"] = len(meta.get("images", []))
                    characters.append(meta)
                except Exception:
                    pass
    return {"characters": characters}


@router.post("/api/video/characters")
async def create_character(body: dict):
    """Create a new character with name and description."""
    name = body.get("name", "").strip()
    description = body.get("description", "").strip()

    if not name:
        return {"error": "Character name is required"}

    # Sanitize directory name
    safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in name)[:50].strip()
    if not safe_name:
        return {"error": "Invalid character name"}

    char_dir = CHARACTERS_DIR / safe_name.lower().replace(" ", "_")
    char_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "name": name,
        "description": description,
        "images": [],
        "dir": safe_name.lower().replace(" ", "_"),
    }
    (char_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {"ok": True, "character": meta}


@router.post("/api/video/characters/{name}/image")
async def upload_character_image(name: str, file: UploadFile = File(...)):
    """Upload a reference image for a character."""
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "File must be an image"}

    # Find character directory
    char_dir = CHARACTERS_DIR / name.lower().replace(" ", "_")
    meta_path = char_dir / "meta.json"
    if not meta_path.exists():
        return {"error": f"Character '{name}' not found"}

    meta = json.loads(meta_path.read_text())

    # Limit to 6 images
    if len(meta.get("images", [])) >= 6:
        return {"error": "Maximum 6 reference images per character"}

    contents = await file.read()

    # Save locally with unique name
    ext = Path(file.filename).suffix or ".png"
    img_name = f"ref_{len(meta['images']) + 1}{ext}"
    local_path = char_dir / img_name
    local_path.write_bytes(contents)

    # Upload to ComfyUI input folder so LoadImage can find it
    import io
    comfyui_filename = f"char_{name.lower().replace(' ', '_')}_{img_name}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{COMFYUI_HOST}/upload/image",
                files={"image": (comfyui_filename, io.BytesIO(contents), file.content_type)},
                data={"overwrite": "true"},
                timeout=30,
            )
        if resp.status_code == 200:
            result = resp.json()
            comfyui_filename = result.get("name", comfyui_filename)
        else:
            logger.warning(f"ComfyUI upload failed for character image: {resp.text}")
    except Exception as e:
        logger.warning(f"Could not upload character image to ComfyUI: {e}")

    # Update metadata — store ComfyUI filename for workflow injection
    meta.setdefault("images", []).append(comfyui_filename)
    meta_path.write_text(json.dumps(meta, indent=2))

    return {"ok": True, "filename": comfyui_filename, "image_count": len(meta["images"])}


@router.put("/api/video/characters/{name}")
async def update_character(name: str, body: dict):
    """Update a character's description."""
    char_dir = CHARACTERS_DIR / name.lower().replace(" ", "_")
    meta_path = char_dir / "meta.json"
    if not meta_path.exists():
        return {"error": f"Character '{name}' not found"}

    meta = json.loads(meta_path.read_text())
    if "description" in body:
        meta["description"] = body["description"]
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"ok": True, "character": meta}


@router.delete("/api/video/characters/{name}")
async def delete_character(name: str):
    """Delete a character and all its reference images."""
    char_dir = CHARACTERS_DIR / name.lower().replace(" ", "_")
    if not char_dir.exists():
        return {"error": f"Character '{name}' not found"}

    import shutil
    shutil.rmtree(str(char_dir), ignore_errors=True)
    return {"ok": True}


@router.delete("/api/video/characters/{name}/image/{index}")
async def delete_character_image(name: str, index: int):
    """Delete a specific reference image from a character."""
    char_dir = CHARACTERS_DIR / name.lower().replace(" ", "_")
    meta_path = char_dir / "meta.json"
    if not meta_path.exists():
        return {"error": f"Character '{name}' not found"}

    meta = json.loads(meta_path.read_text())
    images = meta.get("images", [])

    if index < 0 or index >= len(images):
        return {"error": "Image index out of range"}

    removed = images.pop(index)
    meta["images"] = images
    meta_path.write_text(json.dumps(meta, indent=2))

    # Delete matching local reference file
    local_ref = char_dir / f"ref_{index + 1}{Path(removed).suffix or '.png'}"
    if local_ref.exists():
        try:
            local_ref.unlink()
            logger.info(f"Deleted local character image: {local_ref}")
        except OSError as e:
            logger.warning(f"Could not delete local character image {local_ref}: {e}")

    return {"ok": True, "removed": removed, "remaining": len(images)}


@router.get("/api/video/characters/{name}/image/{filename}")
async def serve_character_image(name: str, filename: str):
    """Serve a character reference image from ComfyUI."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{COMFYUI_HOST}/view",
                params={"filename": filename, "type": "input"},
                timeout=10,
            )
        if resp.status_code == 200:
            from fastapi.responses import Response
            return Response(
                content=resp.content,
                media_type=resp.headers.get("content-type", "image/png"),
            )
    except Exception:
        pass

    # Fallback: try local file
    char_dir = CHARACTERS_DIR / name.lower().replace(" ", "_")
    for f in char_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            return FileResponse(str(f))

    return {"error": "Image not found"}
