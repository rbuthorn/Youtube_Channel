# AGENTS.md

## What this project is

This repository runs an automated AI YouTube pipeline from a single entrypoint: `generate_and_upload_sora.py`.

Given a topic prompt, it can:
1. Generate a documentary-style script.
2. Generate narration and background music.
3. Build a segmented video using Sora 2 plus still-image pan fallbacks.
4. Stitch, upscale, and sync final audio.
5. Optionally upload to YouTube with metadata, thumbnail, and playlist.

## Architecture

- `generate_and_upload_sora.py` contains the full workflow:
  - OpenAI script/prompt/tag generation
  - Sora 2 video job creation, polling, and download
  - DALL-E still image generation and panning conversion
  - ElevenLabs narration/music generation and mixing
  - FFmpeg stitch/upscale/mux operations
  - YouTube OAuth/upload helpers (inlined)
- `video_config.json` stores run configuration between steps.
- `video_output/segment_metadata.json` stores segment state for review/regeneration.

## Workflow model

- **Provider model**: video generation is Sora-only (`sora-2` or `sora-2-pro`).
- **Prompt model**: event/object-centric continuity, not person-identity locking.
- **Segmentation model**: 10-second segments with AI assignment of video vs still segments.
- **Fallback model**: failed segments fall back to still image -> panning video -> emergency placeholder.
- **Execution model**: full run or step-by-step execution (`script`, `narration`, `video`, `review/regenerate`, `upload`).

## Required credentials

- `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY` (optional `ELEVENLABS_VOICE_ID`)
- `client_secrets.json` for YouTube OAuth upload

## Primary artifacts

- `video_output/` final and intermediate video outputs
- `video_output/segment_metadata.json` regeneration/stitch metadata
- `overarching_script.txt` generated/editable script
- `narration_audio.mp3` final narration track used by the workflow
- `video_config.json` persisted settings for step execution
