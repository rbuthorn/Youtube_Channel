# OpenAI Sora 2 API Setup Guide

## Overview

Sora 2 is OpenAI's text-to-video generation model. This guide will help you set up and use the Sora 2 API for generating videos.

## Prerequisites

1. **OpenAI API Account**: You need an OpenAI account with access to Sora 2
2. **API Key**: Obtain your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
3. **Python 3.9+**: Ensure you have Python 3.9 or later installed

## Installation

1. **Install required packages:**
   ```bash
   pip install openai requests
   ```
   
   Or install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API key:**
   
   **Option 1: Environment Variable (Recommended)**
   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your-api-key-here"
   
   # Windows (Command Prompt)
   set OPENAI_API_KEY=your-api-key-here
   
   # Linux/Mac
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option 2: Pass as parameter**
   ```bash
   python generate_and_upload_sora.py "Your prompt" --title "My Video" --api-key your-api-key-here
   ```

## Usage

### Generate and Upload to YouTube

```bash
# Basic usage
python generate_and_upload_sora.py "A serene landscape" --title "Beautiful Landscape" --privacy public

# With custom settings
python generate_and_upload_sora.py "A cat playing piano" \
  --title "Cat Piano" \
  --description "AI generated video using Sora 2" \
  --tags ai sora openai \
  --privacy public \
  --duration 10 \
  --resolution 1920x1080 \
  --model sora-2-pro
```

## Parameters

### Video Generation Parameters

- `prompt` (required): Text description of the video you want to generate
- `--title` (required): YouTube video title
- `--description`: YouTube video description
- `--tags`: Space-separated tags for YouTube
- `--output`: Output video file path (default: temp file, deleted after upload unless `--keep-video` is used)
- `--api-key`: OpenAI API key (default: uses `OPENAI_API_KEY` env var)
- `--model`: Model to use - `sora-2` or `sora-2-pro` (default: `sora-2`)
- `--resolution`: Video resolution (default: `1280x720`)
  - Common options: `1280x720`, `1920x1080`
- `--duration`: Video duration in seconds (default: 8)
  - Max duration varies by model
- `--aspect-ratio`: Aspect ratio (default: `16:9`)
  - Options: `16:9`, `9:16`, `1:1`
- `--poll-interval`: Seconds between status checks (default: 10)
- `--max-wait`: Maximum wait time in seconds (default: 600)
- `--keep-video`: Keep the generated video file after upload

### YouTube Upload Parameters (for generate_and_upload_sora.py)

- `--title` (required): YouTube video title
- `--description`: Video description
- `--tags`: Space-separated tags
- `--category`: YouTube category ID (default: 22)
- `--privacy`: `private`, `public`, or `unlisted` (default: private)
- `--thumbnail`: Path to thumbnail image
- `--playlist`: YouTube playlist ID
- `--keep-video`: Keep generated video file after upload

## Model Comparison

### sora-2
- Standard model
- Good quality and speed balance
- Lower cost

### sora-2-pro
- Higher quality model
- Better for complex scenes
- Higher cost

## Pricing

- Pricing varies by model and video length
- Check current pricing at [OpenAI Pricing](https://platform.openai.com/pricing)
- Example: ~$0.10 per second for 720p videos with sora-2

## How It Works

1. **Job Creation**: Creates a video generation job via OpenAI API
2. **Polling**: Polls the API every 10 seconds (configurable) to check status
3. **Download**: Once completed, downloads the video from the provided URL
4. **Upload** (optional): Uploads to YouTube if using `generate_and_upload_sora.py`

## Status Messages

- `pending`: Job is queued
- `processing`: Video is being generated
- `completed`: Video is ready for download
- `failed`: Generation failed (check error message)

## Troubleshooting

### "API key not found"
- Set `OPENAI_API_KEY` environment variable, or
- Pass `--api-key` parameter

### "Access denied" or "Model not available"
- Ensure your OpenAI account has access to Sora 2
- Check your API key permissions
- Verify your account is in good standing

### "Generation timed out"
- Increase `--max-wait` time (default: 600 seconds)
- Longer videos may take more time

### "Video generation failed"
- Check the error message for details
- Try a different prompt
- Verify your API quota/credits

## Example Prompts

- "A serene landscape with mountains and a river at sunset"
- "A cat playing piano in a cozy living room"
- "Epic space battle with starships and explosions"
- "A peaceful walk through a Japanese garden in spring"
- "Time-lapse of a city skyline from day to night"

## Best Practices

1. **Be specific**: Detailed prompts produce better results
2. **Start simple**: Test with shorter durations first
3. **Monitor costs**: Track your API usage
4. **Save videos**: Use `--keep-video` if you want to keep generated files
5. **Error handling**: Scripts include retry logic and error messages

## Python API Usage

You can also use the functions directly in Python:

```python
from generate_and_upload_sora import generate_and_upload_sora

video_id = generate_and_upload_sora(
    prompt="A serene landscape with mountains",
    title="My AI Video",
    description="Generated with Sora 2",
    tags=['ai', 'sora'],
    privacy_status='public',
    model="sora-2",
    duration=8,
    resolution="1280x720"
)
```

## Support

- OpenAI Documentation: https://platform.openai.com/docs/models/sora-2
- OpenAI API Reference: https://platform.openai.com/docs/api-reference

