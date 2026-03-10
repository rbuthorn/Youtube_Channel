# YouTube Channel - AI Generated Videos

A YouTube channel that automatically generates and uploads videos using AI technology via API integration.

## Overview

This channel leverages artificial intelligence to create video content programmatically, enabling automated video generation and uploads to YouTube through API calls.

## Features

- 🤖 **AI-Powered Content Generation** - Videos are created using advanced AI models
- 📤 **Automated Uploads** - Videos are uploaded directly to YouTube via API
- ⚡ **Scalable Production** - Generate and publish videos at scale
- 🔄 **Automated Workflow** - Streamlined process from generation to publication

## How It Works

1. **Content Generation** - AI models generate video content based on configured parameters
2. **Video Processing** - Generated content is processed and formatted for YouTube
3. **API Upload** - Videos are automatically uploaded to YouTube using the YouTube Data API v3
4. **Publication** - Videos are published according to the configured schedule

## Technical Stack

- **YouTube Data API v3** for video uploads
- **OpenAI Sora 2** for AI video generation from text prompts
- Automated workflow orchestration

## Requirements

- Python 3.7+
- YouTube API credentials (OAuth 2.0)
- Google API Python Client library
- OpenAI API key with Sora 2 access

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get YouTube API credentials:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the YouTube Data API v3
   - Create OAuth 2.0 credentials (Desktop application)
   - Download the credentials JSON file and save it as `client_secrets.json`

3. **First-time authentication:**
   - Run `python generate_and_upload_sora.py` and choose the upload step
   - The OAuth flow opens a browser on first upload
   - Authorize the application and credentials will be saved to `token.pickle`

## Usage

### Video Generation and Upload (OpenAI Sora 2)

Generate a video from a text prompt and upload it to YouTube in one command:

```bash
python generate_and_upload_sora.py "A serene landscape with mountains and a river at sunset" --title "My AI Video" --privacy public
```

**Sora 2 Parameters:**
- `prompt` (required): Text description of the video
- `--title` (required): YouTube video title
- `--description`: YouTube video description
- `--tags`: Space-separated tags for YouTube
- `--model`: Model to use - `sora-2` or `sora-2-pro` (default: `sora-2`)
- `--resolution`: Video resolution (default: `1280x720`)
- `--duration`: Video duration in seconds (default: 8)
- `--aspect-ratio`: Aspect ratio (default: `16:9`)
- `--api-key`: OpenAI API key (default: uses `OPENAI_API_KEY` env var)
- `--privacy`: Privacy status - `private`, `public`, or `unlisted` (default: private)
- `--keep-video`: Keep the generated video file after upload

**Example:**
```bash
python generate_and_upload_sora.py "A cat playing piano" --title "Cat Piano" --description "AI generated video" --tags ai sora --privacy public --duration 10 --resolution 1920x1080
```

**Setup:** Set your OpenAI API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

### Upload Existing Video

Use the interactive workflow in `generate_and_upload_sora.py` and run only the upload step.

### Python Code Example

```python
from generate_and_upload_sora import generate_and_upload_sora

video_id = generate_and_upload_sora(
    prompt="A serene landscape with mountains",
    title="My AI Video",
    description="Generated with Sora 2",
    tags=['ai', 'sora', 'automation'],
    privacy_status='public',
    model="sora-2",
    duration=8,
    resolution="1280x720"
)
```

## Notes

### Sora 2 (OpenAI)
- **API Key Required**: You need an OpenAI API key with Sora 2 access
- **Cloud-based**: No local model download needed, runs on OpenAI servers
- **Text-to-Video**: Generates videos directly from text prompts
- **Pricing**: Pay-per-use based on video length and resolution
- **Setup**: Configure `OPENAI_API_KEY` and run the main script

---

*This channel demonstrates the capabilities of AI-driven content creation and automated video publishing.*

