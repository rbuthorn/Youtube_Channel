
"""
Complete workflow: generate Sora videos and upload to YouTube.
"""

import os
import argparse
import time
import requests
import subprocess
import shutil
import glob
import re
import uuid
import base64
import json
import io
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Audio processing imports (narration generated via ElevenLabs TTS API)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not installed. Install with: pip install openai")

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Warning: Google API client libraries are not installed. Install with: pip install google-api-python-client google-auth-oauthlib")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Thumbnail optimization will be limited. Install with: pip install Pillow")

# Global API key (should be set via environment variable OPENAI_API_KEY or command-line argument)
# For security, do not hardcode API keys in the source code
OPENAI_API_KEY = None  # Will use os.getenv('OPENAI_API_KEY') or command-line argument

# OpenAI Video API configuration (Sora 2)
OPENAI_VIDEO_API_BASE_URL = "https://api.openai.com/v1"
SORA_ALLOWED_SIZES = {"720x1280", "1280x720", "1024x1792", "1792x1024"}
SORA_ALLOWED_SECONDS = {"12"}
FIXED_SEGMENT_DURATION_SECONDS = 12.0
FIXED_SEGMENT_DURATION_INT = int(FIXED_SEGMENT_DURATION_SECONDS)

# ElevenLabs API configuration
# Set ELEVENLABS_API_KEY via environment variable or command-line argument --elevenlabs-api-key
ELEVENLABS_API_KEY = None  # Will use os.getenv('ELEVENLABS_API_KEY') or command-line argument
ELEVENLABS_VOICE_ID = None  # Will use os.getenv('ELEVENLABS_VOICE_ID') or command-line argument --elevenlabs-voice-id
ELEVENLABS_API_BASE_URL = "https://api.elevenlabs.io/v1"
thumbnail_prompt_template = "Create a hyper-realistic, photojournalistic-quality photograph that tells a story and provokes immediate curiosity. The scene should look like it was captured in the real world by a professional photographer — NOT stylized, NOT CGI, NOT cinematic or fantastical. It should show a specific, intriguing moment that makes the viewer think 'What is happening here? I need to know more.' Use natural lighting, authentic textures, real-world environments, and candid human expressions or body language where relevant. The image must feel like a frozen moment from a real story, with visual tension or an unanswered question baked into the composition. Video topic: {description}. The image must comply with content policies: no violence, hate, adult content, illegal activity, or copyrighted characters."
master_image_prompt_template = "Create the most hyperrealistic, ultra-detailed, high-quality, photorealistic reference frame image possible for a video with the description: {description}. The image must be extremely realistic and lifelike, as if photographed by a professional documentary photographer, with maximum detail, photorealism, and natural lighting. Make it look like a real photograph, not an illustration or artwork. The image must comply with OpenAI content policies: no violence, hate, adult content, illegal activity, copyrighted characters, or likenesses of real people. Use generic, artistic representations only, but make them appear completely realistic and photographic."

def get_openai_api_key(api_key=None):
    """Resolve OpenAI API key from argument, env, or global."""
    if api_key:
        return api_key
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    return OPENAI_API_KEY


def get_openai_auth_headers(api_key=None):
    """Build authorization headers for OpenAI REST requests."""
    key = get_openai_api_key(api_key=api_key)
    if not key:
        raise ValueError("OPENAI_API_KEY is required for Sora 2 video generation.")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }


def map_resolution_to_sora_size(resolution):
    """Map arbitrary WxH resolution to nearest allowed Sora API size."""
    if not resolution:
        return "1280x720"
    if resolution in SORA_ALLOWED_SIZES:
        return resolution
    try:
        w, h = resolution.lower().split("x")
        w = int(w.strip())
        h = int(h.strip())
        if w >= h:
            # landscape
            if w >= 1700:
                return "1792x1024"
            return "1280x720"
        # portrait
        if h >= 1700:
            return "1024x1792"
        return "720x1280"
    except Exception:
        return "1280x720"


def map_duration_to_sora_seconds(duration):
    """Return the fixed Sora seconds value used by this workflow."""
    return "12"


def generate_visual_continuity_description(script, video_prompt, api_key=None, model='gpt-5-2025-08-07'):
    """
    Generate a compact event/object-centric continuity anchor that gets prepended to every
    segment prompt. This keeps environmental tone and visual grammar consistent.
    
    Args:
        script: The full script text
        video_prompt: The original video prompt/topic
        api_key: OpenAI API key
        model: Model to use
        
    Returns:
        A detailed visual description string (~800-1000 characters), or empty string if generation fails.
    """
    if not OPENAI_AVAILABLE:
        print("⚠️  OpenAI not available — skipping visual continuity description")
        return ""
    
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    analysis_prompt = f"""Analyze this video script and produce a DETAILED VISUAL CONTINUITY DESCRIPTION.
This text will be prepended to every segment prompt so all segments maintain consistent
style, environment cues, and story-world details.

Video topic: {video_prompt}

Script:
{script[:3000]}

YOUR TASK: Write one dense paragraph that locks continuity for things and events, not people:

1. Prioritize places, objects, events, era details, weather, architecture, materials, and motion cues.
2. Include stable cinematic style controls: lens behavior, camera movement tempo, lighting, color palette, grain, realism level.
3. Include recurring motifs/props/setting markers that should persist across segments.
4. Avoid named people, facial attributes, outfits, or any person-identity continuity requirements.
5. Write as direct descriptive prose, not meta instructions.

CRITICAL CONSTRAINTS:
- Description should be 700-1000 characters.
- Keep it high-information and reusable for every segment.

Provide ONLY the visual description paragraph (no labels, no quotes, no explanation):"""
    
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You are a visual design expert for documentary-style AI video generation. You produce precise event/object-centric continuity anchors and avoid person-specific identity constraints."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_output_tokens=2000
        )
        
        description = response.output_text.strip()
        
        # Clean up any labels or formatting
        for prefix in ["Visual Description:", "Description:", "Visual Continuity:", "Continuity Description:"]:
            if description.startswith(prefix):
                description = description[len(prefix):].strip()
        description = description.strip('"\'')
        
        # Validate length and compress if needed
        if len(description) > 1100:
            print(f"   Visual description is {len(description)} chars — compressing...")
            compress_prompt = f"""Compress this continuity description to 850-1000 characters.
Preserve event/object/environment/style details. Remove person-identity details and filler.
Return only the compressed paragraph:

{description}"""
            compress_response = client.responses.create(
                model='gpt-4o-mini',
                input=[
                    {"role": "system", "content": "Compress text to a specific character count while preserving all visual details."},
                    {"role": "user", "content": compress_prompt}
                ],
                max_output_tokens=1500
            )
            compressed = compress_response.output_text.strip().strip('"\'')
            if 400 <= len(compressed) <= 1200:
                description = compressed
        
        # Hard cap at 1100 characters as safety net
        if len(description) > 1100:
            description = description[:1097] + "..."
        
        if len(description) < 120:
            print(f"⚠️  Visual continuity description too short ({len(description)} chars) — may not be useful")
        
        return description
        
    except Exception as e:
        print(f"⚠️  Visual continuity description generation failed: {e}")
        return ""

def find_ffmpeg():
    """
    Find ffmpeg executable, checking PATH and common installation locations.
    
    Returns:
        Path to ffmpeg executable, or None if not found
    """
    # First try PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # Try common Windows installation locations
    common_paths = [
        # WinGet installation location
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 
                    'Microsoft', 'WinGet', 'Packages', 
                    'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe',
                    'ffmpeg-8.0.1-full_build', 'bin', 'ffmpeg.exe'),
        # Try to find any version in WinGet packages
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 
                    'Microsoft', 'WinGet', 'Packages', 
                    'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe',
                    'ffmpeg-*', 'bin', 'ffmpeg.exe'),
        # Other common locations
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        os.path.join(os.environ.get('ProgramFiles', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
    ]
    
    # Try to find ffmpeg in common locations
    for path in common_paths:
        # Handle wildcard in path
        if '*' in path:
            matches = glob.glob(path)
            if matches:
                # Find the most recent version (sort by path, longest = most recent usually)
                matches.sort(key=len, reverse=True)
                path = matches[0]
        
        if os.path.exists(path):
            return path
    
    return None


def find_thumbnail_file(directory=None):
    """
    Find thumbnail_file image in the directory, checking common image extensions.
    
    Args:
        directory: Directory to search in (default: current working directory)
    
    Returns:
        Path to thumbnail file if found, or None if not found
    """
    if directory is None:
        directory = os.getcwd()
    
    # Common image extensions to check
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    
    # Check for thumbnail_file with various extensions
    for ext in image_extensions:
        possible_names = [
            os.path.join(directory, f"thumbnail_file{ext}"),
            os.path.join(directory, f"thumbnail_file{ext.upper()}"),
            f"thumbnail_file{ext}",
            f"thumbnail_file{ext.upper()}",
        ]
        
        for thumb_file in possible_names:
            if os.path.exists(thumb_file):
                return thumb_file
    
    return None


# YouTube API scopes
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'


def get_authenticated_service(client_secrets_file='client_secrets.json'):
    """
    Authenticate and return a YouTube API service object.
    """
    if not GOOGLE_API_AVAILABLE:
        raise ImportError(
            "Google API libraries are required for uploads. "
            "Install with: pip install google-api-python-client google-auth-oauthlib"
        )

    creds = None
    token_updated = False

    if os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing token: {e}")
            creds = None

    needs_refresh = False
    if creds:
        if not creds.valid or creds.expired:
            needs_refresh = True
        elif creds.expiry:
            time_until_expiry = creds.expiry - datetime.utcnow()
            if time_until_expiry < timedelta(hours=1):
                needs_refresh = True

    if needs_refresh:
        if creds and creds.refresh_token:
            try:
                creds.refresh(Request())
                token_updated = True
            except Exception:
                creds = None
        else:
            creds = None

    if not creds or not creds.valid:
        if not os.path.exists(client_secrets_file):
            raise FileNotFoundError(
                f"Client secrets file not found: {client_secrets_file}\n"
                f"Download OAuth credentials and save as '{client_secrets_file}'."
            )
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
        creds = flow.run_local_server(port=0)
        token_updated = True

    if token_updated or not os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        except Exception as e:
            print(f"⚠️  Warning: Could not save token: {e}")

    return build(API_SERVICE_NAME, API_VERSION, credentials=creds)


def optimize_thumbnail_for_youtube(thumbnail_file, max_size_mb=2, target_size=(1280, 720)):
    """
    Optimize thumbnail image for YouTube upload.
    """
    if not os.path.exists(thumbnail_file):
        raise FileNotFoundError(f"Thumbnail file not found: {thumbnail_file}")

    file_size_mb = os.path.getsize(thumbnail_file) / (1024 * 1024)
    if not PIL_AVAILABLE:
        if file_size_mb > max_size_mb:
            raise ValueError(
                f"Thumbnail too large ({file_size_mb:.2f}MB). "
                f"Install Pillow or compress manually."
            )
        return thumbnail_file

    import tempfile
    with Image.open(thumbnail_file) as img:
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img.thumbnail(target_size, resample)

        if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            x_offset = (target_size[0] - img.size[0]) // 2
            y_offset = (target_size[1] - img.size[1]) // 2
            new_img.paste(img, (x_offset, y_offset))
            img = new_img
        else:
            left = (img.size[0] - target_size[0]) // 2
            top = (img.size[1] - target_size[1]) // 2
            right = left + target_size[0]
            bottom = top + target_size[1]
            img = img.crop((left, top, right, bottom))

        temp_thumbnail = os.path.join(tempfile.gettempdir(), f"youtube_thumbnail_{os.path.basename(thumbnail_file)}")
        quality = 95
        max_size_bytes = max_size_mb * 1024 * 1024
        while quality >= 50:
            img.save(temp_thumbnail, 'JPEG', quality=quality, optimize=True)
            if os.path.getsize(temp_thumbnail) <= max_size_bytes:
                return temp_thumbnail
            quality -= 5
        return temp_thumbnail


def upload_video(
    video_file,
    title,
    description='',
    tags=None,
    category_id='22',
    privacy_status='private',
    thumbnail_file=None,
    playlist_id=None,
    client_secrets_file='client_secrets.json'
):
    """
    Upload a video to YouTube.
    """
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")

    youtube = get_authenticated_service(client_secrets_file)
    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': tags or [],
            'categoryId': category_id
        },
        'status': {
            'privacyStatus': privacy_status
        }
    }

    media = MediaFileUpload(video_file, chunksize=-1, resumable=True, mimetype='video/*')
    insert_request = youtube.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media
    )

    response = None
    retry = 0
    while response is None:
        try:
            status, response = insert_request.next_chunk()
            if response is not None and 'id' in response:
                video_id = response['id']
                print(f"Video uploaded successfully! Video ID: {video_id}")
            elif status:
                print(f"Upload progress: {int(status.progress() * 100)}%")
        except Exception as e:
            if retry > 3:
                print(f"Upload failed after retries: {e}")
                return None
            retry += 1
            print(f"Upload error, retrying ({retry}/3)...")

    if thumbnail_file and os.path.exists(thumbnail_file):
        try:
            optimized_thumbnail = optimize_thumbnail_for_youtube(thumbnail_file)
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(optimized_thumbnail)
            ).execute()
            if optimized_thumbnail != thumbnail_file and os.path.exists(optimized_thumbnail):
                try:
                    os.remove(optimized_thumbnail)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️  Warning: Failed to upload thumbnail: {e}")

    if playlist_id:
        try:
            youtube.playlistItems().insert(
                part='snippet',
                body={
                    'snippet': {
                        'playlistId': playlist_id,
                        'resourceId': {
                            'kind': 'youtube#video',
                            'videoId': video_id
                        }
                    }
                }
            ).execute()
            print(f"Video added to playlist: {playlist_id}")
        except Exception as e:
            print(f"Warning: Failed to add video to playlist: {e}")

    return video_id


# Script file path constant
SCRIPT_FILE_PATH = "overarching_script.txt"
# Narration audio file path constant
NARRATION_AUDIO_PATH = "narration_audio.mp3"
# Config file path constant
CONFIG_FILE_PATH = "video_config.json"

def archive_workflow_files():
    """
    Archive all workflow files (video_output, script, narration, config) to a timestamped folder
    before a new run overwrites them.
    
    Returns:
        Path to the archive folder, or None if archiving failed
    """
    import shutil
    from datetime import datetime
    
    current_dir = os.getcwd()
    
    # Create archive folder in Documents/Youtube_Channel_Archive
    # Or use current directory if Documents is not accessible
    try:
        documents_path = os.path.join(os.path.expanduser("~"), "Documents")
        archive_base = os.path.join(documents_path, "Youtube_Channel_Archive")
    except:
        # Fallback to current directory
        archive_base = os.path.join(current_dir, "Youtube_Channel_Archive")
    
    # Create timestamped archive folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_folder = os.path.join(archive_base, f"workflow_{timestamp}")
    
    try:
        os.makedirs(archive_folder, exist_ok=True)
        print(f"Archiving workflow files...")
        
        files_archived = []
        
        # Archive video_output folder
        video_output_path = os.path.join(current_dir, "video_output")
        if os.path.exists(video_output_path) and os.listdir(video_output_path):
            archive_video_output = os.path.join(archive_folder, "video_output")
            shutil.copytree(video_output_path, archive_video_output, dirs_exist_ok=True)
            files_archived.append("video_output/")
        
        # Archive script file
        script_path = os.path.join(current_dir, SCRIPT_FILE_PATH)
        if os.path.exists(script_path):
            shutil.copy2(script_path, os.path.join(archive_folder, SCRIPT_FILE_PATH))
            files_archived.append(SCRIPT_FILE_PATH)
        
        # Archive narration audio
        narration_path = os.path.join(current_dir, NARRATION_AUDIO_PATH)
        if os.path.exists(narration_path):
            shutil.copy2(narration_path, os.path.join(archive_folder, NARRATION_AUDIO_PATH))
            files_archived.append(NARRATION_AUDIO_PATH)
        
        # Archive config file
        config_path = os.path.join(current_dir, CONFIG_FILE_PATH)
        if os.path.exists(config_path):
            shutil.copy2(config_path, os.path.join(archive_folder, CONFIG_FILE_PATH))
            files_archived.append(CONFIG_FILE_PATH)
        
        # Archive music file if it exists (generated by ElevenLabs or manually provided)
        music_files = ["VIDEO_MUSIC.mp3", "video_music.mp3", "VIDEO_MUSIC.MP3"]
        for music_file in music_files:
            music_path = os.path.join(current_dir, music_file)
            if os.path.exists(music_path):
                shutil.copy2(music_path, os.path.join(archive_folder, music_file))
                files_archived.append(music_file)
                break
        
        if files_archived:
            print(f"✅ Archived {len(files_archived)} items")
            return archive_folder
        else:
            print("⚠️  No files to archive (first run or all files already cleaned)")
            # Remove empty archive folder
            try:
                os.rmdir(archive_folder)
            except:
                pass
            return None
            
    except Exception as e:
        print(f"⚠️  Warning: Could not archive workflow files: {e}")
        print(f"   Archive folder would have been: {archive_folder}")
        return None


def save_config(config_data, config_file_path=None):
    """
    Save configuration data to a JSON file.
    
    Args:
        config_data: Dictionary containing configuration values
        config_file_path: Path to save the config file (default: CONFIG_FILE_PATH)
        
    Returns:
        Path to the saved config file, or None if saving failed
    """
    if config_file_path is None:
        config_file_path = CONFIG_FILE_PATH
    
    import json
    try:
        # Get absolute path for clarity
        abs_path = os.path.abspath(config_file_path)
        
        # Write the config file
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # Verify the file was created
        if os.path.exists(config_file_path):
            print(f"✅ Configuration saved")
            return abs_path
        else:
            print(f"⚠️  Warning: Config file was written but cannot be found at: {abs_path}")
            return None
    except Exception as e:
        print(f"❌ Failed to save config file: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_config(config_file_path=None):
    """
    Load configuration data from a JSON file.
    
    Args:
        config_file_path: Path to the config file (default: CONFIG_FILE_PATH)
        
    Returns:
        Dictionary containing configuration values, or None if file doesn't exist
    """
    if config_file_path is None:
        config_file_path = CONFIG_FILE_PATH
    
    if not os.path.exists(config_file_path):
        return None
    
    import json
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print(f"✅ Configuration loaded from: {config_file_path}")
        return config_data
    except Exception as e:
        print(f"⚠️  Failed to load config file: {e}")
        return None


def generate_and_save_script(video_prompt, duration=12, api_key=None, model='gpt-5.2-2025-12-11', max_tokens=20000, script_file_path=None):
    """
    Generate an overarching script and save it to a text file.
    This is Part 1 of the workflow - allows user to edit the script before continuing.
    
    Args:
        video_prompt: The video prompt/description to base the script on
        duration: Total video duration in seconds (default: 12)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        model: Model to use (default: 'gpt-5.2-2025-12-11')
        max_tokens: Maximum tokens for the response (default: 20000)
        script_file_path: Path to save the script file (default: SCRIPT_FILE_PATH)
        
    Returns:
        Path to the saved script file
    """
    if script_file_path is None:
        script_file_path = SCRIPT_FILE_PATH
    
    # Delete existing script file at the start of each run
    if os.path.exists(script_file_path):
        try:
            os.remove(script_file_path)
        except Exception as e:
            print(f"⚠️  Could not delete previous script file: {e}")
    
    # Generate the script
    print(f"Generating script: {video_prompt[:50]}... ({duration}s)")
    
    generated_script = generate_script_from_prompt(
        video_prompt=video_prompt,
        duration=duration,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens
    )
    
    # Clean dashes from script before saving
    generated_script = clean_script_dashes(generated_script)
    
    # Save script to file
    try:
        with open(script_file_path, 'w', encoding='utf-8') as f:
            f.write(generated_script)
        print(f"✅ Script saved: {script_file_path}")
        return script_file_path
    except Exception as e:
        print(f"❌ Failed to save script to file: {e}")
        raise


def clean_script_dashes(script):
    """
    Clean script by replacing dashes with commas or ellipses.
    Replaces "---" and "--" with "..." (ellipsis), and single "-" with "," (comma).
    
    Args:
        script: The script text to clean
        
    Returns:
        Script with dashes replaced
    """
    if not script:
        return script
    
    # Replace dashes in order from longest to shortest to avoid double replacement
    # Replace "---" (triple dash) with "..." (ellipsis)
    script = script.replace('---', '...')
    # Replace "--" (double dash, often used as em dash) with "..." (ellipsis)
    script = script.replace('--', '...')
    # Replace "-" (single dash) with "," (comma)
    script = script.replace('-', ',')
    
    return script


def clean_script_for_tts(script):
    """
    Clean script to ensure it contains only narration text suitable for TTS.
    Removes any labels, instructions, or extra text that would be read as dialogue by TTS.
    Also formats years (e.g., "2025" -> "20 25") so TTS reads them correctly.
    
    Args:
        script: The script text to clean
        
    Returns:
        Cleaned script containing narration text only
    """
    if not script:
        return script
    
    script = script.strip()
    
    # Remove common script labels at the start
    script_markers = [
        "SCRIPT:", "Script:", "OVERARCHING SCRIPT:", "Overarching Script:",
        "NARRATION:", "Narration:", "DIALOGUE:", "Dialogue:",
        "SCRIPT TEXT:", "Script Text:", "NARRATIVE:", "Narrative:",
        "Here is the script:", "Here's the script:", "The script:",
        "Script for", "Narration for", "Dialogue for"
    ]
    for marker in script_markers:
        if script.startswith(marker):
            script = script[len(marker):].strip()
        # Also check if marker appears after newlines
        if f"\n{marker}" in script:
            script = script.replace(f"\n{marker}", "").strip()
    
    # Remove any lines that are clearly instructions or labels (not dialogue)
    lines = script.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Skip lines that are clearly labels/instructions (all caps, contain colons, etc.)
        if (line.endswith(':') and len(line) < 50 and 
            (line.isupper() or (line[0].isupper() and not any(c.islower() for c in line[1:])))):
            continue
        # Skip lines that start with common instruction patterns
        if line.startswith(('Note:', 'NOTE:', 'Important:', 'IMPORTANT:', 'Remember:', 'REMEMBER:')):
            continue
        # Keep everything else as narration text.
        cleaned_lines.append(line)
    
    script = '\n'.join(cleaned_lines).strip()
    
    # Final pass: remove ALL bracketed stage directions/tags.
    import re
    script = re.sub(r'\[[^\]]+\]', '', script)
    
    # Clean dashes (already handled by clean_script_dashes, but ensure it's done here too)
    script = clean_script_dashes(script)
    # Preserve paragraph breaks for better narration pacing while normalizing spacing.
    script = re.sub(r'[ \t]+', ' ', script)
    script = re.sub(r'\n{3,}', '\n\n', script).strip()
    
    return script


def enforce_story_arc_structure(script, video_prompt=None, api_key=None, model='gpt-5-2025-08-07'):
    """
    Rewrite narration to enforce a strict five-part arc:
    intro -> build-up -> climax -> build-down -> conclusion.
    """
    if not script or not OPENAI_AVAILABLE:
        return script

    key = api_key or os.getenv('OPENAI_API_KEY') or OPENAI_API_KEY
    if not key:
        return script

    target_chars = len(script)
    client = OpenAI(api_key=key)

    arc_prompt = f"""Rewrite the following documentary narration so it clearly follows this exact five-part order:
1) Intro/Hook
2) Build-up
3) Climax
4) Build-down
5) Conclusion

Requirements:
- Keep the same core facts, theme, and documentary tone.
- Keep approximately the same length (target about {target_chars} characters, plus or minus ten percent).
- Ensure the build-down clearly comes after the climax and before the conclusion.
- Make the narration feel cinematic and connected, not choppy.
- Use varied sentence length and cadence, mostly flowing sentences with occasional short impact lines.
- Use smooth transition phrasing so each paragraph naturally leads into the next.
- ALL numbers must be spelled out in words.
- NEVER use dashes, use commas or ellipses instead.
- Output ONLY narration text, no headings, labels, break tags, or explanations.

VIDEO TOPIC:
{video_prompt if video_prompt else "N/A"}

SCRIPT TO REWRITE:
{script}

Return only the rewritten script."""

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert documentary scriptwriter. "
                        "You enforce a strict five-part narrative arc in this exact order: "
                        "intro, build-up, climax, build-down, conclusion. "
                        "Keep facts and theme intact. Avoid staccato sentence fragments, use varied rhythm, "
                        "and output only script text."
                    ),
                },
                {"role": "user", "content": arc_prompt},
            ],
            max_output_tokens=20000,
        )
        rewritten = (response.output_text or "").strip()
        if not rewritten:
            return script
        return clean_script_for_tts(rewritten)
    except Exception as e:
        print(f"⚠️  Story-arc enforcement pass failed: {e}")
        return script


def load_script_from_file(script_file_path=None):
    """
    Load the overarching script from a text file and clean it for TTS.
    Also ensures dashes are cleaned and saves the cleaned version back to file.
    
    Args:
        script_file_path: Path to the script file (default: SCRIPT_FILE_PATH)
        
    Returns:
        The cleaned script text, or None if file doesn't exist
    """
    if script_file_path is None:
        script_file_path = SCRIPT_FILE_PATH
    
    if not os.path.exists(script_file_path):
        return None
    
    try:
        with open(script_file_path, 'r', encoding='utf-8') as f:
            original_script = f.read().strip()
        
        if not original_script:
            return None
        
        # Check if script has dashes that need cleaning
        has_dashes = ('-' in original_script or '--' in original_script or '---' in original_script)
        
        # If script has dashes, clean them and save the cleaned version back to file
        if has_dashes:
            dash_cleaned_script = clean_script_dashes(original_script)
            try:
                with open(script_file_path, 'w', encoding='utf-8') as f:
                    f.write(dash_cleaned_script)
                print(f"✅ Cleaned script saved (dashes removed): {script_file_path}")
                original_script = dash_cleaned_script
            except Exception as e:
                print(f"⚠️  Failed to save cleaned script: {e}")
        
        # Clean the script for TTS use (removes labels, formats years, etc.)
        cleaned_script = clean_script_for_tts(original_script)
        
        return cleaned_script if cleaned_script else None
    except Exception as e:
        print(f"⚠️  Failed to load script from file: {e}")
        return None


def generate_and_save_narration(script_file_path=None, narration_audio_path=None, duration=None, api_key=None):
    """
    Generate narration audio from the script file using ElevenLabs TTS and save it.
    If a target duration is provided, iteratively adjusts the script to match the target duration.
    This is Part 2 of the workflow - generates voiceover from the script.
    
    Args:
        script_file_path: Path to the script file (default: SCRIPT_FILE_PATH)
        narration_audio_path: Path to save the narration audio (default: NARRATION_AUDIO_PATH)
        duration: Expected video duration in seconds (enables iterative duration adjustment if provided)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        
    Returns:
        Path to the saved narration audio file
    """
    if script_file_path is None:
        script_file_path = SCRIPT_FILE_PATH
    
    if narration_audio_path is None:
        narration_audio_path = NARRATION_AUDIO_PATH
    
    # Load script from file
    script = load_script_from_file(script_file_path)
    if not script:
        raise ValueError(f"Script file not found: {script_file_path}. Please generate the script first using --generate-script-only")
    
    # Delete existing narration file at the start of each run
    if os.path.exists(narration_audio_path):
        try:
            os.remove(narration_audio_path)
        except Exception as e:
            print(f"⚠️  Could not delete previous narration file: {e}")
    
    # Generate narration using ElevenLabs TTS with iterative duration adjustment
    print(f"Generating narration audio from script using ElevenLabs TTS ({len(script)} chars)...")
    
    try:
        final_script_for_music = script
        voiceover_only_source = None
        if duration and duration > 0:
            # Use iterative duration loop to match target duration
            voiceover_audio_path, voiceover_only_source, final_script_for_music = generate_narration_with_duration_loop(
                script=script,
                target_duration_seconds=duration,
                output_path=narration_audio_path,
                api_key=api_key,
                max_attempts=3,
                tolerance_seconds=60,
                music_volume=0.08
            )
        else:
            # No target duration - generate narration without duration adjustment
            voiceover_audio_path, voiceover_only_source = generate_voiceover_with_elevenlabs(
                script=script,
                output_path=narration_audio_path,
                music_volume=0.08
            )

        # Generate background music using the final narration duration (not requested video duration).
        current_dir = os.getcwd()
        music_file_path = os.path.join(current_dir, "VIDEO_MUSIC.mp3")
        target_music_duration = get_audio_duration_seconds(voiceover_audio_path)
        if not target_music_duration or target_music_duration <= 0:
            raise RuntimeError(
                "Could not determine final narration duration for narration-step music generation."
            )

        # Load video prompt from config if available
        video_prompt_for_music = None
        config_path = os.path.join(current_dir, "video_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    video_prompt_for_music = config_data.get('prompt', config_data.get('video_prompt', ''))
            except Exception:
                video_prompt_for_music = None
        if not video_prompt_for_music:
            video_prompt_for_music = script[:300]

        print(
            f"🎵 Generating background music in narration step to match final narration "
            f"({float(target_music_duration):.2f}s)..."
        )
        generate_music_for_narration_step(
            video_prompt=video_prompt_for_music,
            script=final_script_for_music,
            target_duration_seconds=float(target_music_duration),
            api_key=api_key,
            output_path=music_file_path
        )

        # Explicitly remix narration with the newly generated music.
        mix_source = (
            voiceover_only_source
            if voiceover_only_source and os.path.exists(voiceover_only_source)
            else voiceover_audio_path
        )
        backup_path = narration_audio_path.replace(".mp3", "_original.mp3")
        if os.path.exists(mix_source):
            try:
                shutil.copy2(mix_source, backup_path)
            except Exception as backup_error:
                print(f"⚠️  Could not save original narration backup: {backup_error}")
        voiceover_audio_path = mix_voiceover_with_background_music(
            voiceover_audio_path=mix_source,
            music_audio_path=music_file_path,
            output_path=narration_audio_path,
            music_volume=0.08
        )
        print(f"✅ Narration remixed with music: {narration_audio_path}")
        
        print(f"✅ Narration audio saved: {narration_audio_path}")
        return voiceover_audio_path
    except Exception as e:
        print(f"❌ Failed to generate narration audio: {e}")
        raise


def load_narration_from_file(narration_audio_path=None):
    """
    Check if narration audio file exists.
    
    Args:
        narration_audio_path: Path to the narration audio file (default: NARRATION_AUDIO_PATH)
        
    Returns:
        Path to the narration audio file if it exists, None otherwise
    """
    if narration_audio_path is None:
        narration_audio_path = NARRATION_AUDIO_PATH
    
    if os.path.exists(narration_audio_path):
        return narration_audio_path
    return None


def adjust_script_for_duration(script, current_duration_seconds, target_duration_seconds, video_prompt=None, api_key=None, model='gpt-5-2025-08-07'):
    """
    Use GPT to adjust a script's length so the resulting narration matches the target duration.
    If the narration is too long, GPT will shorten the script. If too short, GPT will expand it.
    
    Args:
        script: The current script text
        current_duration_seconds: The actual duration of the narration audio in seconds
        target_duration_seconds: The desired duration in seconds
        video_prompt: Original video prompt (for context, optional)
        api_key: OpenAI API key
        model: GPT model to use
        
    Returns:
        Adjusted script text
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    duration_diff = current_duration_seconds - target_duration_seconds
    current_chars = len(script)
    
    # Estimate the character adjustment needed
    # characters per second = current_chars / current_duration_seconds
    if current_duration_seconds > 0:
        chars_per_second = current_chars / current_duration_seconds
        target_chars = int(chars_per_second * target_duration_seconds)
    else:
        # Fallback: 750 chars per minute
        target_chars = int((target_duration_seconds / 60.0) * 750)
    
    if duration_diff > 0:
        direction = "SHORTER"
        action = "condense and shorten"
        detail = (
            f"The narration is currently {abs(duration_diff):.0f} seconds TOO LONG. "
            f"The current script is approximately {current_chars} characters and produces a {current_duration_seconds:.0f}-second narration. "
            f"You need to shorten it to approximately {target_chars} characters to achieve a {target_duration_seconds:.0f}-second narration. "
            f"Cut anything that does not directly serve the central theme first. Tighten phrasing and remove redundancy. "
            f"Keep the most powerful, theme-driven moments and cut tangential details, side context, or generic filler."
        )
    else:
        direction = "LONGER"
        action = "expand and elaborate"
        detail = (
            f"The narration is currently {abs(duration_diff):.0f} seconds TOO SHORT. "
            f"The current script is approximately {current_chars} characters and produces a {current_duration_seconds:.0f}-second narration. "
            f"You need to expand it to approximately {target_chars} characters to achieve a {target_duration_seconds:.0f}-second narration. "
            f"Deepen the central theme: add vivid details, powerful moments, or deeper exploration of existing points that reinforce the theme. "
            f"Do NOT pad with generic context or tangential history. Every added sentence must serve the theme."
        )
    
    context_line = ""
    if video_prompt:
        context_line = f"\n\nOriginal video topic: {video_prompt}"
    
    adjustment_prompt = f"""You must {action} the following script to match a target narration duration.

{detail}{context_line}

THEME PRIORITY: The central theme of this script must remain the dominant focus. When cutting, remove content furthest from the theme first. When expanding, deepen the theme, do not dilute it.

RULES:
1. Output ONLY the adjusted script text. No labels, headers, or explanations.
2. Keep the same style, tone, and theme focus as the original.
3. Do NOT include bracketed tags or stage directions.
4. ALL numbers spelled out in words (e.g., "seventeen eighty three" not "1783").
5. NEVER use dashes. Use commas or ellipses instead.
6. Goes directly to TTS, so any extra text will be spoken.
7. Target approximately {target_chars} characters total.
8. Maintain factual accuracy. Do not invent new facts.
9. Preserve the opening hook and closing impact.
10. Preserve a clear five-part narrative arc in this exact order: intro, build-up, climax, build-down, conclusion.
11. Ensure the build-down comes immediately after the climax and before the conclusion.
12. Avoid choppy one-line fragments, prioritize flowing documentary narration with varied sentence length.
13. Use smooth transitions so each section naturally leads to the next.

CURRENT SCRIPT:
{script}

OUTPUT THE ADJUSTED SCRIPT ONLY:"""

    try:
        print(f"📝 Adjusting script to be {direction} (target: ~{target_chars} chars, currently: {current_chars} chars)...")
        
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": f"Expert documentary scriptwriter. Your #1 rule: the central theme must remain the dominant focus when adjusting length. Make scripts {direction.lower()} by prioritizing theme-relevant content. When cutting, remove what's furthest from the theme. When expanding, deepen the theme with vivid details, not tangential filler. Preserve this exact arc order: intro, build-up, climax, build-down, conclusion. Ensure build-down follows climax before conclusion. Keep rhythm varied and cinematic, avoid staccato short fragments, and keep transitions smooth. Output ONLY the script text. ALL numbers spelled out. No dashes. No bracketed tags or stage directions."},
                {"role": "user", "content": adjustment_prompt}
            ],
            max_output_tokens=20000
        )
        
        adjusted_script = response.output_text.strip()
        
        if not adjusted_script or len(adjusted_script) < 100:
            print(f"⚠️  GPT returned a very short/empty response, keeping original script")
            return script
        
        # Clean dashes from adjusted script
        adjusted_script = clean_script_dashes(adjusted_script)
        # Ensure adjusted script still follows required five-part narrative arc.
        adjusted_script = enforce_story_arc_structure(
            script=adjusted_script,
            video_prompt=video_prompt,
            api_key=api_key,
            model='gpt-5-2025-08-07'
        )
        
        print(f"✅ Script adjusted: {current_chars} → {len(adjusted_script)} chars ({len(adjusted_script) - current_chars:+d})")
        return adjusted_script
        
    except Exception as e:
        print(f"⚠️  Failed to adjust script: {e}")
        print("   Keeping original script...")
        return script


def generate_narration_with_duration_loop(
    script,
    target_duration_seconds,
    output_path=None,
    video_prompt=None,
    api_key=None,
    max_attempts=3,
    tolerance_seconds=60,
    music_volume=0.08):
    """
    Generate narration from script using ElevenLabs TTS, then iteratively adjust the script
    length until the narration duration is within tolerance of the target duration.
    
    Flow:
        1. Generate narration from the current script
        2. Measure the narration duration
        3. If narration is >= target and within tolerance above it, we're done
        4. Otherwise, use GPT to adjust the script (shorter/longer) and regenerate
        5. Repeat up to max_attempts times
    
    Args:
        script: The initial script text
        target_duration_seconds: Desired narration duration in seconds
        output_path: Path to save the final narration audio
        video_prompt: Original video prompt (for GPT context when adjusting)
        api_key: OpenAI API key (for GPT script adjustment)
        max_attempts: Maximum number of generate-check-adjust cycles (default: 3)
        tolerance_seconds: Max seconds the narration may exceed the target (default: 60). Narration must be >= target.
        music_volume: Background music volume (default: 0.08)
        
    Returns:
        Tuple of (narration_audio_path, voiceover_only_path, final_script)
    """
    import tempfile
    
    current_script = script
    temp_dir = tempfile.gettempdir()
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}")
        print(f"🔄 Narration Duration Check - Attempt {attempt}/{max_attempts}")
        print(f"{'='*60}")
        print(f"   Target duration: {target_duration_seconds}s ({target_duration_seconds/60:.1f} min)")
        print(f"   Tolerance: ±{tolerance_seconds}s")
        print(f"   Script length: {len(current_script)} chars")
        
        # Step 1: Generate narration from current script
        timestamp = int(time.time())
        if attempt == max_attempts or attempt == 1:
            # First attempt and last attempt use the final output path
            narration_path = output_path
        else:
            # Intermediate attempts use temp paths
            narration_path = os.path.join(temp_dir, f"narration_attempt_{attempt}_{timestamp}.mp3")
        
        # Always generate to a temp path first so we can check duration before committing
        temp_narration_path = os.path.join(temp_dir, f"narration_check_{attempt}_{timestamp}.mp3")
        
        try:
            narration_audio_path, voiceover_only_path = generate_voiceover_with_elevenlabs(
                script=current_script,
                output_path=temp_narration_path,
                music_volume=music_volume
            )
        except Exception as e:
            print(f"❌ Failed to generate narration on attempt {attempt}: {e}")
            if attempt == 1:
                raise  # If first attempt fails, we can't continue
            else:
                print("   Using previous narration...")
                break
        
        # Step 2: Measure the actual narration duration
        narration_duration = None
        
        # Try pydub first (more reliable, no ffprobe needed)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_narration_path)
            narration_duration = len(audio) / 1000.0  # pydub gives ms
        except Exception:
            pass
        
        # Fallback to ffprobe
        if narration_duration is None:
            narration_duration = get_media_duration(temp_narration_path)
        
        if narration_duration is None:
            print(f"⚠️  Could not measure narration duration. Accepting current narration.")
            # Copy temp to final output
            if output_path and temp_narration_path != output_path:
                shutil.copy2(temp_narration_path, output_path)
                narration_audio_path = output_path
            break
        
        duration_diff = narration_duration - target_duration_seconds
        
        print(f"\n📊 Duration Results (Attempt {attempt}):")
        print(f"   Narration duration: {narration_duration:.1f}s ({narration_duration/60:.1f} min)")
        print(f"   Target duration:    {target_duration_seconds:.1f}s ({target_duration_seconds/60:.1f} min)")
        print(f"   Difference:         {duration_diff:+.1f}s ({abs(duration_diff)/60:.1f} min {'over' if duration_diff > 0 else 'under'})")
        
        # Step 3: Check if within tolerance (must be >= target and within tolerance above it)
        if 0 <= duration_diff <= tolerance_seconds:
            print(f"   ✅ Narration is ≥ target and within +{tolerance_seconds}s tolerance! Narration accepted.")
            # Copy temp to final output
            if output_path and temp_narration_path != output_path:
                shutil.copy2(temp_narration_path, output_path)
                narration_audio_path = output_path
            break
        
        # Step 4: If this was the last attempt, accept what we have
        if attempt >= max_attempts:
            print(f"   ⚠️  Max attempts ({max_attempts}) reached. Accepting current narration.")
            print(f"   Final difference: {duration_diff:+.1f}s from target")
            # Copy temp to final output
            if output_path and temp_narration_path != output_path:
                shutil.copy2(temp_narration_path, output_path)
                narration_audio_path = output_path
            break
        
        # Step 5: Adjust script for next attempt
        if duration_diff < 0:
            print(f"   📝 Narration is {abs(duration_diff):.0f}s shorter than target. Expanding script...")
        elif duration_diff > tolerance_seconds:
            print(f"   📝 Narration is {duration_diff:.0f}s over target (exceeds +{tolerance_seconds}s tolerance). Shortening script...")
        
        current_script = adjust_script_for_duration(
            script=current_script,
            current_duration_seconds=narration_duration,
            target_duration_seconds=target_duration_seconds,
            video_prompt=video_prompt,
            api_key=api_key
        )
        
        # Save the adjusted script to file
        try:
            with open(SCRIPT_FILE_PATH, 'w', encoding='utf-8') as f:
                f.write(current_script)
            print(f"   ✅ Adjusted script saved: {SCRIPT_FILE_PATH}")
        except Exception as e:
            print(f"   ⚠️  Failed to save adjusted script: {e}")
        
        # Clean up temp narration from this attempt
        try:
            if os.path.exists(temp_narration_path) and temp_narration_path != output_path:
                os.remove(temp_narration_path)
        except:
            pass
    
    # Final: ensure the output file exists at the expected path
    if output_path and os.path.exists(temp_narration_path) and temp_narration_path != output_path:
        shutil.copy2(temp_narration_path, output_path)
        narration_audio_path = output_path
        try:
            os.remove(temp_narration_path)
        except:
            pass
    
    return narration_audio_path, voiceover_only_path, current_script


def generate_script_from_prompt(video_prompt, duration=12, api_key=None, model='gpt-5.2-2025-12-11', max_tokens=20000):
    """
    STEP 1: Generate an overarching script for a video based on the video prompt using OpenAI ChatGPT API.
    This is the first of three separate API calls:
    1. This function generates the complete script (separate API call)
    2. segment_script_rule_based() segments the script into X segments (rules-based, no API call)
    3. generate_video_prompts_from_segments() generates video prompts from the segments (separate API calls)
    
    Args:
        video_prompt: The video prompt/description to base the script on
        duration: Total video duration in seconds (default: 12)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        model: Model to use (default: 'gpt-5.2-2025-12-11')
        max_tokens: Maximum tokens for the response (default: 20000)
        
    Returns:
        Generated script as a string
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY  # Use global hardcoded API key as fallback
    
    client = OpenAI(api_key=api_key)
    
    print(f"Generating script ({duration}s)...")
    
    # Create a simplified prompt for script generation only
    # Calculate target character count: 750 characters per minute of video
    # Convert duration from seconds to minutes
    duration_minutes = duration / 60.0
    target_characters = int(duration_minutes * 750)
    
    script_prompt = f"""Write a {duration}-second documentary-style YouTube narration script (~{target_characters} characters) about: {video_prompt}

THE THEME IS EVERYTHING:
The topic above has a CENTRAL THEME. This theme is not just the subject, it is the SOUL of the script. Every single sentence must serve this theme. If a sentence does not directly explore, illuminate, or deepen the theme, cut it. The theme must:
- Open the script. The very first line should immerse the viewer in the theme.
- Drive every section. Do not include tangential history, side characters, or context that does not directly reinforce the theme.
- Build progressively. Each paragraph should deepen the viewer's understanding of why this theme matters, escalating in emotional or intellectual weight.
- Close the script. The final lines should land the theme with impact, leaving the viewer thinking about it.

BE CONCISE, NOT COMPREHENSIVE:
This is NOT an encyclopedia entry. Do not try to cover everything about the topic. Instead:
- Choose the 3-5 most compelling facts, moments, or angles that best serve the theme.
- Go deep on those chosen moments rather than wide across the entire topic.
- Every word must earn its place, but do not sound clipped or abrupt.
- Prefer vivid, specific details over broad generalizations. One powerful detail beats three generic sentences.
- Assume the viewer knows nothing, but explain only what they NEED to understand the theme, not everything about the topic.

MANDATORY STORY ARC, MUST FOLLOW THIS EXACT ORDER:
- Intro/Hook (10-15%): Open with a compelling image, question, or tension that introduces the central theme.
- Build-up (35-45%): Build context and momentum in a clear causal sequence, progressively escalating stakes toward the peak.
- Climax (10-15%): Deliver the single most powerful turning point or revelation, the emotional/intellectual peak of the script.
- Build-down (15-20%): Show the immediate fallout, reflection, or decompression after the climax.
- Conclusion (10-15%): Resolve the narrative thread and land the theme with a meaningful takeaway.

NARRATIVE COHERENCE RULES:
- This must read like ONE continuous story, not a list of facts.
- Every paragraph must logically lead to the next with cause-and-effect transitions.
- If a sentence does not advance the narrative arc, remove it.
- Include any necessary bridge steps between sections so the climax feels earned, not abrupt.

STYLE:
- Documentary tone: authoritative but conversational, like a knowledgeable storyteller
- Blend facts with narrative. Let the facts be compelling on their own.
- Use natural pauses (...) and smooth transitions
- Historically accurate, BBC/National Geographic quality
- Prioritize narrative flow over isolated trivia
- Vary sentence rhythm: mostly flowing medium-to-long sentences, with occasional short impact lines at key moments.
- Avoid strings of disconnected short sentences. Keep each paragraph causally connected to the next.

FORMATTING RULES:
- ALL numbers spelled out (e.g., "seventeen eighty three" not "1783", "nineteen forty five" not "1945")
- NEVER use dashes. Use commas for separations, ellipses for dramatic pauses, or rephrase.
- ~{target_characters} characters total (750 chars per minute of video)

OUTPUT:
- ONLY the narration text
- NO labels, headers, instructions, or formatting markers
- Start with the first word of narration, end with the last word
- This goes directly to text-to-speech; any extra text will be spoken

Script:"""
    
    try:
        # Call Responses API
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You are an expert documentary scriptwriter. Your #1 rule: THE THEME IS EVERYTHING. The user's prompt contains a central theme. Every sentence you write must directly serve, explore, or deepen that theme. If it doesn't, cut it. Be concise, not comprehensive. Pick the strongest angles and go deep. Do not pad with tangential history or generic context. Write like BBC/National Geographic: authoritative, accurate, conversational, and cinematic. Use varied cadence, mostly flowing sentences with occasional short impact lines, and avoid staccato strings of short fragments. Structure: intro hook, build-up, climax, build-down, conclusion. Spell out ALL numbers ('seventeen eighty three' not '1783'). NEVER use dashes; use commas or ellipses instead. Output ONLY narration text. No labels, headers, bracketed tags, or instructions. Goes directly to TTS."},
                {"role": "system", "content": "You are an expert documentary scriptwriter. Your number one rule is THEME DRIVEN STORYTELLING. Every sentence must serve the central theme and move the narrative forward. Write one cohesive story arc in this exact order: intro hook, build-up, climax, build-down, conclusion. The script must never feel like disconnected facts. Use causal flow so each paragraph leads naturally to the next, the climax feels earned, and the build-down clearly follows the climax before the conclusion. Vary sentence length and rhythm for narration performance, and keep transitions smooth between beats. Avoid tangents, filler, and generic summary language. Keep BBC/National Geographic quality: accurate, vivid, authoritative, conversational. Spell out all numbers in words. Never use dashes, use commas or ellipses instead. Output only narration text, no labels, headings, or bracketed tags, because this goes directly to TTS."},
                {"role": "user", "content": script_prompt}
            ],
            max_output_tokens=max_tokens
        )
        
        script = response.output_text.strip()
        
        # Clean script to ensure it only contains narration text.
        script = clean_script_for_tts(script)
        # Enforce five-part narrative arc order explicitly.
        script = enforce_story_arc_structure(
            script=script,
            video_prompt=video_prompt,
            api_key=api_key,
            model='gpt-5-2025-08-07'
        )
        
        print(f"✅ Script generated ({len(script)} chars)")
        
        return script
        
    except Exception as e:
        raise Exception(f"Failed to generate script: {e}")


def segment_script_rule_based(script, num_segments):
    """
    STEP 2: Segment a script into N segments using rule-based approach (word count).
    This uses a rules-based system to divide the script evenly by word count.
    DEPRECATED: Use segment_script_by_narration instead for narration-based segmentation.
    
    Args:
        script: The complete overarching script
        num_segments: Number of segments to create
        
    Returns:
        List of segment texts (one per segment)
    """
    if num_segments <= 1:
        return [script]
    
    # Split script into words
    words = script.split()
    total_words = len(words)
    words_per_segment = total_words / num_segments
    
    segments = []
    for i in range(num_segments):
        start_word = int(i * words_per_segment)
        end_word = int((i + 1) * words_per_segment) if i < num_segments - 1 else total_words
        segment_words = words[start_word:end_word]
        segment_text = " ".join(segment_words)
        segments.append(segment_text)
    
    return segments


def segment_script_by_narration(script, audio_path, segment_duration=FIXED_SEGMENT_DURATION_SECONDS, api_key=None, expected_num_segments=None, narration_offset=0.0):
    """
    Segment a script into segments based on narration audio timing.
    Uses Whisper API to get word-level timestamps and groups words into fixed-length segments.
    This ensures segments align with actual narration timing rather than word count.
    
    The narration_offset parameter centers the narration within the total video duration.
    For example, if the video is 30s and narration is 22s, narration_offset = 4.0s means:
    - Segment 1 (0-12s): first 4s silent opening, then words from narration 0-8s
    - Segment 2 (12-24s): words from narration 8-20s
    - Segment 3 (24-36s): words from narration 20-28s, then 4s silent closing
    
    IMPORTANT: The audio_path should be the final narration audio (narration_audio.mp3)
    that matches the target duration from video_config.json. This ensures word-level timestamps
    are accurate for video prompt generation.
    
    Args:
        script: The complete overarching script
        audio_path: Path to the narration audio file
        segment_duration: Duration of each segment in seconds (default: 12.0)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        expected_num_segments: Expected number of segments (if provided, limits output to this many)
        narration_offset: Seconds to offset narration start within the video (default: 0.0).
            This centers the narration so there's an opening shot before narration begins
            and a closing shot after narration ends.
        
    Returns:
        List of segment texts (one per segment), each containing words spoken during that time window
    """
    if not OPENAI_AVAILABLE:
        print("⚠️  OpenAI library not available, falling back to rule-based segmentation")
        # Fallback: estimate segments based on script length
        estimated_duration = len(script.split()) * 0.5  # Rough estimate: 0.5s per word
        num_segments = max(1, int(estimated_duration / segment_duration))
        return segment_script_rule_based(script, num_segments)
    
    if not os.path.exists(audio_path):
        print(f"⚠️  Audio file not found: {audio_path}, falling back to rule-based segmentation")
        estimated_duration = len(script.split()) * 0.5
        num_segments = max(1, int(estimated_duration / segment_duration))
        return segment_script_rule_based(script, num_segments)
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    if not api_key:
        print("⚠️  No OpenAI API key available, falling back to rule-based segmentation")
        estimated_duration = len(script.split()) * 0.5
        num_segments = max(1, int(estimated_duration / segment_duration))
        return segment_script_rule_based(script, num_segments)
    
    client = OpenAI(api_key=api_key)
    
    try:
        print(f"Transcribing audio with Whisper...")
        
        # Use Whisper API to get word-level timestamps
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        
        # Extract word-level timestamps from response
        words = []
        
        if isinstance(transcript, dict):
            if 'words' in transcript and transcript['words']:
                words = transcript['words']
            elif 'segments' in transcript:
                for segment in transcript['segments']:
                    if 'words' in segment and segment['words']:
                        words.extend(segment['words'])
        else:
            if hasattr(transcript, 'words') and transcript.words:
                words = transcript.words
            elif hasattr(transcript, 'segments') and transcript.segments:
                for segment in transcript.segments:
                    if hasattr(segment, 'words') and segment.words:
                        words.extend(segment.words)
                    elif isinstance(segment, dict) and 'words' in segment:
                        words.extend(segment['words'])
        
        if not words:
            print("⚠️  No word-level timestamps available from Whisper, falling back to rule-based segmentation")
            estimated_duration = len(script.split()) * 0.5
            num_segments = max(1, int(estimated_duration / segment_duration))
            return segment_script_rule_based(script, num_segments)
        
        print(f"✅ Transcribed {len(words)} words")
        
        # Find the total duration of the audio
        total_duration = 0
        for word_data in words:
            if isinstance(word_data, dict):
                end = word_data.get('end', 0)
            else:
                end = getattr(word_data, 'end', 0)
            total_duration = max(total_duration, end)
        
        # Calculate number of segments needed
        if expected_num_segments is not None:
            num_segments_needed = expected_num_segments
        else:
            num_segments_needed = int((total_duration + segment_duration - 0.1) / segment_duration)
        
        # Group words into fixed-length segments based on timestamps
        # Use a more efficient approach: iterate through words once and assign them to segments
        segments = []
        
        # Initialize empty segments (only create the expected number)
        segments = [""] * num_segments_needed
        
        # Assign each word to the appropriate segment based on its start time
        # CRITICAL: Assign ALL words - every word must be assigned to a segment
        # Apply narration_offset: word's video time = word's narration time + offset
        # This centers the narration within the total video duration
        if narration_offset > 0:
            print(f"   📍 Applying narration offset: {narration_offset:.1f}s (narration starts at {narration_offset:.1f}s in video)")
        
        words_assigned = 0
        for word_data in words:
            if isinstance(word_data, dict):
                word = word_data.get('word', '').strip()
                start = word_data.get('start', 0)
            else:
                word = getattr(word_data, 'word', '').strip()
                start = getattr(word_data, 'start', 0)
            
            if not word:
                continue
            
            # Determine which segment this word belongs to (0-indexed)
            # Add narration_offset to convert narration time → video time
            # e.g. word at narration 0s with 4s offset → video time 4s → segment 0 (0-12s)
            video_time = start + narration_offset
            segment_index = int(video_time / segment_duration)
            
            # Ensure segment_index is within bounds
            if segment_index < 0:
                segment_index = 0
            elif segment_index >= num_segments_needed:
                # Word is beyond expected segments - assign to last segment
                segment_index = num_segments_needed - 1
            
            # Add word to the appropriate segment
            if segments[segment_index]:
                segments[segment_index] += " " + word
            else:
                segments[segment_index] = word
            
            words_assigned += 1
        
        # Remove empty segments at the end (if any), but keep at least expected_num_segments
        # Only remove if we have more than expected
        if expected_num_segments is not None:
            # Keep exactly expected_num_segments
            segments = segments[:expected_num_segments]
            # Ensure all segments exist (pad empty ones at end if needed)
            while len(segments) < expected_num_segments:
                segments.append("")
        else:
            # Remove trailing empty segments
            while segments and not segments[-1].strip():
                segments.pop()
        
        # Ensure we have at least one segment
        if not segments:
            segments = [""]
        
        print(f"✅ Segmented into {len(segments)} segments ({total_duration:.1f}s audio)")
        
        return segments
        
    except Exception as e:
        print(f"⚠️  Whisper transcription failed: {e}")
        print("   Falling back to rule-based segmentation...")
        estimated_duration = len(script.split()) * 0.5
        num_segments = max(1, int(estimated_duration / segment_duration))
        return segment_script_rule_based(script, num_segments)


def enforce_strict_video_prompt_constraints(
    video_prompt,
    segment_text,
    segment_duration,
    key_words_phrases=None,
    visual_continuity_description=None,
    narration_beats=None,
):
    """
    Enforce strict realism + narration adherence constraints on a generated Sora prompt.
    """
    prompt = (video_prompt or "").strip()

    # Remove common label noise and audio-specific lines that don't apply.
    label_prefixes = (
        "Sora Prompt:",
        "Video Prompt:",
        "Prompt:",
    )
    for label in label_prefixes:
        if prompt.startswith(label):
            prompt = prompt[len(label):].strip()
    prompt = re.sub(r"(?im)^\s*(sound|audio|music|dialogue|sfx)\s*[:\-].*$", "", prompt).strip()
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()

    realism_anchor = (
        "Ultra-realistic, lifelike live-action documentary footage. "
        "Natural lighting, physically plausible motion, authentic materials/textures, and real camera optics. "
        "No CGI look, no animation, no stylization."
    )
    adherence_anchor = (
        f"Strict narration lock for this {segment_duration:.1f}s segment: "
        "the visuals must directly match the segment narration and must not introduce conflicting events."
    )

    segment_excerpt = ""
    if segment_text and not segment_text.startswith("[SILENT"):
        segment_excerpt = segment_text.strip().replace("\n", " ")
        if len(segment_excerpt) > 520:
            segment_excerpt = segment_excerpt[:517] + "..."
        segment_excerpt = f"Narration source text: {segment_excerpt}"

    key_phrase_lock = ""
    if key_words_phrases:
        filtered = [p.strip() for p in key_words_phrases if p and p.strip()]
        if filtered:
            key_phrase_lock = "Must visibly include: " + ", ".join(filtered[:4]) + "."

    beat_lock = ""
    if narration_beats is None:
        narration_beats = build_narration_visual_beats(segment_text)
    if narration_beats:
        beat_lock = (
            "Ordered narration beats that must appear on-screen in sequence: "
            + " | ".join(f"{idx + 1}) {beat}" for idx, beat in enumerate(narration_beats))
            + "."
        )

    enforced_parts = [realism_anchor, adherence_anchor]
    if segment_excerpt:
        enforced_parts.append(segment_excerpt)
    if key_phrase_lock:
        enforced_parts.append(key_phrase_lock)
    if beat_lock:
        enforced_parts.append(beat_lock)
    enforced_parts.append(prompt)
    enforced_prompt = " ".join(part for part in enforced_parts if part).strip()

    # Ensure visual continuity text remains prepended exactly at the beginning.
    if visual_continuity_description:
        continuity = visual_continuity_description.strip()
        if continuity and not enforced_prompt.startswith(continuity):
            enforced_prompt = f"{continuity} {enforced_prompt}"

    if len(enforced_prompt) > 4000:
        enforced_prompt = enforced_prompt[:3997] + "..."

    return enforced_prompt


def build_narration_visual_beats(segment_text, max_beats=4):
    """
    Convert narration text into short, ordered visual beats for tighter prompt locking.
    """
    if not segment_text:
        return []

    text = segment_text.strip()
    if not text or text.startswith("[SILENT"):
        return []

    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    beats = []
    for sentence in sentences:
        beat = sentence.strip().strip('"\'')
        if not beat:
            continue
        if len(beat) > 140:
            beat = beat[:137].rstrip(",;:- ") + "..."
        beats.append(beat)
        if len(beats) >= max_beats:
            break

    if not beats:
        fallback = text[:140].rstrip(",;:- ")
        if len(text) > 140:
            fallback += "..."
        if fallback:
            beats.append(fallback)

    return beats


def convert_segment_to_video_prompt(
    segment_text,
    segment_id,
    segment_duration,
    total_duration,
    overarching_script=None,
    previous_prompt=None,
    next_segment_text=None,
    api_key=None,
    model='gpt-5-2025-08-07',
    max_tokens=20000,
    total_segments=None,
    visual_continuity_description=None,
    narration_offset=0.0
):
    """
    Convert a segment script text into a video generation prompt using AI.
    Includes full script context and previous segment for chronological continuity.
    Accounts for narration offset in timing calculations.
    
    The visual_continuity_description (if provided) is prepended to every prompt
    so the video model generates visually consistent scene/style constraints across
    all segments.
    
    The narration_offset centers narration within the video. The first segment may
    have a silent opening shot before narration starts, and the last segment may
    have a silent closing shot after narration ends.
    
    Args:
        segment_text: The script text for this segment
        segment_id: Segment number (1-indexed)
        segment_duration: Duration of this segment in seconds
        total_duration: Total video duration in seconds
        overarching_script: The full overarching script (for context and narrative flow)
        previous_prompt: The video prompt from the previous segment (for visual continuity)
        next_segment_text: The script text for the next segment (for forward continuity)
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-5.2-2025-12-11')
        max_tokens: Maximum tokens for the response (default: 2500)
        visual_continuity_description: Detailed visual continuity description prepended to every prompt
        narration_offset: Seconds before narration starts in the video (for centering)
        
    Returns:
        Video generation prompt as a string
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    start_time = (segment_id - 1) * segment_duration
    end_time = segment_id * segment_duration
    
    # Calculate narration timing within this segment
    # narration_offset = time before narration starts in the video
    # narration runs from narration_offset to (total_duration - narration_offset)
    narration_start_in_video = narration_offset
    narration_end_in_video = total_duration - narration_offset
    
    # How much of THIS segment is silent (opening/closing shot)
    silent_opening_in_segment = 0.0
    silent_closing_in_segment = 0.0
    narration_timing_note = ""
    
    if narration_offset > 0:
        seg_start = (segment_id - 1) * segment_duration
        seg_end = segment_id * segment_duration
        
        if narration_start_in_video > seg_start:
            # Narration hasn't started yet at the beginning of this segment
            silent_opening_in_segment = min(narration_start_in_video - seg_start, segment_duration)
        
        if narration_end_in_video < seg_end:
            # Narration ends before this segment ends
            silent_closing_in_segment = min(seg_end - narration_end_in_video, segment_duration)
        
        if silent_opening_in_segment > 0.1:
            narration_timing_note = f"TIMING: First {silent_opening_in_segment:.1f}s of this segment is a SILENT OPENING SHOT (no narration). Narration begins at {silent_opening_in_segment:.1f}s into this segment. Design the opening as an atmospheric establishing shot that sets the scene before the narration starts."
        elif silent_closing_in_segment > 0.1:
            narration_duration_in_segment = segment_duration - silent_closing_in_segment
            narration_timing_note = f"TIMING: Narration ends {silent_closing_in_segment:.1f}s before this segment ends. The last {silent_closing_in_segment:.1f}s is a SILENT CLOSING SHOT. Design the closing as a contemplative, atmospheric shot that lingers after the narration finishes."
    
    # Build context sections (no previous/next segments to avoid confusion)
    context_parts = []
    
    context_text = "\n".join([f"- {part}" for part in context_parts]) if context_parts else ""
    
    # Check if this segment contains musical breaks or is opening/closing
    is_opening_segment = segment_id == 1
    # Determine if this is the closing segment
    if total_segments:
        is_closing_segment = segment_id >= total_segments
    else:
        # Fallback: estimate based on duration
        estimated_segments = int(total_duration / segment_duration) + 1
        is_closing_segment = segment_id >= estimated_segments - 0.5
    
    # Build concise requirements
    requirements_parts = []
    
    if is_opening_segment or is_closing_segment:
        requirements_parts.append(f"Single continuous {segment_duration:.1f}s shot, no cuts, slow camera movement")
    else:
        min_shot_duration = segment_duration * 0.4
        requirements_parts.append(f"Max 1 cut, each shot ≥{min_shot_duration:.1f}s (preferably {segment_duration/2:.1f}s each)")
    
    requirements_parts.append("PHOTOREALISTIC documentary-style (real-life footage, natural lighting, authentic)")
    requirements_parts.append("No diagrams/words, no music/sound descriptions, just the scene")
    requirements_parts.append("Include camera movement, angle, lighting, mood")
    
    requirements_text = "\n".join([f"- {part}" for part in requirements_parts])
    
    # Validate segment_text is provided and not empty
    # Allow empty segments when narration_offset > 0 (silent opening/closing shots)
    if not segment_text or len(segment_text.strip()) == 0:
        if narration_offset > 0:
            # Silent segment — generate an atmospheric establishing/closing shot
            pass  # Will be handled by the narration_timing_note in the prompt
        else:
            raise ValueError(f"Segment {segment_id} text is empty! Cannot generate video prompt.")
    
    # Extract key words/phrases from the segment text using AI
    # This helps identify the main visual focus points for the video
    key_words_phrases = None
    num_shots = 1  # Default to 1 shot
    
    try:
        # Use AI to extract key words/phrases and determine number of shots
        extraction_prompt = f"""Analyze this narration segment and extract the key words or phrases that represent the main visual subjects or concepts:
        
"{segment_text}"

Instructions:
1. Identify 1-3 key words or phrases (nouns, locations, actions, or important concepts) that should be the visual focus
2. Determine if this segment should be 1 continuous shot or 2 separate shots based on content complexity
3. CRITICAL: Only suggest 2 shots if there are 2 DISTINCT and EQUALLY IMPORTANT visual subjects/concepts that truly require separate shots. If in doubt, choose 1 shot to avoid quick cuts.
4. If 2 shots are suggested, each shot must be approximately {segment_duration/2:.1f} seconds long (for a {segment_duration:.1f}-second segment) - NO quick cuts allowed

Respond in this exact format:
KEY_PHRASES: [comma-separated list of 1-3 key phrases]
NUM_SHOTS: [1 or 2]
REASONING: [brief explanation]"""

        extraction_response = client.responses.create(
            model='gpt-4o-mini',  # Use cheaper model for extraction
            input=[
                {"role": "system", "content": "Analyze text to extract key visual concepts and determine shot structure."},
                {"role": "user", "content": extraction_prompt}
            ],
            max_output_tokens=200
        )
        
        extraction_text = extraction_response.output_text.strip()
        
        # Parse the response
        if "KEY_PHRASES:" in extraction_text:
            phrases_line = [line for line in extraction_text.split('\n') if 'KEY_PHRASES:' in line]
            if phrases_line:
                phrases_text = phrases_line[0].split('KEY_PHRASES:')[1].strip()
                key_words_phrases = [p.strip() for p in phrases_text.split(',') if p.strip()]
        
        if "NUM_SHOTS:" in extraction_text:
            shots_line = [line for line in extraction_text.split('\n') if 'NUM_SHOTS:' in line]
            if shots_line:
                shots_text = shots_line[0].split('NUM_SHOTS:')[1].strip()
                try:
                    num_shots = int(shots_text.split()[0])
                    num_shots = max(1, min(2, num_shots))  # Clamp to 1 or 2
                except:
                    num_shots = 1
        
    except Exception as e:
        key_words_phrases = None
        num_shots = 1
    
    # Build concise key phrase instructions
    key_phrase_instructions = ""
    if key_words_phrases and len(key_words_phrases) > 0:
        if num_shots == 1:
            primary_phrase = key_words_phrases[0] if key_words_phrases else "the main subject"
            key_phrase_instructions = f"VISUAL FOCUS: Single continuous shot centered on '{primary_phrase}' (key elements: {', '.join(key_words_phrases)})"
        else:
            first_phrase = key_words_phrases[0] if len(key_words_phrases) > 0 else "the first main subject"
            second_phrase = key_words_phrases[1] if len(key_words_phrases) > 1 else (key_words_phrases[0] if len(key_words_phrases) > 0 else "the second main subject")
            min_shot_duration = max(5.0, segment_duration * 0.4)
            key_phrase_instructions = f"VISUAL FOCUS: 2 shots - Shot 1 ({segment_duration/2:.1f}s): '{first_phrase}' | Shot 2 ({segment_duration/2:.1f}s): '{second_phrase}' (each ≥{min_shot_duration:.1f}s, smooth cut)"
    
    # Create concise prompt with segment_text as primary focus.
    # Full-script context is chronology-only and must never override this segment.
    narration_beats = build_narration_visual_beats(segment_text, max_beats=4)
    narration_beat_instructions = ""
    if narration_beats:
        beat_lines = "\n".join([f"{idx + 1}. {beat}" for idx, beat in enumerate(narration_beats)])
        narration_beat_instructions = f"""MANDATORY NARRATION BEATS (show these visibly and in this order):
{beat_lines}
- Do not skip, swap, or replace these beats with unrelated imagery.
- If a beat is abstract, render a concrete real-world visual equivalent tied to the narration wording."""

    script_preview = overarching_script[:700] + "..." if len(overarching_script) > 700 else overarching_script
    
    # Visual continuity description handling.
    # This gets prepended to every segment prompt as a stable scene/style anchor.
    continuity_section = ""
    continuity_char_budget = 0
    if visual_continuity_description:
        continuity_char_budget = len(visual_continuity_description) + 1  # +1 for space separator
        scene_char_budget = 4000 - continuity_char_budget
        continuity_section = f"""
VISUAL CONTINUITY DESCRIPTION (MUST be included VERBATIM at the START of your output prompt):
\"{visual_continuity_description}\"

This description is {continuity_char_budget} characters. You have {scene_char_budget} remaining characters for the scene-specific content."""
    else:
        scene_char_budget = 4000
        continuity_section = ""
    
    conversion_prompt = f"""Generate a Sora 2 video prompt for segment {segment_id} ({start_time:.1f}-{end_time:.1f}s).

CRITICAL: Keep total prompt length concise (target <= 4000 characters).
{continuity_section}

{narration_timing_note if narration_timing_note else ""}
═══════════════════════════════════════════════════════════════════════════════
PRIMARY FOCUS - NARRATION FOR THIS SEGMENT:
"{segment_text}"

NON-NEGOTIABLE GOALS:
1. EXTREME REALISM: Output must read as ultra-realistic, lifelike live-action documentary footage.
2. STRICT ADHERENCE: Follow the narration for this segment as the source of truth. Do not invent conflicting events.
3. NARRATION ALIGNMENT: The main visible subjects/actions in this clip must be driven by this segment's narration.
4. SHOT DISCIPLINE: Use one clear camera setup and one primary action beat (or two only if explicitly warranted).
5. VISUAL CONCRETENESS: Use concrete nouns/verbs (specific location, objects, actions, lighting, textures) instead of vague wording.
6. Do NOT include any sound/audio/music instructions.
7. PRIORITY ORDER: segment narration + mandatory beats > continuity description > full-script context preview.

{key_phrase_instructions if key_phrase_instructions else ""}
{narration_beat_instructions if narration_beat_instructions else ""}

Brief chronology context only (do NOT pull new events from this unless they are in the segment narration): {script_preview}
{("Additional context:\n" + context_text) if context_text else ""}

Requirements:
{requirements_text}

OUTPUT FORMAT:
{"1. Start with the EXACT visual continuity description above (copy it verbatim)" if visual_continuity_description else ""}
{"2. Follow" if visual_continuity_description else "1. Write"} with a cinematographer-style shot brief: framing, camera move, subject action beats, lighting, palette, and realism cues
{"3." if visual_continuity_description else "2."} Explicitly tie visible actions/subjects to the narration content above
{"4." if visual_continuity_description else "3."} Keep total output concise (target <= 4000 characters)
{"5." if visual_continuity_description else "4."} No labels, no explanations — provide ONLY the video prompt text

Provide ONLY the video prompt:"""
    
    # Retry logic: try up to 3 times to get a valid prompt
    max_retries = 3
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                print(f"Processing segment {segment_id}...")
            
            # Build strict system prompt for realism and prompt adherence.
            system_prompt = (
                "You are an expert Sora 2 prompt writer for documentary clips. "
                "Your prompt must be ultra-realistic and lifelike, with physically plausible motion, natural lighting, "
                "and authentic textures/materials. "
                "The segment narration is the source of truth: strictly align visuals to it and avoid conflicting invented events. "
                "Any provided ordered narration beats are mandatory and must appear on-screen in sequence. "
                "Use concrete cinematography language (shot framing, camera movement, visible subject action beats, lighting/palette). "
                "No audio/sound/music instructions. "
                "Output ONLY the final Sora prompt text."
            )
            if visual_continuity_description:
                system_prompt += " ALWAYS start with the provided visual continuity description verbatim."
            
            # Call Responses API
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversion_prompt}
                ],
                max_output_tokens=max_tokens
            )
            
            # Validate response structure
            if not hasattr(response, 'output_text'):
                raise Exception("API response missing 'output_text' attribute")
            
            # Get raw content
            raw_content = response.output_text
            if raw_content is None:
                raise Exception("API returned None for content")
            
            original_prompt = raw_content.strip()
            video_prompt = original_prompt
            video_prompt = video_prompt.lstrip(":- ").strip()
            video_prompt = enforce_strict_video_prompt_constraints(
                video_prompt=video_prompt,
                segment_text=segment_text,
                segment_duration=segment_duration,
                key_words_phrases=key_words_phrases,
                visual_continuity_description=visual_continuity_description,
                narration_beats=narration_beats,
            )
            
            # Validate that prompt is not empty
            if not video_prompt or len(video_prompt.strip()) == 0:
                if attempt < max_retries:
                    last_error = Exception(f"Generated prompt was empty after cleaning (attempt {attempt})")
                    time.sleep(1)
                    continue
                else:
                    # Last attempt failed - use fallback
                    if segment_text and len(segment_text.strip()) > 0:
                        video_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
                    else:
                        video_prompt = f"Photorealistic documentary-style video scene for segment {segment_id}, as if filmed by a professional camera with natural lighting and realistic textures"
                    video_prompt = enforce_strict_video_prompt_constraints(
                        video_prompt=video_prompt,
                        segment_text=segment_text,
                        segment_duration=segment_duration,
                        key_words_phrases=key_words_phrases,
                        visual_continuity_description=visual_continuity_description,
                        narration_beats=narration_beats,
                    )
                    print(f"⚠️  All attempts failed, using fallback prompt")
                    return video_prompt
            
            print(f"✅ Segment {segment_id} prompt generated")
            return video_prompt
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            error_type = type(e).__name__
            
            if attempt == 1:
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    print(f"⚠️  Rate limit, retrying...")
                elif "token" in error_msg.lower() or "length" in error_msg.lower():
                    print(f"⚠️  Token limit, retrying...")
                elif "timeout" in error_msg.lower():
                    print(f"⚠️  Timeout, retrying...")
                else:
                    print(f"⚠️  Error: {error_type}, retrying...")
            
            if attempt < max_retries:
                # Increase delay on retries, especially for rate limits
                delay = 3 if "rate limit" in error_msg.lower() or "429" in error_msg else 2
                time.sleep(delay)
                continue
            else:
                # All retries exhausted - use fallback
                if segment_text and len(segment_text.strip()) > 0:
                    video_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
                else:
                    video_prompt = f"Photorealistic documentary-style video scene for segment {segment_id}, as if filmed by a professional camera with natural lighting and realistic textures"
                video_prompt = enforce_strict_video_prompt_constraints(
                    video_prompt=video_prompt,
                    segment_text=segment_text,
                    segment_duration=segment_duration,
                    key_words_phrases=key_words_phrases,
                    visual_continuity_description=visual_continuity_description,
                    narration_beats=narration_beats,
                )
                print(f"⚠️  All attempts failed, using fallback prompt")
                return video_prompt
    
    # Should never reach here, but just in case
    if segment_text and len(segment_text.strip()) > 0:
        return f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
    else:
        return f"Photorealistic documentary-style video scene for segment {segment_id}, as if filmed by a professional camera with natural lighting and realistic textures"


def generate_video_prompts_from_segments(
    segment_texts,
    segment_duration,
    total_duration,
    overarching_script=None,
    api_key=None,
    model='gpt-5-2025-08-07',
    max_tokens=20000,
    visual_continuity_description=None,
    narration_offset=0.0
):
    """
    STEP 3: Convert multiple segment scripts into video generation prompts using AI calls.
    This is the second separate API call (after script generation).
    Includes full script context and previous segment prompts for chronological continuity.
    Accounts for narration offset in timing calculations.
    
    Args:
        segment_texts: List of segment script texts
        segment_duration: Duration of each segment in seconds
        total_duration: Total video duration in seconds
        overarching_script: The full overarching script (for context and narrative flow)
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-5.2-2025-12-11')
        max_tokens: Maximum tokens per response (default: 500)
        visual_continuity_description: Detailed continuity description prepended to each prompt
        narration_offset: Seconds before narration starts in the video (for centering)
        
    Returns:
        List of video generation prompts (one per segment)
    """
    video_prompts = []
    
    # Validate that we have the correct number of segments
    if len(segment_texts) == 0:
        raise ValueError("No segment texts provided for video prompt generation")
    
        print(f"Processing {len(segment_texts)} segments...")
    
    for i, segment_text in enumerate(segment_texts, 1):
        # Handle empty segments (can happen with narration offset - silent opening/closing shots)
        if not segment_text or len(segment_text.strip()) == 0:
            if narration_offset > 0:
                # Empty segment is expected when narration is centered — this is a silent shot
                # Use overarching script context to generate an atmospheric establishing/closing shot
                print(f"  ℹ️  Segment {i} has no narration (silent {'opening' if i == 1 else 'closing'} shot)")
                segment_text = f"[SILENT {'OPENING' if i == 1 else 'CLOSING'} SHOT - no narration plays during this segment]"
            else:
                raise ValueError(f"Segment {i} text is empty! Cannot generate video prompt.")
        
        # Calculate expected time range for this segment
        start_time = (i - 1) * segment_duration
        end_time = i * segment_duration
        
        print(f"Converting segment {i}/{len(segment_texts)} ({start_time:.1f}s-{end_time:.1f}s)...")
        
        # Verify this segment is different from previous segments
        # Skip this check for empty/silent segments
        if i > 1 and segment_text and not segment_text.startswith("[SILENT"):
            prev_segment_text = segment_texts[i-2]  # Previous segment (i-2 because i is 1-indexed)
            if segment_text == prev_segment_text:
                raise ValueError(f"Segment {i} text is identical to segment {i-1}! Segmentation may have failed.")
        
        try:
            # Get previous prompt for continuity (if not first segment)
            previous_prompt = video_prompts[-1] if video_prompts else None
            
            # Get next segment text for forward continuity (if not last segment)
            # Note: i is 1-indexed (from enumerate), but segment_texts is 0-indexed
            # Current segment is at index i-1, next segment is at index i
            # For segment 1 (i=1): current is index 0, next is index 1
            # For segment 2 (i=2): current is index 1, next is index 2 (if exists)
            next_segment_index = i  # i is the next index since current is at i-1
            if next_segment_index < len(segment_texts):
                next_segment_text = segment_texts[next_segment_index]
            else:
                next_segment_text = None
            
            
            video_prompt = convert_segment_to_video_prompt(
                segment_text=segment_text,  # This is the correct segment text for segment i
                segment_id=i,
                segment_duration=segment_duration,
                total_duration=total_duration,
                overarching_script=overarching_script,
                previous_prompt=previous_prompt,
                next_segment_text=next_segment_text,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                total_segments=len(segment_texts),
                visual_continuity_description=visual_continuity_description,
                narration_offset=narration_offset
            )
            
            # Validate prompt is not empty (should not happen due to retry logic, but double-check)
            if not video_prompt or len(video_prompt.strip()) == 0:
                # This should not happen due to retry logic, but if it does, use fallback
                if segment_text and len(segment_text.strip()) > 0:
                    video_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
                elif overarching_script and len(overarching_script.strip()) > 0:
                    video_prompt = f"Photorealistic documentary-style video scene, as if filmed by a professional camera with natural lighting and realistic textures: {overarching_script[:300]}"
                else:
                    video_prompt = f"Photorealistic documentary-style video scene for segment {i}, as if filmed by a professional camera with natural lighting and realistic textures"
                video_prompt = enforce_strict_video_prompt_constraints(
                    video_prompt=video_prompt,
                    segment_text=segment_text,
                    segment_duration=segment_duration,
                    key_words_phrases=None,
                    visual_continuity_description=visual_continuity_description,
                )
            
            video_prompts.append(video_prompt)
        except Exception as e:
            print(f"  ⚠️  Segment {i} failed: {e}")
            # Fallback: use a generic prompt based on the segment text
            if segment_text and len(segment_text.strip()) > 0:
                fallback_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
            else:
                # If segment text is also empty, use the overarching script or original prompt
                if overarching_script and len(overarching_script.strip()) > 0:
                    fallback_prompt = f"Photorealistic documentary-style video scene, as if filmed by a professional camera with natural lighting and realistic textures: {overarching_script[:300]}"
                else:
                    fallback_prompt = f"Photorealistic documentary-style video scene for segment {i}, as if filmed by a professional camera with natural lighting and realistic textures"
            fallback_prompt = enforce_strict_video_prompt_constraints(
                video_prompt=fallback_prompt,
                segment_text=segment_text,
                segment_duration=segment_duration,
                key_words_phrases=None,
                visual_continuity_description=visual_continuity_description,
            )
            video_prompts.append(fallback_prompt)
            print(f"  ⚠️  Using fallback prompt for segment {i}")
    
    # Final validation: ensure we have the correct number of prompts
    if len(video_prompts) != len(segment_texts):
        raise ValueError(f"Mismatch: Generated {len(video_prompts)} prompts but expected {len(segment_texts)} segments!")
    
    print(f"✅ Generated {len(video_prompts)} video prompts")
    
    return video_prompts



def get_audio_duration_seconds(audio_path, ffmpeg_path=None):
    """
    Best-effort audio duration resolver in seconds.
    """
    if not audio_path or not os.path.exists(audio_path):
        return None

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception:
        pass

    try:
        return get_media_duration(audio_path, ffmpeg_path=ffmpeg_path)
    except Exception:
        return None


def mix_voiceover_with_background_music(
    voiceover_audio_path,
    music_audio_path,
    output_path=None,
    music_volume=0.08,
    ffmpeg_path=None
):
    """
    Mix a voiceover file with background music and write a synced result.
    """
    if not voiceover_audio_path or not os.path.exists(voiceover_audio_path):
        raise ValueError(f"Voiceover file not found: {voiceover_audio_path}")
    if not music_audio_path or not os.path.exists(music_audio_path):
        raise ValueError(f"Music file not found: {music_audio_path}")

    if output_path is None:
        output_path = voiceover_audio_path

    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is required to mix narration and background music.")

    voiceover_duration = get_audio_duration_seconds(voiceover_audio_path, ffmpeg_path=ffmpeg_path)
    if voiceover_duration is None or voiceover_duration <= 0:
        raise RuntimeError("Could not determine voiceover duration for music mixing.")

    music_duration = get_audio_duration_seconds(music_audio_path, ffmpeg_path=ffmpeg_path)
    if music_duration is None or music_duration <= 0:
        raise RuntimeError("Could not determine music duration for music mixing.")

    import tempfile
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    synced_music_path = music_audio_path
    temp_music_sync_path = None

    # Sync music length to narration length first.
    if abs(music_duration - voiceover_duration) > 0.1:
        temp_music_sync_path = os.path.join(temp_dir, f"music_synced_mix_{timestamp}.mp3")
        if music_duration > voiceover_duration:
            cmd_music = [
                ffmpeg_path,
                "-i", music_audio_path,
                "-t", str(voiceover_duration),
                "-af", f"afade=t=out:st={max(0, voiceover_duration - 2)}:d=2",
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                "-y",
                temp_music_sync_path
            ]
        else:
            loop_count = int((voiceover_duration / music_duration) + 1)
            cmd_music = [
                ffmpeg_path,
                "-stream_loop", str(loop_count - 1),
                "-i", music_audio_path,
                "-t", str(voiceover_duration),
                "-af", f"afade=t=out:st={max(0, voiceover_duration - 2)}:d=2",
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                "-y",
                temp_music_sync_path
            ]
        subprocess.run(cmd_music, capture_output=True, text=True, check=True)
        synced_music_path = temp_music_sync_path

    output_tmp_path = output_path
    output_same_as_input = (
        os.path.abspath(output_path) == os.path.abspath(voiceover_audio_path)
    )
    if output_same_as_input:
        output_tmp_path = os.path.join(temp_dir, f"narration_mixed_{timestamp}.mp3")

    filter_complex = (
        f"[0:a]aresample=44100,volume=1.0[voice];"
        f"[1:a]aresample=44100,volume={music_volume}[music];"
        f"[voice][music]amix=inputs=2:duration=first:dropout_transition=2,"
        f"volume=2.0"
    )
    cmd_mix = [
        ffmpeg_path,
        "-i", voiceover_audio_path,
        "-i", synced_music_path,
        "-filter_complex", filter_complex,
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-y",
        output_tmp_path
    ]
    subprocess.run(cmd_mix, capture_output=True, text=True, check=True)

    if output_same_as_input:
        shutil.copy2(output_tmp_path, output_path)
        try:
            os.remove(output_tmp_path)
        except Exception:
            pass

    try:
        if temp_music_sync_path and os.path.exists(temp_music_sync_path):
            os.remove(temp_music_sync_path)
    except Exception:
        pass

    return output_path


def normalize_audio_duration_exact(
    input_audio_path,
    target_duration_seconds,
    output_audio_path,
    ffmpeg_path=None,
    tolerance_seconds=0.15
):
    """
    Force an audio file to an exact target duration using looping + trim.
    """
    if not input_audio_path or not os.path.exists(input_audio_path):
        raise ValueError(f"Input audio file not found: {input_audio_path}")
    if target_duration_seconds is None or target_duration_seconds <= 0:
        raise ValueError("target_duration_seconds must be > 0")

    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is required to enforce exact music duration.")

    fade_out_start = max(0.0, float(target_duration_seconds) - 2.0)
    cmd = [
        ffmpeg_path,
        "-stream_loop", "-1",
        "-i", input_audio_path,
        "-t", f"{float(target_duration_seconds):.3f}",
        "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start:.3f}:d=2",
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-y",
        output_audio_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=180)

    normalized_duration = get_audio_duration_seconds(output_audio_path, ffmpeg_path=ffmpeg_path)
    if normalized_duration is None:
        raise RuntimeError("Could not verify normalized music duration.")
    if abs(normalized_duration - float(target_duration_seconds)) > tolerance_seconds:
        raise RuntimeError(
            f"Normalized music duration mismatch: got {normalized_duration:.3f}s, "
            f"expected {float(target_duration_seconds):.3f}s"
        )
    return output_audio_path


def infer_video_vibe(video_prompt, script=None):
    """
    Infer a short vibe label from topic/script for music prompt conditioning.
    """
    context = f"{video_prompt or ''} {script or ''}".lower()

    if any(k in context for k in ["war", "battle", "conflict", "genocide", "tragedy", "disaster"]):
        return "somber and reflective"
    if any(k in context for k in ["space", "cosmos", "galaxy", "astronomy", "nasa", "planet"]):
        return "awe-filled and contemplative"
    if any(k in context for k in ["ocean", "sea", "reef", "forest", "wildlife", "nature", "amazon"]):
        return "organic and serene"
    if any(k in context for k in ["egypt", "arab", "middle east", "persia", "ottoman"]):
        return "ancient and mysterious"
    if any(k in context for k in ["japan", "china", "korea", "asia", "samurai", "dynasty"]):
        return "elegant and restrained"
    if any(k in context for k in ["medieval", "rome", "greece", "viking", "renaissance"]):
        return "historical and dignified"
    if any(k in context for k in ["technology", "ai", "computer", "cyber", "future", "robot"]):
        return "modern and focused"
    return "calm and curious"


def generate_music_prompt(video_prompt, script=None, api_key=None, model='gpt-5-2025-08-07', video_vibe=None):
    """
    Generate a concise, vibe-aligned documentary music prompt.
    """
    video_vibe = (video_vibe or infer_video_vibe(video_prompt, script)).strip()

    def _build_documentary_fallback_music_prompt(topic, vibe_label):
        return (
            f"{vibe_label} documentary underscore for {topic[:140]}; "
            "instrumental, slow steady pulse, subtle dynamics, seamless, no vocals or dramatic hits."
        )

    key = api_key or OPENAI_API_KEY or os.getenv('OPENAI_API_KEY')
    if not key:
        print("⚠️  No OpenAI API key available for music prompt generation. Using default prompt.")
        return _build_documentary_fallback_music_prompt(video_prompt, video_vibe)

    try:
        client = OpenAI(api_key=key)

        system_prompt = """You are an expert documentary music director.
Write ONE simple and concise music prompt.

Hard requirements:
- Use this vibe phrase exactly: {VIBE}
- Documentary background only (underscore under narration)
- Instrumental only, slow steady pulse, subtle dynamics
- Follow topic geography/era mood with 1-2 fitting instruments
- No vocals, no dramatic hits, no risers, no abrupt transitions
- Keep it <= 28 words
- Output only the prompt text""".replace("{VIBE}", video_vibe)

        user_content = f"Video topic: {video_prompt}"
        if script:
            script_excerpt = script[:350].strip()
            user_content += f"\nScript excerpt: {script_excerpt}"
        user_content += "\nReturn the final prompt now."

        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_output_tokens=120
        )

        music_prompt = response.output_text.strip().strip('"\'')
        print(f"🎵 Video vibe: {video_vibe}")
        print(f"🎵 Generated music prompt: {music_prompt}")
        return music_prompt

    except Exception as e:
        print(f"⚠️  Failed to generate music prompt via GPT: {e}")
        return _build_documentary_fallback_music_prompt(video_prompt, video_vibe)


def generate_music_for_narration_step(
    video_prompt,
    script,
    target_duration_seconds,
    api_key=None,
    output_path=None
):
    """
    Narration-step music workflow:
    1) AI generates a vibe-matched prompt
    2) ElevenLabs generates music
    3) Audio is normalized and verified to exact video duration
    """
    import tempfile

    if target_duration_seconds is None or target_duration_seconds <= 0:
        raise ValueError("target_duration_seconds must be provided for narration-step music generation.")

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "VIDEO_MUSIC.mp3")

    print("\n🎵 Narration step: generating vibe-matched background music...")
    video_vibe = infer_video_vibe(video_prompt, script)
    print(f"🎚️  Detected video vibe: {video_vibe}")
    music_prompt = generate_music_prompt(
        video_prompt=video_prompt,
        script=script,
        api_key=api_key,
        video_vibe=video_vibe
    )

    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    raw_music_path = os.path.join(temp_dir, f"music_raw_{timestamp}.mp3")

    script_excerpt_for_music = None
    if script:
        try:
            script_excerpt_for_music = re.sub(r"\s+", " ", str(script)).strip()[:260]
        except Exception:
            script_excerpt_for_music = str(script)[:260]

    generated_path = generate_background_music_with_elevenlabs(
        music_prompt=music_prompt,
        target_duration_seconds=target_duration_seconds,
        output_path=raw_music_path,
        video_vibe=video_vibe,
        script_excerpt_for_music=script_excerpt_for_music
    )
    if not generated_path or not os.path.exists(generated_path):
        raise RuntimeError("ElevenLabs music generation failed in narration step.")

    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        normalize_audio_duration_exact(
            input_audio_path=generated_path,
            target_duration_seconds=target_duration_seconds,
            output_audio_path=output_path,
            ffmpeg_path=ffmpeg_path
        )
    else:
        # Without ffmpeg we cannot guarantee exact duration unless already exact.
        raw_duration = get_audio_duration_seconds(generated_path, ffmpeg_path=None)
        if raw_duration is None or abs(raw_duration - float(target_duration_seconds)) > 0.15:
            raise RuntimeError("FFmpeg is required to enforce exact music duration for narration step.")
        shutil.copy2(generated_path, output_path)

    final_duration = get_audio_duration_seconds(output_path, ffmpeg_path=ffmpeg_path)
    if final_duration is None or abs(final_duration - float(target_duration_seconds)) > 0.15:
        raise RuntimeError(
            f"Final music duration mismatch: got {final_duration}, expected {target_duration_seconds}"
        )

    print(f"✅ Narration-step music ready: {output_path} ({final_duration:.2f}s target {float(target_duration_seconds):.2f}s)")
    try:
        if os.path.exists(raw_music_path):
            os.remove(raw_music_path)
    except Exception:
        pass
    return output_path, music_prompt


def generate_background_music_with_elevenlabs(
    music_prompt,
    target_duration_seconds,
    output_path=None,
    elevenlabs_api_key=None,
    prompt_influence=0.92,
    video_vibe=None,
    script_excerpt_for_music=None
):
    """
    Generate background music using ElevenLabs.
    Primary path uses the dedicated music endpoint/model, and falls back to
    sound-generation if needed.
    
    Args:
        music_prompt: Text description of the music to generate
        target_duration_seconds: Desired total music duration in seconds
        output_path: Path to save the final music file (default: VIDEO_MUSIC.mp3 in cwd)
        elevenlabs_api_key: ElevenLabs API key
        prompt_influence: Sound-generation fallback prompt adherence (0.0-1.0).
        script_excerpt_for_music: Optional narration excerpt included in final music prompt.
        
    Returns:
        Path to the generated music file, or None if generation failed
    """
    import tempfile
    
    # Match narration key resolution order exactly:
    # explicit arg -> env var -> global fallback.
    api_key = elevenlabs_api_key
    if api_key is None:
        api_key = os.getenv('ELEVENLABS_API_KEY')
    if api_key is None:
        api_key = ELEVENLABS_API_KEY
    if not api_key:
        print("❌ No ElevenLabs API key available for music generation.")
        return None
    
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "VIDEO_MUSIC.mp3")
    
    # Check if music file already exists
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # More than 1KB
            print(f"✅ Music file already exists: {output_path} ({file_size / 1024:.1f} KB)")
            print(f"   Skipping music generation. Delete the file to regenerate.")
            return output_path
    
    print(f"\n🎵 Generating background music with ElevenLabs...")
    print(f"   Prompt: {music_prompt[:100]}{'...' if len(music_prompt) > 100 else ''}")
    print(f"   Target duration: {target_duration_seconds:.1f}s")

    # Keep the final generation prompt short and strongly conditioned on vibe.
    resolved_vibe = (video_vibe or "calm and curious").strip()
    enhanced_prompt = (
        f"{resolved_vibe} documentary underscore. {music_prompt}. "
        "Instrumental only, slow steady pulse, subtle dynamics, seamless, no vocals or dramatic hits."
    )
    if script_excerpt_for_music:
        enhanced_prompt += f" Narration context: {script_excerpt_for_music}"

    print("   Final ElevenLabs music prompt (full):")
    print(f"   {enhanced_prompt}")

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    temp_clip_path = os.path.join(temp_dir, f"music_clip_{timestamp}.mp3")
    music_model_generated = False
    clip_duration = None

    # Primary path: dedicated music model.
    music_length_ms = int(max(3000, min(600000, round(float(target_duration_seconds) * 1000.0))))
    music_url = f"{ELEVENLABS_API_BASE_URL}/music/stream"
    music_payload = {
        "prompt": enhanced_prompt,
        "music_length_ms": music_length_ms,
        "model_id": "music_v1",
        "force_instrumental": True
    }
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"   Calling ElevenLabs Music API (model=music_v1, attempt {attempt}/{max_retries})...")
            response = requests.post(
                music_url,
                headers=headers,
                params={"output_format": "mp3_44100_128"},
                json=music_payload,
                timeout=180
            )
            if response.status_code == 200:
                with open(temp_clip_path, 'wb') as f:
                    f.write(response.content)
                file_size = os.path.getsize(temp_clip_path)
                print(f"   ✅ Music model clip generated ({file_size / 1024:.1f} KB)")
                music_model_generated = True
                break
            if response.status_code == 401:
                print("   ❌ Authentication failed. Check your ElevenLabs API key.")
                return None
            if response.status_code == 429:
                wait_time = 10 * attempt
                print(f"   ⚠️  Rate limited on Music API. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            error_msg = response.text[:300] if response.text else "Unknown error"
            print(f"   ⚠️  Music API returned {response.status_code}: {error_msg}")
            # Continue retries; if still failing we fall back to sound-generation.
            if attempt < max_retries:
                time.sleep(5 * attempt)
        except requests.exceptions.Timeout:
            print(f"   ⚠️  Music API request timed out (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(5 * attempt)
        except Exception as e:
            print(f"   ⚠️  Music API error: {e} (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(5 * attempt)

    # Fallback path: sound-generation model with high prompt adherence (lower variability).
    if not music_model_generated:
        print("   ℹ️  Falling back to ElevenLabs sound-generation endpoint.")
        max_clip_duration = 30.0
        clip_duration = min(max_clip_duration, target_duration_seconds)
        sound_url = f"{ELEVENLABS_API_BASE_URL}/sound-generation"
        sound_payload = {
            "text": enhanced_prompt,
            "duration_seconds": clip_duration,
            "prompt_influence": max(0.0, min(1.0, float(prompt_influence))),
            "model_id": "eleven_text_to_sound_v2",
            "loop": True
        }
        for attempt in range(1, max_retries + 1):
            try:
                print(f"   Calling Sound Generation API fallback (attempt {attempt}/{max_retries})...")
                response = requests.post(
                    sound_url,
                    headers=headers,
                    params={"output_format": "mp3_44100_128"},
                    json=sound_payload,
                    timeout=120
                )
                if response.status_code == 200:
                    with open(temp_clip_path, 'wb') as f:
                        f.write(response.content)
                    file_size = os.path.getsize(temp_clip_path)
                    print(f"   ✅ Sound fallback clip generated ({file_size / 1024:.1f} KB)")
                    break
                if response.status_code == 401:
                    print("   ❌ Authentication failed. Check your ElevenLabs API key.")
                    return None
                if response.status_code == 429:
                    wait_time = 10 * attempt
                    print(f"   ⚠️  Rate limited on fallback API. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                error_msg = response.text[:300] if response.text else "Unknown error"
                print(f"   ⚠️  Fallback API returned {response.status_code}: {error_msg}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                else:
                    print(f"   ❌ Failed after {max_retries} fallback attempts.")
                    return None
            except requests.exceptions.Timeout:
                print(f"   ⚠️  Fallback request timed out (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                else:
                    print(f"   ❌ Timed out after {max_retries} fallback attempts.")
                    return None
            except Exception as e:
                print(f"   ⚠️  Fallback error: {e} (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                else:
                    print(f"   ❌ Failed after {max_retries} fallback attempts: {e}")
                    return None

    if not os.path.exists(temp_clip_path):
        print("❌ Music clip was not generated.")
        return None

    measured_clip_duration = get_audio_duration_seconds(temp_clip_path, ffmpeg_path=None)
    if measured_clip_duration and measured_clip_duration > 0:
        clip_duration = measured_clip_duration
    elif clip_duration is None:
        clip_duration = min(float(target_duration_seconds), 30.0)

    # If the generated clip already covers target duration, trim/fade and return.
    if target_duration_seconds <= clip_duration + 0.5:
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path:
            fade_out_start = max(0, target_duration_seconds - 2.0)
            cmd = [
                ffmpeg_path,
                "-i", temp_clip_path,
                "-t", str(target_duration_seconds),
                "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=2",
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                "-y",
                output_path
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
                print(f"   ✅ Music saved: {output_path}")
                try:
                    os.remove(temp_clip_path)
                except Exception:
                    pass
                return output_path
            except Exception as e:
                print(f"   ⚠️  FFmpeg trim failed: {e}, using raw clip")
                shutil.copy2(temp_clip_path, output_path)
                return output_path
        shutil.copy2(temp_clip_path, output_path)
        return output_path

    # Clip is shorter than target, loop it.
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        print("⚠️  FFmpeg not found. Using single clip as music (may be shorter than video).")
        shutil.copy2(temp_clip_path, output_path)
        return output_path

    print(f"   Looping music clip to {target_duration_seconds:.1f}s with crossfade...")
    loop_count = int((target_duration_seconds / clip_duration) + 1)
    fade_out_start = max(0, target_duration_seconds - 2.0)
    looped_path = os.path.join(temp_dir, f"music_looped_{timestamp}.mp3")
    cmd_loop = [
        ffmpeg_path,
        "-stream_loop", str(loop_count - 1),
        "-i", temp_clip_path,
        "-t", str(target_duration_seconds),
        "-af", f"afade=t=in:st=0:d=1.5,afade=t=out:st={fade_out_start}:d=2",
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-y",
        looped_path
    ]

    try:
        subprocess.run(cmd_loop, capture_output=True, text=True, check=True, timeout=120)
        print(f"   ✅ Music looped to {target_duration_seconds:.1f}s")
        shutil.copy2(looped_path, output_path)
        try:
            os.remove(temp_clip_path)
            os.remove(looped_path)
        except Exception:
            pass
        file_size = os.path.getsize(output_path)
        print(f"   ✅ Background music saved: {output_path} ({file_size / 1024:.1f} KB)")
        return output_path
    except Exception as e:
        print(f"   ⚠️  Music looping failed: {e}")
        print("   Using single clip as fallback.")
        shutil.copy2(temp_clip_path, output_path)
        try:
            os.remove(temp_clip_path)
        except Exception:
            pass
        return output_path


def _clamp_voice_setting(value):
    """Clamp ElevenLabs voice setting values to the valid 0.0-1.0 range."""
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def build_arc_aware_voice_settings(
    segment_index,
    total_segments,
    base_stability,
    base_similarity_boost,
    base_style,
    use_speaker_boost=True,
):
    """
    Shape ElevenLabs voice settings across the narration arc for better intonation.
    """
    denominator = max(1, total_segments - 1)
    progress = segment_index / denominator

    # Intro -> build-up -> climax -> build-down -> conclusion.
    if progress < 0.18:
        stability_delta, style_delta, similarity_delta = -0.08, 0.18, 0.00
    elif progress < 0.60:
        stability_delta, style_delta, similarity_delta = -0.12, 0.24, 0.02
    elif progress < 0.75:
        stability_delta, style_delta, similarity_delta = -0.18, 0.34, 0.04
    elif progress < 0.90:
        stability_delta, style_delta, similarity_delta = -0.05, 0.16, 0.01
    else:
        stability_delta, style_delta, similarity_delta = 0.03, 0.10, 0.00

    return {
        "stability": _clamp_voice_setting(base_stability + stability_delta),
        "similarity_boost": _clamp_voice_setting(base_similarity_boost + similarity_delta),
        "style": _clamp_voice_setting(base_style + style_delta),
        "use_speaker_boost": bool(use_speaker_boost),
    }


def generate_voiceover_with_elevenlabs(
    script,
    output_path=None,
    music_volume=0.08,
    voice_id=None,
    elevenlabs_api_key=None,
    model_id='eleven_multilingual_v2',
    stability=0.42,
    similarity_boost=0.75,
    style=0.18,
    use_speaker_boost=True):
    """
    Generate voiceover audio from script text using ElevenLabs text-to-speech API.
    Splits long narration into API-safe chunks, generates TTS for each chunk, and
    stitches them together before mixing with music.
    
    Args:
        script: The full narration script text
        output_path: Path to save the final audio file (default: temp file)
        music_volume: Volume of background music relative to voiceover (0.0-1.0) (default: 0.08, 8%)
        voice_id: ElevenLabs voice ID to use (default: uses ELEVENLABS_VOICE_ID global/env var)
        elevenlabs_api_key: ElevenLabs API key (default: uses ELEVENLABS_API_KEY global/env var)
        model_id: ElevenLabs model ID (default: 'eleven_multilingual_v2')
        stability: Voice stability setting 0.0-1.0 (default: 0.42)
        similarity_boost: Voice similarity boost 0.0-1.0 (default: 0.75)
        style: Style exaggeration 0.0-1.0 (default: 0.18)
        use_speaker_boost: Whether to use speaker boost (default: True)
        
    Returns:
        Tuple of (path to final audio file with music, path to voiceover-only file without music)
    """
    import tempfile
    
    try:
        from pydub import AudioSegment
    except ImportError as e:
        error_msg = str(e)
        if 'pyaudioop' in error_msg or 'audioop' in error_msg:
            raise ImportError(
                "pydub requires audioop module which was removed in Python 3.13. "
                "Install audioop-lts with: pip install audioop-lts"
            )
        else:
            raise ImportError(
                "pydub library is required for audio processing. Install with: pip install pydub"
            )
    
    # Resolve ElevenLabs API key
    if elevenlabs_api_key is None:
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
    if elevenlabs_api_key is None:
        elevenlabs_api_key = ELEVENLABS_API_KEY
    if not elevenlabs_api_key:
        raise ValueError(
            "ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable "
            "or use --elevenlabs-api-key argument."
        )
    
    # Resolve voice ID
    if voice_id is None:
        voice_id = os.getenv('ELEVENLABS_VOICE_ID')
    if voice_id is None:
        voice_id = ELEVENLABS_VOICE_ID
    if not voice_id:
        # Default to ElevenLabs voice "Brian" - narrative male voice
        voice_id = "nPczCjzI2devNBz1zQrb"
        print(f"ℹ️  No voice ID specified, using default voice (Brian): {voice_id}")
    
    # Determine output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        output_path = os.path.join(temp_dir, f"voiceover_elevenlabs_{timestamp}.mp3")
    
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    
    try:
        # Step 1: Clean and split narration into API-safe chunks
        print(f"🎙️  Generating narration with ElevenLabs TTS...")
        print(f"   Voice ID: {voice_id}")
        print(f"   Model: {model_id}")

        sanitized_script = re.sub(r'\[[^\]]+\]', ' ', script or '')
        sanitized_script = re.sub(r'\s+', ' ', sanitized_script).strip()
        if not sanitized_script:
            raise ValueError("Narration script is empty after cleanup.")

        max_chars_per_chunk = 2400
        sentences = re.split(r'(?<=[.!?])\s+', sanitized_script)
        text_segments = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            if len(candidate) <= max_chars_per_chunk:
                current_chunk = candidate
                continue

            if current_chunk:
                text_segments.append(current_chunk)
                current_chunk = ""

            if len(sentence) <= max_chars_per_chunk:
                current_chunk = sentence
                continue

            # Extremely long sentence fallback: split by words.
            words = sentence.split()
            word_chunk = ""
            for word in words:
                word_candidate = f"{word_chunk} {word}".strip() if word_chunk else word
                if len(word_candidate) <= max_chars_per_chunk:
                    word_chunk = word_candidate
                else:
                    if word_chunk:
                        text_segments.append(word_chunk)
                    word_chunk = word
            if word_chunk:
                current_chunk = word_chunk

        if current_chunk:
            text_segments.append(current_chunk)

        if not text_segments:
            raise ValueError("Unable to build narration chunks for TTS.")

        print(f"✅ Prepared {len(text_segments)} narration chunk(s) for TTS generation")
        
        # Step 2: Generate TTS audio for each text segment via ElevenLabs API
        print(f"Generating TTS for {len(text_segments)} segments...")
        
        audio_segments = []
        tts_url = f"{ELEVENLABS_API_BASE_URL}/text-to-speech/{voice_id}"
        
        headers = {
            "xi-api-key": elevenlabs_api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        
        for seg_idx, text_segment in enumerate(text_segments):
            print(f"   Generating TTS for segment {seg_idx + 1}/{len(text_segments)} ({len(text_segment)} chars)...")
            voice_settings = build_arc_aware_voice_settings(
                segment_index=seg_idx,
                total_segments=len(text_segments),
                base_stability=stability,
                base_similarity_boost=similarity_boost,
                base_style=style,
                use_speaker_boost=use_speaker_boost,
            )
            print(
                "      Voice settings, "
                f"stability={voice_settings['stability']:.2f}, "
                f"style={voice_settings['style']:.2f}"
            )
            
            payload = {
                "text": text_segment,
                "model_id": model_id,
                "voice_settings": voice_settings
            }
            
            # Make API request with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        tts_url,
                        json=payload,
                        headers=headers,
                        timeout=120  # 2 minute timeout per segment
                    )
                    
                    if response.status_code == 200:
                        # Keep segment audio in memory; avoid persisted segment artifacts.
                        segment_audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
                        audio_segments.append(segment_audio)
                        print(f"   ✅ Segment {seg_idx + 1}: {len(segment_audio) / 1000:.1f}s generated")
                        break
                    elif response.status_code == 429:
                        # Rate limited - wait and retry
                        wait_time = (attempt + 1) * 10
                        print(f"   ⏳ Rate limited, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    elif response.status_code == 401:
                        raise ValueError(
                            "ElevenLabs API authentication failed. Check your API key."
                        )
                    else:
                        error_detail = ""
                        try:
                            error_json = response.json()
                            error_detail = error_json.get('detail', {})
                            if isinstance(error_detail, dict):
                                error_detail = error_detail.get('message', str(error_detail))
                        except:
                            error_detail = response.text[:200]
                        
                        if attempt < max_retries - 1:
                            print(f"   ⚠️  API error (status {response.status_code}): {error_detail}")
                            print(f"   Retrying in {(attempt + 1) * 5}s...")
                            time.sleep((attempt + 1) * 5)
                            continue
                        else:
                            raise Exception(
                                f"ElevenLabs API error (status {response.status_code}): {error_detail}"
                            )
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        print(f"   ⏳ Request timed out, retrying...")
                        time.sleep((attempt + 1) * 5)
                        continue
                    else:
                        raise Exception(f"ElevenLabs API timed out after {max_retries} attempts for segment {seg_idx + 1}")
                except requests.exceptions.ConnectionError as e:
                    if attempt < max_retries - 1:
                        print(f"   ⏳ Connection error, retrying...")
                        time.sleep((attempt + 1) * 5)
                        continue
                    else:
                        raise Exception(f"ElevenLabs API connection failed: {e}")
        
        if not audio_segments:
            raise Exception("No audio segments were generated successfully.")
        
        print(f"✅ Generated {len(audio_segments)} TTS audio segments")
        
        # Step 3: Check for CTA_AUDIO file
        cta_audio_path = None
        current_dir = os.getcwd()
        possible_cta_paths = [
            os.path.join(current_dir, "CTA_AUDIO.mp3"),
            os.path.join(current_dir, "cta_audio.mp3"),
            os.path.join(current_dir, "CTA_AUDIO.MP3"),
        ]
        
        for cta_path in possible_cta_paths:
            if os.path.exists(cta_path):
                cta_audio_path = cta_path
                print(f"✅ Found CTA_AUDIO file: {cta_audio_path}")
                break
        
        if not cta_audio_path:
            print("ℹ️  CTA_AUDIO.mp3 not found - skipping CTA")
        
        # Step 4: Stitch audio segments together
        print("Stitching TTS segments together...")
        
        final_audio = AudioSegment.empty()
        for i, segment_audio in enumerate(audio_segments):
            final_audio += segment_audio
            
            # Insert CTA_AUDIO after the first segment (before segment 2)
            if i == 0 and cta_audio_path and len(audio_segments) > 1:
                print(f"   Inserting CTA_AUDIO after segment 1...")
                try:
                    cta_audio = AudioSegment.from_file(cta_audio_path)
                    final_audio += cta_audio
                    print(f"   ✅ Added CTA_AUDIO ({len(cta_audio) / 1000:.1f}s)")
                except Exception as e:
                    print(f"   ⚠️  Failed to load CTA_AUDIO: {e}")
                    print("   Continuing without CTA audio...")
            
        # Save voiceover-only file (before mixing with music)
        voiceover_only_path = os.path.join(temp_dir, f"voiceover_only_{timestamp}.mp3")
        final_audio.export(voiceover_only_path, format='mp3', bitrate='192k')
        print(f"✅ Stitched narration ({len(final_audio) / 1000:.1f}s)")
        
        # Step 5: Mix with background music if available
        print("Mixing with background music...")
        music_path = None
        current_dir = os.getcwd()
        possible_names = [
            "VIDEO_MUSIC.mp3",
            "video_music.mp3",
            "VIDEO_MUSIC.MP3",
            os.path.join(current_dir, "VIDEO_MUSIC.mp3"),
            os.path.join(current_dir, "video_music.mp3"),
        ]
        
        for music_file in possible_names:
            if os.path.exists(music_file):
                music_path = music_file
                file_size = os.path.getsize(music_file)
                print(f"✅ Found music file: {music_path} ({file_size / 1024:.1f} KB)")
                break
        
        if music_path and os.path.exists(music_path):
            # Mix voiceover and music
            ffmpeg_path = find_ffmpeg()
            if ffmpeg_path:
                voiceover_duration = len(final_audio) / 1000.0
                music_duration = get_media_duration(music_path, ffmpeg_path)
                
                if music_duration:
                    # Sync music to voiceover duration
                    synced_music_path = os.path.join(temp_dir, f"music_synced_{timestamp}.mp3")
                    
                    if abs(music_duration - voiceover_duration) > 0.1:
                        if music_duration > voiceover_duration:
                            # Trim music
                            cmd_music = [
                                ffmpeg_path,
                                "-i", music_path,
                                "-t", str(voiceover_duration),
                                "-af", f"afade=t=out:st={max(0, voiceover_duration-2)}:d=2",
                                "-c:a", "libmp3lame",
                                "-b:a", "192k",
                                "-y",
                                synced_music_path
                            ]
                        else:
                            # Loop music to extend
                            loop_count = int((voiceover_duration / music_duration) + 1)
                            cmd_music = [
                                ffmpeg_path,
                                "-stream_loop", str(loop_count - 1),
                                "-i", music_path,
                                "-t", str(voiceover_duration),
                                "-af", f"afade=t=out:st={max(0, voiceover_duration-2)}:d=2",
                                "-c:a", "libmp3lame",
                                "-b:a", "192k",
                                "-y",
                                synced_music_path
                            ]
                        
                        try:
                            subprocess.run(cmd_music, capture_output=True, text=True, check=True)
                            music_path = synced_music_path
                            print(f"✅ Music synced")
                        except Exception as e:
                            print(f"⚠️  Music sync failed, using original")
                    else:
                        pass
                    
                    # Mix voiceover and music
                    filter_complex = (
                        f"[0:a]aresample=44100,volume=1.0[voice];"
                        f"[1:a]aresample=44100,volume={music_volume}[music];"
                        f"[voice][music]amix=inputs=2:duration=first:dropout_transition=2,"
                        f"volume=2.0"  # Boost volume by 2x after mixing
                    )
                    
                    cmd_mix = [
                        ffmpeg_path,
                        "-i", voiceover_only_path,
                        "-i", music_path,
                        "-filter_complex", filter_complex,
                        "-c:a", "libmp3lame",
                        "-b:a", "192k",
                        "-ar", "44100",
                        "-ac", "2",
                        "-y",
                        output_path
                    ]
                    
                    try:
                        subprocess.run(cmd_mix, capture_output=True, text=True, check=True)
                        print(f"✅ Final audio saved")
                        return output_path, voiceover_only_path
                    except Exception as e:
                        print(f"⚠️  Music mixing failed, using voiceover-only")
                        import shutil
                        shutil.copy2(voiceover_only_path, output_path)
                        return output_path, voiceover_only_path
                else:
                    print(f"⚠️  Could not determine music duration")
                    import shutil
                    shutil.copy2(voiceover_only_path, output_path)
                    return output_path, voiceover_only_path
            else:
                print(f"⚠️  FFmpeg not found, using voiceover-only")
                import shutil
                shutil.copy2(voiceover_only_path, output_path)
                return output_path, voiceover_only_path
        else:
            print(f"⚠️  Music file not found, using voiceover-only")
            import shutil
            shutil.copy2(voiceover_only_path, output_path)
            return output_path, voiceover_only_path
        
    except Exception as e:
        raise Exception(f"Failed to generate voiceover with ElevenLabs: {e}")


def get_media_duration(media_path, ffmpeg_path=None):
    """
    Get the duration of a video or audio file in seconds using ffprobe.
    
    Args:
        media_path: Path to the video or audio file
        ffmpeg_path: Path to ffmpeg executable (ffprobe is usually in the same directory)
        
    Returns:
        Duration in seconds as float, or None if unable to determine
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        return None
    
    # ffprobe is usually in the same directory as ffmpeg
    ffprobe_path = ffmpeg_path.replace('ffmpeg.exe', 'ffprobe.exe').replace('ffmpeg', 'ffprobe')
    
    # If ffprobe not found, try to find it
    if not os.path.exists(ffprobe_path):
        ffprobe_path = shutil.which('ffprobe')
        if not ffprobe_path:
            return None
    
    try:
        cmd = [
            ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"⚠️  Could not determine duration of {media_path}: {e}")
        return None


def center_audio_in_duration(audio_path, target_duration, output_path=None, ffmpeg_path=None):
    """
    Center shorter audio within a longer duration by adding silence at start and end.
    This ensures the audio starts later and ends earlier, avoiding cutting content.
    
    Args:
        audio_path: Path to the input audio file
        target_duration: Target duration in seconds (must be longer than audio)
        output_path: Path to save the adjusted audio
        ffmpeg_path: Path to ffmpeg executable
        
    Returns:
        Path to the adjusted audio file
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise Exception("FFmpeg not found. Cannot center audio.")
    
    if output_path is None:
        base, ext = os.path.splitext(audio_path)
        output_path = f"{base}_centered{ext}"
    
    # Get current audio duration
    current_duration = get_media_duration(audio_path, ffmpeg_path)
    if current_duration is None:
        raise Exception("Could not determine audio duration")
    
    if current_duration >= target_duration:
        # Audio is longer or equal - this function is for centering shorter audio
        print(f"   Audio ({current_duration:.2f}s) is not shorter than target ({target_duration:.2f}s)")
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path
    
    # Calculate silence padding needed
    total_padding = target_duration - current_duration
    padding_start = total_padding / 2
    padding_end = total_padding / 2
    
    print(f"   Centering audio: {current_duration:.2f}s within {target_duration:.2f}s")
    print(f"   Adding {padding_start:.2f}s silence at start and {padding_end:.2f}s at end")
    
    # Create silence and concatenate: [silence_start] + [audio] + [silence_end]
    import tempfile
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    
    silence_start = os.path.join(temp_dir, f"silence_start_{timestamp}.mp3")
    silence_end = os.path.join(temp_dir, f"silence_end_{timestamp}.mp3")
    
    # Generate silence at start
    cmd_silence_start = [
        ffmpeg_path,
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(padding_start),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-y",
        silence_start
    ]
    
    # Generate silence at end
    cmd_silence_end = [
        ffmpeg_path,
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(padding_end),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-y",
        silence_end
    ]
    
    try:
        subprocess.run(cmd_silence_start, capture_output=True, text=True, check=True)
        subprocess.run(cmd_silence_end, capture_output=True, text=True, check=True)
        
        # Concatenate: silence_start + audio + silence_end
        concat_file = os.path.join(temp_dir, f"concat_center_{timestamp}.txt")
        with open(concat_file, 'w') as f:
            f.write(f"file '{silence_start.replace(chr(92), '/')}'\n")
            f.write(f"file '{audio_path.replace(chr(92), '/')}'\n")
            f.write(f"file '{silence_end.replace(chr(92), '/')}'\n")
        
        cmd_concat = [
            ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-y",
            output_path
        ]
        
        subprocess.run(cmd_concat, capture_output=True, text=True, check=True)
        
        # Clean up temp files
        for temp_file in [silence_start, silence_end, concat_file]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        # Verify output
        final_duration = get_media_duration(output_path, ffmpeg_path)
        if final_duration:
            print(f"   ✅ Centered audio duration: {final_duration:.2f}s (target: {target_duration:.2f}s)")
        
        return output_path
        
    except Exception as e:
        print(f"⚠️  Failed to center audio: {e}")
        # Clean up on error
        for temp_file in [silence_start, silence_end]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        raise


def extend_video_duration(video_path, target_duration, output_path=None, ffmpeg_path=None):
    """
    Extend shorter video to match longer duration by looping/extending frames at start and end.
    This ensures the video starts later and ends earlier relative to audio, avoiding cutting content.
    
    Args:
        video_path: Path to the input video file
        target_duration: Target duration in seconds (must be longer than video)
        output_path: Path to save the extended video
        ffmpeg_path: Path to ffmpeg executable
        
    Returns:
        Path to the extended video file
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise Exception("FFmpeg not found. Cannot extend video.")
    
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_extended{ext}"
    
    # Get current video duration
    current_duration = get_media_duration(video_path, ffmpeg_path)
    if current_duration is None:
        raise Exception("Could not determine video duration")
    
    if current_duration >= target_duration:
        # Video is longer or equal - this function is for extending shorter video
        print(f"   Video ({current_duration:.2f}s) is not shorter than target ({target_duration:.2f}s)")
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path
    
    # Calculate extension needed
    total_extension = target_duration - current_duration
    extension_start = total_extension / 2
    extension_end = total_extension / 2
    
    print(f"   Extending video: {current_duration:.2f}s to {target_duration:.2f}s")
    print(f"   Adding {extension_start:.2f}s at start and {extension_end:.2f}s at end (looping frames)")
    
    # Use a simpler approach: extract first and last frames, then loop them
    import tempfile
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    
    first_frame = os.path.join(temp_dir, f"first_frame_{timestamp}.png")
    last_frame = os.path.join(temp_dir, f"last_frame_{timestamp}.png")
    
    # Extract first frame (frame 0)
    cmd_first = [
        ffmpeg_path,
        "-i", video_path,
        "-vf", "select=eq(n\\,0)",
        "-vframes", "1",
        "-y",
        first_frame
    ]
    
    # Extract last frame (use -ss to get near the end, then select last frame)
    cmd_last = [
        ffmpeg_path,
        "-sseof", "-1",  # Seek to 1 second before end
        "-i", video_path,
        "-vf", "select=eq(n\\,0)",
        "-vframes", "1",
        "-y",
        last_frame
    ]
    
    try:
        # Extract frames
        subprocess.run(cmd_first, capture_output=True, text=True, check=True)
        subprocess.run(cmd_last, capture_output=True, text=True, check=True)
        
        # Create video segments: [first_frame_loop] + [original_video] + [last_frame_loop]
        first_loop = os.path.join(temp_dir, f"first_loop_{timestamp}.mp4")
        last_loop = os.path.join(temp_dir, f"last_loop_{timestamp}.mp4")
        
        # Get video resolution for consistent output
        # Use probe to get resolution
        probe_cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-hide_banner"
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        # Extract resolution from probe output (format: "1920x1080")
        resolution = "1280x720"  # Default
        for line in probe_result.stderr.split('\n'):
            if 'Video:' in line and 'x' in line:
                parts = line.split()
                for part in parts:
                    if 'x' in part and part.replace('x', '').replace('.', '').isdigit():
                        resolution = part.split(',')[0]
                        break
        
        # Create loop of first frame
        cmd_first_loop = [
            ffmpeg_path,
            "-loop", "1",
            "-i", first_frame,
            "-t", str(extension_start),
            "-vf", f"scale={resolution}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", "30",  # Match typical frame rate
            "-y",
            first_loop
        ]
        
        # Create loop of last frame
        cmd_last_loop = [
            ffmpeg_path,
            "-loop", "1",
            "-i", last_frame,
            "-t", str(extension_end),
            "-vf", f"scale={resolution}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", "30",  # Match typical frame rate
            "-y",
            last_loop
        ]
        
        subprocess.run(cmd_first_loop, capture_output=True, text=True, check=True)
        subprocess.run(cmd_last_loop, capture_output=True, text=True, check=True)
        
        # Concatenate: first_loop + video + last_loop
        concat_file = os.path.join(temp_dir, f"concat_extend_{timestamp}.txt")
        with open(concat_file, 'w') as f:
            f.write(f"file '{first_loop.replace(chr(92), '/')}'\n")
            f.write(f"file '{video_path.replace(chr(92), '/')}'\n")
            f.write(f"file '{last_loop.replace(chr(92), '/')}'\n")
        
        cmd_concat = [
            ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-y",
            output_path
        ]
        
        subprocess.run(cmd_concat, capture_output=True, text=True, check=True)
        
        # Clean up temp files
        for temp_file in [first_frame, last_frame, first_loop, last_loop, concat_file]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        # Verify output
        final_duration = get_media_duration(output_path, ffmpeg_path)
        if final_duration:
            print(f"   ✅ Extended video duration: {final_duration:.2f}s (target: {target_duration:.2f}s)")
        
        return output_path
        
    except Exception as e:
        print(f"⚠️  Failed to extend video: {e}")
        # Clean up on error
        for temp_file in [first_frame, last_frame]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        raise


def remove_audio_from_video(video_path, output_path=None, ffmpeg_path=None):
    """
    Remove audio track from a video file using ffmpeg.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the output video (default: overwrite input)
        ffmpeg_path: Path to ffmpeg executable (default: auto-detect)
        
    Returns:
        Path to the output video without audio
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        # If ffmpeg not available, return original path
        print("⚠️  FFmpeg not found. Cannot remove audio. Video may contain original audio.")
        return video_path
    
    if output_path is None:
        # Create new file with _no_audio suffix
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_no_audio{ext}"
    
    # Use ffmpeg to remove audio from video
    cmd = [
        ffmpeg_path,
        "-i", video_path,
        "-c:v", "copy",      # Copy video stream without re-encoding
        "-an",                # Remove all audio streams
        "-y",                 # Overwrite output
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        # If successful, remove original and rename
        if output_path != video_path and os.path.exists(output_path):
            # Replace original with no-audio version
            try:
                os.remove(video_path)
                os.rename(output_path, video_path)
                return video_path
            except:
                return output_path
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to remove audio: {e.stderr}")
        # Return original path if removal failed
        return video_path


def sync_mixed_audio_to_video(mixed_audio_path, video_duration, output_path=None, ffmpeg_path=None, voiceover_padding=2.0):
    """
    Synchronize mixed audio (voiceover + music) to video duration.
    Music MUST match video exactly (start/end together).
    Voiceover can have padding (start before, end after video).
    
    Args:
        mixed_audio_path: Path to the mixed audio file (voiceover + music)
        video_duration: Target video duration in seconds
        output_path: Path to save the synchronized audio
        ffmpeg_path: Path to ffmpeg executable
        voiceover_padding: Seconds of padding before/after video for voiceover (default: 2.0)
        
    Returns:
        Path to the synchronized audio file
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise Exception("FFmpeg not found. Cannot synchronize audio.")
    
    if output_path is None:
        base, ext = os.path.splitext(mixed_audio_path)
        output_path = f"{base}_synced{ext}"
    
    # Target duration: video duration + padding (2s before + 2s after)
    target_duration = video_duration + (voiceover_padding * 2)
    
    current_duration = get_media_duration(mixed_audio_path, ffmpeg_path)
    if current_duration is None:
        raise Exception("Could not determine audio duration")
    
    print(f"Synchronizing audio to video...")
    
    duration_diff = target_duration - current_duration
    
    if abs(duration_diff) < 0.1:
        import shutil
        shutil.copy2(mixed_audio_path, output_path)
        return output_path
    
    import tempfile
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    
    if duration_diff > 0:
        # Audio is shorter - pad with silence
        print(f"Padding audio...")
        cmd = [
            ffmpeg_path,
            "-i", mixed_audio_path,
            "-af", "apad",
            "-t", str(target_duration),
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-y",
            output_path
        ]
    else:
        # Audio is longer - trim it
        print(f"Trimming audio...")
        cmd = [
            ffmpeg_path,
            "-i", mixed_audio_path,
            "-t", str(target_duration),
            "-c:a", "copy",
            "-y",
            output_path
        ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Audio synchronized")
        return output_path
    except Exception as e:
        print(f"⚠️  Audio synchronization failed: {e}")
        import shutil
        shutil.copy2(mixed_audio_path, output_path)
        return output_path


def apply_ending_fade(video_path, output_path=None, ffmpeg_path=None, fade_duration=2.0):
    """
    Apply fade to black and audio fade out at the end of the video.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video (if None, overwrites input)
        ffmpeg_path: Path to ffmpeg executable (if None, will try to find it)
        fade_duration: Duration of fade in seconds (default: 2.0)
        
    Returns:
        Path to output video file
    """
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_faded{ext}"
    
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH. "
            "Please install ffmpeg: https://ffmpeg.org/download.html"
        )
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video duration
    video_duration = get_media_duration(video_path, ffmpeg_path)
    if not video_duration:
        raise RuntimeError("Could not determine video duration")
    
    if video_duration <= fade_duration:
        print(f"⚠️  Video duration ({video_duration:.1f}s) is too short for {fade_duration}s fade. Skipping fade.")
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path
    
    # Calculate fade out start time
    fade_out_start = max(0, video_duration - fade_duration)
    
    print(f"   Applying fade to black ({fade_duration}s) and audio fade out ({fade_duration}s) at end...")
    print(f"   Fade starts at {fade_out_start:.1f}s (video duration: {video_duration:.1f}s)")
    
    # Use faster preset for long videos
    if video_duration > 300:  # 5 minutes
        fade_preset = 'ultrafast'
    else:
        fade_preset = 'veryfast'
    
    # Apply video fade to black and audio fade out
    # Video: fade to black over last fade_duration seconds
    # Audio: fade from 100% to 0% over last fade_duration seconds
    cmd = [
        ffmpeg_path,
        '-i', video_path,
        '-vf', f'fade=t=out:st={fade_out_start}:d={fade_duration}:color=black',  # Fade to black
        '-af', f'afade=t=out:st={fade_out_start}:d={fade_duration}',  # Audio fade out
        '-c:v', 'libx264',
        '-preset', fade_preset,
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-y',
        output_path
    ]
    
    try:
        print(f"   Processing ending fade (using {fade_preset} preset)...")
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,  # Suppress verbose ffmpeg output
            text=True,
            check=True
        )
        print(f"   ✅ Ending fade applied successfully")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else 'Unknown error')
        print(f"   ⚠️  WARNING: Ending fade application failed: {error_msg[:200] if error_msg else 'Unknown error'}")
        print(f"   SKIPPING: Using video without ending fade")
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path
    except Exception as e:
        print(f"   ⚠️  WARNING: Ending fade processing error: {e}")
        print(f"   SKIPPING: Using video without ending fade")
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path


def add_audio_to_video(video_path, audio_path, output_path=None, ffmpeg_path=None, remove_existing_audio=True, sync_duration=True, audio_delay_ms=1000):
    """
    Add audio track to a video file using ffmpeg, removing any existing audio.
    Optionally adjusts audio duration to match video duration to prevent cutoff.
    
    Args:
        video_path: Path to the video file
        audio_path: Path to the audio file
        output_path: Path to save the output video (default: overwrite input)
        ffmpeg_path: Path to ffmpeg executable (default: auto-detect)
        remove_existing_audio: If True, remove any existing audio from video (default: True)
        sync_duration: If True, adjust audio duration to match video (default: True)
        audio_delay_ms: Delay in milliseconds before audio starts (default: 1000ms).
            Used to center narration within the video — set to narration_offset * 1000.
        
    Returns:
        Path to the output video with audio
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise Exception("FFmpeg not found. Cannot add audio to video.")
    
    if output_path is None:
        # Overwrite input video
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_with_audio{ext}"
    
    # Step 1: Get video duration and adjust audio if needed
    adjusted_audio_path = audio_path
    if sync_duration:
        video_duration = get_media_duration(video_path, ffmpeg_path)
        audio_duration = get_media_duration(audio_path, ffmpeg_path)
        
        if video_duration and audio_duration:
            duration_diff = abs(video_duration - audio_duration)
            # Calculate percentage difference
            percent_diff = (duration_diff / max(video_duration, audio_duration)) * 100
            
            # If difference is more than 5%, synchronize without cutting more than 5%
            if percent_diff > 5.0:
                print(f"Synchronizing audio and video...")
                
                import tempfile
                temp_dir = tempfile.gettempdir()
                timestamp = int(time.time())
                
                # Strategy: Center shorter media within longer, or adjust if difference > 5%
                # Never cut more than 5% of either media
                if audio_duration < video_duration:
                    # Audio is shorter - center it within video (no cutting)
                    print(f"Centering audio within video...")
                    adjusted_audio_path = os.path.join(temp_dir, f"audio_centered_{timestamp}.mp3")
                    adjusted_audio_path = center_audio_in_duration(
                        audio_path=audio_path,
                        target_duration=video_duration,
                        output_path=adjusted_audio_path,
                        ffmpeg_path=ffmpeg_path
                    )
                elif video_duration < audio_duration:
                    # Video is shorter - extend video to match audio (no cutting)
                    print(f"Extending video to match audio...")
                    extended_video_path = os.path.join(temp_dir, f"video_extended_{timestamp}.mp4")
                    # Extend video and update video_path to use the extended version
                    extended_video = extend_video_duration(
                        video_path=video_path,
                        target_duration=audio_duration,
                        output_path=extended_video_path,
                        ffmpeg_path=ffmpeg_path
                    )
                    # Update video_path to use extended version for the rest of the function
                    video_path = extended_video
                    adjusted_audio_path = audio_path  # No need to adjust audio
                else:
                    # Durations are equal (shouldn't happen if percent_diff > 5%, but handle it)
                    adjusted_audio_path = audio_path
                
                try:
                    if adjusted_audio_path != audio_path:
                        input_size = 0
                        if os.path.exists(audio_path):
                            input_size = os.path.getsize(audio_path)
                    
                    # Verify adjusted audio was created and has content
                    if adjusted_audio_path and os.path.exists(adjusted_audio_path):
                        adjusted_size = os.path.getsize(adjusted_audio_path)
                        if input_size > 0 and adjusted_size < input_size * 0.5:
                            print(f"⚠️  Warning: Adjusted audio is much smaller than input")
                    
                    print(f"✅ Audio synchronized")
                    
                except Exception as e:
                    print(f"⚠️  Audio sync failed, using original")
                    adjusted_audio_path = audio_path
            else:
                pass
        else:
            print("⚠️  Could not determine durations, skipping synchronization")
    
    # Step 2: Add audio to video with volume boost and narration offset delay
    # Use ffmpeg to add audio to video
    # Remove any existing audio and replace with new audio track
    # Apply volume boost to ensure audio is loud enough (compensate for any volume loss during encoding)
    # Delay audio by audio_delay_ms so narration is centered in the video
    delay_ms = int(audio_delay_ms)
    print(f"   Audio delay: {delay_ms}ms ({delay_ms / 1000.0:.1f}s)")
    if remove_existing_audio:
        # Map only video from first input, audio from second input (removes existing audio)
        # Apply volume boost and narration offset delay
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-i", adjusted_audio_path,
            "-c:v", "copy",      # Copy video stream without re-encoding
            "-af", f"volume=1.5,adelay={delay_ms}|{delay_ms}",  # Boost audio by 1.5x and delay by narration offset
            "-c:a", "aac",       # Encode audio as AAC
            "-b:a", "192k",      # High quality audio bitrate
            "-map", "0:v:0",     # Use video from first input (video only, no audio)
            "-map", "1:a:0",     # Use audio from second input
            "-shortest",         # Finish encoding when the shortest input stream ends (should match now)
            "-y",                # Overwrite output
            output_path
        ]
    else:
        # Keep existing audio and mix (not recommended for our use case)
        # Apply narration offset delay
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-i", adjusted_audio_path,
            "-c:v", "copy",
            "-af", f"adelay={delay_ms}|{delay_ms}",  # Delay audio by narration offset
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            output_path
        ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"OK: Audio added to video")
        
        # Clean up adjusted audio if it was created
        if adjusted_audio_path != audio_path and os.path.exists(adjusted_audio_path):
            try:
                os.remove(adjusted_audio_path)
            except:
                pass
        
        return output_path
    except subprocess.CalledProcessError as e:
        # Clean up adjusted audio if it was created
        if adjusted_audio_path != audio_path and os.path.exists(adjusted_audio_path):
            try:
                os.remove(adjusted_audio_path)
            except:
                pass
        raise Exception(f"Failed to add audio to video: {e.stderr}")


def generate_image_from_prompt(prompt, output_path=None, api_key=None, model='dall-e-3', size='1536x1024'):
    """
    Generate an image from a text prompt using OpenAI DALL-E API.
    
    Args:
        prompt: Text prompt describing the image to generate
        output_path: Path to save the generated image (default: temp file)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        model: Model to use ('dall-e-3' or 'dall-e-2', default: 'dall-e-3')
        size: Image size (default: '1024x1024')
        
    Returns:
        Path to generated image file
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    import tempfile
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY  # Use global hardcoded API key as fallback
    
    client = OpenAI(api_key=api_key)
    
    # Determine output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        output_path = os.path.join(temp_dir, f"generated_image_{timestamp}.png")
    
    print(f"Generating image with prompt: {prompt}")
    try:
        # Generate image using DALL-E
        response = client.images.generate(
            model="gpt-image-1.5",
            prompt=prompt,
            n=1,
            size=size
        )
        
        # Get image data from response - handle both b64_json and url
        image_data = response.data[0]
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Check if we have base64 JSON (new format) or URL (old format)
        if hasattr(image_data, 'b64_json') and image_data.b64_json:
            # New format: decode base64 JSON and save directly
            print(f"Decoding base64 image data and saving to: {output_path}")
            image_bytes = base64.b64decode(image_data.b64_json)
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            print(f"✅ Image saved successfully from base64 data to: {output_path}")
        elif hasattr(image_data, 'url') and image_data.url:
            # Old format: download from URL
            print(f"Downloading image from URL to: {output_path}")
            image_response = requests.get(image_data.url, stream=True)
            image_response.raise_for_status()
            
            # Save image to file
            with open(output_path, 'wb') as f:
                for chunk in image_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"✅ Image downloaded and saved successfully to: {output_path}")
        else:
            raise ValueError("Image response has neither b64_json nor url - cannot save image")
        
        return output_path
        
    except Exception as e:
        error_str = str(e)
        # Check if it's a content policy violation
        if 'content_policy_violation' in error_str or 'safety system' in error_str.lower():
            raise Exception(
                f"Image generation blocked by OpenAI content policy.\n"
                f"Your prompt may contain content that violates OpenAI's safety guidelines.\n"
                f"Try adjusting your prompt to be less explicit or use different wording.\n"
                f"Original error: {error_str}"
            )
        else:
            raise Exception(f"Failed to generate image: {e}")


def generate_curiosity_thumbnail_prompt(description, script=None, api_key=None, model='gpt-4o'):
    """
    Use GPT to analyze the video content and generate a curiosity-inducing thumbnail concept.
    The goal is to create a thumbnail that looks like a real photograph from a real story,
    raises an immediate question in the viewer's mind, and hooks them into clicking.
    
    Args:
        description: Description/topic of the video
        script: The full script text (optional, provides richer context)
        api_key: OpenAI API key
        model: GPT model to use
        
    Returns:
        A detailed DALL-E prompt for a curiosity-inducing, realistic thumbnail
    """
    if not OPENAI_AVAILABLE:
        # Fallback to template if GPT is unavailable
        return thumbnail_prompt_template.format(description=description)
    
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    script_context = ""
    if script:
        # Use up to 2000 chars of the script for context
        script_context = f"""

FULL SCRIPT (use this to find the most intriguing story moment):
{script[:2000]}"""
    
    analysis_prompt = f"""You are a world-class YouTube thumbnail strategist. Your job is to design a thumbnail that:

1. Looks like a REAL PHOTOGRAPH taken by a professional photojournalist — NOT CGI, NOT stylized, NOT artistic, NOT cinematic
2. Captures a single, specific STORY MOMENT that immediately raises a QUESTION in the viewer's mind
3. Makes the viewer think "What happened here? I NEED to click this"
4. Uses visual tension, mystery, or an unanswered situation to create an irresistible hook

VIDEO TOPIC: {description}
{script_context}

ANALYZE the content above and identify the single most curiosity-inducing, question-raising moment or concept from the story.

Then write a DALL-E image generation prompt that describes a hyper-realistic photograph of that moment. Your prompt must:

- Describe a specific, concrete scene (not abstract concepts)
- Specify natural lighting, real-world environment, authentic textures
- Include specific details: facial expressions, body language, environmental clues
- Create visual tension or mystery — something that feels "mid-story" and unresolved
- Look like it was shot by a documentary photographer or photojournalist
- NOT use words like "epic," "cinematic," "dramatic lighting," "CGI," "render," "illustration," "artwork"
- NOT describe fantasy, sci-fi, or impossible scenarios unless the video topic demands it
- Keep the scene grounded and believable — as if someone actually took this photo

IMPORTANT CONSTRAINTS:
- The image must comply with content policies: no violence, gore, hate, adult content, illegal activity, copyrighted characters, or likenesses of real living people
- Use generic but realistic-looking people if humans are needed
- The thumbnail should work WITHOUT any text overlay — the image alone should hook the viewer

Respond with ONLY the DALL-E prompt, nothing else. No labels, no explanations."""
    
    try:
        response = client.responses.create(
            model=model,
            input=analysis_prompt,
        )
        
        thumbnail_dalle_prompt = response.output_text.strip()
        
        # Clean up any labels
        for label in ["DALL-E Prompt:", "Prompt:", "Thumbnail Prompt:", "Image Prompt:"]:
            if thumbnail_dalle_prompt.startswith(label):
                thumbnail_dalle_prompt = thumbnail_dalle_prompt[len(label):].strip()
        
        thumbnail_dalle_prompt = thumbnail_dalle_prompt.strip('"').strip("'").strip()
        
        if not thumbnail_dalle_prompt or len(thumbnail_dalle_prompt) < 20:
            print("⚠️  GPT returned empty/short thumbnail prompt, using template fallback")
            return thumbnail_prompt_template.format(description=description)
        
        print(f"✅ Generated curiosity-inducing thumbnail concept")
        print(f"   Concept: {thumbnail_dalle_prompt[:150]}...")
        return thumbnail_dalle_prompt
        
    except Exception as e:
        print(f"⚠️  Failed to generate thumbnail concept via GPT: {e}")
        print(f"   Falling back to template-based thumbnail prompt")
        return thumbnail_prompt_template.format(description=description)


def generate_thumbnail_from_prompt(description, output_path=None, api_key=None, script=None):
    """
    Generate a realistic, story-driven, curiosity-inducing thumbnail image.
    
    Uses GPT to analyze the video content and identify the most intriguing story moment,
    then generates a hyper-realistic photograph-style thumbnail that hooks the viewer
    by raising an immediate question in their mind.
    
    Args:
        description: Description of the video (used to generate thumbnail prompt)
        output_path: Path to save the generated thumbnail (default: temp file)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        script: Optional full script text for richer context in thumbnail generation
        
    Returns:
        Path to generated thumbnail image file
    """
    # Use GPT to generate a smart, curiosity-inducing thumbnail prompt
    prompt = generate_curiosity_thumbnail_prompt(
        description=description,
        script=script,
        api_key=api_key
    )
    
    # Sanitize for content policy compliance
    prompt = sanitize_image_prompt(prompt)
    
    return generate_image_from_prompt(prompt, output_path=output_path, api_key=api_key)


def sanitize_image_prompt(prompt):
    """
    Sanitize an image prompt to ensure it complies with OpenAI content policies.
    Removes or neutralizes potentially problematic content.
    
    Args:
        prompt: The original image prompt
        
    Returns:
        Sanitized prompt that should comply with content policies
    """
    import re
    
    # Convert to lowercase for case-insensitive matching
    prompt_lower = prompt.lower()
    
    # List of problematic terms that should be removed or replaced
    problematic_patterns = [
        # Violence-related
        (r'\b(violence|violent|blood|gore|weapon|gun|knife|sword|war|battle|fight|attack|kill|death|dead|corpse)\b', ''),
        # Hate-related
        (r'\b(hate|hateful|discrimination|racist|sexist)\b', ''),
        # Adult content
        (r'\b(nude|naked|explicit|sexual|adult|nsfw)\b', ''),
        # Real people/celebrities (common names)
        (r'\b(celebrity|famous person|real person|actual person|likeness of|portrait of)\b', 'generic person'),
        # Copyrighted characters (common ones)
        (r'\b(mickey|disney|superman|batman|spiderman|pokemon|nintendo|marvel|dc comics)\b', 'generic character'),
    ]
    
    sanitized = prompt
    for pattern, replacement in problematic_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    # Add safety suffix to ensure compliance
    if not re.search(r'(safe|appropriate|compliant|policy)', sanitized, re.IGNORECASE):
        sanitized += ". Safe, appropriate, and compliant with content policies. No violence, hate, adult content, illegal activity, copyrighted characters, or likenesses of real people."
    
    # Clean up extra whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    return sanitized


def generate_master_image_from_prompt(description=None, image_prompt=None, output_path=None, api_key=None, resolution='1024x1024'):
    """
    Generate a master/reference image from a video description or custom image prompt using OpenAI API.
    This image can be used as an optional reference frame for visual planning.
    
    Args:
        description: Description of the video (used to generate master image prompt if image_prompt not provided)
        image_prompt: Custom DALL-E prompt for the reference image (takes precedence over description)
        output_path: Path to save the generated master image (default: temp file)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        resolution: Image resolution (should match video resolution, default: '1024x1024')
        
    Returns:
        Path to generated master image file
    """
    # Use custom image prompt if provided, otherwise generate from description
    if image_prompt:
        prompt = image_prompt
    elif description:
        prompt = master_image_prompt_template.format(description=description)
    else:
        raise ValueError("Either description or image_prompt must be provided")
    
    # CRITICAL: Sanitize the prompt to ensure content policy compliance
    prompt = sanitize_image_prompt(prompt)
    print(f"   Sanitized prompt for content policy compliance")
    
    # Parse resolution to get width and height for image size
    # DALL-E 3 supports: '1024x1024', '1024x1792', '1792x1024'
    # We'll use the closest match or default to 1024x1024
    if resolution:
        # Try to match video resolution, but DALL-E has limited sizes
        # For now, use 1024x1024 as default, but we could map common resolutions
        size = '1024x1024'  # Default DALL-E 3 size
    else:
        size = '1024x1024'
    
    return generate_image_from_prompt(prompt, output_path=output_path, api_key=api_key, size=size)


def generate_tags_from_script(script, video_prompt=None, api_key=None, model='gpt-4o'):
    """
    Generate 5 YouTube tags from the script using AI.
    Tags should be 1-3 words each and synonyms/related to each other.
    
    Args:
        script: The full video script
        video_prompt: The original video prompt/topic (optional, for context)
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-4o')
        
    Returns:
        List of 5 tags (strings, 1-3 words each)
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    # Create prompt for tag generation
    tag_prompt = f"""Generate exactly 5 YouTube video tags based on this script.

Script:
{script[:1000]}{'...' if len(script) > 1000 else ''}

{f'Video Topic: {video_prompt}' if video_prompt else ''}

REQUIREMENTS:
- Generate exactly 5 tags
- Each tag should be 1-3 words (no more than 3 words)
- Tags should be synonyms or related to each other (same theme/topic)
- Tags should be relevant to the script content
- Use lowercase, no special characters except spaces
- Make tags searchable and relevant for YouTube discovery
- Tags should reflect the main topic, style, and content of the video

EXAMPLES:
If the video is about Pokemon in a National Geographic style:
- pokemon documentary
- pokemon biology
- fictional wildlife
- pokémon ecosystem
- national geographic style

If the video is about ancient history:
- ancient history
- historical documentary
- ancient civilizations
- history explained
- educational history

Output format: Return ONLY a comma-separated list of exactly 5 tags, nothing else.
Example: tag1, tag2, tag3, tag4, tag5"""

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You are a YouTube SEO expert who creates relevant, searchable tags for videos. Generate exactly 5 tags that are 1-3 words each, synonyms/related to each other, and relevant to the video content."},
                {"role": "user", "content": tag_prompt}
            ],
            max_output_tokens=200
        )
        
        # Parse the response
        tags_text = response.output_text.strip()
        
        # Clean up the response (remove any labels, quotes, etc.)
        tags_text = tags_text.replace("Tags:", "").replace("tags:", "").replace("TAGS:", "").strip()
        tags_text = tags_text.strip('"').strip("'").strip()
        
        # Split by comma and clean each tag
        tags = [tag.strip().lower() for tag in tags_text.split(',') if tag.strip()]
        
        # Ensure we have exactly 5 tags (pad or trim if needed)
        if len(tags) < 5:
            # If we have fewer than 5, duplicate the last one or add variations
            while len(tags) < 5:
                tags.append(tags[-1] if tags else "documentary")
        elif len(tags) > 5:
            # If we have more than 5, take the first 5
            tags = tags[:5]
        
        # Validate each tag is 1-3 words
        validated_tags = []
        for tag in tags:
            words = tag.split()
            if len(words) <= 3:
                validated_tags.append(tag)
            else:
                # If more than 3 words, take first 3
                validated_tags.append(' '.join(words[:3]))
        
        return validated_tags[:5]  # Ensure exactly 5 tags
        
    except Exception as e:
        print(f"⚠️  Failed to generate tags from script: {e}")
        # Fallback: generate simple tags from video prompt or script keywords
        if video_prompt:
            # Simple fallback: use video prompt words
            words = video_prompt.lower().split()[:5]
            return words if len(words) >= 3 else [video_prompt.lower()] * 5
        else:
            # Generic fallback tags
            return ["documentary", "educational", "informative", "video essay", "explained"]


def build_all_video_segment_assignments(num_segments):
    """
    Build segment assignments where every segment is generated as Sora video.
    """
    return [{'segment_id': i, 'type': 'video'} for i in range(1, num_segments + 1)]


def start_video_generation_job(
    prompt,
    api_key=None,
    model='sora-2',
    resolution='1280x720',
    duration=12,
    reference_image_path=None,
    mode='std'
):
    """
    Start a Sora 2 video generation job and return the job ID (non-blocking).
    
    Args:
        prompt: Text prompt describing the video to generate
        api_key: OpenAI API key
        model: Sora model ('sora-2' or 'sora-2-pro')
        resolution: Target output resolution
        duration: Requested duration in seconds
        reference_image_path: Optional first-frame guide image path
        mode: Deprecated. Ignored.
        
    Returns:
        Video job ID (string)
    """
    headers = get_openai_auth_headers(api_key=api_key)
    sora_size = map_resolution_to_sora_size(resolution)
    sora_seconds = map_duration_to_sora_seconds(duration)

    if model not in ("sora-2", "sora-2-pro"):
        print(f"⚠️  Invalid model '{model}', defaulting to sora-2")
        model = "sora-2"

    payload = {
        "prompt": (prompt or "").strip()[:4000],
        "model": model,
        "size": sora_size,
        "seconds": sora_seconds,
    }

    # Keep optional input reference support for non-human scenes.
    files = None
    if reference_image_path and os.path.exists(reference_image_path):
        try:
            files = {
                "input_reference": open(reference_image_path, "rb"),
            }
            payload_for_multipart = {
                "prompt": payload["prompt"],
                "model": payload["model"],
                "size": payload["size"],
                "seconds": payload["seconds"],
            }
        except Exception:
            files = None

    endpoint = f"{OPENAI_VIDEO_API_BASE_URL}/videos"
    print(f"   📡 Sora API: model={model}, size={sora_size}, seconds={sora_seconds}s")

    try:
        if files:
            multipart_headers = {"Authorization": headers["Authorization"]}
            response = requests.post(
                endpoint,
                data=payload_for_multipart,
                files=files,
                headers=multipart_headers,
                timeout=120,
            )
        else:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()

        video_id = result.get("id")
        if not video_id:
            raise Exception(f"No video id in Sora API response: {result}")
        return video_id
    except requests.exceptions.HTTPError as e:
        try:
            payload = e.response.json()
            error_detail = payload.get("error", {}).get("message") or str(payload)
        except Exception:
            error_detail = str(e)
        raise Exception(f"Sora API request failed: {error_detail}")
    except Exception as e:
        raise e
    finally:
        if files:
            try:
                files["input_reference"].close()
            except Exception:
                pass


def wait_for_video_completion(
    video_id,
    output_path,
    api_key=None,
    poll_interval=10,
    max_wait_time=600
):
    """
    Wait for a Sora video generation task to complete and download the result.
    
    Args:
        video_id: Sora video job ID
        output_path: Path to save the output video
        api_key: OpenAI API key
        poll_interval: Seconds to wait between status checks
        max_wait_time: Maximum time to wait in seconds
        
    Returns:
        Path to generated video file
    """
    status_endpoint = f"{OPENAI_VIDEO_API_BASE_URL}/videos/{video_id}"
    download_endpoint = f"{OPENAI_VIDEO_API_BASE_URL}/videos/{video_id}/content"
    headers = get_openai_auth_headers(api_key=api_key)

    print(f"Polling Sora job {video_id} for completion (checking every {poll_interval} seconds)...")
    start_time = time.time()
    last_status = None
    
    while True:
        elapsed_time = time.time() - start_time
        
        if elapsed_time > max_wait_time:
            raise TimeoutError(
                f"Video generation timed out after {max_wait_time} seconds. Task ID: {video_id}"
            )
        
        try:
            response = requests.get(status_endpoint, headers=headers, timeout=30)
            response.raise_for_status()
            video_info = response.json()
            status = (video_info.get("status") or "unknown").lower()
            
            if status != last_status:
                print(f"  Task {video_id}: Status: {status} (elapsed: {int(elapsed_time)}s)")
                last_status = status
            
            if status == "completed":
                print(f"  ✅ Task {video_id} completed! Downloading video content...")
                stream_video_content(api_key, download_endpoint, output_path)

                # Remove generated audio so narration/music pipeline controls final mix.
                print(f"  Removing source audio from generated video...")
                output_path = remove_audio_from_video(output_path, ffmpeg_path=find_ffmpeg())
                print(f"  ✅ Video saved (no audio): {output_path}")
                return output_path
                
            elif status == "failed":
                error_payload = video_info.get("error") or {}
                error_msg = error_payload.get("message", "Unknown error")

                exception_msg = f"Video generation failed for job {video_id}: {error_msg}"
                exception = Exception(exception_msg)
                
                if any(keyword in error_msg.lower() for keyword in ['content', 'policy', 'moderation', 'blocked', 'sensitive']):
                    exception.code = 'moderation_blocked'
                
                raise exception
            
            time.sleep(poll_interval)
            
        except requests.exceptions.RequestException as e:
            if 'not found' in str(e).lower() or '404' in str(e):
                print(f"  ⚠️  Warning: Could not retrieve task {video_id} status: {e}")
                time.sleep(poll_interval)
                continue
            else:
                raise
        except Exception as e:
            if 'not found' in str(e).lower():
                print(f"  ⚠️  Warning: Could not retrieve task {video_id} status: {e}")
                time.sleep(poll_interval)
                continue
            else:
                raise


def generate_video_from_prompt(
    prompt,
    output_path,
    api_key=None,
    model='sora-2',
    resolution='1280x720',
    duration=12,
    aspect_ratio='16:9',
    poll_interval=10,
    max_wait_time=600,
    reference_image_path=None,
    mode='std'
):
    """
    Generate a video from a text prompt using Sora API (blocking).
    This is a convenience function that combines start_video_generation_job and wait_for_video_completion.
    
    Args:
        prompt: Text prompt describing the video to generate
        output_path: Path to save the output video (MP4)
        api_key: OpenAI API key
        model: Model to use ('sora-2', 'sora-2-pro')
        resolution: Video resolution (default: '1280x720')
        duration: Video duration in seconds (default: 12)
        aspect_ratio: Aspect ratio (default: '16:9')
        poll_interval: Seconds to wait between status checks (default: 10)
        max_wait_time: Maximum time to wait for completion in seconds (default: 600)
        reference_image_path: Path to reference image to use as first frame (optional)
        mode: Deprecated. Ignored.
        
    Returns:
        Path to generated video file
    """
    print("Creating Sora video generation task...")
    video_id = start_video_generation_job(
        prompt=prompt,
        api_key=api_key,
        model=model,
        resolution=resolution,
        duration=duration,
        reference_image_path=reference_image_path,
        mode=mode
    )
    print(f"✅ Video generation started! Task ID: {video_id}")
    
    return wait_for_video_completion(
        video_id=video_id,
        output_path=output_path,
        api_key=api_key,
        poll_interval=poll_interval,
        max_wait_time=max_wait_time
    )


def upscale_video(input_path, output_path=None, target_resolution='1920x1080', method='lanczos'):
    """
    Upscale a video to a higher resolution using ffmpeg.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to save the upscaled video (default: replaces input with _1080p suffix)
        target_resolution: Target resolution in format 'WIDTHxHEIGHT' (default: '1920x1080')
        method: Upscaling algorithm ('lanczos', 'bicubic', 'spline', 'neighbor') (default: 'lanczos')
        
    Returns:
        Path to the upscaled video file
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Check if ffmpeg is available
    ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH.\n"
            "Please install ffmpeg: https://ffmpeg.org/download.html\n"
            "Or use: winget install --id=Gyan.FFmpeg -e\n"
            "After installation, restart your IDE/terminal for PATH changes to take effect."
        )
    
    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_1080p{ext}"
    
    # Map method to ffmpeg scale filter
    scale_methods = {
        'lanczos': 'lanczos',
        'bicubic': 'bicubic',
        'spline': 'spline',
        'neighbor': 'neighbor'
    }
    scale_method = scale_methods.get(method.lower(), 'lanczos')
    
    print(f"Upscaling video from {input_path} to {target_resolution}...")
    print(f"Using {method} upscaling algorithm...")
    
    # Build ffmpeg command
    # Using high-quality settings for upscaling
    cmd = [
        ffmpeg_path,
        '-i', input_path,
        '-vf', f'scale={target_resolution}:flags={scale_method}',
        '-c:v', 'libx264',  # H.264 codec
        '-preset', 'slow',  # Better quality, slower encoding
        '-crf', '18',  # High quality (lower = better quality, 18 is visually lossless)
        '-c:a', 'copy',  # Copy audio without re-encoding
        '-y',  # Overwrite output file if it exists
        output_path
    ]
    
    try:
        # Run ffmpeg
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"✅ Video upscaled successfully to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else "Unknown error"
        raise RuntimeError(f"Failed to upscale video: {error_msg}")
    except Exception as e:
        raise RuntimeError(f"Error during video upscaling: {e}")


def save_segment_metadata(output_folder, segment_id_to_prompt, generated_video_segments, 
                          still_image_videos, segment_assignments, generated_segment_texts,
                          generated_script, num_segments, num_videos, num_still_images,
                          output_video_path,
                          narration_offset=0.0):
    """
    Save segment metadata to a JSON file for later use in regeneration and stitching.
    
    Args:
        output_folder: Folder where metadata will be saved
        segment_id_to_prompt: Mapping from segment_id to video prompt
        generated_video_segments: List of dicts with segment_id, prompt, video_path
        still_image_videos: Deprecated. Ignored.
        segment_assignments: List of segment assignment dicts
        generated_segment_texts: List of segment text strings
        generated_script: Full script text
        num_segments: Total number of segments
        num_videos: Number of video segments
        num_still_images: Deprecated. Ignored.
        output_video_path: Base output video path
        narration_offset: Seconds before narration starts in the video (for centering)
    """
    metadata = {
        'segment_id_to_prompt': segment_id_to_prompt,
        'generated_video_segments': generated_video_segments,
        'still_image_videos': {},
        'segment_assignments': segment_assignments,
        'generated_segment_texts': generated_segment_texts,
        'generated_script': generated_script,
        'num_segments': num_segments,
        'num_videos': num_videos,
        'num_still_images': 0,
        'output_video_path': output_video_path,
        'narration_offset': narration_offset
    }
    
    metadata_path = os.path.join(output_folder, 'segment_metadata.json')
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ Segment metadata saved to: {metadata_path}")
        return metadata_path
    except Exception as e:
        print(f"⚠️  Warning: Failed to save segment metadata: {e}")
        return None


def load_segment_metadata(output_folder):
    """
    Load segment metadata from JSON file.
    
    Args:
        output_folder: Folder where metadata is saved
        
    Returns:
        Dictionary with segment metadata, or None if not found
    """
    metadata_path = os.path.join(output_folder, 'segment_metadata.json')
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"✅ Segment metadata loaded from: {metadata_path}")
        return metadata
    except Exception as e:
        print(f"⚠️  Warning: Failed to load segment metadata: {e}")
        return None


def ensure_audio_on_video(video_path, ffmpeg_path=None, narration_audio_path=None, music_volume=0.08, audio_delay_ms=1000):
    """
    Ensure audio (narration + music) is added to a video file.
    This function automatically finds narration and music files and adds them to the video.
    
    Args:
        video_path: Path to the video file
        ffmpeg_path: Path to ffmpeg executable (if None, will try to find it)
        narration_audio_path: Path to narration audio (if None, will try to find it)
        music_volume: Volume level for background music (default: 0.08 = 8%)
        audio_delay_ms: Delay in milliseconds before narration starts (default: 1000ms).
            Used to center narration within the video — set to narration_offset * 1000.
        
    Returns:
        Path to the video with audio, or original path if audio addition failed
    """
    if not video_path or not os.path.exists(video_path):
        print(f"[WARNING] Video file not found: {video_path}")
        return video_path
    
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        print(f"[WARNING] FFmpeg not found. Cannot add audio to video.")
        return video_path
    
    # Check if video already has audio
    try:
        cmd_check = [
            ffmpeg_path.replace('ffmpeg.exe', 'ffprobe.exe').replace('ffmpeg', 'ffprobe'),
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd_check, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            # Video already has audio stream
            print(f"[OK] Video already has audio: {video_path}")
            return video_path
    except:
        pass  # If check fails, proceed to add audio
    
    print(f"Adding audio to video: {os.path.basename(video_path)}")
    
    # Find narration audio file
    if not narration_audio_path:
        current_dir = os.getcwd()
        possible_narration = [
            os.path.join(current_dir, "narration_audio.mp3"),
            os.path.join(current_dir, NARRATION_AUDIO_PATH),
            "narration_audio.mp3",
            NARRATION_AUDIO_PATH
        ]
        for nar_path in possible_narration:
            if os.path.exists(nar_path):
                narration_audio_path = nar_path
                break
    
    if not narration_audio_path or not os.path.exists(narration_audio_path):
        print(f"[WARNING] Narration audio not found. Looking for: narration_audio.mp3")
        # Try to find any narration file
        current_dir = os.getcwd()
        for root, dirs, files in os.walk(current_dir):
            for file in files:
                if file.startswith("narration") and file.endswith(".mp3"):
                    narration_audio_path = os.path.join(root, file)
                    print(f"   Found narration: {narration_audio_path}")
                    break
            if narration_audio_path:
                break
    
    if not narration_audio_path or not os.path.exists(narration_audio_path):
        print(f"[WARNING] Cannot add audio: narration file not found")
        return video_path
    
    # Get video duration
    video_duration = get_media_duration(video_path, ffmpeg_path)
    if not video_duration:
        print(f"[WARNING] Cannot determine video duration")
        return video_path
    
    print(f"   Video duration: {video_duration:.2f}s")
    
    # Find music file
    current_dir = os.getcwd()
    music_source = None
    for music_file in ["VIDEO_MUSIC.mp3", "video_music.mp3", "VIDEO_MUSIC.MP3"]:
        music_path_check = os.path.join(current_dir, music_file)
        if os.path.exists(music_path_check):
            music_source = music_path_check
            break
    
    import tempfile
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    
    # Prepare final audio
    final_audio_path = None
    
    if music_source:
        # Mix narration + music
        print(f"   Mixing narration + music (music at {music_volume*100:.0f}% volume)...")
        
        # Sync music to video duration
        synced_music_path = os.path.join(temp_dir, f"music_synced_{timestamp}.mp3")
        music_duration = get_media_duration(music_source, ffmpeg_path)
        
        if music_duration:
            if abs(music_duration - video_duration) > 0.1:
                if music_duration > video_duration:
                    # Trim music
                    fade_out_start = max(0, video_duration - 1.0)
                    cmd_music = [
                        ffmpeg_path,
                        "-i", music_source,
                        "-t", str(video_duration),
                        "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1",
                        "-c:a", "libmp3lame",
                        "-b:a", "192k",
                        "-y",
                        synced_music_path
                    ]
                else:
                    # Loop music
                    loop_count = int((video_duration / music_duration) + 1)
                    fade_out_start = max(0, video_duration - 1.0)
                    cmd_music = [
                        ffmpeg_path,
                        "-stream_loop", str(loop_count - 1),
                        "-i", music_source,
                        "-t", str(video_duration),
                        "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1",
                        "-c:a", "libmp3lame",
                        "-b:a", "192k",
                        "-y",
                        synced_music_path
                    ]
                
                try:
                    subprocess.run(cmd_music, capture_output=True, text=True, check=True, timeout=300)
                    print(f"   [OK] Music synced to video duration")
                except Exception as e:
                    print(f"   [WARNING] Music sync failed: {e}, using original music")
                    synced_music_path = music_source
            else:
                # Apply fade in/out
                fade_out_start = max(0, video_duration - 1.0)
                faded_music_path = os.path.join(temp_dir, f"music_faded_{timestamp}.mp3")
                cmd_fade = [
                    ffmpeg_path,
                    "-i", music_source,
                    "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1",
                    "-c:a", "libmp3lame",
                    "-b:a", "192k",
                    "-y",
                    faded_music_path
                ]
                try:
                    subprocess.run(cmd_fade, capture_output=True, text=True, check=True, timeout=300)
                    synced_music_path = faded_music_path
                except:
                    synced_music_path = music_source
        
        # Mix narration + music
        final_audio_path = os.path.join(temp_dir, f"audio_mixed_{timestamp}.mp3")
        filter_complex = (
            f"[0:a]aresample=44100,volume=1.0[voice];"
            f"[1:a]aresample=44100,volume={music_volume}[music];"
            f"[voice][music]amix=inputs=2:duration=longest:dropout_transition=2,"
            f"volume=2.0"
        )
        
        cmd_mix = [
            ffmpeg_path,
            "-i", narration_audio_path,
            "-i", synced_music_path,
            "-filter_complex", filter_complex,
            "-t", str(video_duration),
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            "-y",
            final_audio_path
        ]
        
        try:
            subprocess.run(cmd_mix, capture_output=True, text=True, check=True, timeout=300)
            print(f"   [OK] Audio mixed: narration + music")
        except Exception as e:
            print(f"   [WARNING] Audio mixing failed: {e}, using narration only")
            final_audio_path = narration_audio_path
    else:
        # No music, use narration only
        print(f"   [WARNING] Music file not found, using narration only")
        final_audio_path = narration_audio_path
    
    # Add audio to video
    base, ext = os.path.splitext(video_path)
    video_with_audio_path = f"{base}_with_audio{ext}"
    
    try:
        print(f"   Audio delay: {audio_delay_ms}ms ({audio_delay_ms/1000:.1f}s)")
        result_path = add_audio_to_video(
            video_path=video_path,
            audio_path=final_audio_path,
            output_path=video_with_audio_path,
            ffmpeg_path=ffmpeg_path,
            sync_duration=False,  # Audio already synced
            audio_delay_ms=audio_delay_ms  # Center narration in video
        )
        print(f"[OK] Audio added to video: {os.path.basename(result_path)}")
        
        # Clean up temp files
        try:
            if final_audio_path and final_audio_path != narration_audio_path and os.path.exists(final_audio_path):
                os.remove(final_audio_path)
            if 'synced_music_path' in locals() and synced_music_path != music_source and os.path.exists(synced_music_path):
                os.remove(synced_music_path)
        except:
            pass
        
        return result_path
    except Exception as e:
        print(f"[WARNING] Failed to add audio to video: {e}")
        return video_path


def stitch_videos(video_paths, output_path, ffmpeg_path=None, upscale_to_1080p=False, narration_offset=0.0):
    """
    Stitch multiple video files together into one video using ffmpeg.
    
    Args:
        video_paths: List of video file paths in order (should be sorted by segment ID)
        output_path: Path to save the stitched video
        ffmpeg_path: Path to ffmpeg executable (if None, will try to find it)
        upscale_to_1080p: If True, upscale the stitched video to 1080p using lanczos algorithm
        narration_offset: Seconds before narration starts in the video (for centering audio)
        
    Returns:
        Path to the stitched (and optionally upscaled) video file
    """
    if not video_paths:
        raise ValueError("No video paths provided for stitching")
    
    if len(video_paths) == 1:
        # If only one video, just copy it
        import shutil
        shutil.copy2(video_paths[0], output_path)
        print(f"✅ Single video copied to: {output_path}")
        return output_path
    
    print(f"Stitching {len(video_paths)} video segments in order:")
    for i, vp in enumerate(video_paths, 1):
        print(f"  {i}. {os.path.basename(vp)}")
    
    # Find ffmpeg if not provided
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH. "
            "Please install ffmpeg: https://ffmpeg.org/download.html"
        )
    
    # Verify all input videos exist
    for vp in video_paths:
        if not os.path.exists(vp):
            raise FileNotFoundError(f"Video file not found: {vp}")
    
    print(f"Stitching {len(video_paths)} video segments together...")
    
    # Create a temporary file list for ffmpeg concat
    import tempfile
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    concat_file = os.path.join(temp_dir, f"concat_list_{timestamp}.txt")
    
    try:
        # Write concat file (ffmpeg format)
        with open(concat_file, 'w', encoding='utf-8') as f:
            for video_path in video_paths:
                # Convert to absolute path to avoid path issues
                abs_path = os.path.abspath(video_path)
                # Escape single quotes and backslashes for ffmpeg
                # Use forward slashes for ffmpeg (works on Windows too)
                escaped_path = abs_path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        # First, stitch videos together
        temp_stitched = os.path.join(temp_dir, f"stitched_temp_{timestamp}.mp4")
        cmd = [
            ffmpeg_path,
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # Copy streams without re-encoding (faster)
            '-y',  # Overwrite output file
            temp_stitched
        ]
        
        # Run ffmpeg to stitch (suppress verbose output)
        process = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True
        )
        
        # Apply fade in and fade out
        # Use faster preset to avoid long encoding times for long videos
        print(f"   Applying fade in (1s) and fade out (1s)...")
        video_duration = get_media_duration(temp_stitched, ffmpeg_path)
        if video_duration and video_duration > 2.0:  # Only apply fades if video is longer than 2 seconds
            # Calculate fade out start time
            fade_out_start = max(0, video_duration - 1.0)
            
            # Use 'veryfast' preset for faster encoding (especially important for long videos)
            # For videos longer than 5 minutes, use 'ultrafast' to save time
            if video_duration > 300:  # 5 minutes
                fade_preset = 'ultrafast'
                print(f"   Using ultrafast preset for fade (video is {video_duration/60:.1f} minutes long)")
            else:
                fade_preset = 'veryfast'
            
            cmd_fade = [
                ffmpeg_path,
                '-i', temp_stitched,
                '-vf', f'fade=t=in:st=0:d=1,fade=t=out:st={fade_out_start}:d=1',  # Fade in 1s and fade out 1s
                '-c:v', 'libx264',
                '-preset', fade_preset,  # Use faster preset
                '-crf', '23',
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-y',
                output_path
            ]
            
            try:
                print(f"   Processing fade effects (using {fade_preset} preset for faster encoding)...")
                # Run ffmpeg - suppress verbose output
                result = subprocess.run(
                    cmd_fade,
                    stdout=subprocess.DEVNULL,  # Suppress stdout
                    stderr=subprocess.DEVNULL,  # Suppress stderr (verbose ffmpeg output)
                    text=True,
                    check=True
                )
                print(f"   OK: Fade in/out applied successfully")
                # Clean up temp file
                if os.path.exists(temp_stitched):
                    try:
                        os.remove(temp_stitched)
                    except:
                        pass
            except subprocess.TimeoutExpired:
                print(f"   WARNING: Fade processing timed out (took longer than expected)")
                print(f"   SKIPPING: Skipping fade in/out - using stitched video without fade effects")
                import shutil
                shutil.copy2(temp_stitched, output_path)
                if os.path.exists(temp_stitched):
                    try:
                        os.remove(temp_stitched)
                    except:
                        pass
            except subprocess.CalledProcessError as e:
                error_msg = e.stdout if e.stdout else (e.stderr if e.stderr else 'Unknown error')
                print(f"   WARNING: Fade application failed: {error_msg[:200] if error_msg else 'Unknown error'}")
                print(f"   SKIPPING: Skipping fade in/out - using stitched video without fade effects")
                import shutil
                shutil.copy2(temp_stitched, output_path)
                if os.path.exists(temp_stitched):
                    try:
                        os.remove(temp_stitched)
                    except:
                        pass
            except Exception as e:
                print(f"   WARNING: Fade processing error: {e}")
                print(f"   SKIPPING: Skipping fade in/out - using stitched video without fade effects")
                import shutil
                shutil.copy2(temp_stitched, output_path)
                if os.path.exists(temp_stitched):
                    try:
                        os.remove(temp_stitched)
                    except:
                        pass
        else:
            # Video is too short for fades, just copy
            import shutil
            shutil.copy2(temp_stitched, output_path)
            if os.path.exists(temp_stitched):
                try:
                    os.remove(temp_stitched)
                except:
                    pass
        
        print(f"Videos stitched successfully to: {output_path}")
        
        # Automatically add audio to stitched video if available
        try:
            narration_delay_ms = int(narration_offset * 1000) if narration_offset > 0 else 1000
            if narration_offset > 0:
                print(f"   🎬 Centering narration: {narration_delay_ms}ms ({narration_offset:.1f}s) delay before narration starts")
            ensure_audio_on_video(output_path, ffmpeg_path=ffmpeg_path, audio_delay_ms=narration_delay_ms)
        except Exception as audio_error:
            print(f"⚠️  Warning: Could not automatically add audio to stitched video: {audio_error}")
            print(f"   Video stitched successfully but may be missing audio")
        
        # Apply upscaling if requested (right after stitching, during same execution)
        if upscale_to_1080p:
            try:
                print(f"   Upscaling stitched video to 1080p using lanczos algorithm...")
                original_output_path = output_path
                base, ext = os.path.splitext(output_path)
                upscaled_output_path = f"{base}_1080p{ext}"
                
                upscaled_path = upscale_video(
                    input_path=output_path,
                    output_path=upscaled_output_path,
                    target_resolution='1920x1080',
                    method='lanczos'
                )
                
                # Replace output_path with upscaled version
                output_path = upscaled_path
                print(f"   ✅ Video upscaled to 1080p: {output_path}")
                
                # Clean up original 720p video if upscaling succeeded
                if os.path.exists(original_output_path) and original_output_path != output_path:
                    try:
                        os.remove(original_output_path)
                        print(f"   Cleaned up original 720p stitched video")
                    except Exception as cleanup_error:
                        print(f"   ⚠️  Could not clean up original video: {cleanup_error}")
            except Exception as upscale_error:
                print(f"   ⚠️  Video upscaling failed: {upscale_error}")
                print(f"   Continuing with original resolution video...")
                # Keep original output_path if upscaling fails
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else "Unknown error"
        raise RuntimeError(f"Failed to stitch videos: {error_msg}")
    except Exception as e:
        raise RuntimeError(f"Error during video stitching: {e}")
    finally:
        # Clean up concat file
        if os.path.exists(concat_file):
            try:
                os.remove(concat_file)
            except:
                pass


# OBSOLETE: Subtitle/caption generation code - commented out
# def generate_srt_from_audio(audio_path, script, output_path=None, api_key=None, segment_duration=10.0):
#     """
#     Generate an SRT subtitle file with word-by-word timing using OpenAI Whisper API.
#     Displays 1-2 words at a time as they are narrated, with words always side by side.
#     Uses word-level timestamps for precise synchronization between captions and audio.
#     
#     Args:
#         audio_path: Path to the audio file (voiceover) to analyze
#         script: The narration script text (for reference/validation)
#         output_path: Path to save the SRT file (default: temp file)
#         api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
#         segment_duration: Not used (kept for compatibility, word-level timing is used instead)
#         
#     Returns:
#         Path to the generated SRT file, or None if generation fails
#     """
#     if not OPENAI_AVAILABLE:
#         print("⚠️  OpenAI library not available, falling back to estimated timing")
#         return None
#     
#     import tempfile
#     
#     if output_path is None:
#         temp_dir = tempfile.gettempdir()
#         timestamp = int(time.time())
#         output_path = os.path.join(temp_dir, f"subtitles_{timestamp}.srt")
#     
#     # Initialize OpenAI client
#     if api_key is None:
#         api_key = os.getenv('OPENAI_API_KEY')
#     if api_key is None:
#         api_key = OPENAI_API_KEY
#     
#     if not api_key:
#         print("⚠️  No OpenAI API key available, falling back to estimated timing")
#         return None
#     
#     client = OpenAI(api_key=api_key)
#     
#     try:
#         print("🎤 Transcribing audio with Whisper for word-level timestamps...")
#         
#         # Use Whisper API to get word-level timestamps
#         with open(audio_path, 'rb') as audio_file:
#             transcript = client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=audio_file,
#                 response_format="verbose_json",  # Get detailed JSON response
#                 timestamp_granularities=["word"]  # Request word-level timestamps
#             )
#         
#         # Extract word-level timestamps from response
#         words = []
#         
#         # Check if response is a dict (JSON) or object
#         if isinstance(transcript, dict):
#             # Handle dict response
#             if 'words' in transcript and transcript['words']:
#                 words = transcript['words']
#             elif 'segments' in transcript:
#                 # Extract words from segments
#                 for segment in transcript['segments']:
#                     if 'words' in segment and segment['words']:
#                         words.extend(segment['words'])
#         else:
#             # Handle object response
#             if hasattr(transcript, 'words') and transcript.words:
#                 words = transcript.words
#             elif hasattr(transcript, 'segments') and transcript.segments:
#                 # Fallback: extract words from segments
#                 for segment in transcript.segments:
#                     if hasattr(segment, 'words') and segment.words:
#                         words.extend(segment.words)
#                     elif isinstance(segment, dict) and 'words' in segment:
#                         words.extend(segment['words'])
#         
#         if not words:
#             print("⚠️  No word-level timestamps available from Whisper")
#             print(f"   Response type: {type(transcript)}")
#             print(f"   Response keys/attrs: {dir(transcript) if not isinstance(transcript, dict) else list(transcript.keys())}")
#             print("   Falling back to estimated timing")
#             return None
#         
#         print(f"✅ Transcribed {len(words)} words with timestamps")
#         
#         # Format time as SRT format: HH:MM:SS,mmm
#         def format_srt_time(seconds):
#             hours = int(seconds // 3600)
#             minutes = int((seconds % 3600) // 60)
#             secs = int(seconds % 60)
#             millis = int((seconds % 1) * 1000)
#             return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
#         
#         # Create word-by-word subtitles (1-2 words at a time, side by side)
#         srt_content = []
#         subtitle_index = 1
#         
#         # Parse words with their timestamps
#         word_list = []
#         for word_data in words:
#             # Extract word and timing
#             if isinstance(word_data, dict):
#                 word = word_data.get('word', '').strip()
#                 start = word_data.get('start', 0)
#                 end = word_data.get('end', start + 0.3)
#             else:
#                 word = getattr(word_data, 'word', '').strip()
#                 start = getattr(word_data, 'start', 0)
#                 end = getattr(word_data, 'end', start + 0.3)
#             
#             if word:
#                 word_list.append({
#                     'word': word,
#                     'start': start,
#                     'end': end
#                 })
#         
#         if not word_list:
#             print("⚠️  No words found in transcription")
#             return None
#         
#         # Generate subtitles: 1-2 words at a time, side by side
#         # Add 1 second delay to fix narration lag (subtitles appear 1 second before narration)
#         SUBTITLE_DELAY = 1.0  # 1 second delay to sync with narration
#         i = 0
#         while i < len(word_list):
#             current_word = word_list[i]
#             word_text = current_word['word']
#             start_time = current_word['start'] + SUBTITLE_DELAY
#             end_time = current_word['end'] + SUBTITLE_DELAY
#             
#             # Try to group with next word (max 2 words per subtitle)
#             if i + 1 < len(word_list):
#                 next_word = word_list[i + 1]
#                 # Group if next word starts within 0.5 seconds (natural speech flow)
#                 if next_word['start'] - current_word['end'] < 0.5:
#                     # Group 2 words together, side by side
#                     word_text = f"{current_word['word']} {next_word['word']}"
#                     end_time = next_word['end']
#                     i += 2  # Skip both words
#                 else:
#                     # Single word (gap is too large)
#                     i += 1
#             else:
#                 # Last word, single
#                 i += 1
#             
#             # Create SRT entry (words always side by side, not stacked)
#             srt_content.append(f"{subtitle_index}")
#             srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
#             srt_content.append(word_text)  # Single line, words side by side
#             srt_content.append("")  # Empty line between entries
#             subtitle_index += 1
#         
#         # Write SRT file
#         with open(output_path, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(srt_content))
#         
#         print(f"✅ Generated SRT with {subtitle_index - 1} word-by-word subtitles (1-2 words each)")
#         return output_path
#         
#     except Exception as e:
#         print(f"⚠️  Whisper transcription failed: {e}")
#         print("   Falling back to estimated timing...")
#         return None
# End of obsolete generate_srt_from_audio function


# OBSOLETE: Subtitle/caption generation code - commented out
# def generate_srt_from_script(script, video_duration, output_path=None):
#     """
#     Generate an SRT subtitle file from a script, timing it based on video duration.
#     This is a fallback method when audio-based timing is not available.
#     Splits script into natural phrases/sentences and distributes them evenly.
#     
#     Args:
#         script: The narration script text
#         video_duration: Total video duration in seconds
#         output_path: Path to save the SRT file (default: temp file)
#         
#     Returns:
#         Path to the generated SRT file
#     """
#     import tempfile
#     import re
#     
#     if output_path is None:
#         temp_dir = tempfile.gettempdir()
#         timestamp = int(time.time())
#         output_path = os.path.join(temp_dir, f"subtitles_{timestamp}.srt")
#     
#     # Split script into sentences - preserve punctuation
#     # Split on sentence-ending punctuation (. ! ?) but keep the punctuation with the sentence
#     sentences = re.split(r'([.!?]+)', script)
#     
#     # Recombine sentences with their punctuation
#     sentence_list = []
#     i = 0
#     while i < len(sentences):
#         sentence = sentences[i].strip()
#         if i + 1 < len(sentences) and re.match(r'^[.!?]+$', sentences[i + 1].strip()):
#             sentence += sentences[i + 1].strip()
#             i += 2
#         else:
#             i += 1
#         if sentence:
#             sentence_list.append(sentence)
#     
#     # If no sentences found, split by commas or just use the whole script
#     if not sentence_list:
#         sentence_list = [p.strip() for p in re.split(r'[,;]', script) if p.strip()]
#     if not sentence_list:
#         sentence_list = [script]
#     
#     # Calculate timing for each sentence
#     # Reserve 0.5 seconds at the start and end
#     usable_duration = video_duration - 1.0
#     if usable_duration <= 0:
#         usable_duration = video_duration
#     
#     # Distribute sentences evenly, but allow for natural variation
#     sentence_count = len(sentence_list)
#     if sentence_count == 0:
#         return None
#     
#     # Calculate timing based on natural speech patterns
#     # Average reading speed: ~2.3 words/second, but with pauses (ellipses, dashes) it's slower
#     words_per_second = 2.3  # Base reading speed
#     
#     # Format time as SRT format: HH:MM:SS,mmm
#     def format_srt_time(seconds):
#         hours = int(seconds // 3600)
#         minutes = int((seconds % 3600) // 60)
#         secs = int(seconds % 60)
#         millis = int((seconds % 1) * 1000)
#         return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
#     
#     # Generate SRT content - split by sentences, ensure sentence ends on its own line
#     # Add 1 second delay to fix narration lag (subtitles appear 1 second before narration)
#     SUBTITLE_DELAY = 1.0  # 1 second delay to sync with narration
#     srt_content = []
#     current_time = 0.2 + SUBTITLE_DELAY  # Start slightly earlier (0.2 seconds in) + delay
#     subtitle_index = 1
#     MIN_GAP = 0.01  # Minimum gap between subtitles (10ms) to prevent any overlap
#     SENTENCE_GAP = 0.1  # Gap between sentences (100ms)
#     last_end_time = 0.0 + SUBTITLE_DELAY
#     
#     for sentence in sentence_list:
#         # Preserve punctuation - don't remove it
#         sentence = sentence.strip()
#         
#         if not sentence:
#             continue
#         
#         # Split sentence into words (preserving punctuation attached to words)
#         # Use regex to split on whitespace but keep punctuation with words
#         words = re.findall(r'\S+', sentence)
#         
#         if not words:
#             continue
#         
#         # Check if last word ends with sentence-ending punctuation
#         last_word = words[-1]
#         is_sentence_end = bool(re.search(r'[.!?]+', last_word))
#         
#         # Calculate total duration for this sentence
#         word_count = len(words)
#         pause_count = sentence.count('...') + sentence.count('..') + sentence.count('—') + sentence.count('-')
#         pause_duration = pause_count * 0.5  # Each pause adds 0.5 seconds
#         base_duration = word_count / words_per_second
#         sentence_duration = base_duration + pause_duration
#         
#         # Apply bounds: minimum 0.8 seconds, maximum 4 seconds per sentence
#         min_duration = 0.8
#         max_duration = 4.0
#         sentence_duration = min(max_duration, max(min_duration, sentence_duration))
#         
#         # Don't exceed video duration
#         if current_time + sentence_duration > video_duration - 0.2:
#             sentence_duration = max(0.3, video_duration - current_time - 0.2)
#         
#         # Calculate duration per word
#         word_duration = sentence_duration / word_count
#         word_duration = min(0.7, max(0.35, word_duration))  # 0.35-0.7 seconds per word
#         
#         # Process words in this sentence
#         word_idx = 0
#         while word_idx < len(words):
#             if current_time >= video_duration - 0.2:
#                 break
#             
#             current_word = words[word_idx]
#             is_last_word = (word_idx == len(words) - 1)
#             
#             # If this is the last word in the sentence, it MUST be on its own line
#             if is_last_word:
#                 # Last word of sentence - ensure it displays long enough
#                 start_time = max(current_time, last_end_time + MIN_GAP)
#                 end_time = start_time + max(word_duration, 0.5)  # At least 0.5s for last word
#                 end_time = min(end_time, video_duration - 0.1)
#                 
#                 srt_content.append(f"{subtitle_index}")
#                 srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
#                 srt_content.append(current_word)  # Includes punctuation
#                 srt_content.append("")  # Empty line between entries
#                 subtitle_index += 1
#                 
#                 # CRITICAL: Update last_end_time - next sentence MUST start after this ends + gap
#                 last_end_time = end_time + SENTENCE_GAP
#                 current_time = last_end_time
#                 word_idx += 1
#             else:
#                 # Not the last word - can group with next word (max 2 words)
#                 next_word = words[word_idx + 1] if word_idx + 1 < len(words) else None
#                 
#                 # Try to group with next word (max 2 words per subtitle, side by side)
#                 if next_word:
#                     # Group 2 words together, side by side
#                     group_text = f"{current_word} {next_word}"  # Words side by side, not stacked
#                     start_time = max(current_time, last_end_time + MIN_GAP)
#                     end_time = start_time + (word_duration * 1.8)  # Slightly longer for 2 words
#                     end_time = min(end_time, video_duration - 0.1)
#                     
#                     srt_content.append(f"{subtitle_index}")
#                     srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
#                     srt_content.append(group_text)  # Single line: words side by side
#                     srt_content.append("")  # Empty line between entries
#                     subtitle_index += 1
#                     
#                     # Move to next subtitle start time (end of current + gap)
#                     last_end_time = end_time
#                     current_time = end_time + MIN_GAP
#                     word_idx += 2  # Skip both words
#                 else:
#                     # Single word subtitle (not last in sentence)
#                     start_time = max(current_time, last_end_time + MIN_GAP)
#                     end_time = start_time + word_duration
#                     end_time = min(end_time, video_duration - 0.1)
#                     
#                     srt_content.append(f"{subtitle_index}")
#                     srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
#                     srt_content.append(current_word)  # Single word, side by side (not stacked)
#                     srt_content.append("")  # Empty line between entries
#                     subtitle_index += 1
#                     
#                     # Move to next subtitle start time (end of current + gap)
#                     last_end_time = end_time
#                     current_time = end_time + MIN_GAP
#                     word_idx += 1
#     
#     # Write SRT file
#     try:
#         with open(output_path, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(srt_content))
#         return output_path
#     except Exception as e:
#         print(f"⚠️  Failed to create SRT file: {e}")
#         return None
# End of obsolete generate_srt_from_script function


# OBSOLETE: Subtitle/caption generation code - commented out
# def add_subtitles_to_video(video_path, script, video_duration, output_path=None, ffmpeg_path=None, audio_path=None, api_key=None):
#     """
#     Add styled subtitles to a video using FFmpeg.
#     Creates an SRT file with word-level timestamps from audio (if available) or estimates from script.
#     
#     Args:
#         video_path: Path to the input video
#         script: The narration script text
#         video_duration: Total video duration in seconds
#         output_path: Path to save the output video (default: overwrite input)
#         ffmpeg_path: Path to ffmpeg executable (default: auto-detect)
#         audio_path: Path to the voiceover audio file for accurate word-level timing (optional)
#         api_key: OpenAI API key for Whisper transcription (optional)
#         
#     Returns:
#         Path to the output video with subtitles
#     """
#     if ffmpeg_path is None:
#         ffmpeg_path = find_ffmpeg()
#     
#     if not ffmpeg_path:
#         raise Exception("FFmpeg not found. Cannot add subtitles to video.")
#     
#     if not script or len(script.strip()) == 0:
#         print("⚠️  No script provided, skipping subtitle generation")
#         return video_path
#     
#     # Clean script: remove musical break and visual break markers (these shouldn't appear in captions)
#     import re
#     cleaned_script = script
#     # Remove [MUSICAL BREAK] and [VISUAL BREAK] markers (case-insensitive) and any text that follows them
#     # This handles cases where the marker might have explanatory text after it
#     cleaned_script = re.sub(r'\[MUSICAL\s+BREAK\][^\.!?\n\[\]]*[\.!?\n]?', '', cleaned_script, flags=re.IGNORECASE)
#     cleaned_script = re.sub(r'\[VISUAL\s+BREAK\][^\.!?\n\[\]]*[\.!?\n]?', '', cleaned_script, flags=re.IGNORECASE)
#     # Also remove any standalone markers (exact matches)
#     cleaned_script = cleaned_script.replace('[MUSICAL BREAK]', '')
#     cleaned_script = cleaned_script.replace('[VISUAL BREAK]', '')
#     cleaned_script = cleaned_script.replace('[musical break]', '')
#     cleaned_script = cleaned_script.replace('[visual break]', '')
#     # Remove any phrases that might have been generated describing these breaks (without brackets)
#     # This catches cases like "visual break look over castle" that might appear in transcriptions
#     cleaned_script = re.sub(r'(?i)\b(visual\s+break|musical\s+break)\s+[^\.!?\n]*', '', cleaned_script)
#     # Clean up any extra whitespace, newlines, or punctuation artifacts
#     cleaned_script = re.sub(r'\s+', ' ', cleaned_script)
#     cleaned_script = re.sub(r'\s*\.\s*\.\s*\.\s*', '...', cleaned_script)  # Normalize ellipses
#     cleaned_script = cleaned_script.strip()
#     
#     if output_path is None:
#         base, ext = os.path.splitext(video_path)
#         # If video already has "_with_subtitles" in the name, remove it first to avoid duplicates
#         if "_with_subtitles" in base:
#             # Remove the last occurrence of "_with_subtitles" and any trailing numbers
#             base = base.rsplit("_with_subtitles", 1)[0]
#         output_path = f"{base}_with_subtitles{ext}"
#     
#     import tempfile
#     temp_dir = tempfile.gettempdir()
#     timestamp = int(time.time())
#     srt_path = os.path.join(temp_dir, f"subtitles_{timestamp}.srt")
#     
#     try:
#         # Try to generate SRT with word-level timestamps from audio first
#         srt_path = None
#         if audio_path and os.path.exists(audio_path):
#             print("🎯 Attempting to generate captions with exact audio timing...")
#             srt_path = generate_srt_from_audio(audio_path, cleaned_script, srt_path, api_key)
#         
#         # Fallback to estimated timing if audio-based generation failed
#         if not srt_path or not os.path.exists(srt_path):
#             print("📝 Using estimated timing from script (audio-based timing unavailable)...")
#             srt_path = generate_srt_from_script(cleaned_script, video_duration, srt_path)
#         
#         if not srt_path or not os.path.exists(srt_path):
#             print("⚠️  Failed to generate SRT file, skipping subtitles")
#             return video_path
#         
#         # FFmpeg subtitle styling
#         # Professional, clean appearance suitable for YouTube
#         # Much lower on screen, refined typography
#         subtitle_style = (
#             "FontName=Segoe UI,"
#             "FontSize=20,"
#             "PrimaryColour=&H00F5F5F5,"  # Soft white (slightly off-white for better readability)
#             "OutlineColour=&H00000000,"  # Black outline
#             "BackColour=&H90000000,"  # More opaque black background for better contrast
#             "Bold=0,"  # Not bold for cleaner, more professional look
#             "Alignment=2,"  # Bottom center (1=bottom-left, 2=bottom-center, 3=bottom-right)
#             "MarginV=20,"  # 20 pixels from bottom (ensures captions are at the bottom)
#             "Outline=1.5,"  # 1.5 pixel outline (slightly thinner for elegance)
#             "Shadow=0.3,"  # Subtle shadow
#             "MarginL=10,"  # Left margin to prevent cutoff
#             "MarginR=10"  # Right margin to prevent cutoff
#         )
#         
#         # Escape SRT path for Windows (replace backslashes and escape special characters)
#         srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
#         
#         # Build FFmpeg command to burn subtitles
#         # Use subtitles filter with force_style for consistent appearance
#         # Ensure subtitles are always visible and don't get cut off
#         cmd = [
#             ffmpeg_path,
#             "-i", video_path,
#             "-vf", f"subtitles='{srt_path_escaped}':force_style='{subtitle_style}'",
#             "-c:a", "copy",  # Copy audio without re-encoding
#             "-y",  # Overwrite output
#             output_path
#         ]
#         
#         # Run FFmpeg
#         subprocess.run(cmd, capture_output=True, text=True, check=True)
#         print(f"✅ Subtitles added to video: {output_path}")
#         return output_path
#         
#     except subprocess.CalledProcessError as e:
#         print(f"⚠️  Failed to add subtitles: {e.stderr if e.stderr else 'Unknown error'}")
#         print("   Continuing without subtitles...")
#         return video_path
#     except Exception as e:
#         print(f"⚠️  Error adding subtitles: {e}")
#         print("   Continuing without subtitles...")
#         return video_path
#     finally:
#         # Clean up SRT file
#         if os.path.exists(srt_path):
#             try:
#                 os.remove(srt_path)
#             except:
#                 pass
# End of obsolete add_subtitles_to_video function


def stream_video_content(api_key, video_url, filepath):
    """
    Download generated video content from OpenAI to a local file.
    
    Args:
        api_key: OpenAI API key
        video_url: Direct URL to video content endpoint
        filepath: Local path to save the video
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        headers = get_openai_auth_headers(api_key=api_key)
        
        # Stream the video content from the URL
        response = requests.get(video_url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {percent:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        
    except Exception as e:
        raise Exception(f"Failed to download video content: {e}")


def stitch_all_segments(
    generated_video_segments,
    still_image_videos,
    segment_assignments,
    num_segments,
    output_video_path,
    duration,
    upscale_to_1080p=False,
    narration_offset=0.0
):
    """
    Stitch all generated Sora video segments together into final video.
    
    Args:
        generated_video_segments: List of dicts with segment_id, prompt, video_path
        still_image_videos: Deprecated. Ignored.
        segment_assignments: Deprecated. Ignored.
        num_segments: Total number of segments
        output_video_path: Base output video path
        duration: Expected total duration
        upscale_to_1080p: If True, upscale the stitched video to 1080p using lanczos algorithm
        narration_offset: Seconds before narration starts in the video (for centering audio)
        
    Returns:
        Path to stitched (and optionally upscaled) video file
    """
    # Create a mapping of segment_id -> generated video path.
    video_segment_map = {}
    for seg_info in generated_video_segments:
        seg_id = int(seg_info['segment_id']) if seg_info.get('segment_id') is not None else None
        if seg_id is not None:
            video_segment_map[seg_id] = seg_info['video_path']

    # Combine video segments in strict numeric order.
    all_segment_paths = []
    print(f"\nBuilding segment order (total segments: {num_segments})...")

    for segment_id in range(1, num_segments + 1):
        if segment_id not in video_segment_map:
            raise RuntimeError(f"Missing generated Sora segment {segment_id}; cannot stitch final video.")
        video_path = video_segment_map[segment_id]
        if not os.path.exists(video_path):
            raise RuntimeError(f"Generated Sora segment file missing for segment {segment_id}: {video_path}")
        all_segment_paths.append(video_path)
        print(f"  [{segment_id}] Added video: {os.path.basename(video_path)}")

    print(f"\nTotal segments to stitch: {len(all_segment_paths)}")
    if not all_segment_paths:
        raise RuntimeError("No Sora video segments were generated.")
    elif len(all_segment_paths) > 1:
        print(f"Stitching {len(all_segment_paths)} video segments together...")
        
        # Create final stitched video path
        base, ext = os.path.splitext(output_video_path)
        stitched_video_path = f"{base}_stitched{ext}"
        
        try:
            video_path = stitch_videos(
                video_paths=all_segment_paths,
                output_path=stitched_video_path,
                upscale_to_1080p=upscale_to_1080p,
                narration_offset=narration_offset
            )
        except Exception as stitch_error:
            raise RuntimeError(f"Video stitching failed: {stitch_error}")
        
        # Verify stitched video exists and has content
        if not os.path.exists(video_path):
            raise RuntimeError(f"Stitched video was not created: {video_path}")
        
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            raise RuntimeError(f"Stitched video is empty: {video_path}")
        
        # Verify stitched video duration matches input duration
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path:
            stitched_duration = get_media_duration(video_path, ffmpeg_path)
            if stitched_duration:
                duration_diff = abs(stitched_duration - duration)
                if duration_diff > 0.5:  # More than 0.5s difference
                    print(f"⚠️  Warning: Stitched video duration ({stitched_duration:.1f}s) doesn't match input duration ({duration:.1f}s)")
                    print(f"   Difference: {duration_diff:.1f}s")
                else:
                    print(f"✅ Stitched video duration matches input: {stitched_duration:.1f}s (target: {duration:.1f}s)")
    elif len(all_segment_paths) == 1:
        # Only one segment, no stitching needed
        video_path = all_segment_paths[0]
        # Ensure audio is added even for single segment
        try:
            ffmpeg_path = find_ffmpeg()
            if ffmpeg_path:
                narration_delay_ms = int(narration_offset * 1000) if narration_offset > 0 else 1000
                ensure_audio_on_video(video_path, ffmpeg_path=ffmpeg_path, audio_delay_ms=narration_delay_ms)
        except Exception as audio_error:
            print(f"⚠️  Warning: Could not automatically add audio to video: {audio_error}")
    else:
        raise RuntimeError("No segments available for stitching")
    
    return video_path


def generate_video_segments(
    segment_id_to_prompt,
    segment_assignments,
    num_segments,
    num_videos,
    output_folder,
    output_video_path,
    generated_segment_texts,
    generated_script,
    api_key,
    model,
    resolution,
    poll_interval,
    max_wait_time,
    segments_to_regenerate=None
):
    """
    Generate video segments with Sora. Can generate all segments or regenerate specific ones.
    Every segment is Sora-only; no image/panning fallback path.
    
    Args:
        segment_id_to_prompt: Mapping from segment_id to video prompt
        segment_assignments: List of segment assignment dicts
        num_segments: Total number of segments
        num_videos: Number of video segments
        output_folder: Folder where videos will be saved
        output_video_path: Base output video path
        generated_segment_texts: List of segment text strings
        generated_script: Full script text
        api_key: OpenAI API key
        model: Sora model to use
        resolution: Video resolution
        poll_interval: Polling interval for status checks
        max_wait_time: Maximum wait time for generation
        segments_to_regenerate: Optional list of segment IDs to regenerate. If None, generates all.
        
    Returns:
        List of dicts with segment_id, prompt, video_path
    """
    # All segments are always video segments in Sora-only mode.
    video_segment_ids = list(range(1, num_segments + 1))

    # If segments_to_regenerate is specified, filter to only those segments.
    if segments_to_regenerate is not None:
        video_segment_ids = [seg_id for seg_id in video_segment_ids if seg_id in segments_to_regenerate]
        print(f"Regenerating {len(video_segment_ids)} specific video segment(s): {video_segment_ids}")
    else:
        print(f"Generating {len(video_segment_ids)} video segment(s)")

    # Rate limiting: 4 requests per minute = 15 seconds between requests
    rate_limit_delay = 15  # seconds

    generated_video_segments = []
    video_jobs = []
    # Step 2a: Start all video generation jobs 15 seconds apart (non-blocking)
    print("Starting video generation jobs...")

    for video_idx, segment_id in enumerate(video_segment_ids, 1):
        if segment_id_to_prompt and segment_id in segment_id_to_prompt:
            segment_prompt = segment_id_to_prompt[segment_id]
        else:
            segment_prompt = "A cinematic scene"

        if not segment_prompt or len(segment_prompt.strip()) == 0:
            segment_prompt = "A cinematic scene"

        print(f"Segment {segment_id}/{num_segments} (video {video_idx}/{len(video_segment_ids)}): Text-to-video")
        print(f"  Prompt: {segment_prompt[:100]}...")

        base, ext = os.path.splitext(output_video_path)
        segment_output_path = f"{base}_segment_{segment_id:03d}{ext}"

        video_id = start_video_generation_job(
            prompt=segment_prompt,
            api_key=api_key,
            model=model,
            resolution=resolution,
            duration=FIXED_SEGMENT_DURATION_INT,
        )

        video_jobs.append({
            'segment_id': segment_id,
            'video_id': video_id,
            'output_path': segment_output_path,
            'prompt': segment_prompt,
        })

        if video_idx < len(video_segment_ids):
            time.sleep(rate_limit_delay)

    # Step 2b: Wait for all video generation jobs to complete with retry logic.
    print(f"Waiting for {len(video_jobs)} video generation job(s) to complete...")

    for job in video_jobs:
        segment_id = job['segment_id']
        video_id = job['video_id']
        segment_output_path = job['output_path']
        segment_prompt = job['prompt']

        print(f"\n--- Processing Segment {segment_id} (Job {video_id}) ---")
        max_retries = 3
        segment_video_path = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1:
                    segment_video_path = wait_for_video_completion(
                        video_id=video_id,
                        output_path=segment_output_path,
                        api_key=api_key,
                        poll_interval=poll_interval,
                        max_wait_time=max_wait_time
                    )
                else:
                    print(f"   Retry attempt {attempt}/{max_retries}: Starting new video generation job...")
                    retry_video_id = start_video_generation_job(
                        prompt=segment_prompt,
                        api_key=api_key,
                        model=model,
                        resolution=resolution,
                        duration=FIXED_SEGMENT_DURATION_INT,
                    )
                    print(f"   New job ID: {retry_video_id}")
                    base, ext = os.path.splitext(segment_output_path)
                    retry_output_path = f"{base}_retry{attempt}{ext}"
                    segment_video_path = wait_for_video_completion(
                        video_id=retry_video_id,
                        output_path=retry_output_path,
                        api_key=api_key,
                        poll_interval=poll_interval,
                        max_wait_time=max_wait_time
                    )

                generated_video_segments.append({
                    'segment_id': segment_id,
                    'prompt': segment_prompt,
                    'video_path': segment_video_path
                })
                print(f"✅ Segment {segment_id} completed: {segment_video_path}")
                break

            except Exception as e:
                last_error = e
                error_msg = str(e)
                error_type = type(e).__name__
                if attempt < max_retries:
                    print(f"   ⚠️  Attempt {attempt} failed ({error_type}): {error_msg[:200]}")
                    print("   Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"   ❌ All {max_retries} attempts failed for segment {segment_id}")

        if segment_video_path is None:
            raise RuntimeError(
                f"Sora generation failed for segment {segment_id} after {max_retries} attempts: {last_error}"
            )

    return generated_video_segments


def generate_and_upload_sora(
    prompt,
    title,
    description='',
    tags=None,
    category_id='27',
    privacy_status='private',
    thumbnail_file=None,
    playlist_id=None,
    output_video_path=None,
    api_key=None,
    model='sora-2',
    resolution='1280x720',
    duration=12,
    aspect_ratio='16:9',
    poll_interval=10,
    max_wait_time=600,
    keep_video=False,
    upscale_to_1080p=True,
    skip_narration=False,
    skip_upload=False,
):
    """
    Generate a video from a text prompt using Sora and upload it to YouTube.
    
    Args:
        prompt: Text prompt for video generation
        title: YouTube video title
        description: YouTube video description
        tags: List of tags for YouTube
        category_id: YouTube category ID
        privacy_status: Privacy status ('private', 'public', 'unlisted')
        thumbnail_file: Optional thumbnail image path
        playlist_id: Optional YouTube playlist ID
        output_video_path: Path to save generated video (default: temp file)
        api_key: OpenAI API key
        model: Sora model to use ('sora-2' or 'sora-2-pro')
        resolution: Video resolution
        duration: Video duration in seconds
        aspect_ratio: Aspect ratio
        poll_interval: Polling interval for status checks
        max_wait_time: Maximum wait time for generation
        keep_video: If True, keep the generated video file after upload
        upscale_to_1080p: If True, upscale video from 720p to 1080p (default: True)
        num_videos: Number of video segments to generate and stitch together (default: 1)
        
    Returns:
        YouTube video ID
    """
    import tempfile
    import shutil

    if model not in ("sora-2", "sora-2-pro"):
        print(f"⚠️  Unsupported model '{model}', defaulting to sora-2")
        model = "sora-2"
    print(f"🎬 Video model: {model}")
    
    # Create output folder for reference image and final video
    current_dir = os.getcwd()
    output_folder = os.path.join(current_dir, "video_output")
    
    # Archive workflow files before cleanup (save previous run's files)
    if os.path.exists(output_folder):
        print("Archiving previous workflow files...")
        archive_workflow_files()
    
    # Cleanup: Delete output folder from last run if it exists
    if os.path.exists(output_folder):
        try:
            shutil.rmtree(output_folder)
        except Exception as e:
            print(f"⚠️  Warning: Could not delete previous output folder: {e}")
    
    # Create fresh output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Determine output path
    if output_video_path is None:
        timestamp = int(time.time())
        output_video_path = os.path.join(output_folder, f"sora_video_{timestamp}.mp4")
    else:
        # If output_video_path is provided, use the output folder as the directory
        output_video_path = os.path.join(output_folder, os.path.basename(output_video_path))
        keep_video = False  # Auto-delete temp files
    
    print(f"Prompt: {prompt}")
    print(f"Model: {model}")
    
    # Simple duration validation: must be divisible by 12
    if duration % FIXED_SEGMENT_DURATION_INT != 0:
        raise ValueError(
            f"Duration must be divisible by {FIXED_SEGMENT_DURATION_INT}. "
            f"Provided duration: {duration} seconds"
        )
    
    # Fixed parameters
    SEGMENT_DURATION = FIXED_SEGMENT_DURATION_SECONDS  # Each segment is 12 seconds (video)
    
    # Calculate total number of segments (each segment is 12 seconds)
    num_segments = int(duration / FIXED_SEGMENT_DURATION_INT)
    
    num_still_images = 0
    num_videos = num_segments
    
    segment_duration = SEGMENT_DURATION
    
    print(f"Duration: {duration}s | Total segments: {num_segments} | Video segments: {num_videos}")
    
    # Step 0: Generate overarching script from video prompt (AI call)
    generated_script = None
    generated_segment_texts = []
    generated_video_prompts = []
    segment_id_to_prompt = {}  # Mapping from segment_id to prompt (only for video segments)
    narration_audio_path = None  # Will be set in Step 0.1 (narration generation - MUST happen before video generation)
    original_voiceover_backup = None  # Will be set in Step 0.1 (narration generation)
    narration_offset = 0.0  # Seconds before narration starts (centers narration in total video)
    generated_video_segments = []  # Will be populated in Step 1
    still_image_videos = {}  # Deprecated placeholder for metadata compatibility
    segment_assignments = []  # All segments are videos in Sora-only mode
    
    print("Step 0: Loading or generating overarching script...")
    # CRITICAL: If any API call fails before video generation, exit the program
    try:
        # Try to load script from file first
        generated_script = load_script_from_file()
        
        if generated_script:
            print(f"Loaded script from file: {SCRIPT_FILE_PATH}")
        else:
            generated_script = generate_script_from_prompt(
                video_prompt=prompt,
                duration=duration,
                api_key=api_key,
                model='gpt-5-2025-08-07'
            )
            # Clean dashes from script
            generated_script = clean_script_dashes(generated_script)
            # Save cleaned script to file
            try:
                with open(SCRIPT_FILE_PATH, 'w', encoding='utf-8') as f:
                    f.write(generated_script)
                print(f"✅ Script saved: {SCRIPT_FILE_PATH}")
            except Exception as e:
                print(f"⚠️  Failed to save script to file: {e}")
            print("Generated script:")
        
        print(f"Script ({len(generated_script)} chars)")
        
        # CRITICAL: Step 0.1 - Generate and save narration FIRST (before any video generation)
        # Music is generated after narration, using the measured final narration duration.
        print("Step 0.1: Generating narration...")
        current_dir = os.getcwd()
        music_file_path = os.path.join(current_dir, "VIDEO_MUSIC.mp3")
        music_target_duration = None
        
        narration_audio_path = None
        original_voiceover_backup = None
        
        # Try to load narration from file first
        narration_file = load_narration_from_file()
        
        if narration_file and os.path.exists(narration_file):
            narration_audio_path = narration_file
            print(f"✅ Loaded existing narration from file: {narration_file}")
            # Try to find the original voiceover backup (without music) if it exists
            backup_path = narration_file.replace('.mp3', '_original.mp3')
            if os.path.exists(backup_path):
                original_voiceover_backup = backup_path
                print(f"✅ Found original voiceover backup: {backup_path}")
            
        elif not skip_narration:
            # Generate narration now with iterative duration adjustment
            # This will generate TTS, check duration vs target, adjust script if needed, and repeat
            try:
                current_dir = os.getcwd()
                narration_audio_path = os.path.join(current_dir, NARRATION_AUDIO_PATH)
                
                print(f"🎙️  Generating narration with duration targeting ({duration}s target)...")
                narration_audio_path, original_voiceover_backup, generated_script = generate_narration_with_duration_loop(
                    script=generated_script,
                    target_duration_seconds=duration,
                    output_path=narration_audio_path,
                    video_prompt=prompt,
                    api_key=api_key,
                    max_attempts=3,
                    tolerance_seconds=60,
                    music_volume=0.08
                )
                
                print(f"✅ Narration generated and saved: {narration_audio_path}")
                if original_voiceover_backup:
                    print(f"✅ Original voiceover backup saved: {original_voiceover_backup}")
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Narration generation failed: {e}")
                print("   Cannot proceed without narration. Exiting...")
                import sys
                sys.exit(1)
        else:
            # skip_narration is True - but we still need narration for segmentation
            # Generate it temporarily with duration adjustment
            try:
                import tempfile
                temp_dir = tempfile.gettempdir()
                timestamp = int(time.time())
                narration_audio_path = os.path.join(temp_dir, f"voiceover_segmentation_{timestamp}.mp3")
                
                print(f"⚠️  skip_narration=True, generating temporary narration for segmentation only...")
                narration_audio_path, _, generated_script = generate_narration_with_duration_loop(
                    script=generated_script,
                    target_duration_seconds=duration,
                    output_path=narration_audio_path,
                    video_prompt=prompt,
                    api_key=api_key,
                    max_attempts=3,
                    tolerance_seconds=60,
                    music_volume=0.08
                )
                print(f"✅ Generated temporary narration for segmentation: {narration_audio_path}")
                print(f"   Note: Narration will be regenerated in Step 3")
            except Exception as e:
                print(f"⚠️  Failed to generate temporary narration for segmentation: {e}")
                print("   Falling back to rule-based segmentation...")
                narration_audio_path = None
        
        # CRITICAL: Recalculate duration and segment counts based on ACTUAL narration length
        # The video duration should match the narration, not the originally requested duration
        if narration_audio_path and os.path.exists(narration_audio_path):
            actual_narration_duration = None
            
            # Try pydub first
            try:
                from pydub import AudioSegment as _AudioSeg
                _audio = _AudioSeg.from_file(narration_audio_path)
                actual_narration_duration = len(_audio) / 1000.0
            except Exception:
                pass
            
            # Fallback to ffprobe
            if actual_narration_duration is None:
                actual_narration_duration = get_media_duration(narration_audio_path)
            
            if actual_narration_duration and actual_narration_duration > 0:
                music_target_duration = float(actual_narration_duration)
                # Round UP to the nearest multiple of 12 so all narration is covered
                import math
                adjusted_duration = int(
                    math.ceil(actual_narration_duration / FIXED_SEGMENT_DURATION_SECONDS)
                    * FIXED_SEGMENT_DURATION_INT
                )
                
                if adjusted_duration != duration:
                    print(f"\n🔄 Adjusting video duration to match narration:")
                    print(f"   Original requested duration: {duration}s")
                    print(f"   Actual narration duration:   {actual_narration_duration:.1f}s")
                    print(
                        f"   Adjusted video duration:     {adjusted_duration}s "
                        f"(rounded up to nearest {FIXED_SEGMENT_DURATION_INT}s)"
                    )
                    
                    duration = adjusted_duration
                    
                    # Recalculate segment counts
                    num_segments = int(duration / FIXED_SEGMENT_DURATION_INT)
                    num_still_images = 0
                    num_videos = num_segments
                    print(f"   New segments: {num_segments} total | {num_videos} video")
                else:
                    print(f"✅ Narration duration ({actual_narration_duration:.1f}s) matches video duration ({duration}s)")
                
                # CENTER narration within the total video duration
                # e.g. 22s narration in 30s video → 4s opening, 22s narration, 4s closing
                narration_offset = (duration - actual_narration_duration) / 2.0
                narration_offset = max(0.0, narration_offset)  # Safety: never negative
                print(f"\n🎬 Narration centering:")
                print(f"   Narration offset:  {narration_offset:.1f}s (narration starts at {narration_offset:.1f}s)")
                print(f"   Opening shot:      0.0s - {narration_offset:.1f}s ({narration_offset:.1f}s silent)")
                print(f"   Narration:         {narration_offset:.1f}s - {narration_offset + actual_narration_duration:.1f}s ({actual_narration_duration:.1f}s)")
                print(f"   Closing shot:      {narration_offset + actual_narration_duration:.1f}s - {duration:.1f}s ({narration_offset:.1f}s silent)")
            else:
                narration_offset = 0.0
                print(f"⚠️  Could not measure narration duration. Using original duration: {duration}s")
        else:
            narration_offset = 0.0

        if not music_target_duration or music_target_duration <= 0:
            music_target_duration = float(duration)
            print(
                f"\n⚠️  Falling back to requested video duration for music generation "
                f"because final narration duration was unavailable ({music_target_duration:.1f}s)."
            )

        print(
            f"\n🎵 Generating narration-step music to match final narration duration "
            f"({music_target_duration:.1f}s)..."
        )
        music_generated_this_run = False
        try:
            generate_music_for_narration_step(
                video_prompt=prompt,
                script=generated_script,
                target_duration_seconds=music_target_duration,
                api_key=api_key,
                output_path=music_file_path
            )
            music_generated_this_run = True
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to generate exact-length music: {e}")
            print("   Cannot proceed without exact-duration background music. Exiting...")
            import sys
            sys.exit(1)

        # Ensure narration audio actually includes the newly generated music track.
        if (
            music_generated_this_run
            and narration_audio_path
            and os.path.exists(narration_audio_path)
            and os.path.exists(music_file_path)
        ):
            remix_source = (
                original_voiceover_backup
                if original_voiceover_backup and os.path.exists(original_voiceover_backup)
                else narration_audio_path
            )
            try:
                narration_audio_path = mix_voiceover_with_background_music(
                    voiceover_audio_path=remix_source,
                    music_audio_path=music_file_path,
                    output_path=narration_audio_path,
                    music_volume=0.08
                )
                print(f"✅ Narration audio remixed with generated music: {narration_audio_path}")
            except Exception as remix_error:
                print(f"⚠️  Failed to remix narration with music: {remix_error}")
                print("   Continuing with existing narration audio.")
        
        # Step 0.5: Segment script based on narration timing (or fallback to rule-based)
        print(f"\nStep 0.5: Segmenting script into {num_segments} segments...")
        
        if narration_audio_path and os.path.exists(narration_audio_path):
            print(f"   Using narration-based segmentation ({FIXED_SEGMENT_DURATION_INT}-second segments)...")
            # CRITICAL: This extracts segments from the ACTUAL NARRATION AUDIO using Whisper timestamps
            # The segments contain words that were actually spoken, not the original script text
            # These narration-based segments are used for segment-level Sora video prompts.
            # IMPORTANT: narration_audio_path is the STRETCHED/SHRUNK version (if time-stretching was applied)
            # This ensures word-level timestamps match the final audio duration from video_config.json
            print(f"   📍 Using audio file: {narration_audio_path} (this is the stretched/shrunk version if time-stretching was applied)")
            generated_segment_texts = segment_script_by_narration(
                script=generated_script,
                audio_path=narration_audio_path,  # This is the stretched version that matches config duration
                segment_duration=segment_duration,
                api_key=api_key,
                expected_num_segments=num_segments,  # Pass expected number to limit segments
                narration_offset=narration_offset  # Center narration within total video duration
            )
            # Ensure we have the correct number of segments
            if len(generated_segment_texts) != num_segments:
                print(f"⚠️  Narration-based segmentation produced {len(generated_segment_texts)} segments, expected {num_segments}")
                print("   Adjusting to match expected segment count...")
                if len(generated_segment_texts) < num_segments:
                    # Pad with empty segments or repeat last segment
                    while len(generated_segment_texts) < num_segments:
                        generated_segment_texts.append(generated_segment_texts[-1] if generated_segment_texts else "")
                else:
                    # Truncate to expected number
                    generated_segment_texts = generated_segment_texts[:num_segments]
        else:
            print("   Using rule-based segmentation (word count)...")
            generated_segment_texts = segment_script_rule_based(
                script=generated_script,
                num_segments=num_segments
            )
        
        # Step 0.52: Generate tags from script and combine with user-provided tags
        # Initialize tags if not provided
        if tags is None:
            tags = []
        
        # Convert to list if it's not already
        if not isinstance(tags, list):
            tags = [tags] if tags else []
        
        # Generate tags if not provided or if we have a script/prompt to generate from
        if not tags or generated_script or prompt:
            print("Step 0.52: Generating YouTube tags...")
            try:
                # Use script if available, otherwise use prompt
                script_for_tags = generated_script if generated_script else (prompt if prompt else "")
                
                if script_for_tags:
                    generated_tags = generate_tags_from_script(
                        script=script_for_tags,
                        video_prompt=prompt,
                        api_key=api_key,
                        model='gpt-4o'
                    )
                    
                    # Combine user-provided tags with generated tags
                    user_tags = tags if tags else []
                    
                    # Combine and remove duplicates (case-insensitive)
                    all_tags = user_tags + generated_tags
                    # Remove duplicates while preserving order (case-insensitive)
                    seen = set()
                    unique_tags = []
                    for tag in all_tags:
                        tag_lower = tag.lower()
                        if tag_lower not in seen:
                            seen.add(tag_lower)
                            unique_tags.append(tag)
                    
                    tags = unique_tags
                    
                    if user_tags:
                        print(f"✅ Combined {len(user_tags)} user-provided tag(s) with {len(generated_tags)} generated tag(s):")
                        print(f"   User tags: {', '.join(user_tags)}")
                        print(f"   Generated tags: {', '.join(generated_tags)}")
                        print(f"   Total unique tags: {len(tags)}")
                    else:
                        print(f"✅ Generated {len(generated_tags)} tags:")
                        print(f"   {', '.join(generated_tags)}")
                    print(f"   Final tags: {', '.join(tags)}")
                else:
                    print("⚠️  No script or prompt available for tag generation")
                    if not tags:
                        print("   Continuing without tags...")
            except Exception as e:
                print(f"⚠️  Failed to generate tags: {e}")
                import traceback
                traceback.print_exc()
                if not tags:
                    print("   Continuing without tags...")
                else:
                    print(f"   Using only user-provided tags: {', '.join(tags)}")
        
        # Step 0.55: SKIPPED — person-specific reference-image analysis was removed.
        print("\n" + "="*60)
        print("Step 0.55: Skipped (reference image analysis not used — text-to-video only)")
        
        # Step 0.57: Generate visual continuity description for object/event-centric consistency.
        print("\n" + "="*60)
        print("Step 0.57: Generating visual continuity description...")
        print("="*60 + "\n")
        
        visual_continuity_description = ""
        try:
            visual_continuity_description = generate_visual_continuity_description(
                script=generated_script,
                video_prompt=prompt,
                api_key=api_key,
                model='gpt-5-2025-08-07'
            )
            if visual_continuity_description:
                print(f"✅ Visual continuity description generated ({len(visual_continuity_description)} chars)")
                print(f"   This description will be prepended to EVERY video prompt for consistency.")
                print(f"   Remaining budget per prompt: {4000 - len(visual_continuity_description) - 1} chars for scene content.")
                print(f"\n   Description preview:")
                # Print first 200 chars for preview
                preview = visual_continuity_description[:200]
                print(f"   \"{preview}...\"")
            else:
                print(f"⚠️  No visual continuity description generated.")
                print(f"   Video prompts will use the full 4000-char budget for scene content.")
        except Exception as e:
            print(f"⚠️  Visual continuity description generation failed: {e}")
            visual_continuity_description = ""
        
        # Sora-only mode: every segment is generated as video.
        segment_assignments = build_all_video_segment_assignments(len(generated_segment_texts))

        # Step 0.6: Convert each segment text to video generation prompt (AI call per segment)
        print("\n" + "="*60)
        print(f"Step 0.6: Converting segment texts to video generation prompts...")
        print("="*60 + "\n")

        print(f"   Generating prompts for {len(generated_segment_texts)} video segment(s)")

        generated_video_prompts = generate_video_prompts_from_segments(
            segment_texts=generated_segment_texts,
            segment_duration=segment_duration,
            total_duration=duration,
            overarching_script=generated_script,
            api_key=api_key,
            model='gpt-5-2025-08-07',
            visual_continuity_description=visual_continuity_description,
            narration_offset=narration_offset
        )

        # Segment ID maps 1:1 with prompt index in Sora-only mode.
        segment_id_to_prompt = {idx + 1: p for idx, p in enumerate(generated_video_prompts)}

        print(f"\nVideo Prompts ({len(generated_video_prompts)} segments):")
        print("-" * 60)
        for segment_id, prompt in sorted(segment_id_to_prompt.items()):
            print(f"\nSegment {segment_id}: {prompt[:100]}...")
        print("-" * 60)
        
        # Save video generation prompts and visual continuity description to JSON
        prompts_json_path = os.path.join(output_folder, "video_generation_prompts.json")
        prompts_output = {
            "visual_continuity_description": visual_continuity_description if visual_continuity_description else None,
            "visual_continuity_description_char_count": len(visual_continuity_description) if visual_continuity_description else 0,
            "narration_offset_seconds": narration_offset,
            "total_video_duration_seconds": duration,
            "segment_duration_seconds": segment_duration,
            "total_segments": len(generated_segment_texts),
            "video_segments": len(generated_video_prompts),
            "prompts": []
        }
        for segment_id, vid_prompt in sorted(segment_id_to_prompt.items()):
            prompts_output["prompts"].append({
                "video_segment_index": segment_id,
                "segment_id": segment_id,
                "prompt": vid_prompt,
                "prompt_char_count": len(vid_prompt),
                "segment_narration_text": generated_segment_texts[segment_id - 1] if segment_id <= len(generated_segment_texts) else None
            })
        
        try:
            with open(prompts_json_path, 'w', encoding='utf-8') as f:
                json.dump(prompts_output, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Video generation prompts saved to: {prompts_json_path}")
        except Exception as json_err:
            print(f"⚠️  Failed to save prompts JSON: {json_err}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: API call failed before video generation: {e}")
        print("   Exiting program as requested. All API calls must succeed before video generation.")
        import sys
        sys.exit(1)
    
    # Step 0.7: Use the narration that was already generated in Step 0.1
    # CRITICAL: Narration was already generated and saved in Step 0.1 (before video generation)
    # At this point, narration_audio_path should already exist from Step 0.1
    voiceover_audio_path = None
    original_voiceover_backup = None
    
    if not skip_narration:
        # Narration should already exist from Step 0.1
        if narration_audio_path and os.path.exists(narration_audio_path):
            voiceover_audio_path = narration_audio_path
            # Try to find the original voiceover backup (without music) if it exists
            backup_path = narration_audio_path.replace('.mp3', '_original.mp3')
            if os.path.exists(backup_path):
                original_voiceover_backup = backup_path
            print(f"✅ Using narration from Step 0.1: {voiceover_audio_path}")
            if original_voiceover_backup:
                print(f"✅ Original voiceover backup available: {original_voiceover_backup}")
        else:
            # Fallback: try to load from file (shouldn't happen if Step 0.1 worked)
            print("⚠️  Narration not found from Step 0.1, attempting to load from file...")
            narration_file = load_narration_from_file()
            if narration_file and os.path.exists(narration_file):
                voiceover_audio_path = narration_file
                backup_path = narration_file.replace('.mp3', '_original.mp3')
                if os.path.exists(backup_path):
                    original_voiceover_backup = backup_path
                print(f"✅ Loaded narration from file: {voiceover_audio_path}")
            else:
                print(f"❌ CRITICAL ERROR: Narration should have been generated in Step 0.1")
                print(f"   Cannot proceed without narration. Exiting...")
                import sys
                sys.exit(1)
    else:
        print("⏭️  Step 0.7: Skipping narration (skip_narration=True)")
        # Note: If skip_narration is True, narration was generated temporarily in Step 0.1 for segmentation only
        # It will be regenerated in Step 3 if needed
    
    # ============================================================================
    # VIDEO GENERATION WORKFLOW BEGINS HERE
    # ============================================================================
    # CRITICAL ASSUMPTION: Narration has already been generated and saved in Step 0.1
    # At this point, narration_audio_path should exist and be ready for use
    # All video generation steps
    # assume narration already exists
    # ============================================================================
    
    # Step 1: SKIPPED — Character reference image generation is no longer used.
    # Text-to-video generation uses prompt continuity for consistency.
    # visual_continuity_description that is prepended to every video prompt.
    print("\n" + "="*60)
    print("Step 1: Skipped (reference image generation not used — text-to-video only)")
    print("="*60 + "\n")
    print("📌 NOTE: Narration was already generated in Step 0.1 (before video generation)")
    if narration_audio_path and os.path.exists(narration_audio_path):
        print(f"   ✅ Narration ready: {os.path.basename(narration_audio_path)}")
    else:
        print(f"   ⚠️  WARNING: Narration path not found, but continuing with video generation...")
    
    still_image_videos = {}

    # Step 2: Generate multiple videos with rate limiting
    print(f"Step 2: Generating {num_videos} video segment(s) using Sora...")
    
    # Each segment is 12 seconds
    video_segment_duration = FIXED_SEGMENT_DURATION_INT
    
    # Use generated video prompts if available, otherwise use original prompt
    # CRITICAL: segment_id_to_prompt maps actual segment IDs to prompts (only for video segments)
    # If available, use it; otherwise fall back to generated_video_prompts or original prompt
    if segment_id_to_prompt:
        # segment_id_to_prompt will be used directly in the video generation loop
        video_prompts_to_use = None  # Will use segment_id_to_prompt mapping instead
        print(f"   Using segment_id_to_prompt mapping with {len(segment_id_to_prompt)} video segment(s)")
    elif generated_video_prompts:
        # Fallback: use generated_video_prompts (should only contain video segments now)
        video_prompts_to_use = generated_video_prompts
        print(f"   Using generated_video_prompts list with {len(generated_video_prompts)} prompt(s)")
    else:
        video_prompts_to_use = [prompt] * num_videos
        print(f"   Using original prompt for all {num_videos} video segment(s)")
    
    # Ensure we have the right number of prompts (only if using list, not mapping)
    if video_prompts_to_use is not None:
        if len(video_prompts_to_use) < num_videos:
            # Pad with the last prompt or original prompt
            while len(video_prompts_to_use) < num_videos:
                video_prompts_to_use.append(video_prompts_to_use[-1] if video_prompts_to_use else prompt)
        elif len(video_prompts_to_use) > num_videos:
            # Take only the first num_videos
            video_prompts_to_use = video_prompts_to_use[:num_videos]
    
    # Generate Video Segments
    print("\nGenerating video segments...")
    
    # Ask if user wants to generate segments
    run_step1 = False
    if not skip_narration:  # Only ask if not in non-interactive mode
        try:
            step1_input = input("Run Step 1? (y/n, default: n): ").strip().lower()
            run_step1 = step1_input in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            run_step1 = False
    
    if run_step1:
        # Check if metadata exists (from previous run)
        metadata = load_segment_metadata(output_folder)
        
        # Generate all video segments
        print("Generating all video segments...")
        generated_video_segments = generate_video_segments(
            segment_id_to_prompt=segment_id_to_prompt,
            segment_assignments=segment_assignments,
            num_segments=num_segments,
            num_videos=num_videos,
            output_folder=output_folder,
            output_video_path=output_video_path,
            generated_segment_texts=generated_segment_texts,
            generated_script=generated_script,
            api_key=api_key,
            model=model,
            resolution=resolution,
            poll_interval=poll_interval,
            max_wait_time=max_wait_time,
            segments_to_regenerate=None,  # Generate all
        )
        
        # Save metadata after generation
        save_segment_metadata(
            output_folder=output_folder,
            segment_id_to_prompt=segment_id_to_prompt,
            generated_video_segments=generated_video_segments,
            still_image_videos=still_image_videos,
            segment_assignments=segment_assignments,
            generated_segment_texts=generated_segment_texts,
            generated_script=generated_script,
            num_segments=num_segments,
            num_videos=num_videos,
            num_still_images=num_still_images,
            output_video_path=output_video_path,
            narration_offset=narration_offset
        )
        print("✅ Step 1 complete: Video segments generated")
        return None  # Stop execution after Step 1
    else:
        print("⏭️  Skipping segment generation")
        # Try to load existing segments from metadata (from a previous run)
        # If metadata doesn't exist, proceed with generating videos using data from Step 0
        metadata = load_segment_metadata(output_folder)
        if metadata:
            print("   Loading existing video segments from previous run...")
            generated_video_segments = metadata.get('generated_video_segments', [])
            # Filter out segments that don't exist on disk
            existing_segments = [seg for seg in generated_video_segments if os.path.exists(seg.get('video_path', ''))]
            missing_segments = [seg for seg in generated_video_segments if not os.path.exists(seg.get('video_path', ''))]
            
            if missing_segments:
                print(f"   [WARNING] {len(missing_segments)} segment(s) from metadata no longer exist on disk:")
                for seg in missing_segments:
                    print(f"      - Segment {seg.get('segment_id', '?')}: {seg.get('video_path', 'unknown')}")
            
            generated_video_segments = existing_segments
            still_image_videos = {}
            
            # Use metadata values if available, otherwise keep values from Step 0
            segment_assignments = metadata.get('segment_assignments', segment_assignments)
            generated_segment_texts = metadata.get('generated_segment_texts', generated_segment_texts)
            generated_script = metadata.get('generated_script', generated_script)
            num_segments = metadata.get('num_segments', num_segments)
            num_videos = num_segments
            num_still_images = 0
            segment_id_to_prompt = metadata.get('segment_id_to_prompt', segment_id_to_prompt)
            
            print(f"   [OK] Loaded {len(generated_video_segments)} video segment(s) from metadata")
            print(f"   Proceeding to next step with existing segments...")
        else:
            # No metadata exists - this is fine! Proceed with generating videos using data from Step 0
            print("   [INFO] No metadata found from previous run.")
            print("   [INFO] Metadata is optional and is only used to resume from previous runs.")
            print("   [INFO] Proceeding to generate videos using data from Step 0...")
            # Variables are already initialized at the start of the function
            # Proceed to generate videos
            print("Generating all video segments...")
            generated_video_segments = generate_video_segments(
                segment_id_to_prompt=segment_id_to_prompt,
                segment_assignments=segment_assignments,
                num_segments=num_segments,
                num_videos=num_videos,
                output_folder=output_folder,
                output_video_path=output_video_path,
                generated_segment_texts=generated_segment_texts,
                generated_script=generated_script,
                api_key=api_key,
                model=model,
                resolution=resolution,
                poll_interval=poll_interval,
                max_wait_time=max_wait_time,
                segments_to_regenerate=None,  # Generate all
            )
            
            # Save metadata AFTER generating videos (for future use)
            save_segment_metadata(
                output_folder=output_folder,
                segment_id_to_prompt=segment_id_to_prompt,
                generated_video_segments=generated_video_segments,
                still_image_videos=still_image_videos,
                segment_assignments=segment_assignments,
                generated_segment_texts=generated_segment_texts,
                generated_script=generated_script,
                num_segments=num_segments,
                num_videos=num_videos,
                num_still_images=num_still_images,
                output_video_path=output_video_path,
                narration_offset=narration_offset
            )
            print("✅ Video segments generated (metadata saved for future use)")
            return None
    
    # Review, Regenerate, or Stitch Video Segments
    print("\nReviewing segments...")
    
    # Ask if user wants to review segments
    run_step2 = False
    if not skip_narration:  # Only ask if not in non-interactive mode
        try:
            step2_input = input("Run Step 2? (y/n, default: n): ").strip().lower()
            run_step2 = step2_input in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            run_step2 = False
    
    if run_step2:
        # Show available segments
        print(f"\nAvailable segments:")
        print(f"  Video segments: {len(generated_video_segments)}")
        print(f"  Total segments: {num_segments}")
        
        # Show segment details if metadata is available
        if generated_video_segments:
            print(f"\nVideo segment details:")
            for seg in sorted(generated_video_segments, key=lambda x: x.get('segment_id', 0)):
                seg_id = seg.get('segment_id', '?')
                video_path = seg.get('video_path', 'unknown')
                exists = os.path.exists(video_path) if video_path != 'unknown' else False
                status = "[EXISTS]" if exists else "[MISSING]"
                print(f"  Segment {seg_id}: {os.path.basename(video_path)} {status}")
        
        # Ask user what they want to do
        if not skip_narration:
            print(f"\nWhat would you like to do?")
            print(f"  1. Stitch all segments together")
            print(f"  2. Regenerate specific segments (then stitch)")
            print(f"  3. Skip (continue to next step)")
            
            try:
                choice = input("Enter choice (1/2/3, default: 3): ").strip()
            except (EOFError, KeyboardInterrupt):
                choice = "3"
        else:
            choice = "3"  # Default to skip in non-interactive mode
        
        if choice == "1":
            # Stitch all segments
            print("\nStitching all video segments together...")
            try:
                video_path = stitch_all_segments(
                    generated_video_segments=generated_video_segments,
                    still_image_videos=still_image_videos,
                    segment_assignments=segment_assignments,
                    num_segments=num_segments,
                    output_video_path=output_video_path,
                    duration=duration,
                    upscale_to_1080p=upscale_to_1080p,
                    narration_offset=narration_offset
                )
                print("✅ Video segments stitched")
                return None
            except Exception as e:
                print(f"❌ Error during stitching: {e}")
                import traceback
                traceback.print_exc()
                print("⚠️  Step 2 failed, but continuing...")
                return None  # Stop execution even on error
        
        elif choice == "2":
            # Regenerate specific segments
            segments_to_regenerate = None
            if not skip_narration:
                try:
                    regenerate_input = input("\nEnter segment numbers to regenerate (comma-separated, e.g., 1,3,5) or press Enter to cancel: ").strip()
                    if regenerate_input:
                        try:
                            segments_to_regenerate = [int(x.strip()) for x in regenerate_input.split(',')]
                            print(f"Will regenerate segments: {segments_to_regenerate}")
                        except ValueError:
                            print("Invalid input. Cancelling regeneration.")
                            segments_to_regenerate = None
                except (EOFError, KeyboardInterrupt):
                    segments_to_regenerate = None
            
            if segments_to_regenerate:
                # Regenerate specific segments
                print(f"\nRegenerating segments: {segments_to_regenerate}")
                regenerated_segments = generate_video_segments(
                    segment_id_to_prompt=segment_id_to_prompt,
                    segment_assignments=segment_assignments,
                    num_segments=num_segments,
                    num_videos=num_videos,
                    output_folder=output_folder,
                    output_video_path=output_video_path,
                    generated_segment_texts=generated_segment_texts,
                    generated_script=generated_script,
                    api_key=api_key,
                    model=model,
                    resolution=resolution,
                    poll_interval=poll_interval,
                    max_wait_time=max_wait_time,
                    segments_to_regenerate=segments_to_regenerate,
                )
                
                # Update generated_video_segments with regenerated ones
                # Remove old segments and add new ones
                generated_video_segments = [seg for seg in generated_video_segments if seg['segment_id'] not in segments_to_regenerate]
                generated_video_segments.extend(regenerated_segments)
                
                # Save updated metadata
                save_segment_metadata(
                    output_folder=output_folder,
                    segment_id_to_prompt=segment_id_to_prompt,
                    generated_video_segments=generated_video_segments,
                    still_image_videos=still_image_videos,
                    segment_assignments=segment_assignments,
                    generated_segment_texts=generated_segment_texts,
                    generated_script=generated_script,
                    num_segments=num_segments,
                    num_videos=num_videos,
                    num_still_images=num_still_images,
                    output_video_path=output_video_path,
                    narration_offset=narration_offset
                )
                
                # After regenerating, ask if they want to stitch
                if not skip_narration:
                    try:
                        stitch_input = input("\nSegments regenerated. Stitch all segments together now? (y/n, default: y): ").strip().lower()
                        stitch_now = stitch_input in ['y', 'yes', '']
                    except (EOFError, KeyboardInterrupt):
                        stitch_now = True
                else:
                    stitch_now = True  # Default to stitching in non-interactive mode
                
                if stitch_now:
                    print("\nStitching all video segments together...")
                    try:
                        video_path = stitch_all_segments(
                            generated_video_segments=generated_video_segments,
                            still_image_videos=still_image_videos,
                            segment_assignments=segment_assignments,
                            num_segments=num_segments,
                            output_video_path=output_video_path,
                            duration=duration,
                            upscale_to_1080p=upscale_to_1080p,
                            narration_offset=narration_offset
                        )
                        print("✅ Segments regenerated and stitched")
                        return None
                    except Exception as e:
                        print(f"❌ Error during stitching: {e}")
                        import traceback
                        traceback.print_exc()
                        print("⚠️  Stitching failed, but segments were regenerated")
                        return None
                else:
                    print("✅ Segments regenerated (stitching skipped)")
                    return None
            else:
                print("⏭️  Regeneration cancelled")
        else:
            print("⏭️  Skipping segment review")
    else:
        print("⏭️  Skipping segment review")
    
    # Stitch Video Segments (if not done earlier)
    print("\nStitching video segments...")
    
    # Ask if user wants to stitch segments
    run_step3 = False
    if not skip_narration:  # Only ask if not in non-interactive mode
        try:
            step3_input = input("Run Step 3? (y/n, default: n): ").strip().lower()
            run_step3 = step3_input in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            run_step3 = False
    
    if run_step3:
        try:
            video_path = stitch_all_segments(
                generated_video_segments=generated_video_segments,
                still_image_videos=still_image_videos,
                segment_assignments=segment_assignments,
                num_segments=num_segments,
                output_video_path=output_video_path,
                duration=duration,
                upscale_to_1080p=upscale_to_1080p,
                narration_offset=narration_offset
            )
            print("✅ Video segments stitched")
            return None
        except Exception as e:
            print(f"❌ Error during stitching: {e}")
            import traceback
            traceback.print_exc()
            video_path = None
            print("⚠️  Stitching failed, but continuing...")
            return None
    else:
        print("⏭️  Skipping stitching")
        video_path = None  # Set to None since we didn't stitch
    
    # Add Voiceover
    print("\nAdding voiceover...")
    
    # Ask if user wants to add voiceover
    run_step4 = False
    if not skip_narration:  # Only ask if not in non-interactive mode
        try:
            step4_input = input("Run Step 4? (y/n, default: n): ").strip().lower()
            run_step4 = step4_input in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            run_step4 = False
    
    if run_step4:
        # Check if voiceover audio exists
        if not voiceover_audio_path:
            # Try to find narration audio file
            current_dir = os.getcwd()
            narration_file = os.path.join(current_dir, NARRATION_AUDIO_PATH)
            if os.path.exists(narration_file):
                voiceover_audio_path = narration_file
                print(f"✅ Found narration audio: {voiceover_audio_path}")
            else:
                print("⚠️  No voiceover audio path available")
        
        # Check if video exists
        if not video_path or not os.path.exists(video_path):
            print("⚠️  No stitched video found - cannot add voiceover")
            print("⏭️  Skipping Step 4")
            add_voiceover = False
        else:
            add_voiceover = True  # User said yes, so proceed
        
        # Add audio to stitched video (mix narration + music at 8%)
        if add_voiceover and video_path and os.path.exists(video_path) and voiceover_audio_path and os.path.exists(voiceover_audio_path):
            print("Mixing narration and music, then adding to stitched video...")
            
            ffmpeg_path = find_ffmpeg()
            if not ffmpeg_path:
                print("⚠️  FFmpeg not found. Cannot add audio.")
            else:
                video_duration = get_media_duration(video_path, ffmpeg_path)
            
            if video_duration:
                import tempfile
                temp_dir = tempfile.gettempdir()
                timestamp = int(time.time())
                
                # Use narration as-is (no speed adjustment)
                print(f"   Using narration as-is (no speed adjustment)...")
                print(f"   Video duration: {video_duration:.2f}s")
                
                # Get original voiceover source (without music)
                voiceover_source = None
                if original_voiceover_backup and os.path.exists(original_voiceover_backup):
                    voiceover_source = original_voiceover_backup
                    print(f"   Using original voiceover from backup")
                else:
                    # Fallback: try to find original voiceover in temp directory
                    import glob
                    temp_dir_check = tempfile.gettempdir()
                    original_pattern = os.path.join(temp_dir_check, "original_voiceover_*.mp3")
                    original_files = glob.glob(original_pattern)
                    if original_files:
                        voiceover_source = max(original_files, key=os.path.getmtime)
                        print(f"   Using original voiceover from temp directory")
                    else:
                        # Final fallback: use mixed audio (will extract narration if possible)
                        voiceover_source = voiceover_audio_path
                        print(f"   ⚠️  Using mixed audio as source (original not found)")
                
                # Use narration as-is (no speed adjustment)
                voiceover_duration = get_media_duration(voiceover_source, ffmpeg_path)
                if voiceover_duration:
                    print(f"   Narration duration: {voiceover_duration:.2f}s (used as-is, no speed adjustment)")
                
                # Now proceed with music mixing
                # Re-mix audio with proper synchronization:
                # 1. Narration is used as-is (no speed adjustment)
                # 2. Sync music to video duration exactly
                # 3. Re-mix with voiceover at 8% music volume
                
                # Try to re-extract music from VIDEO_MUSIC.mp3 and sync it
                current_dir = os.getcwd()
                music_source = None
                for music_file in ["VIDEO_MUSIC.mp3", "video_music.mp3", "VIDEO_MUSIC.MP3"]:
                    music_path_check = os.path.join(current_dir, music_file)
                    if os.path.exists(music_path_check):
                        music_source = music_path_check
                        break
                
                if music_source:
                        voiceover_tolerance = 2.0  # Narration can start/end within 2 seconds of video bounds
                        # Sync music to video duration exactly
                        synced_music_path = os.path.join(temp_dir, f"music_synced_{timestamp}.mp3")
                        music_duration = get_media_duration(music_source, ffmpeg_path)
                        
                        if music_duration:
                            if abs(music_duration - video_duration) > 0.1:
                                # Adjust music to match video exactly
                                if music_duration > video_duration:
                                    # Trim music with fade in (1s) and fade out (1s)
                                    fade_out_start = max(0, video_duration - 1.0)
                                    cmd_music = [
                                        ffmpeg_path,
                                        "-i", music_source,
                                        "-t", str(video_duration),
                                        "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1",
                                        "-c:a", "libmp3lame",
                                        "-b:a", "192k",
                                        "-y",
                                        synced_music_path
                                    ]
                                else:
                                    # Loop music to extend with fade in (1s) and fade out (1s)
                                    loop_count = int((video_duration / music_duration) + 1)
                                    fade_out_start = max(0, video_duration - 1.0)
                                    cmd_music = [
                                        ffmpeg_path,
                                        "-stream_loop", str(loop_count - 1),
                                        "-i", music_source,
                                        "-t", str(video_duration),
                                        "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1",
                                        "-c:a", "libmp3lame",
                                        "-b:a", "192k",
                                        "-y",
                                        synced_music_path
                                    ]
                                
                                try:
                                    subprocess.run(cmd_music, capture_output=True, text=True, check=True)
                                    print(f"   ✅ Music synced to video duration: {video_duration:.2f}s")
                                    
                                    # Check if we're using mixed audio - if so, don't add music again!
                                    using_mixed_audio_as_source = False
                                    if voiceover_source == voiceover_audio_path:
                                        using_mixed_audio_as_source = True
                                    
                                    if using_mixed_audio_as_source:
                                        # The source already has music mixed in, so just sync it to video duration
                                        print(f"   ℹ️  Source already contains music - syncing to video without re-adding music")
                                        synced_audio_path = os.path.join(temp_dir, f"audio_synced_no_remix_{timestamp}.mp3")
                                        
                                        # Just trim/extend the mixed audio to match video duration exactly
                                        audio_duration = get_media_duration(voiceover_source, ffmpeg_path)
                                        if audio_duration and abs(audio_duration - video_duration) > 0.1:
                                            if audio_duration > video_duration:
                                                # Trim to video duration
                                                cmd_sync = [
                                                    ffmpeg_path,
                                                    "-i", voiceover_source,
                                                    "-t", str(video_duration),
                                                    "-c:a", "copy",
                                                    "-y",
                                                    synced_audio_path
                                                ]
                                            else:
                                                # Extend with silence
                                                cmd_sync = [
                                                    ffmpeg_path,
                                                    "-i", voiceover_source,
                                                    "-af", "apad",
                                                    "-t", str(video_duration),
                                                    "-c:a", "libmp3lame",
                                                    "-b:a", "192k",
                                                    "-y",
                                                    synced_audio_path
                                                ]
                                            try:
                                                subprocess.run(cmd_sync, capture_output=True, text=True, check=True)
                                                voiceover_audio_path = synced_audio_path
                                                print(f"   ✅ Mixed audio synced to video duration ({video_duration:.2f}s) without doubling music")
                                            except Exception as e:
                                                print(f"   ⚠️  Audio sync failed: {e}, using original")
                                                voiceover_audio_path = voiceover_source
                                        else:
                                            # Duration already matches, just use it
                                            voiceover_audio_path = voiceover_source
                                            print(f"   ✅ Mixed audio duration already matches video ({video_duration:.2f}s)")
                                    else:
                                        # Original voiceover found - safe to add music
                                        # Mix: voiceover + music (8% volume)
                                        synced_audio_path = os.path.join(temp_dir, f"audio_resynced_{timestamp}.mp3")
                                        
                                        # Mix music and narration together at 8% music volume
                                        filter_complex = (
                                            f"[0:a]aresample=44100,volume=1.0[voice];"
                                            f"[1:a]aresample=44100,volume={0.08}[music];"  # 8% volume for background music
                                            f"[voice][music]amix=inputs=2:duration=longest:dropout_transition=2,"
                                            f"volume=2.0"  # Boost volume by 2x after mixing
                                        )
                                        
                                        cmd_remix = [
                                            ffmpeg_path,
                                            "-i", voiceover_source,
                                            "-i", synced_music_path,
                                            "-filter_complex", filter_complex,
                                            "-t", str(video_duration),
                                            "-c:a", "libmp3lame",
                                            "-b:a", "192k",
                                            "-ar", "44100",
                                            "-ac", "2",
                                            "-y",
                                            synced_audio_path
                                        ]
                                        
                                        subprocess.run(cmd_remix, capture_output=True, text=True, check=True)
                                        voiceover_audio_path = synced_audio_path
                                        print(f"   ✅ Audio mixed: narration + music (8% volume) synced to video ({video_duration:.2f}s)")
                                    
                                except Exception as e:
                                    print(f"   ⚠️  Music re-sync failed: {e}")
                                    print(f"   Using original mixed audio")
                            else:
                                # Music duration matches, but we still need to add fade in/out
                                print(f"   Music duration matches video, applying fade in/out...")
                                fade_out_start = max(0, video_duration - 1.0)
                                faded_music_path = os.path.join(temp_dir, f"music_faded_{timestamp}.mp3")
                                cmd_fade = [
                                    ffmpeg_path,
                                    "-i", music_source,
                                    "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1",
                                    "-c:a", "libmp3lame",
                                    "-b:a", "192k",
                                    "-y",
                                    faded_music_path
                                ]
                                try:
                                    subprocess.run(cmd_fade, capture_output=True, text=True, check=True)
                                    synced_music_path = faded_music_path
                                    print(f"   ✅ Music fade in/out applied")
                                except Exception as e:
                                    print(f"   ⚠️  Music fade failed: {e}")
                                    print(f"   ⏭️  Skipping music fade - using original music without fade effects")
                                    synced_music_path = music_source
                        else:
                            print(f"   ⚠️  VIDEO_MUSIC.mp3 not found - cannot mix music separately")
                            print(f"   Using original mixed audio")
                
                # Add mixed audio to video
                try:
                        base, ext = os.path.splitext(video_path)
                        video_with_audio_path = f"{base}_with_audio{ext}"
                        
                        # Add audio to video with narration offset delay
                        # narration_offset centers the narration within the video
                        narration_delay_ms = int(narration_offset * 1000)
                        print(f"   🎬 Narration delay: {narration_delay_ms}ms ({narration_offset:.1f}s) to center narration in video")
                        video_path = add_audio_to_video(
                            video_path=video_path,
                            audio_path=voiceover_audio_path,
                            output_path=video_with_audio_path,
                            ffmpeg_path=ffmpeg_path,
                            sync_duration=False,  # Already synced above
                            audio_delay_ms=narration_delay_ms  # Center narration in video
                        )
                        print(f"✅ Video with mixed audio (narration + 8% music): {video_path}")
                        
                        # Apply ending fade: fade to black and audio fade out over last 2 seconds
                        try:
                            print("\n" + "="*60)
                            print("Applying ending fade (fade to black + audio fade out)...")
                            print("="*60 + "\n")
                            faded_video_path = apply_ending_fade(
                                video_path=video_path,
                                output_path=None,  # Will create new file
                                ffmpeg_path=ffmpeg_path,
                                fade_duration=2.0
                            )
                            # Replace video_path with faded version
                            if os.path.exists(faded_video_path):
                                # Clean up the non-faded version
                                try:
                                    if video_path != faded_video_path:
                                        os.remove(video_path)
                                except:
                                    pass
                                video_path = faded_video_path
                                print(f"✅ Ending fade applied: {video_path}")
                        except Exception as fade_error:
                            print(f"⚠️  Failed to apply ending fade: {fade_error}")
                            print(f"   Continuing with video without ending fade...")
                except Exception as e:
                    print(f"⚠️  Failed to add audio to video: {e}")
                    print(f"   Continuing with video without audio...")
        print("✅ Voiceover added")
        return None
    else:
        print("⏭️  Skipping voiceover")
    
    # Note: Upscaling now happens during stitching (in stitch_videos function)
    # No separate upscaling step needed here
    
    if video_path is None:
        # CRITICAL: Never stop execution - create emergency placeholder
        print(f"⚠️  WARNING: Video generation completed but no video path was set")
        print(f"   🔄 Creating emergency placeholder video to continue...")
        try:
            timestamp = int(time.time())
            emergency_video_path = os.path.join(output_folder, f"emergency_placeholder_final_{timestamp}.mp4")
            ffmpeg_path = find_ffmpeg()
            if ffmpeg_path:
                cmd_emergency = [
                    ffmpeg_path,
                    "-f", "lavfi",
                    "-i", f"color=c=0x1a1a2e:s=1280x720:d={duration}",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    emergency_video_path
                ]
                subprocess.run(cmd_emergency, capture_output=True, text=True, check=True)
                video_path = emergency_video_path
                print(f"   ✅ Emergency placeholder video created: {emergency_video_path}")
            else:
                raise RuntimeError("FFmpeg not available - cannot create emergency placeholder")
        except Exception as emergency_error:
            print(f"   ❌ CRITICAL: Emergency placeholder creation failed: {emergency_error}")
            raise RuntimeError("Video generation completed but no video path was set and emergency placeholder failed")
    
    # Upscaling is already handled above after stitching and audio addition
        
        # Note: Audio may have been added in Step 4 (after stitching, before upscaling)
        # No need to add audio again - it's already on the video
                
                # OBSOLETE: Step 2.7 - Add subtitles/captions to video (commented out)
                # if generated_script and video_duration:
                #     print("\n" + "="*60)
                #     print("Step 2.7: Adding subtitles/captions to video...")
                #     print("="*60 + "\n")
                #     try:
                #         # Check if video already has subtitles - if so, use the original video without subtitles
                #         # FFmpeg subtitles filter will add subtitles on top, so we need a clean video
                #         base, ext = os.path.splitext(video_path)
                #         if "_with_subtitles" in base:
                #             # Video already has subtitles - find the original video without subtitles
                #             original_base = base.rsplit("_with_subtitles", 1)[0]
                #             original_video_path = f"{original_base}{ext}"
                #             if os.path.exists(original_video_path):
                #                 print(f"   Using original video without subtitles: {os.path.basename(original_video_path)}")
                #                 video_path_for_subtitles = original_video_path
                #             else:
                #                 # Original not found, use current video (will create new one)
                #                 video_path_for_subtitles = video_path
                #         else:
                #             video_path_for_subtitles = video_path
                #         
                #         # Create output path (will be handled by add_subtitles_to_video if None)
                #         base_for_output, ext_for_output = os.path.splitext(video_path_for_subtitles)
                #         if "_with_subtitles" in base_for_output:
                #             base_for_output = base_for_output.rsplit("_with_subtitles", 1)[0]
                #         video_with_subtitles_path = f"{base_for_output}_with_subtitles{ext_for_output}"
                #         
                #         # Use narration_audio_path for accurate word-level timing
                #         # Prefer narration_audio_path over original_voiceover_backup
                #         audio_for_subtitles = narration_audio_path if narration_audio_path and os.path.exists(narration_audio_path) else (original_voiceover_backup if original_voiceover_backup and os.path.exists(original_voiceover_backup) else voiceover_audio_path)
                #         
                #         video_with_subtitles = add_subtitles_to_video(
                #             video_path=video_path_for_subtitles,
                #             script=generated_script,
                #             video_duration=video_duration,
                #             output_path=video_with_subtitles_path,
                #             ffmpeg_path=ffmpeg_path,
                #             audio_path=audio_for_subtitles,
                #             api_key=api_key
                #         )
                #         
                #         if video_with_subtitles and os.path.exists(video_with_subtitles):
                #             # Remove the video without subtitles (keep the one with subtitles)
                #             video_without_subtitles = video_path
                #             video_path = video_with_subtitles
                #             print(f"✅ Video with subtitles: {video_path}")
                #             
                #             # Remove the video without subtitles
                #             if video_without_subtitles != video_path and os.path.exists(video_without_subtitles):
                #                 try:
                #                     os.remove(video_without_subtitles)
                #                     print(f"  Removed video without subtitles: {os.path.basename(video_without_subtitles)}")
                #                 except Exception as e:
                #                     print(f"  Warning: Could not remove {video_without_subtitles}: {e}")
                #         else:
                #             print("⚠️  Subtitle generation returned no output, keeping video without subtitles")
                #     except Exception as e:
                #         print(f"⚠️  Failed to add subtitles to video: {e}")
                #         print("   Continuing with video without subtitles...")
        
        # Clean up individual segment files if upscaling was disabled or failed
        # Note: Individual segment files are kept for potential regeneration
        # They can be cleaned up manually if needed
        
        # Step 3: Generate thumbnail if not provided (YouTube auto-generates thumbnail)
        generated_thumbnail = None
        # Only set thumbnail_file to None if it wasn't already provided
        # This preserves the thumbnail_file found in the directory
        if thumbnail_file is None:
            print("\n" + "="*60)
            print("Step 3: Thumbnail Generation (SKIPPED - Using YouTube auto-generated thumbnail)")
            print("="*60 + "\n")
        else:
            print(f"Step 3: Using provided thumbnail: {thumbnail_file}")
        
        # Step 4: Upload to YouTube (SKIP if skip_upload is True)
        if not skip_upload:
            print("Step 4: Uploading to YouTube...")
            
            # Verify we're uploading the correct video (stitched, not individual segment)
            if '_stitched' not in video_path and '_1080p' not in video_path:
                # Check if video_path is actually a segment (shouldn't happen, but verify)
                if '_segment_' in video_path:
                    raise RuntimeError(
                        f"ERROR: Attempting to upload individual segment instead of stitched video!\n"
                        f"Segment path: {video_path}\n"
                        f"This should not happen. Please check the stitching logic."
                    )
            
            try:
                # Ensure tags is a list (not None) before uploading
                upload_tags = tags if tags else []
                if not isinstance(upload_tags, list):
                    upload_tags = [upload_tags] if upload_tags else []
                
                print(f"📤 Uploading with {len(upload_tags)} tag(s): {', '.join(upload_tags) if upload_tags else 'none'}")
                
                video_id = upload_video(
                    video_file=video_path,
                    title=title,
                    description=description,
                    tags=upload_tags,
                    category_id=category_id,
                    privacy_status=privacy_status,
                    thumbnail_file=thumbnail_file,
                    playlist_id=playlist_id
                )
            except Exception as e:
                print(f"❌ YouTube upload failed: {e}")
                # Clean up video file if upload fails and it's a temp file
                if not keep_video and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        print(f"Cleaned up temporary video file: {video_path}")
                    except:
                        pass
                
                # Clean up temporary image files even if upload fails
                # Note: Character reference images are kept for potential regeneration
                
                if generated_thumbnail and os.path.exists(generated_thumbnail):
                    try:
                        os.remove(generated_thumbnail)
                        print(f"Cleaned up generated thumbnail: {generated_thumbnail}")
                    except:
                        pass
                
                raise
        else:
            print("⏭️  Step 4: Skipping YouTube upload (will be done in Step 5)")
            video_id = None
        
        # Move final video to output_video_path (in output folder) if it's in a different location
        if video_path and os.path.exists(video_path) and video_path != output_video_path:
            try:
                # Ensure output directory exists (should already exist, but double-check)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder, exist_ok=True)
                
                # Move final video to output path in output folder
                if os.path.exists(output_video_path):
                    os.remove(output_video_path)  # Remove existing file if present
                # Ensure shutil is available (it's imported at top, but ensure it's accessible)
                import shutil
                shutil.move(video_path, output_video_path)
                video_path = output_video_path
                print(f"✅ Final video saved to: {output_video_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not move video to output path: {e}")
                print(f"   Video is at: {video_path}")
        
        print("\n" + "="*60)
        print("✅ Complete! Video generated successfully!")
        print("="*60)
        print(f"📹 Video file: {video_path}")
        if video_id:
            print(f"🎬 YouTube Video ID: {video_id}")
            print(f"🔗 Video URL: https://www.youtube.com/watch?v={video_id}")
        elif skip_upload:
            print("⏭️  Video upload skipped (will be done in Step 5)")
        print("="*60)
        
        # Comprehensive cleanup of all temporary files and folders
        print("\nCleaning up all temporary files and folders...")
        cleaned_items = []
    
        # Final video is saved in the output folder
        if video_path and os.path.exists(video_path):
            print(f"✅ Final video saved: {video_path}")
            print(f"📁 Both files are in the output folder: {output_folder}")
        
        # Clean up generated thumbnail (only if it was auto-generated, not user-provided)
        if generated_thumbnail and os.path.exists(generated_thumbnail) and generated_thumbnail != video_path:
            try:
                os.remove(generated_thumbnail)
                cleaned_items.append(f"Generated Thumbnail: {os.path.basename(generated_thumbnail)}")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete thumbnail: {e}")
        
        # Note: Individual segment video files are kept for potential regeneration
        # They can be cleaned up manually if needed
        
        # Clean up intermediate video files (_stitched, _1080p, _with_audio)
        if video_path and os.path.exists(video_path):
            video_dir = os.path.dirname(video_path)
            video_base = os.path.basename(video_path)
            if video_dir:
                for file in os.listdir(video_dir):
                    file_path = os.path.join(video_dir, file)
                    if os.path.isfile(file_path):
                        # Check if it's an intermediate file related to this video
                        base_name = os.path.splitext(video_base)[0]
                        if (file.startswith(base_name) and file != video_base and 
                            ('_stitched' in file or '_1080p' in file or '_with_audio' in file or '_segment_' in file)):
                            try:
                                os.remove(file_path)
                                cleaned_items.append(f"Intermediate Video: {file}")
                            except Exception as e:
                                print(f"⚠️  Warning: Could not delete intermediate file {file}: {e}")
        
        # Clean up audio_review folders
        try:
            import glob
            cwd = os.getcwd()
            review_pattern = os.path.join(cwd, "audio_review_*")
            review_folders = glob.glob(review_pattern)
            for review_folder in review_folders:
                if os.path.isdir(review_folder):
                    try:
                        import shutil
                        shutil.rmtree(review_folder)
                        cleaned_items.append(f"Review Folder: {os.path.basename(review_folder)}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not delete review folder {review_folder}: {e}")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up review folders: {e}")
        
        # Clean up temporary audio files in temp directory
        try:
            import glob
            temp_dir = tempfile.gettempdir()
            # Clean up common temp audio patterns
            temp_patterns = [
                os.path.join(temp_dir, "voiceover_*.mp3"),
                os.path.join(temp_dir, "music_*.mp3"),
                os.path.join(temp_dir, "mixed_audio_*.mp3"),
                os.path.join(temp_dir, "audio_*.mp3"),
                os.path.join(temp_dir, "original_voiceover_*.mp3"),
                os.path.join(temp_dir, "silence_*.mp3"),
                os.path.join(temp_dir, "chord_*.wav"),
                os.path.join(temp_dir, "tone_*.wav"),
                os.path.join(temp_dir, "concat_*.txt"),
                os.path.join(temp_dir, "music_synced_*.mp3"),
                os.path.join(temp_dir, "audio_resynced_*.mp3"),
                os.path.join(temp_dir, "voiceover_adjusted_*.mp3"),
                os.path.join(temp_dir, "voiceover_padded_*.mp3"),
            ]
            for pattern in temp_patterns:
                for temp_file in glob.glob(pattern):
                    if os.path.isfile(temp_file):
                        try:
                            os.remove(temp_file)
                            cleaned_items.append(f"Temp Audio: {os.path.basename(temp_file)}")
                        except Exception as e:
                            pass  # Silently ignore - might be in use
        except Exception as e:
            pass  # Silently ignore temp cleanup errors
        
        # Clean up temporary video files in temp directory
        try:
            import glob
            temp_dir = tempfile.gettempdir()
            temp_video_patterns = [
                os.path.join(temp_dir, "video_*.mp4"),
                os.path.join(temp_dir, "first_*.png"),
                os.path.join(temp_dir, "last_*.png"),
                os.path.join(temp_dir, "first_loop_*.mp4"),
                os.path.join(temp_dir, "last_loop_*.mp4"),
            ]
            for pattern in temp_video_patterns:
                for temp_file in glob.glob(pattern):
                    if os.path.isfile(temp_file) and temp_file != video_path and temp_file != output_video_path:
                        try:
                            os.remove(temp_file)
                            cleaned_items.append(f"Temp Video: {os.path.basename(temp_file)}")
                        except Exception as e:
                            pass  # Silently ignore
        except Exception as e:
            pass  # Silently ignore temp cleanup errors
        
        print(f"Final video: {output_video_path}")
        
        # Try to move video to output path (in output folder) if needed
        if video_path and os.path.exists(video_path) and video_path != output_video_path:
            try:
                # Ensure output folder exists
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder, exist_ok=True)
                if os.path.exists(output_video_path):
                    os.remove(output_video_path)
                # Ensure shutil is available (it's imported at top, but ensure it's accessible)
                import shutil
                shutil.move(video_path, output_video_path)
                video_path = output_video_path
                print(f"✅ Final video saved to: {output_video_path}")
            except Exception as e2:
                print(f"⚠️  Could not move video to output path: {e2}")
                print(f"   Video is at: {video_path}")
    
    # Return video_id if uploaded, otherwise return video_path
    if skip_upload:
        return video_path
    else:
        return video_id if video_id else video_path


def validate_and_cap_duration(duration, max_duration=600):
    """
    Validate and cap video duration to prevent excessive costs.
    
    Args:
        duration: The requested duration in seconds
        max_duration: Maximum allowed duration in seconds (default: 600 = 10 minutes)
    
    Returns:
        int: The validated duration, capped at max_duration
    """
    if duration is None:
        return FIXED_SEGMENT_DURATION_INT  # Default duration
    duration = int(duration)
    if duration < FIXED_SEGMENT_DURATION_INT:
        print(
            f"⚠️  Warning: Duration {duration}s is too short. "
            f"Setting to minimum of {FIXED_SEGMENT_DURATION_INT} seconds."
        )
        duration = FIXED_SEGMENT_DURATION_INT
    if duration > max_duration:
        print(f"⚠️  Warning: Duration {duration}s exceeds maximum of {max_duration}s (10 minutes).")
        print(f"   Capping duration to {max_duration}s to prevent excessive costs.")
        duration = max_duration

    if duration % FIXED_SEGMENT_DURATION_INT != 0:
        import math
        rounded = int(
            math.ceil(duration / FIXED_SEGMENT_DURATION_INT) * FIXED_SEGMENT_DURATION_INT
        )
        if rounded > max_duration:
            rounded = (max_duration // FIXED_SEGMENT_DURATION_INT) * FIXED_SEGMENT_DURATION_INT
        if rounded < FIXED_SEGMENT_DURATION_INT:
            rounded = FIXED_SEGMENT_DURATION_INT
        print(
            f"⚠️  Warning: Duration {duration}s is not a multiple of {FIXED_SEGMENT_DURATION_INT}s. "
            f"Rounding to {rounded}s."
        )
        duration = rounded

    return duration


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate video using Sora 2 and upload to YouTube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and upload with minimal parameters:
  python generate_and_upload_sora.py "A serene landscape" --title "My Video"
  
  # Full example:
  python generate_and_upload_sora.py "A cat playing piano" --title "Cat Piano" \\
    --description "AI generated video" --privacy public \\
    --duration 12 --resolution 1920x1080 --keep-video

Environment Variables:
  OPENAI_API_KEY: Your OpenAI API key (required for script/image/video generation)
  ELEVENLABS_API_KEY: Your ElevenLabs API key (required for narration TTS)
  ELEVENLABS_VOICE_ID: Your ElevenLabs voice ID (optional, defaults to "Brian")
        """
    )
    
    # Prompt and YouTube parameters
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Text prompt describing the video to generate'
    )
    
    parser.add_argument(
        '--title',
        help='YouTube video title'
    )
    
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Disable interactive prompts (fail if required parameters missing)'
    )
    
    parser.add_argument(
        '--generate-script-only',
        action='store_true',
        help='Only generate and save the overarching script to a file. Do not proceed with video generation. Allows editing the script before running the full workflow.'
    )
    
    parser.add_argument(
        '--generate-narration-only',
        action='store_true',
        help='Only generate and save the narration audio from the script file. Do not proceed with video generation. Requires the script file to exist. Allows editing the script before generating narration.'
    )
    
    
    parser.add_argument(
        '--description',
        default='',
        help='YouTube video description'
    )
    
    
    parser.add_argument(
        '--category',
        default='27',
        help='YouTube category ID (default: 27 - Education)'
    )
    
    parser.add_argument(
        '--privacy',
        choices=['private', 'public', 'unlisted'],
        default='private',
        help='Privacy status (default: private)'
    )
    
    parser.add_argument(
        '--thumbnail',
        help='Path to thumbnail image file'
    )
    
    parser.add_argument(
        '--playlist',
        help='YouTube playlist ID'
    )
    
    # Video generation parameters
    parser.add_argument(
        '--output',
        help='Output video file path (default: temp file, deleted after upload)'
    )
    
    parser.add_argument(
        '--api-key',
        default=None,
        help='OpenAI API key for script/video generation (default: uses OPENAI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--elevenlabs-api-key',
        default=None,
        help='ElevenLabs API key for narration TTS (default: uses ELEVENLABS_API_KEY env var)'
    )
    
    parser.add_argument(
        '--elevenlabs-voice-id',
        default=None,
        help='ElevenLabs voice ID for narration (default: uses ELEVENLABS_VOICE_ID env var, or "Adam" voice)'
    )
    
    parser.add_argument(
        '--model',
        choices=[
            'sora-2',
            'sora-2-pro',
        ],
        default='sora-2',
        help='Sora model to use (default: sora-2). Available: sora-2, sora-2-pro'
    )
    
    parser.add_argument(
        '--resolution',
        default='1280x720',
        help='Video resolution (default: 1280x720)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Video duration in seconds (default: from config file, or 12 if no config, max: 600)'
    )
    
    # Removed --num-videos argument - now calculated automatically from duration
    
    parser.add_argument(
        '--aspect-ratio',
        default='16:9',
        help='Aspect ratio (default: 16:9)'
    )
    
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=10,
        help='Seconds to wait between status checks (default: 10)'
    )
    
    parser.add_argument(
        '--max-wait',
        type=int,
        default=600,
        help='Maximum time to wait for completion in seconds (default: 600)'
    )
    
    parser.add_argument(
        '--keep-video',
        action='store_true',
        help='Keep the generated video file after upload'
    )
    
    parser.add_argument(
        '--no-upscale',
        action='store_true',
        help='Disable automatic upscaling to 1080p (default: upscale is enabled)'
    )
    
    args = parser.parse_args()
    
    # Set global ElevenLabs API key and voice ID from command-line argument or environment variable
    global ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
    if hasattr(args, 'elevenlabs_api_key') and args.elevenlabs_api_key:
        ELEVENLABS_API_KEY = args.elevenlabs_api_key
    elif os.getenv('ELEVENLABS_API_KEY'):
        ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
    
    if hasattr(args, 'elevenlabs_voice_id') and args.elevenlabs_voice_id:
        ELEVENLABS_VOICE_ID = args.elevenlabs_voice_id
    elif os.getenv('ELEVENLABS_VOICE_ID'):
        ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID')
    
    # Validate and cap duration immediately after parsing arguments
    if args.duration:
        args.duration = validate_and_cap_duration(args.duration)
    
    # Check command-line flags first (these take precedence over interactive prompts)
    # Check if we should only generate the script
    if args.generate_script_only:
        if not args.prompt:
            print("❌ Error: --prompt is required when using --generate-script-only")
            return 1
        
        prompt = args.prompt
        
        # Collect ALL inputs once during script generation and save to config
        # Delete existing config file at the start (cleanup from last run)
        if os.path.exists(CONFIG_FILE_PATH):
            print(f"🧹 Deleting existing config file: {CONFIG_FILE_PATH}")
            try:
                os.remove(CONFIG_FILE_PATH)
            except Exception as e:
                print(f"⚠️  Warning: Could not delete config file: {e}")
        
        print("\n📋 Collecting video configuration (this will be saved for later steps)...")
        
        # Always ask for title and description
        if not args.title:
            try:
                title = input("Enter YouTube video title: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Error: Video title is required!")
                return 1
            if not title:
                print("❌ Error: Video title is required!")
                return 1
        else:
            title = args.title
        
        if not args.description:
            print("\nEnter video description (press Enter twice when done, or just Enter for empty):")
            description_lines = []
            while True:
                try:
                    line = input()
                    if line == '' and description_lines:
                        break
                    description_lines.append(line)
                except EOFError:
                    break
            description = '\n'.join(description_lines)
        else:
            description = args.description
        
        # Always ask for duration
        duration = args.duration if args.duration else FIXED_SEGMENT_DURATION_INT
        try:
            duration_input = input(f"\nEnter video duration in seconds (default: {duration}, max: 600): ").strip()
            if duration_input:
                duration = int(duration_input)
        except (EOFError, ValueError):
            pass
        duration = validate_and_cap_duration(duration)
        
        # Ask for tags
        try:
            tags_input = input("Enter tags (comma or space-separated, or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            tags_input = ""
        if tags_input:
            tags = [tag.strip() for tag in tags_input.replace(',', ' ').split() if tag.strip()]
        else:
            tags = None
        
        # Ask for privacy status
        try:
            privacy_input = input("Privacy status [private/public/unlisted] (default: private): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            privacy_input = ""
        if privacy_input in ['private', 'public', 'unlisted']:
            privacy_status = privacy_input
        else:
            privacy_status = args.privacy if args.privacy else 'private'
        
        # Ask for category
        try:
            category_input = input("Category ID (default: 27 - Education, or press Enter to use default): ").strip()
        except (EOFError, KeyboardInterrupt):
            category_input = ""
        category_id = category_input if category_input else (args.category if args.category else '27')
        
        # Ask for model
        try:
            model_input = input(f"Model [sora-2/sora-2-pro] (default: {args.model if args.model else 'sora-2'}): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            model_input = ""
        if model_input in ['sora-2', 'sora-2-pro']:
            model = model_input
        else:
            model = args.model if args.model else 'sora-2'
        
        # Ask for resolution
        try:
            resolution_input = input(f"Resolution (default: {args.resolution if args.resolution else '1920x1080'}): ").strip()
        except (EOFError, KeyboardInterrupt):
            resolution_input = ""
        resolution = resolution_input if resolution_input else (args.resolution if args.resolution else '1920x1080')
        
        # Check for thumbnail_file in directory first
        thumbnail_file = find_thumbnail_file()
        if not thumbnail_file:
            # Ask for thumbnail
            try:
                thumbnail_input = input("Thumbnail image path (optional, press Enter to skip): ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                thumbnail_input = ""
            thumbnail_file = thumbnail_input if thumbnail_input else (args.thumbnail if args.thumbnail else None)
        
        # Ask for playlist
        try:
            playlist_input = input("Playlist ID (optional, press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            playlist_input = ""
        playlist_id = playlist_input if playlist_input else (args.playlist if args.playlist else None)
        
        print(f"\nConfiguration: {title} ({duration}s, {model})")
        
        # Save all configuration to file
        config_data = {
            'prompt': prompt,
            'title': title,
            'description': description,
            'duration': duration,
            'tags': tags,
            'privacy_status': privacy_status,
            'category_id': category_id,
            'model': model,
            'resolution': resolution,
            'thumbnail_file': thumbnail_file,
            'playlist_id': playlist_id,
        }
        
        # Save config and verify it was created
        config_path = save_config(config_data)
        if config_path and os.path.exists(config_path):
            print(f"✅ Config file verified at: {os.path.abspath(config_path)}")
        else:
            print(f"⚠️  Warning: Config file may not have been created properly!")
            print(f"   Expected path: {os.path.abspath(CONFIG_FILE_PATH)}")
        
        try:
            script_file = generate_and_save_script(
                video_prompt=prompt,
                duration=duration,
                api_key=args.api_key,
                model='gpt-5-2025-08-07'
            )
            print(f"✅ Script generation complete")
            return 0
        except Exception as e:
            print(f"\n❌ Error generating script: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Check if we should only generate the narration
    if args.generate_narration_only:
        # Narration generation only - load config if available for target duration
        config = load_config()
        narration_target_duration = args.duration
        if narration_target_duration is None and config:
            narration_target_duration = config.get('duration', None)
        
        try:
            narration_file = generate_and_save_narration(
                script_file_path=SCRIPT_FILE_PATH,
                narration_audio_path=NARRATION_AUDIO_PATH,
                duration=narration_target_duration,
                api_key=args.api_key,
            )
            print(f"✅ Narration generation complete")
            print(f"\n💡 Next steps:")
            print(f"   1. Edit {SCRIPT_FILE_PATH} if needed (then regenerate narration)")
            print(f"   2. Run the script again WITHOUT --generate-narration-only to continue with video generation")
            return 0
        except Exception as e:
            print(f"\n❌ Error generating narration: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Interactive mode: prompt for missing required parameters
    if not args.non_interactive:
        # Check if we're in a non-interactive environment (like debugger)
        import sys
        if not sys.stdin.isatty():
            print("⚠️  Running in non-interactive mode")
            print("Provide command-line arguments or use --non-interactive flag")
            if not args.prompt or not args.title:
                print("\n❌ Error: Missing required arguments (prompt and title)")
                print("   Run with: --non-interactive --prompt '...' --title '...'")
                return 1
        
        # ============================================================
        # WORKFLOW: 5 Sequential Steps - Run one and exit
        # ============================================================
        
        # Step 1: Generate script
        print("\nSTEP 1: Generate Script")
        step1_input = 'n'
        if not args.non_interactive:
            try:
                step1_input = input("Generate script? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step1_input = 'n'
        
        if step1_input in ['y', 'yes']:
            # Archive workflow files before starting new script generation
            # This ensures narration_audio.mp3 and other files from previous run are saved
            print("Archiving previous workflow files...")
            archive_workflow_files()
            # Always ask for all configuration inputs and save to config
            # Delete existing config file at the start (cleanup from last run)
            if os.path.exists(CONFIG_FILE_PATH):
                try:
                    os.remove(CONFIG_FILE_PATH)
                except Exception as e:
                    print(f"⚠️  Could not delete config file: {e}")
            
            print("Collecting video configuration...")
            
            # Get prompt
            if not args.prompt:
                try:
                    print("Enter video prompt (text description of the video, including the central theme that will be the focus of the story):")
                    prompt_lines = []
                    while True:
                        try:
                            line = input()
                            if line == '' and prompt_lines:
                                break
                            prompt_lines.append(line)
                        except EOFError:
                            break
                    prompt = '\n'.join(prompt_lines).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Error: Video prompt is required!")
                    return 1
            else:
                prompt = args.prompt
            
            if not prompt:
                print("❌ Error: Video prompt is required!")
                return 1
            
            # Always ask for title
            if not args.title:
                try:
                    title = input("Enter YouTube video title: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Error: Video title is required!")
                    return 1
                if not title:
                    print("❌ Error: Video title is required!")
                    return 1
            else:
                title = args.title
            
            # Always ask for description
            if not args.description:
                print("\nEnter video description (press Enter twice when done, or just Enter for empty):")
                description_lines = []
                while True:
                    try:
                        line = input()
                        if line == '' and description_lines:
                            break
                        description_lines.append(line)
                    except EOFError:
                        break
                description = '\n'.join(description_lines)
            else:
                description = args.description
            
            # Always ask for duration
            duration = args.duration if args.duration else FIXED_SEGMENT_DURATION_INT
            try:
                duration_input = input(f"\nEnter video duration in seconds (default: {duration}, max: 600): ").strip()
                if duration_input:
                    duration = int(duration_input)
            except (EOFError, ValueError):
                pass
            duration = validate_and_cap_duration(duration)
            
            # Ask for tags
            try:
                tags_input = input("Enter tags (comma or space-separated, or press Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                tags_input = ""
            if tags_input:
                tags = [tag.strip() for tag in tags_input.replace(',', ' ').split() if tag.strip()]
            else:
                tags = None
            
            # Ask for privacy status
            try:
                privacy_input = input("Privacy status [private/public/unlisted] (default: private): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                privacy_input = ""
            if privacy_input in ['private', 'public', 'unlisted']:
                privacy_status = privacy_input
            else:
                privacy_status = args.privacy if args.privacy else 'private'
            
            # Ask for category
            try:
                category_input = input("Category ID (default: 27 - Education, or press Enter to use default): ").strip()
            except (EOFError, KeyboardInterrupt):
                category_input = ""
            category_id = category_input if category_input else (args.category if args.category else '27')
            
            # Ask for model
            try:
                model_input = input(f"Model [sora-2/sora-2-pro] (default: {args.model if args.model else 'sora-2'}): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                model_input = ""
            if model_input in ['sora-2', 'sora-2-pro']:
                model = model_input
            else:
                model = args.model if args.model else 'sora-2'
            
            # Ask for resolution
            try:
                resolution_input = input(f"Resolution (default: {args.resolution if args.resolution else '1920x1080'}): ").strip()
            except (EOFError, KeyboardInterrupt):
                resolution_input = ""
            resolution = resolution_input if resolution_input else (args.resolution if args.resolution else '1920x1080')
            
            # Check for thumbnail_file in directory first
            thumbnail_file = find_thumbnail_file()
            if not thumbnail_file:
                # Ask for thumbnail
                try:
                    thumbnail_input = input("Thumbnail image path (optional, press Enter to skip): ").strip().strip('"').strip("'")
                except (EOFError, KeyboardInterrupt):
                    thumbnail_input = ""
                thumbnail_file = thumbnail_input if thumbnail_input else (args.thumbnail if args.thumbnail else None)
            
            # Ask for playlist
            try:
                playlist_input = input("Playlist ID (optional, press Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                playlist_input = ""
            playlist_id = playlist_input if playlist_input else (args.playlist if args.playlist else None)
            
            print(f"Configuration: {title} ({duration}s, {model})")
            
            # Save all configuration to file
            config_data = {
                'prompt': prompt,
                'title': title,
                'description': description,
                'duration': duration,
                'tags': tags,
                'privacy_status': privacy_status,
                'category_id': category_id,
                'model': model,
                'resolution': resolution,
                'thumbnail_file': thumbnail_file,
                'playlist_id': playlist_id,
            }
            
            # Save config and verify it was created
            config_path = save_config(config_data)
            if config_path and os.path.exists(config_path):
                print(f"✅ Config file verified at: {os.path.abspath(config_path)}")
            else:
                print(f"⚠️  Warning: Config file may not have been created properly!")
                print(f"   Expected path: {os.path.abspath(CONFIG_FILE_PATH)}")
            
            try:
                script_file = generate_and_save_script(
                    video_prompt=prompt,
                    duration=duration,
                    api_key=args.api_key,
                    model='gpt-5-2025-08-07'
                )
                print(f"✅ Step 1 complete: Script saved")
                return 0
            except Exception as e:
                print(f"\n❌ Error generating script: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Step 2: Generate narration (generate narration audio)
        print("\nSTEP 2: Generate Narration")
        step2_input = 'n'
        if not args.non_interactive:
            try:
                step2_input = input("Generate narration? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step2_input = 'n'
        
        if step2_input in ['y', 'yes']:
            # Load duration from config if not already set
            try:
                duration
            except NameError:
                config = load_config()
                duration = (
                    config.get('duration', FIXED_SEGMENT_DURATION_INT)
                    if config
                    else (args.duration if args.duration else FIXED_SEGMENT_DURATION_INT)
                )
            try:
                narration_file = generate_and_save_narration(
                    script_file_path=SCRIPT_FILE_PATH,
                    narration_audio_path=NARRATION_AUDIO_PATH,
                    duration=duration,  # Pass target duration for iterative adjustment
                    api_key=args.api_key,
                )
                print(f"✅ Step 2 complete: Narration saved")
                return 0
            except Exception as e:
                print(f"\n❌ Error generating narration: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Step 3: Generate video
        print("\nSTEP 3: Generate Video")
        step3_input = 'n'
        if not args.non_interactive:
            try:
                step3_input = input("Generate video? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step3_input = 'n'
        
        if step3_input in ['y', 'yes']:
            # Load config and script
            config = load_config()
            if not config:
                print("⚠️  No config found. Please generate script first.")
                return 1
            
            prompt = args.prompt if args.prompt else config.get('prompt')
            title = args.title if args.title else config.get('title')
            description = args.description if args.description else config.get('description', '')
            duration = args.duration if args.duration else config.get('duration', FIXED_SEGMENT_DURATION_INT)
            
            if not prompt or not title:
                print("⚠️  Missing required configuration. Please generate script first.")
                return 1
            tags = config.get('tags')
            privacy_status = args.privacy if args.privacy != 'private' else config.get('privacy_status', 'private')
            category_id = args.category if args.category != '27' else config.get('category_id', '27')
            model = args.model if args.model != 'sora-2' else config.get('model', 'sora-2')
            resolution = args.resolution if args.resolution != '1920x1080' else config.get('resolution', '1920x1080')
            # Check for thumbnail_file in directory
            thumbnail_file = find_thumbnail_file()
            if not thumbnail_file:
                thumbnail_file = args.thumbnail if args.thumbnail else None
            playlist_id = args.playlist if args.playlist else None
            
            duration = validate_and_cap_duration(duration)
            
            try:
                # Generate video WITHOUT narration and WITHOUT YouTube upload
                video_path = generate_and_upload_sora(
                    prompt=prompt,
                    title=title,
                    description=description,
                    tags=tags,
                    category_id=category_id,
                    privacy_status=privacy_status,
                    thumbnail_file=thumbnail_file,
                    playlist_id=playlist_id,
                    output_video_path=args.output,
                    api_key=args.api_key,
                    model=model,
                    resolution=resolution,
                    duration=duration,
                    aspect_ratio=args.aspect_ratio,
                    poll_interval=args.poll_interval,
                    max_wait_time=args.max_wait,
                    keep_video=args.keep_video,
                    upscale_to_1080p=not args.no_upscale,
                    skip_narration=True,  # Skip narration generation
                    skip_upload=True,  # Skip YouTube upload
                )
                print(f"✅ Step 3 complete: Video saved")
                return 0
            except Exception as e:
                print(f"\n❌ Error generating video: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Step 4: Review, Regenerate, or Stitch Segments
        print("\nSTEP 4: Review, Regenerate, or Stitch Segments")
        step4_input = 'n'
        if not args.non_interactive:
            try:
                step4_input = input("Review and stitch segments? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step4_input = 'n'
        
        if step4_input in ['y', 'yes']:
            try:
                # Load config and metadata
                config = load_config()
                if not config:
                    print("⚠️  No config found. Please generate video first.")
                    return 1
                
                output_folder = os.path.join(os.getcwd(), "video_output")
                metadata = load_segment_metadata(output_folder)
                
                if not metadata:
                    print("⚠️  No segment metadata found. Please generate video segments first.")
                    return 1
                
                # Extract data from metadata
                generated_video_segments = metadata.get('generated_video_segments', [])
                still_image_videos = {}
                segment_assignments = metadata.get('segment_assignments', [])
                num_segments = metadata.get('num_segments', 0)
                segment_id_to_prompt = metadata.get('segment_id_to_prompt', {})
                generated_segment_texts = metadata.get('generated_segment_texts', [])
                generated_script = metadata.get('generated_script', '')
                num_videos = num_segments
                num_still_images = 0
                output_video_path = metadata.get('output_video_path', os.path.join(output_folder, 'sora_video.mp4'))
                duration = config.get('duration', 60)
                narration_offset = metadata.get('narration_offset', 0.0)
                
                # If narration_offset not in metadata, compute it from narration audio
                if narration_offset == 0.0:
                    narration_audio_path = None
                    for nar_name in ["narration_audio.mp3", NARRATION_AUDIO_PATH]:
                        nar_check = os.path.join(os.getcwd(), nar_name)
                        if os.path.exists(nar_check):
                            narration_audio_path = nar_check
                            break
                    
                    if narration_audio_path:
                        ffmpeg_path = find_ffmpeg()
                        if ffmpeg_path:
                            actual_narration_duration = get_media_duration(narration_audio_path, ffmpeg_path)
                            if actual_narration_duration and actual_narration_duration > 0:
                                import math
                                adjusted_duration = int(
                                    math.ceil(actual_narration_duration / FIXED_SEGMENT_DURATION_SECONDS)
                                    * FIXED_SEGMENT_DURATION_INT
                                )
                                if adjusted_duration != duration:
                                    duration = adjusted_duration
                                narration_offset = (duration - actual_narration_duration) / 2.0
                                narration_offset = max(0.0, narration_offset)
                
                if narration_offset > 0:
                    print(f"\n🎬 Narration centering (from metadata):")
                    print(f"   Narration offset: {narration_offset:.1f}s (narration starts at {narration_offset:.1f}s in video)")
                
                # Determine upscale_to_1080p from command-line argument
                upscale_to_1080p = not args.no_upscale
                
                # Show available segments
                print(f"\nAvailable segments:")
                print(f"  Video segments: {len(generated_video_segments)}")
                print(f"  Total segments: {num_segments}")
                
                if generated_video_segments:
                    print(f"\nVideo segment details:")
                    for seg in sorted(generated_video_segments, key=lambda x: x.get('segment_id', 0)):
                        seg_id = seg.get('segment_id', '?')
                        video_path = seg.get('video_path', 'unknown')
                        exists = os.path.exists(video_path) if video_path != 'unknown' else False
                        status = "[EXISTS]" if exists else "[MISSING]"
                        print(f"  Segment {seg_id}: {os.path.basename(video_path)} {status}")
                
                # Ask user what they want to do
                if not args.non_interactive:
                    print(f"\nWhat would you like to do?")
                    print(f"  1. Stitch all segments together")
                    print(f"  2. Regenerate specific segments (then stitch)")
                    print(f"  3. Skip")
                    
                    try:
                        choice = input("Enter choice (1/2/3, default: 3): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        choice = "3"
                else:
                    choice = "3"
                
                if choice == "1":
                    # Stitch all segments
                    print("\nStitching all video segments together...")
                    video_path = stitch_all_segments(
                        generated_video_segments=generated_video_segments,
                        still_image_videos=still_image_videos,
                        segment_assignments=segment_assignments,
                        num_segments=num_segments,
                        output_video_path=output_video_path,
                        duration=duration,
                        upscale_to_1080p=upscale_to_1080p,
                        narration_offset=narration_offset
                    )
                    print(f"✅ Step 4 complete: Video segments stitched")
                    return 0
                
                elif choice == "2":
                    # Regenerate specific segments
                    segments_to_regenerate = None
                    if not args.non_interactive:
                        try:
                            regenerate_input = input("\nEnter segment numbers to regenerate (comma-separated, e.g., 1,3,5) or press Enter to cancel: ").strip()
                            if regenerate_input:
                                try:
                                    segments_to_regenerate = [int(x.strip()) for x in regenerate_input.split(',')]
                                    print(f"Will regenerate segments: {segments_to_regenerate}")
                                except ValueError:
                                    print("Invalid input. Cancelling regeneration.")
                                    segments_to_regenerate = None
                        except (EOFError, KeyboardInterrupt):
                            segments_to_regenerate = None
                    
                    if segments_to_regenerate:
                        # Regenerate specific segments
                        print(f"\nRegenerating segments: {segments_to_regenerate}")
                        regenerated_segments = generate_video_segments(
                            segment_id_to_prompt=segment_id_to_prompt,
                            segment_assignments=segment_assignments,
                            num_segments=num_segments,
                            num_videos=num_videos,
                            output_folder=output_folder,
                            output_video_path=output_video_path,
                            generated_segment_texts=generated_segment_texts,
                            generated_script=generated_script,
                            api_key=args.api_key,
                            model=config.get('model', 'sora-2'),
                            resolution=config.get('resolution', '1280x720'),
                            poll_interval=args.poll_interval,
                            max_wait_time=args.max_wait,
                            segments_to_regenerate=segments_to_regenerate,
                        )
                        
                        # Update generated_video_segments with regenerated ones
                        generated_video_segments = [seg for seg in generated_video_segments if seg['segment_id'] not in segments_to_regenerate]
                        generated_video_segments.extend(regenerated_segments)
                        
                        # Save updated metadata
                        save_segment_metadata(
                            output_folder=output_folder,
                            segment_id_to_prompt=segment_id_to_prompt,
                            generated_video_segments=generated_video_segments,
                            still_image_videos=still_image_videos,
                            segment_assignments=segment_assignments,
                            generated_segment_texts=generated_segment_texts,
                            generated_script=generated_script,
                            num_segments=num_segments,
                            num_videos=num_videos,
                            num_still_images=num_still_images,
                            output_video_path=output_video_path,
                            narration_offset=narration_offset
                        )
                        
                        # After regenerating, ask if they want to stitch
                        if not args.non_interactive:
                            try:
                                stitch_input = input("\nSegments regenerated. Stitch all segments together now? (y/n, default: y): ").strip().lower()
                                stitch_now = stitch_input in ['y', 'yes', '']
                            except (EOFError, KeyboardInterrupt):
                                stitch_now = True
                        else:
                            stitch_now = True
                        
                        if stitch_now:
                            print("\nStitching all video segments together...")
                            video_path = stitch_all_segments(
                                generated_video_segments=generated_video_segments,
                                still_image_videos=still_image_videos,
                                segment_assignments=segment_assignments,
                                num_segments=num_segments,
                                output_video_path=output_video_path,
                                duration=duration,
                                upscale_to_1080p=upscale_to_1080p,
                                narration_offset=narration_offset
                            )
                            print(f"✅ Step 4 complete: Segments regenerated and stitched")
                            return 0
                        else:
                            print(f"✅ Step 4 complete: Segments regenerated (stitching skipped)")
                            return 0
                    else:
                        print("⏭️  Regeneration cancelled")
                        return 0
                else:
                    print("⏭️  Skipping Step 4")
                    return 0
                    
            except Exception as e:
                print(f"\n❌ Error in Step 4: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # OBSOLETE: Step 5 - Add captions based on narration (commented out)
        # print("\nSTEP 5: Add Captions")
        # step4_input = 'n'
        # if not args.non_interactive:
        #     try:
        #         step4_input = input("Generate captions/subtitles? (y/n, default: n): ").strip().lower()
        #     except (EOFError, KeyboardInterrupt):
        #         step4_input = 'n'
        # 
        # if step4_input in ['y', 'yes']:
        #     # Find video file
        #     video_path = None
        #     output_folder = os.path.join(os.getcwd(), "video_output")
        #     if os.path.exists(output_folder):
        #         video_files = [f for f in os.listdir(output_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        #         if video_files:
        #             video_path = os.path.join(output_folder, sorted(video_files)[-1])
        #     
        #     if not video_path or not os.path.exists(video_path):
        #         print("❌ Error: Video file not found. Please generate video first.")
        #         return 1
        #     
        #     if not os.path.exists(NARRATION_AUDIO_PATH):
        #         print("❌ Error: Narration file not found. Please generate narration first.")
        #         return 1
        #     
        #     try:
        #         # Add narration, music, and captions to video
        #         # This function adds narration audio, background music, and captions based on the narration timing
        #         (legacy captions helper removed; workflow is now single-file)
        #         final_video_path = add_narration_music_and_captions_to_video(
        #             video_path=video_path,
        #             output_path=None,
        #             api_key=args.api_key,
        #             ffmpeg_path=None
        #         )
        #         video_path = final_video_path  # Update video_path to final version
        #         print(f"✅ Step 4 complete: Captions added")
        #         return 0
        #     except Exception as e:
        #         print(f"\n❌ Error adding captions to video: {e}")
        #         import traceback
        #         traceback.print_exc()
        #         return 1
        
        # Step 5: Upload to YouTube
        print("\nSTEP 5: Upload to YouTube")
        step5_input = 'n'
        if not args.non_interactive:
            try:
                step5_input = input("Upload to YouTube? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step5_input = 'n'
        
        if step5_input in ['y', 'yes']:
            # Find video file
            video_path = None
            output_folder = os.path.join(os.getcwd(), "video_output")
            if os.path.exists(output_folder):
                video_files = [f for f in os.listdir(output_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
                if video_files:
                    video_path = os.path.join(output_folder, sorted(video_files)[-1])
            
            if not video_path or not os.path.exists(video_path):
                print("❌ Error: Video file not found. Please generate video first.")
                return 1
            
            # Load config for upload
            config = load_config()
            if not config:
                print("⚠️  No config found. Please generate script first.")
                return 1
            
            title = args.title if args.title else config.get('title')
            description = args.description if args.description else config.get('description', '')
            tags = config.get('tags')
            privacy_status = args.privacy if args.privacy != 'private' else config.get('privacy_status', 'private')
            category_id = args.category if args.category != '27' else config.get('category_id', '27')
            
            if not title:
                print("⚠️  Missing title. Please generate script first.")
                return 1
            
            # Generate tags if not provided
            if not tags:
                print("📝 No tags found in config. Generating tags from script...")
                try:
                    script = load_script_from_file()
                    prompt = config.get('prompt', '')
                    if script or prompt:
                        generated_tags = generate_tags_from_script(
                            script=script if script else prompt,
                            video_prompt=prompt,
                            api_key=args.api_key,
                            model='gpt-4o'
                        )
                        tags = generated_tags
                        print(f"✅ Generated {len(tags)} tags: {', '.join(tags)}")
                    else:
                        print("⚠️  No script or prompt available for tag generation")
                        tags = []
                except Exception as e:
                    print(f"⚠️  Failed to generate tags: {e}")
                    tags = []
            
            # Ensure tags is a list
            if tags and not isinstance(tags, list):
                tags = [tags] if tags else []
            elif not tags:
                tags = []
            
            thumbnail_file = find_thumbnail_file()
            if not thumbnail_file:
                thumbnail_file = args.thumbnail if args.thumbnail else None
            playlist_id = args.playlist if args.playlist else None
            
            try:
                print(f"📤 Uploading with {len(tags)} tag(s): {', '.join(tags) if tags else 'none'}")
                video_id = upload_video(
                    video_file=video_path,
                    title=title,
                    description=description,
                    tags=tags,
                    category_id=category_id,
                    privacy_status=privacy_status,
                    thumbnail_file=thumbnail_file,
                    playlist_id=playlist_id
                )
                print(f"✅ Step 5 complete: Uploaded to YouTube")
                print(f"Video URL: https://www.youtube.com/watch?v={video_id}")
                return 0
            except Exception as e:
                print(f"\n❌ Error uploading to YouTube: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # If we get here, user answered 'no' to all steps
        print("No steps selected. Exiting.")
        return 0


if __name__ == '__main__':
    exit(main())

