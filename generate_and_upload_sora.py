"""
Complete Workflow: Generate video using OpenAI Sora 2 and upload to YouTube
Combines Sora 2 video generation and YouTube upload in one script.
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
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not installed. Install with: pip install openai")

try:
    from upload_video import upload_video
except ImportError as e:
    print(f"Error importing upload_video: {e}")
    print("Make sure upload_video.py is in the same directory")
    exit(1)

# Global API key (should be set via environment variable OPENAI_API_KEY or command-line argument)
# For security, do not hardcode API keys in the source code
OPENAI_API_KEY = None  # Will use os.getenv('OPENAI_API_KEY') or command-line argument
thumbnail_prompt_template = "Create an epic ultrarealistic cinematic-style thumbnail image for a video with the description: {description}. The image must be safe, appropriate, and comply with OpenAI content policies: no violence, hate, adult content, illegal activity, or copyrighted characters."
master_image_prompt_template = "Create the most hyperrealistic, ultra-detailed, high-quality, photorealistic reference frame image possible for a video with the description: {description}. The image must be extremely realistic and lifelike, as if photographed by a professional documentary photographer, with maximum detail, photorealism, and natural lighting. Make it look like a real photograph, not an illustration or artwork. The image must comply with OpenAI content policies: no violence, hate, adult content, illegal activity, copyrighted characters, or likenesses of real people. Use generic, artistic representations only, but make them appear completely realistic and photographic."

def analyze_script_for_reference_images(script, video_prompt, api_key=None, model='gpt-4o'):
    """
    Analyze script to determine what set of reference images should be created for visual consistency.
    For example, if the video is about Blackbeard, it might need: his face and his ship.
    
    Args:
        script: The full script text
        video_prompt: The original video prompt/topic
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-4o')
        
    Returns:
        List of dictionaries, each with 'id', 'type' ('character' or 'subject'), 'description', 'image_prompt', and 'reasoning'
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    analysis_prompt = f"""Analyze this script to determine what reference images should be created for visual consistency across all video segments.

Video topic: {video_prompt}

Full script: {script[:2000]}{'...' if len(script) > 2000 else ''}

Think about what visual elements need to remain consistent throughout the video. For example:
- If the video is about a person's life history, you might need: their face/character and important locations/objects
- If the video is about a historical event, you might need: key characters and important locations/objects
- If the video is about a place, you might need: the location and key objects/features

Output JSON array:
[
    {{
        "id": "ref_1",
        "type": "character" or "subject",
        "description": "Clear description of what this reference image represents (e.g., 'Blackbeard's face', 'Blackbeard's ship Queen Anne's Revenge')",
        "image_prompt": "Detailed DALL-E prompt for the most hyperrealistic, ultra-detailed, photorealistic image possible, as if photographed by a professional documentary photographer. Generic enough to avoid copyright, specific enough for consistency. Make it look like a real photograph with natural lighting, realistic textures, and lifelike detail. MUST comply with OpenAI content policy: no violence, hate, adult content, illegal activity, copyrighted characters, or likenesses of real, living people. Use generic, artistic representations only, but make them appear completely realistic and photographic with maximum detail and photorealism.",
        "reasoning": "Why this reference image is needed for visual consistency"
    }}
]

Rules:
- Identify 1-4 reference images that are most critical for visual consistency
- Character type = specific person/character that appears throughout (e.g., "Blackbeard's face", "Napoleon's appearance")
- Subject type = important locations, objects, or visual elements (e.g., "Blackbeard's ship", "battlefield location", "ancient temple")
- Each reference image should be something that appears in multiple segments and needs to look identical across all videos
- image_prompt MUST be safe, appropriate, and comply with OpenAI DALL-E content policies
- NEVER include likenesses of living people or celebrities. Historical non-living figures (Napoleon, Blackbeard, Cleopatra, etc.) are okay.
- Avoid: violence, hate speech, adult content, illegal activities, living people, copyrighted characters, weapons, gore
- Use: generic, artistic, educational, and appropriate visual descriptions
- Be specific enough that the same reference image can be used consistently, but generic enough to avoid copyright issues

Provide ONLY valid JSON array:"""
    
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Video production assistant. Analyze scripts to determine what set of reference images should be created for visual consistency across video segments. Identify the most important visual elements (characters, locations, objects) that need to remain consistent throughout the video. Create detailed, hyperrealistic image prompts that comply with OpenAI content policies. CRITICAL: All images must be hyperrealistic and photorealistic, as if photographed by a professional documentary photographer - make them look like real photographs with natural lighting, realistic textures, and maximum detail. Never create prompts with likenesses of real people, celebrities, or historical figures. Always use generic, artistic, stylized representations, but make them appear completely realistic and photographic. Ensure all image prompts comply with OpenAI content policies: no violence, hate, adult content, illegal activity, copyrighted characters, or real person likenesses."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_output_tokens=20000,
            temperature=1
        )
        
        import json
        reference_images = json.loads(response.output_text)
        
        # Ensure it's a list
        if isinstance(reference_images, dict):
            reference_images = [reference_images]
        
        # Validate and add default id if missing
        for i, ref_img in enumerate(reference_images):
            if 'id' not in ref_img:
                ref_img['id'] = f"ref_{i+1}"
        
        return reference_images
        
    except Exception as e:
        print(f"⚠️  Script analysis for reference images failed: {e}")
        # Fallback: return single general reference image
        return [{
            "id": "ref_1",
            "type": "subject",
            "description": f"Visual representation of {video_prompt}",
            "image_prompt": f"The most hyperrealistic, ultra-detailed, photorealistic reference image possible representing {video_prompt}, as if photographed by a professional documentary photographer, suitable for video generation, with maximum detail, natural lighting, realistic textures, and lifelike quality. Make it look like a real photograph, not an illustration.",
            "reasoning": "Analysis failed, defaulting to general subject"
        }]


def analyze_script_for_reference_image(script, video_prompt, api_key=None, model='gpt-4o'):
    """
    Analyze script to determine what reference image is needed (main character or general subject).
    
    Args:
        script: The full script text
        video_prompt: The original video prompt/topic
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-4o')
        
    Returns:
        Dictionary with 'type' ('character' or 'subject'), 'description', and 'image_prompt'
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    analysis_prompt = f"""Analyze script to determine reference image type (character or subject) for video: {video_prompt}

Script: {script[:500]}{'...' if len(script) > 500 else ''}

Output JSON:
{{
    "type": "character" or "subject",
    "reasoning": "Brief explanation",
    "description": "Generic description (for character: generic person, not specific. For subject: main visual element/location)",
    "image_prompt": "Detailed DALL-E prompt for the most hyperrealistic, ultra-detailed, photorealistic image possible, as if photographed by a professional documentary photographer. Generic enough to avoid copyright, specific enough for consistency. Make it look like a real photograph with natural lighting, realistic textures, and lifelike detail. MUST comply with OpenAI content policy: no violence, hate, adult content, illegal activity, copyrighted characters, or likenesses of real, living people. Use generic, artistic representations only, but make them appear completely realistic and photographic with maximum detail and photorealism."
}}

Rules: 
- Character = follows specific person's journey. Subject = general topic/location. Use generic descriptions only.
- image_prompt MUST be safe, appropriate, and comply with OpenAI DALL-E content policies.
- NEVER include likenesses of living people or celebrities. It is okay however to include likeness of historical non living figures (Napolean, Blackbeard, Cleopatra, etc.).
- Avoid: violence, hate speech, adult content, illegal activities, living people, copyrighted characters, weapons, gore, or any content that could violate policies.
- Use: generic, artistic, educational, and appropriate visual descriptions.

Provide ONLY valid JSON:"""
    
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Video production assistant. Analyze scripts to determine reference image type (character/subject) and create generic descriptions for visual consistency. CRITICAL: All images must be hyperrealistic and photorealistic, as if photographed by a professional documentary photographer - make them look like real photographs with natural lighting, realistic textures, and maximum detail. Never create prompts with likenesses of real people, celebrities, or historical figures. Always use generic, artistic, stylized representations, but make them appear completely realistic and photographic. Ensure all image prompts comply with OpenAI content policies: no violence, hate, adult content, illegal activity, copyrighted characters, or real person likenesses."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_output_tokens=20000,
            temperature=1
        )
        
        import json
        analysis = json.loads(response.output_text)
        return analysis
        
    except Exception as e:
        print(f"⚠️  Script analysis for reference image failed: {e}")
        # Fallback: assume it's a general subject
        return {
            "type": "subject",
            "reasoning": "Analysis failed, defaulting to general subject",
            "description": f"Visual representation of {video_prompt}",
            "image_prompt": f"The most hyperrealistic, ultra-detailed, photorealistic reference image possible representing {video_prompt}, as if photographed by a professional documentary photographer, suitable for video generation, with maximum detail, natural lighting, realistic textures, and lifelike quality. Make it look like a real photograph, not an illustration."
        }
# NOTE: Script generation is now separated into three distinct steps:
# 1. generate_script_from_prompt() - Generates the overarching script (separate API call)
# 2. segment_script_by_narration() - Segments the script into 12-second segments based on narration audio timing (uses Whisper API)
#    Falls back to segment_script_rule_based() if narration is not available
# 3. generate_sora_prompts_from_segments() - Generates Sora 2 prompts from narration-based segments (separate API calls)
# This separation allows for better control, error handling, and modularity.
# The narration-based segmentation ensures segments align with actual 12-second narration windows rather than word count.


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
        print(f"📦 Archiving workflow files to: {archive_folder}")
        
        files_archived = []
        
        # Archive video_output folder
        video_output_path = os.path.join(current_dir, "video_output")
        if os.path.exists(video_output_path) and os.listdir(video_output_path):
            archive_video_output = os.path.join(archive_folder, "video_output")
            shutil.copytree(video_output_path, archive_video_output, dirs_exist_ok=True)
            files_archived.append("video_output/")
            print(f"   ✅ Archived video_output folder")
        
        # Archive script file
        script_path = os.path.join(current_dir, SCRIPT_FILE_PATH)
        if os.path.exists(script_path):
            shutil.copy2(script_path, os.path.join(archive_folder, SCRIPT_FILE_PATH))
            files_archived.append(SCRIPT_FILE_PATH)
            print(f"   ✅ Archived {SCRIPT_FILE_PATH}")
        
        # Archive narration audio
        narration_path = os.path.join(current_dir, NARRATION_AUDIO_PATH)
        if os.path.exists(narration_path):
            shutil.copy2(narration_path, os.path.join(archive_folder, NARRATION_AUDIO_PATH))
            files_archived.append(NARRATION_AUDIO_PATH)
            print(f"   ✅ Archived {NARRATION_AUDIO_PATH}")
        
        # Archive config file
        config_path = os.path.join(current_dir, CONFIG_FILE_PATH)
        if os.path.exists(config_path):
            shutil.copy2(config_path, os.path.join(archive_folder, CONFIG_FILE_PATH))
            files_archived.append(CONFIG_FILE_PATH)
            print(f"   ✅ Archived {CONFIG_FILE_PATH}")
        
        # Archive music file if it exists
        music_files = ["VIDEO_MUSIC.mp3", "video_music.mp3", "VIDEO_MUSIC.MP3"]
        for music_file in music_files:
            music_path = os.path.join(current_dir, music_file)
            if os.path.exists(music_path):
                shutil.copy2(music_path, os.path.join(archive_folder, music_file))
                files_archived.append(music_file)
                print(f"   ✅ Archived {music_file}")
                break
        
        if files_archived:
            print(f"✅ Archived {len(files_archived)} item(s) to: {archive_folder}")
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
            file_size = os.path.getsize(config_file_path)
            print(f"✅ Configuration saved to: {abs_path}")
            print(f"   File size: {file_size} bytes")
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


def generate_and_save_script(video_prompt, duration=8, api_key=None, model='gpt-5.2-2025-12-11', max_tokens=20000, script_file_path=None):
    """
    Generate an overarching script and save it to a text file.
    This is Part 1 of the workflow - allows user to edit the script before continuing.
    
    Args:
        video_prompt: The video prompt/description to base the script on
        duration: Total video duration in seconds (default: 8)
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
        print(f"🧹 Deleting existing script file: {script_file_path}")
        try:
            os.remove(script_file_path)
            print(f"✅ Deleted previous script file")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete previous script file: {e}")
    
    # Generate the script
    print("="*60)
    print("📝 Part 1: Generating Overarching Script")
    print("="*60)
    print(f"Video Prompt: {video_prompt}")
    print(f"Duration: {duration} seconds")
    print("="*60 + "\n")
    
    generated_script = generate_script_from_prompt(
        video_prompt=video_prompt,
        duration=duration,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens
    )
    
    # Save script to file
    try:
        with open(script_file_path, 'w', encoding='utf-8') as f:
            f.write(generated_script)
        print(f"\n✅ Script saved to: {script_file_path}")
        print(f"📝 You can now edit this file before running the rest of the workflow.")
        print(f"   When ready, run the script again without --generate-script-only to continue.")
        return script_file_path
    except Exception as e:
        print(f"❌ Failed to save script to file: {e}")
        raise


def clean_script_for_tts(script):
    """
    Clean script to ensure it contains ONLY dialogue, [MUSICAL BREAK], and [VISUAL BREAK].
    Removes any labels, instructions, or extra text that would be read as dialogue by TTS.
    Also formats years (e.g., "2025" -> "20 25") so TTS reads them correctly.
    
    Args:
        script: The script text to clean
        
    Returns:
        Cleaned script containing only dialogue, [MUSICAL BREAK], and [VISUAL BREAK]
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
        # Keep everything else (dialogue, [MUSICAL BREAK], [VISUAL BREAK])
        cleaned_lines.append(line)
    
    script = '\n'.join(cleaned_lines).strip()
    
    # Final pass: ensure only dialogue, [MUSICAL BREAK], and [VISUAL BREAK] remain
    # Remove any remaining bracket patterns that aren't our markers
    import re
    # Keep only [MUSICAL BREAK] and [VISUAL BREAK], remove other bracket patterns
    script = re.sub(r'\[(?!MUSICAL BREAK|VISUAL BREAK)[^\]]+\]', '', script)
    
    # Format years: Replace 4-digit years (1000-2099) with space-separated versions
    # Pattern: word boundary, 4 digits (1000-2099), word boundary
    # This ensures we only match standalone years, not parts of larger numbers
    def format_year(match):
        year = match.group(0)
        # Split into two 2-digit parts
        return f"{year[:2]} {year[2:]}"
    
    # Match years between 1000 and 2099 (reasonable year range)
    # Use word boundaries to avoid matching years within larger numbers
    script = re.sub(r'\b(1[0-9]{3}|20[0-9]{2})\b', format_year, script)
    
    return script


def load_script_from_file(script_file_path=None):
    """
    Load the overarching script from a text file and clean it for TTS.
    
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
            script = f.read().strip()
        if script:
            # Clean the script to ensure it only contains dialogue, [MUSICAL BREAK], and [VISUAL BREAK]
            script = clean_script_for_tts(script)
        return script if script else None
    except Exception as e:
        print(f"⚠️  Failed to load script from file: {e}")
        return None


def generate_and_save_narration(script_file_path=None, narration_audio_path=None, duration=None, api_key=None):
    """
    Generate narration audio from the script file and save it.
    This is Part 2 of the workflow - generates voiceover from the script.
    
    Args:
        script_file_path: Path to the script file (default: SCRIPT_FILE_PATH)
        narration_audio_path: Path to save the narration audio (default: NARRATION_AUDIO_PATH)
        duration: Expected video duration in seconds (for music generation)
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
        print(f"🧹 Deleting existing narration audio file: {narration_audio_path}")
        try:
            os.remove(narration_audio_path)
            print(f"✅ Deleted previous narration audio file")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete previous narration audio file: {e}")
    
    # Generate narration
    print("="*60)
    print("🎙️  Part 2: Generating Narration Audio")
    print("="*60)
    print(f"Script file: {script_file_path}")
    print(f"Script length: {len(script)} characters")
    print("="*60 + "\n")
    
    try:
        # Stitch narration files from folder
        voiceover_audio_path, _ = generate_voiceover_from_folder(
            script=script,
            output_path=narration_audio_path,
            narration_folder=None,  # Uses 'narration_segments' folder in current directory
            break_duration=1000,  # 1 second for breaks
            music_volume=0.07  # 7% volume for background music
        )
        
        print(f"\n✅ Narration audio saved to: {narration_audio_path}")
        print(f"📝 You can now edit the script file if needed, then run the script again")
        print(f"   without --generate-narration-only to continue with video generation.")
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


def generate_script_from_prompt(video_prompt, duration=8, api_key=None, model='gpt-5.2-2025-12-11', max_tokens=20000):
    """
    STEP 1: Generate an overarching script for a video based on the video prompt using OpenAI ChatGPT API.
    This is the first of three separate API calls:
    1. This function generates the complete script (separate API call)
    2. segment_script_rule_based() segments the script into X segments (rules-based, no API call)
    3. generate_sora_prompts_from_segments() generates Sora 2 prompts from the segments (separate API calls)
    
    Args:
        video_prompt: The video prompt/description to base the script on
        duration: Total video duration in seconds (default: 8)
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
    
    print(f"Generating overarching script for video: {video_prompt[:50]}...")
    print(f"Total duration: {duration} seconds")
    
    # Create a simplified prompt for script generation only
    # Calculate target character count: 900 characters per minute of video
    # Convert duration from seconds to minutes
    duration_minutes = duration / 60.0
    target_characters = int(duration_minutes * 900)
    
    script_prompt = f"""Create a {duration}-second documentary-style YouTube script (approximately {target_characters} characters) for: {video_prompt}

IMPORTANT: The script should be approximately {target_characters} characters long (900 characters per minute of video). This allows for 2-3 second opening and closing shots with no narration.

CRITICAL ASSUMPTION: Assume the viewer knows NOTHING about this topic. Provide comprehensive context, background, and explanations throughout.

REQUIREMENTS:
- Tell the COMPLETE story - cover the full narrative from beginning to end
- Historically accurate and informational - like BBC or National Geographic documentaries
- Provide extensive context and background:
  * Explain who the key people/characters are and why they matter
  * Explain what important terms, concepts, or locations mean
  * Explain when this happened and what the world was like at that time
  * Explain where this took place and why location matters
  * Explain why events happened - the causes, motivations, and context
  * Provide historical background - what led up to these events
  * Explain cultural, political, or social context that helps understanding
  * Define technical terms, historical periods, or specialized concepts
  * Don't assume prior knowledge - explain everything clearly
- Structure:
  * Hook (5-10s): Compelling opening that grabs attention while providing initial context
  * Introduction (10-15%): Set the scene, context, background, and key players. Explain who/what/where/when/why thoroughly
  * What Happened (40-50%): Detailed narrative of events, key moments, and developments. Continuously provide context and explanations as the story unfolds
  * Climax (20-25%): The pivotal moment, turning point, or most dramatic event. Explain why this moment was significant
  * Conclusion (10-15%): How it ended and immediate aftermath. Explain the consequences
  * Impact (5-10%): Lasting significance, consequences, and why it matters. Connect to broader historical or cultural context
- Musical breaks and visual moments:
  * CRITICAL: Include a [MUSICAL BREAK] or [VISUAL BREAK] after no more than every 2000 characters of narration
  * This ensures regular pacing and prevents narration from becoming too dense
  * Place these breaks after dramatic moments, before transitions, or during visually stunning scenes
  * These breaks should be 2-4 seconds long - let the visuals and music tell part of the story
  * Flow: narration → pause/musical break → narration continues naturally
  * Use breaks to build tension, emphasize key moments, or transition between story sections
  * Count characters carefully: after every 2000 characters of narration text, you MUST include a break
- Style: Informative yet engaging. Blend facts with storytelling. Use natural pauses (...), varied pacing, and smooth transitions
- Tone: Authoritative but accessible - like a knowledgeable expert sharing a fascinating story to someone who's never heard it before
- Be educational and explanatory - prioritize clarity and understanding over brevity
- Provide context continuously - don't just state facts, explain them
- Make it entertaining without sacrificing accuracy - facts should be compelling on their own
- One continuous script, approximately {target_characters} characters (900 characters per minute of video)

CRITICAL OUTPUT REQUIREMENTS:
- Output ONLY the script text itself
- Include ONLY: dialogue/narration text, [MUSICAL BREAK], and [VISUAL BREAK]
- NO labels, NO instructions, NO explanations, NO section headers, NO formatting markers
- NO text like "SCRIPT:", "NARRATION:", "DIALOGUE:", etc.
- The output will be read directly by text-to-speech, so any extra text will be spoken as dialogue
- Start immediately with the first word of narration, end with the last word

Provide ONLY the script text:"""
    
    try:
        # Call Responses API
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Expert documentary scriptwriter. Write informative, historically accurate scripts that tell complete stories in an engaging way. Structure content with hook, introduction, narrative, climax, conclusion, and impact. Blend factual accuracy with compelling storytelling - like BBC or National Geographic documentaries. Be authoritative yet accessible, informative yet entertaining. CRITICAL: Assume the viewer knows NOTHING about the topic. Provide extensive context, background, and explanations throughout. Explain who people are, what terms mean, when/where events occurred, why they happened, and the historical/cultural context. Don't assume prior knowledge - explain everything clearly and thoroughly. CRITICAL BREAK REQUIREMENT: You MUST include a [MUSICAL BREAK] or [VISUAL BREAK] after no more than every 2000 characters of narration text. Count characters carefully and ensure breaks occur regularly to maintain pacing. Include strategic musical breaks marked with [MUSICAL BREAK] or [VISUAL BREAK] where narration stops for 2-4 seconds to let visuals and music shine. These breaks should flow naturally - place them after dramatic moments, before transitions, or during visually stunning scenes, but ALWAYS ensure they occur at least every 2000 characters. Cover the full story comprehensively with rich context and explanations. CRITICAL: Output ONLY the script text - dialogue/narration, [MUSICAL BREAK], and [VISUAL BREAK] markers only. NO labels, NO instructions, NO explanations. The output will be read directly by text-to-speech, so any extra text will be spoken as dialogue."},
                {"role": "user", "content": script_prompt}
            ],
            max_output_tokens=max_tokens,
            temperature=1
        )
        
        script = response.output_text.strip()
        
        # Clean script to ensure it only contains dialogue, [MUSICAL BREAK], and [VISUAL BREAK]
        script = clean_script_for_tts(script)
        
        print(f"✅ Script generated successfully")
        print(f"   Script length: {len(script)} characters")
        
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


def segment_script_by_narration(script, audio_path, segment_duration=12.0, api_key=None, expected_num_segments=None):
    """
    Segment a script into segments based on narration audio timing.
    Uses Whisper API to get word-level timestamps and groups words into 12-second segments.
    This ensures segments align with actual narration timing rather than word count.
    
    Args:
        script: The complete overarching script
        audio_path: Path to the narration audio file
        segment_duration: Duration of each segment in seconds (default: 12.0)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        expected_num_segments: Expected number of segments (if provided, limits output to this many)
        
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
        print(f"🎤 Transcribing audio with Whisper for {segment_duration}-second segmentation...")
        
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
        
        print(f"✅ Transcribed {len(words)} words with timestamps")
        
        # Find the total duration of the audio
        total_duration = 0
        for word_data in words:
            if isinstance(word_data, dict):
                end = word_data.get('end', 0)
            else:
                end = getattr(word_data, 'end', 0)
            total_duration = max(total_duration, end)
        
        print(f"   Audio duration: {total_duration:.2f}s")
        
        # Calculate number of segments needed
        # If expected_num_segments is provided, use that (narration should match video duration)
        # Otherwise, calculate based on audio duration
        if expected_num_segments is not None:
            num_segments_needed = expected_num_segments
            print(f"   Using expected number of segments: {num_segments_needed}")
            # Calculate expected duration based on segments
            expected_duration = num_segments_needed * segment_duration
            print(f"   Expected narration duration: {expected_duration:.2f}s (for {num_segments_needed} segments)")
        else:
            # Calculate based on actual audio duration
            num_segments_needed = int((total_duration + segment_duration - 0.1) / segment_duration)  # Round up
            print(f"   Calculated segments from audio duration: {num_segments_needed}")
        
        # Group words into 12-second segments based on timestamps
        # Use a more efficient approach: iterate through words once and assign them to segments
        segments = []
        
        # Initialize empty segments (only create the expected number)
        segments = [""] * num_segments_needed
        
        # Assign each word to the appropriate segment based on its start time
        # CRITICAL: Assign ALL words - every word must be assigned to a segment
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
            # Simply divide start time by segment duration
            segment_index = int(start / segment_duration)
            
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
        
        print(f"✅ Segmented script into {len(segments)} {segment_duration}-second segments based on narration timing")
        print(f"   Total audio duration: {total_duration:.2f}s")
        print(f"   Words assigned to segments: {words_assigned}/{len(words)}")
        if expected_num_segments:
            print(f"   Expected segments: {expected_num_segments}, Created: {len(segments)}")
        
        return segments
        
    except Exception as e:
        print(f"⚠️  Whisper transcription failed: {e}")
        print("   Falling back to rule-based segmentation...")
        estimated_duration = len(script.split()) * 0.5
        num_segments = max(1, int(estimated_duration / segment_duration))
        return segment_script_rule_based(script, num_segments)


def convert_segment_to_sora_prompt(
    segment_text,
    segment_id,
    segment_duration,
    total_duration,
    overarching_script=None,
    previous_prompt=None,
    next_segment_text=None,
    reference_image_info=None,
    still_image_segments=None,
    api_key=None,
    model='gpt-5-2025-08-07',
    max_tokens=20000,
    total_segments=None
):
    """
    Convert a segment script text into a Sora-2 video generation prompt using AI.
    Includes full script context and previous segment for chronological continuity.
    Accounts for still image gaps in timing calculations.
    
    Args:
        segment_text: The script text for this segment
        segment_id: Segment number (1-indexed)
        segment_duration: Duration of this segment in seconds
        total_duration: Total video duration in seconds
        overarching_script: The full overarching script (for context and narrative flow)
        previous_prompt: The Sora prompt from the previous segment (for visual continuity)
        next_segment_text: The script text for the next segment (for forward continuity)
        reference_image_info: Dict with 'type' ('character' or 'subject') and 'description' of reference image
        still_image_segments: List of still image segment info dicts (with 'segment_id' indicating position)
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-5.2-2025-12-11')
        max_tokens: Maximum tokens for the response (default: 2500)
        
    Returns:
        Sora-2 video generation prompt as a string
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    # Calculate timing accounting for still image gaps
    # Each still image adds 12 seconds before subsequent segments
    STILL_IMAGE_DURATION = 12.0
    still_image_offset = 0.0
    
    if still_image_segments:
        # Calculate total still image duration before this segment
        # Still images are placed AFTER certain segment IDs (e.g., after segment 3, 6, 9)
        # Opening still image (segment_id=0) is at the beginning
        for seg_info in still_image_segments:
            still_seg_id = seg_info.get('segment_id', -1)
            if still_seg_id == 0:
                # Opening still image: adds 12s at the beginning
                still_image_offset += STILL_IMAGE_DURATION
            elif still_seg_id > 0 and still_seg_id < segment_id:
                # Still image after segment still_seg_id: adds 12s before segment segment_id
                still_image_offset += STILL_IMAGE_DURATION
    
    start_time = (segment_id - 1) * segment_duration + still_image_offset
    end_time = segment_id * segment_duration + still_image_offset
    
    # Build context sections
    context_sections = []
    
    if previous_prompt:
        # Truncate previous prompt if too long to avoid token limit issues
        # Keep last 1000 chars for narrative context only (NOT for visual continuity)
        max_prev_prompt_length = 1000
        if len(previous_prompt) > max_prev_prompt_length:
            truncated_prev = "..." + previous_prompt[-max_prev_prompt_length:]
        else:
            truncated_prev = previous_prompt
        
        context_sections.append(f"""PREVIOUS SEGMENT PROMPT (for narrative context only - NOT for visual continuity): {truncated_prev}""")

    if next_segment_text:
        context_sections.append(f"""NEXT SEGMENT SCRIPT (for narrative context only):
{next_segment_text}

NOTE: This is provided ONLY for narrative context to understand where the story is heading. The current segment must be a STANDALONE video. Do NOT try to visually connect to the next segment - each segment is generated independently.""")
    
    if reference_image_info:
        ref_type = reference_image_info.get('type', 'subject')
        ref_desc = reference_image_info.get('description', 'the main visual element')
        if ref_type == 'character':
            context_sections.append(f"""═══════════════════════════════════════════════════════════════════════════════
CRITICAL - REFERENCE IMAGE FOR CHARACTER MATCHING:
A reference image will be provided showing the main character: {ref_desc}

ABSOLUTELY CRITICAL - EXACT CHARACTER MATCHING REQUIRED:
- The character in the generated video MUST be THE EXACT SAME PERSON as shown in the reference image
- This is NOT a look-alike, similar person, or someone with matching features - it MUST be THE EXACT SAME INDIVIDUAL
- Every facial feature, body type, hair, clothing style, and physical characteristic must match EXACTLY
- The character's face, build, posture, and appearance must be IDENTICAL to the reference image
- When describing the character in your Sora prompt, you MUST emphasize that this is the EXACT SAME PERSON from the reference image
- Use phrases like "the exact same person as shown in the reference image", "identical to the reference character", "the precise individual from the reference image"
- Do NOT describe the character as "similar to" or "resembling" - it must be THE SAME PERSON
- This is the SINGLE MOST IMPORTANT requirement for character-based reference images
═══════════════════════════════════════════════════════════════════════════════""")
        else:
            context_sections.append(f"""REFERENCE IMAGE CONTEXT:
A reference image will be provided showing: {ref_desc}

IMPORTANT: The video must maintain visual consistency with this reference image. The main visual elements, style, and atmosphere should align with the reference image throughout all segments.""")
    
    context_text = "\n\n".join(context_sections) if context_sections else ""
    
    # Check if this segment contains musical breaks or is opening/closing
    is_opening_segment = segment_id == 1
    # Determine if this is the closing segment
    if total_segments:
        is_closing_segment = segment_id >= total_segments
    else:
        # Fallback: estimate based on duration
        estimated_segments = int(total_duration / segment_duration) + 1
        is_closing_segment = segment_id >= estimated_segments - 0.5
    
    # Build requirements based on segment type
    # MOST IMPORTANT: Script matching must be the top priority
    visual_requirements = []
    # Add character matching requirement FIRST if reference image is character-based
    if reference_image_info and reference_image_info.get('type') == 'character':
        visual_requirements.append("- ═══════════════════════════════════════════════════════════════════════════════")
        visual_requirements.append("- SINGLE MOST IMPORTANT REQUIREMENT - EXACT CHARACTER MATCHING:")
        visual_requirements.append("- The character shown in this video MUST be THE EXACT SAME PERSON as in the reference image")
        visual_requirements.append("- This is NOT a look-alike or similar person - it MUST be IDENTICAL to the reference image")
        visual_requirements.append("- Every facial feature, body type, hair, clothing, and physical characteristic must match EXACTLY")
        visual_requirements.append("- In your prompt, explicitly state that the character is 'the exact same person as shown in the reference image'")
        visual_requirements.append("- Use phrases like 'identical to the reference character', 'the precise individual from the reference image'")
        visual_requirements.append("- Do NOT use words like 'similar', 'resembling', 'looks like' - use 'is the same person', 'identical', 'exact match'")
        visual_requirements.append("- ═══════════════════════════════════════════════════════════════════════════════")
    visual_requirements.append("- MOST CRITICAL: The video MUST make perfect sense with what the script narration is saying during this segment. Every visual element must directly correspond to and support the narration.")
    visual_requirements.append("- ABSOLUTELY CRITICAL: The video MUST be PHOTOREALISTIC and look like REAL LIFE. It must appear as if filmed by a professional documentary camera with natural lighting, realistic textures, authentic details, and genuine photographic quality. Use terms like 'photorealistic', 'hyperrealistic', 'documentary-style', 'as if filmed by a professional camera', 'real-life footage', 'authentic', 'natural lighting', 'realistic textures' to ensure Sora generates real-world appearance.")
    visual_requirements.append("- NEVER use artistic, stylized, illustrative, animated, or CGI-like descriptions. Always emphasize that this is real-world footage.")
    visual_requirements.append("- Briefly include camera movement, angle, lighting, mood")
    visual_requirements.append("- Be concise and cinematic")
    # Maximum scenes/cuts per segment (but first and last must be continuous)
    if is_opening_segment or is_closing_segment:
        visual_requirements.append("- MOST CRITICAL: This shot must be LONG and can have slow, smooth camera movement, but NO cuts or transitions. Hold the same scene throughout the entire segment. Let the visual breathe before narration begins.")
    else:
        visual_requirements.append("- MOST CRITICAL: The video MUST have at most ONE cut. There should be at the MOST one or two scenes in the video segment.")
        visual_requirements.append(f"- ABSOLUTELY CRITICAL - NO QUICK CUTS: If there are 2 shots in this {segment_duration:.1f}-second segment, each shot MUST be at least {segment_duration * 0.4:.1f} seconds long (preferably {segment_duration/2:.1f} seconds each). NO shots shorter than {segment_duration * 0.4:.1f} seconds. NO rapid cuts or transitions.")
        visual_requirements.append("- DURATION REQUIREMENT: Each shot/scene must have adequate duration to be visually coherent. Quick cuts (shots shorter than 3-4 seconds) are ABSOLUTELY FORBIDDEN.")
    
    # Each segment (except the first and last) should start with a new scene/cut
        
    visual_requirements.append(f"- Optimized for a {segment_duration:.1f}s clip")
    
    requirements_text = "\n".join(visual_requirements)
    
    # Validate segment_text is provided and not empty
    if not segment_text or len(segment_text.strip()) == 0:
        raise ValueError(f"Segment {segment_id} text is empty! Cannot generate Sora prompt.")
    
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
4. If 2 shots are suggested, each shot must be approximately 6 seconds long (for a 12-second segment) - NO quick cuts allowed

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
            max_output_tokens=200,
            temperature=0.7
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
        
        if key_words_phrases:
            print(f"   📌 Extracted key phrases: {', '.join(key_words_phrases)}")
            print(f"   🎬 Number of shots: {num_shots}")
    except Exception as e:
        print(f"   ⚠️  Key phrase extraction failed: {e}, using default approach")
        key_words_phrases = None
        num_shots = 1
    
    # Build key phrase instructions for the prompt
    key_phrase_instructions = ""
    if key_words_phrases and len(key_words_phrases) > 0:
        if num_shots == 1:
            # Single shot - focus on the primary key phrase(s)
            primary_phrase = key_words_phrases[0] if key_words_phrases else "the main subject"
            key_phrase_instructions = f"""
═══════════════════════════════════════════════════════════════════════════════
CRITICAL VISUAL FOCUS - KEY PHRASE ANALYSIS:
The narration segment contains these key visual elements: {', '.join(key_words_phrases)}

ABSOLUTELY CRITICAL: This segment should be 1 CONTINUOUS SHOT focused on: {primary_phrase}
- The entire video must visually center around and emphasize "{primary_phrase}"
- Every visual element should relate to and support showing "{primary_phrase}"
- The camera should focus on "{primary_phrase}" throughout the entire shot
- If multiple key phrases are mentioned, prioritize "{primary_phrase}" as the primary visual focus
- The shot should be continuous with no cuts - maintain focus on "{primary_phrase}" for the full duration
═══════════════════════════════════════════════════════════════════════════════"""
        else:
            # Two shots - first shot on first phrase, second shot on second phrase
            first_phrase = key_words_phrases[0] if len(key_words_phrases) > 0 else "the first main subject"
            second_phrase = key_words_phrases[1] if len(key_words_phrases) > 1 else (key_words_phrases[0] if len(key_words_phrases) > 0 else "the second main subject")
            # Calculate minimum duration per shot (at least 5 seconds each for a 12-second segment)
            min_shot_duration = max(5.0, segment_duration * 0.4)  # At least 40% of segment duration, minimum 5 seconds
            key_phrase_instructions = f"""
═══════════════════════════════════════════════════════════════════════════════════
CRITICAL VISUAL FOCUS - KEY PHRASE ANALYSIS:
The narration segment contains these key visual elements: {', '.join(key_words_phrases)}

ABSOLUTELY CRITICAL: This segment should be 2 SEPARATE SHOTS with EQUAL DURATION:
- FIRST SHOT: Focus entirely on "{first_phrase}"
  * The first part of the video must visually center around and emphasize "{first_phrase}"
  * Every visual element in the first shot should relate to and support showing "{first_phrase}"
  * The camera should focus on "{first_phrase}" throughout the first shot
  * DURATION: This first shot MUST last for approximately {segment_duration/2:.1f} seconds (half of the {segment_duration:.1f}-second segment)
  
- SECOND SHOT: Focus entirely on "{second_phrase}"
  * The second part of the video must visually center around and emphasize "{second_phrase}"
  * Every visual element in the second shot should relate to and support showing "{second_phrase}"
  * The camera should focus on "{second_phrase}" throughout the second shot
  * DURATION: This second shot MUST last for approximately {segment_duration/2:.1f} seconds (half of the {segment_duration:.1f}-second segment)
  
- CRITICAL DURATION REQUIREMENT: Each shot MUST be approximately {segment_duration/2:.1f} seconds long. NO quick cuts. NO shots shorter than {min_shot_duration:.1f} seconds.
- There should be ONE clear cut/transition between the two shots at approximately {segment_duration/2:.1f} seconds
- ABSOLUTELY NO quick cuts or rapid transitions - each shot must have adequate duration to be visually coherent
- The cut between shots should be smooth and natural, not jarring or abrupt
═══════════════════════════════════════════════════════════════════════════════════"""
    
    # Create prompt to convert segment script to Sora video prompt
    # CRITICAL: Put segment_text FIRST and make it the primary focus
    conversion_prompt = f"""Create a Sora-2 video prompt for segment {segment_id} ({start_time:.1f}-{end_time:.1f}s) of a {total_duration}s video.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: Here is the whole script for this video(just for narrative context):
{overarching_script}

CRITICAL: Here is the portion of the script for which a video must be generated. You must generate a photorealistic video based on the key idea of this segment of the script:
{segment_text}
{key_phrase_instructions}

CRITICAL: The prompt you generate should never tell sora 2 to generate diagrams or words. All of the shots must be a natural scene that could come legitimately from a camera.
CRITICAL: ABSOLUTELY CRITICAL - The video MUST be PHOTOREALISTIC and look like REAL LIFE. It must appear as if filmed by a professional documentary camera. Use terms like 'photorealistic', 'hyperrealistic', 'documentary-style', 'as if filmed by a professional camera', 'real-life footage', 'authentic', 'natural lighting', 'realistic textures' to ensure Sora generates real-world appearance. NEVER use artistic, stylized, illustrative, animated, or CGI-like descriptions.
CRITICAL: Do not ever describe the music or sound effects the video should have. Just describe the scene.

Context:
{context_text}

Requirements:
{requirements_text}

═══════════════════════════════════════════════════════════════════════════════
FINAL REMINDER: The video MUST make perfect sense with the script narration above.
If the script describes a main location, the video MUST show that location.
If the script mentions an action, the video MUST show that action.
The video segment MUST be 1 or 2 different scenes or shots.

ABSOLUTELY CRITICAL - NO QUICK CUTS:
- If the video has 1 shot: It must be a continuous {segment_duration:.1f}-second shot with NO cuts
- If the video has 2 shots: Each shot MUST be approximately {segment_duration/2:.1f} seconds long. NO shots shorter than {segment_duration * 0.4:.1f} seconds. The cut must occur at approximately {segment_duration/2:.1f} seconds into the segment
- ABSOLUTELY FORBIDDEN: Quick cuts, rapid transitions, or shots shorter than {segment_duration * 0.4:.1f} seconds
- Each shot must have adequate duration to be visually coherent and meaningful

ABSOLUTELY CRITICAL: The video MUST be PHOTOREALISTIC and look like REAL LIFE - as if filmed by a professional documentary camera with natural lighting, realistic textures, and authentic details. Use terms like 'photorealistic', 'hyperrealistic', 'documentary-style', 'real-life footage' in your prompt.
{f"═══════════════════════════════════════════════════════════════════════════════\nSINGLE MOST IMPORTANT REQUIREMENT - EXACT CHARACTER MATCHING:\nIf a character reference image is provided, the character in this video MUST be THE EXACT SAME PERSON as shown in the reference image. This is NOT a look-alike - it MUST be IDENTICAL. Every feature must match EXACTLY. In your prompt, explicitly state the character is the exact same person from the reference image. Use phrases like 'identical to the reference character', 'the precise individual from the reference image'. Do NOT use 'similar' or 'resembling' - use 'is the same person', 'identical', 'exact match'.\n═══════════════════════════════════════════════════════════════════════════════\n" if (reference_image_info and reference_image_info.get('type') == 'character') else ""}Do not ever describe the music or sound effects the video should have. Just describe the scene.
═══════════════════════════════════════════════════════════════════════════════

Provide ONLY the Sora-2 prompt (no labels):"""
    
    # Retry logic: try up to 3 times to get a valid prompt
    max_retries = 3
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                # Show input only on first attempt
                print(f"\n📥 INPUT (Segment {segment_id}):")
                print(f"   Segment text: {segment_text[:150]}{'...' if len(segment_text) > 150 else ''}")
                print(f"   Previous prompt length: {len(previous_prompt) if previous_prompt else 0} chars")
                print(f"   Next segment text length: {len(next_segment_text) if next_segment_text else 0} chars")
                print(f"\n   Full API Prompt ({len(conversion_prompt)} chars):")
                print("   " + "="*70)
                # Show the full conversion prompt (truncated if too long)
                prompt_preview = conversion_prompt[:800] + ("\n   ... [truncated]" if len(conversion_prompt) > 800 else "")
                for line in prompt_preview.split('\n'):
                    print(f"   {line}")
                print("   " + "="*70)
            
            # Call Responses API
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": f"Professional Sora 2 Video Prompter. Prompt Sora 2 to create detailed cinematic prompts matching script narration. CRITICAL: ALL videos must be ABSOLUTELY PHOTOREALISTIC - they must look like real-life footage captured by a professional documentary camera. Every video must appear as if it were filmed in real life, with natural lighting, realistic textures, authentic details, and genuine photographic quality. Never use artistic, stylized, or illustrative styles. Always emphasize hyperrealism, photorealistic quality, and documentary-style authenticity. {'ABSOLUTELY CRITICAL - CHARACTER REFERENCE IMAGE: When a character-based reference image is provided, the character in the generated video MUST be THE EXACT SAME PERSON as shown in the reference image. This is NOT a look-alike or similar person - it MUST be IDENTICAL. Every facial feature, body type, hair, clothing, and physical characteristic must match EXACTLY. In your prompts, explicitly state that the character is the exact same person from the reference image. Use phrases like identical to the reference character, the precise individual from the reference image. Do NOT use words like similar, resembling, looks like - use is the same person, identical, exact match. This is the SINGLE MOST IMPORTANT requirement when a character reference image is provided.' if (reference_image_info and reference_image_info.get('type') == 'character') else ''} Always provide complete prompts."},
                    {"role": "user", "content": conversion_prompt}
                ],
                max_output_tokens=max_tokens,
                temperature=1
            )
            
            # Validate response structure
            if not hasattr(response, 'output_text'):
                raise Exception("API response missing 'output_text' attribute")
            
            # Get raw content
            raw_content = response.output_text
            if raw_content is None:
                raise Exception("API returned None for content")
            
            original_prompt = raw_content.strip()
            
            # Show raw output
            if attempt == 1:
                print(f"\n📤 RAW OUTPUT:")
                print(f"   Length: {len(original_prompt)} characters")
                print(f"   Content: {original_prompt[:200]}{'...' if len(original_prompt) > 200 else ''}")
            
            sora_prompt = original_prompt
            
            # Clean up any labels or formatting
            labels_to_remove = [
                f"Segment {segment_id}:",
                f"Segment {segment_id}",
                "Sora-2 Prompt:",
                "Sora Prompt:",
                "Prompt:",
                "Video Prompt:"
            ]
            
            for label in labels_to_remove:
                if sora_prompt.startswith(label):
                    sora_prompt = sora_prompt[len(label):].strip()
            
            # Additional cleanup: remove common prefixes
            sora_prompt = sora_prompt.lstrip(":- ").strip()
            
            # Show cleaned output
            if attempt == 1:
                print(f"\n🧹 CLEANED OUTPUT:")
                print(f"   Length: {len(sora_prompt)} characters")
                print(f"   Content: {sora_prompt[:200]}{'...' if len(sora_prompt) > 200 else ''}")
            
            # Validate that prompt is not empty
            if not sora_prompt or len(sora_prompt.strip()) == 0:
                if attempt < max_retries:
                    last_error = Exception(f"Generated prompt was empty after cleaning (attempt {attempt})")
                    time.sleep(1)
                    continue
                else:
                    # Last attempt failed - use fallback
                    if segment_text and len(segment_text.strip()) > 0:
                        sora_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
                    else:
                        sora_prompt = f"Photorealistic documentary-style video scene for segment {segment_id}, as if filmed by a professional camera with natural lighting and realistic textures"
                    print(f"\n❌ FAILED: All {max_retries} attempts returned empty content")
                    print(f"✅ Using fallback prompt based on segment text")
                    return sora_prompt
            
            # Success
            print(f"\n✅ SUCCESS: Prompt generated ({len(sora_prompt)} chars)")
            return sora_prompt
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            error_type = type(e).__name__
            
            if attempt == 1:
                print(f"\n❌ API ERROR: {error_type}: {error_msg}")
                # Check for common issues
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    print(f"   ⚠️  Rate limit detected - will retry with longer delay")
                elif "token" in error_msg.lower() or "length" in error_msg.lower() or "context_length" in error_msg.lower():
                    print(f"   ⚠️  Token/length issue - prompt may be too long ({len(conversion_prompt)} chars)")
                    print(f"   ⚠️  Previous prompt length: {len(previous_prompt) if previous_prompt else 0} chars")
                elif "timeout" in error_msg.lower():
                    print(f"   ⚠️  Timeout detected - will retry")
            
            if attempt < max_retries:
                # Increase delay on retries, especially for rate limits
                delay = 3 if "rate limit" in error_msg.lower() or "429" in error_msg else 2
                time.sleep(delay)
                continue
            else:
                # All retries exhausted - use fallback
                if segment_text and len(segment_text.strip()) > 0:
                    sora_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
                else:
                    sora_prompt = f"Photorealistic documentary-style video scene for segment {segment_id}, as if filmed by a professional camera with natural lighting and realistic textures"
                print(f"\n❌ FAILED: All {max_retries} attempts failed with errors ({error_type})")
                print(f"✅ Using fallback prompt based on segment text")
                return sora_prompt
    
    # Should never reach here, but just in case
    if segment_text and len(segment_text.strip()) > 0:
        return f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
    else:
        return f"Photorealistic documentary-style video scene for segment {segment_id}, as if filmed by a professional camera with natural lighting and realistic textures"


def generate_sora_prompts_from_segments(
    segment_texts,
    segment_duration,
    total_duration,
    overarching_script=None,
    reference_image_info=None,
    still_image_segments=None,
    api_key=None,
    model='gpt-5-2025-08-07',
    max_tokens=20000
):
    """
    STEP 3: Convert multiple segment scripts into Sora-2 video prompts using AI calls.
    This is the second separate API call (after script generation).
    Includes full script context and previous segment prompts for chronological continuity.
    Accounts for still image gaps in timing calculations.
    
    Args:
        segment_texts: List of segment script texts
        segment_duration: Duration of each segment in seconds
        total_duration: Total video duration in seconds
        overarching_script: The full overarching script (for context and narrative flow)
        reference_image_info: Dict with 'type' and 'description' of reference image
        still_image_segments: List of still image segment info dicts (with 'segment_id' indicating position)
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-5.2-2025-12-11')
        max_tokens: Maximum tokens per response (default: 500)
        
    Returns:
        List of Sora-2 video generation prompts (one per segment)
    """
    sora_prompts = []
    
    # Validate that we have the correct number of segments
    if len(segment_texts) == 0:
        raise ValueError("No segment texts provided for Sora prompt generation")
    
    print(f"📋 Processing {len(segment_texts)} segment(s) for Sora prompt generation")
    print(f"   Total script length: {sum(len(seg) for seg in segment_texts)} characters")
    
    for i, segment_text in enumerate(segment_texts, 1):
        # Validate segment text is not empty
        if not segment_text or len(segment_text.strip()) == 0:
            raise ValueError(f"Segment {i} text is empty! Cannot generate Sora prompt.")
        
        # Calculate expected time range for this segment
        # Account for still image gaps: each still image adds 12 seconds before subsequent segments
        STILL_IMAGE_DURATION = 12.0
        still_image_offset = 0.0
        
        if still_image_segments:
            # Calculate total still image duration before this segment
            # Still images are placed AFTER certain segment IDs (e.g., after segment 3, 6, 9)
            # Opening still image (segment_id=0) is at the beginning
            for seg_info in still_image_segments:
                still_seg_id = seg_info.get('segment_id', -1)
                if still_seg_id == 0:
                    # Opening still image: adds 12s at the beginning
                    still_image_offset += STILL_IMAGE_DURATION
                elif still_seg_id > 0 and still_seg_id < i:
                    # Still image after segment still_seg_id: adds 12s before segment i
                    still_image_offset += STILL_IMAGE_DURATION
        
        start_time = (i - 1) * segment_duration + still_image_offset
        end_time = i * segment_duration + still_image_offset
        
        print(f"\n{'='*60}")
        print(f"Converting segment {i}/{len(segment_texts)} script to Sora-2 prompt...")
        print(f"   Time range: {start_time:.1f}s - {end_time:.1f}s (accounting for {still_image_offset:.1f}s of still images)")
        print(f"   Segment text length: {len(segment_text)} characters")
        print(f"   Segment text preview: {segment_text[:150]}{'...' if len(segment_text) > 150 else ''}")
        print(f"   Segment text ending: ...{segment_text[-100:] if len(segment_text) > 100 else segment_text}")
        
        # Verify this segment is different from previous segments
        if i > 1:
            prev_segment_text = segment_texts[i-2]  # Previous segment (i-2 because i is 1-indexed)
            if segment_text == prev_segment_text:
                raise ValueError(f"Segment {i} text is identical to segment {i-1}! Segmentation may have failed.")
            print(f"   ✅ Verified: Segment {i} is different from segment {i-1}")
        
        try:
            # Get previous prompt for continuity (if not first segment)
            previous_prompt = sora_prompts[-1] if sora_prompts else None
            
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
            
            # CRITICAL: Verify we're passing the correct segment text
            print(f"   🔍 Verifying segment text for segment {i}...")
            print(f"      Expected segment index: {i-1} (0-indexed)")
            print(f"      Actual segment text from list: {segment_text[:50]}...")
            
            sora_prompt = convert_segment_to_sora_prompt(
                segment_text=segment_text,  # This is the correct segment text for segment i
                segment_id=i,
                segment_duration=segment_duration,
                total_duration=total_duration,
                overarching_script=overarching_script,
                previous_prompt=previous_prompt,
                next_segment_text=next_segment_text,
                reference_image_info=reference_image_info,
                still_image_segments=still_image_segments,  # Pass still image info for timing
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                total_segments=len(segment_texts)
            )
            
            # Validate prompt is not empty (should not happen due to retry logic, but double-check)
            if not sora_prompt or len(sora_prompt.strip()) == 0:
                # This should not happen due to retry logic, but if it does, use fallback
                print(f"  ⚠️  Warning: Prompt validation failed for segment {i}, using fallback...")
                if segment_text and len(segment_text.strip()) > 0:
                    sora_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
                elif overarching_script and len(overarching_script.strip()) > 0:
                    sora_prompt = f"Photorealistic documentary-style video scene, as if filmed by a professional camera with natural lighting and realistic textures: {overarching_script[:300]}"
                else:
                    sora_prompt = f"Photorealistic documentary-style video scene for segment {i}, as if filmed by a professional camera with natural lighting and realistic textures"
            
            sora_prompts.append(sora_prompt)
            print(f"  ✅ Segment {i} Sora prompt generated ({len(sora_prompt)} characters)")
        except Exception as e:
            print(f"  ⚠️  Failed to convert segment {i} to Sora prompt after retries: {e}")
            # Fallback: use a generic prompt based on the segment text
            if segment_text and len(segment_text.strip()) > 0:
                fallback_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
            else:
                # If segment text is also empty, use the overarching script or original prompt
                if overarching_script and len(overarching_script.strip()) > 0:
                    fallback_prompt = f"Photorealistic documentary-style video scene, as if filmed by a professional camera with natural lighting and realistic textures: {overarching_script[:300]}"
                else:
                    fallback_prompt = f"Photorealistic documentary-style video scene for segment {i}, as if filmed by a professional camera with natural lighting and realistic textures"
            sora_prompts.append(fallback_prompt)
            print(f"  ⚠️  Using fallback prompt for segment {i}: {fallback_prompt[:100]}...")
    
    # Final validation: ensure we have the correct number of prompts
    if len(sora_prompts) != len(segment_texts):
        raise ValueError(f"Mismatch: Generated {len(sora_prompts)} prompts but expected {len(segment_texts)} segments!")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Successfully generated {len(sora_prompts)} Sora prompts from {len(segment_texts)} script segments")
    print(f"{'='*60}")
    for i, (seg_text, sora_prompt) in enumerate(zip(segment_texts, sora_prompts), 1):
        start_time = (i - 1) * segment_duration
        end_time = i * segment_duration
        print(f"   Segment {i} ({start_time:.1f}s-{end_time:.1f}s):")
        print(f"      Script: {seg_text[:60]}{'...' if len(seg_text) > 60 else ''}")
        print(f"      Sora prompt: {sora_prompt[:60]}{'...' if len(sora_prompt) > 60 else ''}")
    
    return sora_prompts



def generate_voiceover_from_folder(
    script,
    output_path=None,
    narration_folder=None,
    break_duration=1000,
    music_volume=0.07):
    """
    Generate voiceover audio by stitching together narration files from a folder.
    Looks for files named narration_0, narration_1, narration_2, etc. in the specified folder,
    stitches them together in order, adds breaks based on script markers, and mixes with music.
    
    Args:
        script: The script text (used to determine break positions)
        output_path: Path to save the final audio file (default: temp file)
        narration_folder: Folder containing narration files (default: 'narration_segments' in current directory)
        break_duration: Duration of silence for breaks in milliseconds (default: 3000ms = 3 seconds)
        music_volume: Volume of background music relative to voiceover (0.0-1.0) (default: 0.07, 7%)
        
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
                "pydub library is required for audio stitching. Install with: pip install pydub"
            )
    
    # Determine narration folder
    if narration_folder is None:
        narration_folder = os.path.join(os.getcwd(), "narration_segments")
    
    # Create folder if it doesn't exist
    if not os.path.exists(narration_folder):
        os.makedirs(narration_folder)
        print(f"📁 Created narration folder: {narration_folder}")
        print(f"   Please add your narration files as: narration_0.mp3, narration_1.mp3, narration_2.mp3, etc.")
        raise FileNotFoundError(
            f"Narration folder is empty: {narration_folder}\n"
            f"Please add your narration files named: narration_0.mp3, narration_1.mp3, narration_2.mp3, etc."
        )
    
    # Determine output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        output_path = os.path.join(temp_dir, f"voiceover_stitched_{timestamp}.mp3")
    
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    
    try:
        # Step 1: Find all narration files in order
        print(f"📂 Looking for narration files in: {narration_folder}")
        
        # Find all narration files (narration_0, narration_1, etc.)
        narration_files = []
        i = 0
        while True:
            # Try various extensions
            for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']:
                file_path = os.path.join(narration_folder, f"narration_{i}{ext}")
                if os.path.exists(file_path):
                    narration_files.append({
                        'path': file_path,
                        'index': i
                    })
                    break
            else:
                # No file found for this index
                if i == 0:
                    raise FileNotFoundError(
                        f"No narration files found in {narration_folder}.\n"
                        f"Expected files: narration_0.mp3, narration_1.mp3, etc."
                    )
                # We've reached the end
                break
            i += 1
        
        if not narration_files:
            raise FileNotFoundError(
                f"No narration files found in {narration_folder}.\n"
                f"Expected files: narration_0.mp3, narration_1.mp3, etc."
            )
        
        print(f"✅ Found {len(narration_files)} narration files")
        for nf in narration_files:
            file_size = os.path.getsize(nf['path']) / 1024
            print(f"   - {os.path.basename(nf['path'])} ({file_size:.1f} KB)")
        
        # Step 2: Split script by break markers to determine where breaks should go
        print("📝 Analyzing script for break markers...")
        break_positions = []
        
        # Find all break markers and their positions
        for match in re.finditer(r'\[(MUSICAL|VISUAL)\s+BREAK\]', script, re.IGNORECASE):
            break_positions.append((match.start(), match.end(), match.group(1).upper()))
        
        # Determine break positions relative to narration segments
        # Each narration file corresponds to one segment (before a break or at the end)
        num_segments = len(narration_files)
        break_types = []
        
        # Map breaks to segments (each break comes after a segment)
        for i in range(num_segments - 1):  # Last segment has no break after it
            if i < len(break_positions):
                break_types.append(break_positions[i][2])  # MUSICAL or VISUAL
            else:
                # If we have more segments than breaks, assume MUSICAL BREAK
                break_types.append('MUSICAL')
        
        print(f"✅ Script has {len(break_positions)} break markers")
        if break_types:
            print(f"   Break sequence: {' → '.join(break_types)}")
        
        # Step 3: Stitch narration files together with breaks
        print("🔗 Stitching narration files together...")
        final_audio = AudioSegment.empty()
        
        for i, narration_file in enumerate(narration_files):
            # Load narration audio
            print(f"   Loading {os.path.basename(narration_file['path'])}...")
            segment_audio = AudioSegment.from_file(narration_file['path'])
            final_audio += segment_audio
            
            # Add break (silence) after segment if not the last one
            if i < len(narration_files) - 1:
                break_type = break_types[i] if i < len(break_types) else 'MUSICAL'
                silence = AudioSegment.silent(duration=break_duration)
                final_audio += silence
                print(f"   Added {break_duration/1000:.1f}s {break_type} break")
        
        # Save voiceover-only file (before mixing with music)
        voiceover_only_path = os.path.join(temp_dir, f"voiceover_only_{timestamp}.mp3")
        final_audio.export(voiceover_only_path, format='mp3', bitrate='192k')
        print(f"✅ Stitched narration saved: {voiceover_only_path}")
        print(f"   Total duration: {len(final_audio) / 1000:.2f}s")
        
        # Step 4: Mix with music if available
        print("🎵 Mixing narration with background music...")
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
                            print(f"   ✅ Music synced to narration duration ({voiceover_duration:.2f}s)")
                        except Exception as e:
                            print(f"   ⚠️  Music sync failed: {e}")
                            print(f"   Using original music file")
                    else:
                        print(f"   ✅ Music duration already matches narration")
                    
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
                        print(f"✅ Final narration with music saved: {output_path}")
                        return output_path, voiceover_only_path
                    except Exception as e:
                        print(f"   ⚠️  Music mixing failed: {e}")
                        print(f"   Using voiceover-only audio")
                        import shutil
                        shutil.copy2(voiceover_only_path, output_path)
                        return output_path, voiceover_only_path
                else:
                    print(f"   ⚠️  Could not determine music duration, using voiceover-only")
                    import shutil
                    shutil.copy2(voiceover_only_path, output_path)
                    return output_path, voiceover_only_path
            else:
                print(f"   ⚠️  FFmpeg not found, using voiceover-only")
                import shutil
                shutil.copy2(voiceover_only_path, output_path)
                return output_path, voiceover_only_path
        else:
            print(f"   ⚠️  VIDEO_MUSIC.mp3 not found, using voiceover-only")
            import shutil
            shutil.copy2(voiceover_only_path, output_path)
            return output_path, voiceover_only_path
        
    except Exception as e:
        raise Exception(f"Failed to generate voiceover from folder: {e}")


def generate_voiceover_with_music(
    script,
    output_path=None,
    api_key=None,
    voice='echo',  # Deep, manly male voice
    music_style='cinematic',
    music_volume=0.07,  # 7% volume for background music
    duration=None,
    instructions="Use a very passionate and very exciting story telling style."): 
    """
    [DEPRECATED] Generate voiceover audio from script using OpenAI TTS, add background music with dynamic swells, and combine them.
    This function is deprecated. Use generate_voiceover_from_folder instead.
    
    Args:
        script: The script text to convert to voiceover
        output_path: Path to save the final audio file (default: temp file)
        api_key: OpenAI API key
        voice: TTS voice to use ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer') (default: 'onyx' - deep, manly male voice)
        music_style: Style of background music (default: 'cinematic', will be overridden by script analysis)
        music_volume: Base volume of background music relative to voiceover (0.0-1.0) (default: 0.07, 7%)
        duration: Expected duration in seconds (for music generation) (default: None, auto-detect)
        instructions: Instructions for voice characteristics (default: British documentary narration tone)
        
    Returns:
        Path to the final audio file with voiceover and music
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    import tempfile
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    # Determine output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        output_path = os.path.join(temp_dir, f"voiceover_with_music_{timestamp}.mp3")
    
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    voiceover_path = os.path.join(temp_dir, f"voiceover_{timestamp}.mp3")
    
    try:
        # Step 1: Process script to handle musical breaks
        # Remove [MUSICAL BREAK] and [VISUAL BREAK] markers and replace with natural pauses
        # This allows music/visuals to play while TTS pauses
        processed_script = script
        processed_script = processed_script.replace('[MUSICAL BREAK]', '... ... ...')  # Long pause for musical break
        processed_script = processed_script.replace('[VISUAL BREAK]', '... ... ...')  # Long pause for visual break
        
        # Step 2: Generate voiceover using OpenAI TTS
        print(f"Generating voiceover using TTS (voice: {voice})...")
        
        # Use TTS model - gpt-4o-mini-tts supports instructions, or use tts-1/tts-1-hd
        tts_model = "gpt-4o-mini-tts-2025-12-15"  # or "tts-1"/"tts-1-hd" per docs
        
        # Generate speech using the new API format
        response = client.audio.speech.create(
            model=tts_model,
            input=processed_script,
            voice=voice,
            instructions=instructions,  # only works on models that support it
            response_format="mp3"
        )
        
        # Write the response to file
        with open(voiceover_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Voiceover generated: {voiceover_path}")
        
        # Step 1.5: Get actual voiceover duration (may differ from estimated duration)
        actual_voiceover_duration = get_media_duration(voiceover_path, find_ffmpeg())
        if actual_voiceover_duration:
            print(f"   Actual voiceover duration: {actual_voiceover_duration:.2f}s")
            if duration and abs(actual_voiceover_duration - duration) > 0.5:
                print(f"   Note: Voiceover duration ({actual_voiceover_duration:.2f}s) differs from target ({duration:.2f}s)")
                print(f"   Audio will be synchronized with video duration during final mixing")
        
        # Save original voiceover path for later re-mixing (before any mixing occurs)
        original_voiceover_backup = None
        if voiceover_path and os.path.exists(voiceover_path):
            original_voiceover_backup = os.path.join(temp_dir, f"original_voiceover_backup_{timestamp}.mp3")
            import shutil
            shutil.copy2(voiceover_path, original_voiceover_backup)
        
        # Step 2: Load VIDEO_MUSIC.mp3 file from directory
        print("Loading VIDEO_MUSIC.mp3 file...")
        music_path = None
        original_music_path = None
        
        # Look for VIDEO_MUSIC.mp3 in current directory
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
                original_music_path = music_file
                file_size = os.path.getsize(music_file)
                print(f"✅ Found music file: {music_path} ({file_size / 1024:.1f} KB)")
                break
        
        if not music_path:
            print("⚠️  VIDEO_MUSIC.mp3 not found in directory")
            print(f"   Searched in: {current_dir}")
            print(f"   Continuing without music")
        
        # Step 3: Mix voiceover and music together
        ffmpeg_path = find_ffmpeg()
        if not ffmpeg_path:
            print("⚠️  FFmpeg not found. Saving voiceover as-is...")
            import shutil
            shutil.copy2(voiceover_path, output_path)
            # Cleanup temp files
            try:
                if os.path.exists(voiceover_path) and voiceover_path != output_path:
                    os.remove(voiceover_path)
                if music_path and os.path.exists(music_path):
                    os.remove(music_path)
            except:
                pass
            return output_path
        
        # Mix voiceover and music
        if music_path and os.path.exists(music_path):
            # Verify music file is valid
            try:
                music_file_size = os.path.getsize(music_path)
                if music_file_size == 0:
                    print(f"⚠️  Music file is empty, skipping music mixing")
                    music_path = None
                else:
                    print(f"✅ Music file is valid ({music_file_size / 1024:.1f} KB)")
            except Exception as e:
                print(f"⚠️  Cannot verify music file: {e}, skipping music mixing")
                music_path = None
        else:
            if not music_path:
                print(f"⚠️  No music path available, skipping music mixing")
            elif not os.path.exists(music_path):
                print(f"⚠️  Music file does not exist: {music_path}, skipping music mixing")
        
        if music_path and os.path.exists(music_path):
            print("Mixing voiceover and music...")
            # Get actual durations to ensure proper synchronization
            voiceover_duration = get_media_duration(voiceover_path, ffmpeg_path)
            music_duration_actual = get_media_duration(music_path, ffmpeg_path)
            
            if voiceover_duration and music_duration_actual:
                print(f"   Voiceover: {voiceover_duration:.2f}s, Music: {music_duration_actual:.2f}s")
                
                # Note: Music will be synced to video duration later (after video is generated)
                # For now, sync music to voiceover duration as initial estimate
                # Music will be re-synced to match video exactly in Step 2.6
                if abs(music_duration_actual - voiceover_duration) > 0.1:
                    print(f"   Adjusting music duration to match voiceover (will re-sync to video later)...")
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    timestamp = int(time.time())
                    adjusted_music_path = os.path.join(temp_dir, f"music_trimmed_{timestamp}.mp3")
                    
                    # Trim music to voiceover duration and add fade-out (1-2 seconds before end)
                    fade_out_start = max(0, voiceover_duration - 2)
                    fade_out_duration = min(2, voiceover_duration)
                    
                    cmd_adjust = [
                        ffmpeg_path,
                        "-i", music_path,
                        "-af", f"afade=t=out:st={fade_out_start}:d={fade_out_duration}",
                        "-t", str(voiceover_duration),
                        "-y",
                        adjusted_music_path
                    ]
                    
                    try:
                        subprocess.run(cmd_adjust, capture_output=True, text=True, check=True)
                        music_path = adjusted_music_path
                        print(f"   ✅ Music trimmed to match voiceover (temporary - will sync to video later)")
                    except subprocess.CalledProcessError as e:
                        print(f"   ⚠️  Music adjustment failed: {e.stderr}")
                        print(f"   Continuing with original music (will sync to video later)")
                else:
                    # Music duration is close, but ensure it has a fade-out
                    print(f"   Music duration matches, ensuring fade-out...")
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    timestamp = int(time.time())
                    faded_music_path = os.path.join(temp_dir, f"music_faded_{timestamp}.mp3")
                    
                    # Add fade-out to music (1-2 seconds before end)
                    fade_out_start = max(0, voiceover_duration - 2)
                    fade_out_duration = min(2, voiceover_duration)
                    
                    cmd_fade = [
                        ffmpeg_path,
                        "-i", music_path,
                        "-af", f"afade=t=out:st={fade_out_start}:d={fade_out_duration}",
                        "-t", str(voiceover_duration),
                        "-y",
                        faded_music_path
                    ]
                    
                    try:
                        subprocess.run(cmd_fade, capture_output=True, text=True, check=True)
                        # Only use faded version if it's different from original
                        if faded_music_path != music_path:
                            # Clean up original if it was a temp file
                            try:
                                if music_path != voiceover_path and os.path.exists(music_path):
                                    os.remove(music_path)
                            except:
                                pass
                            music_path = faded_music_path
                        print(f"   ✅ Music fade-out applied")
                    except subprocess.CalledProcessError as e:
                        print(f"   ⚠️  Music fade-out failed: {e.stderr}")
                        print(f"   Continuing without fade-out")
            else:
                # Could not determine durations, but still try to add fade-out to music
                # Use estimated duration or a safe default
                estimated_duration = duration or 10  # Fallback to provided duration or 10 seconds
                print(f"   ⚠️  Could not determine durations, using estimated duration: {estimated_duration}s")
                print(f"   Adding fade-out to music as safety measure...")
                
                import tempfile
                temp_dir = tempfile.gettempdir()
                timestamp = int(time.time())
                faded_music_path = os.path.join(temp_dir, f"music_faded_{timestamp}.mp3")
                
                fade_out_start = max(0, estimated_duration - 2)
                fade_out_duration = min(2, estimated_duration)
                
                cmd_fade = [
                    ffmpeg_path,
                    "-i", music_path,
                    "-af", f"afade=t=out:st={fade_out_start}:d={fade_out_duration}",
                    "-t", str(estimated_duration),
                    "-y",
                    faded_music_path
                ]
                
                try:
                    subprocess.run(cmd_fade, capture_output=True, text=True, check=True)
                    if faded_music_path != music_path:
                        try:
                            if music_path != voiceover_path and os.path.exists(music_path):
                                os.remove(music_path)
                        except:
                            pass
                        music_path = faded_music_path
                    print(f"   ✅ Music fade-out applied (estimated duration)")
                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️  Music fade-out failed: {e.stderr}")
                    print(f"   Continuing without fade-out")
            
            # Verify music file still exists and is readable before mixing
            if not os.path.exists(music_path):
                print(f"   ⚠️  ERROR: Music file no longer exists: {music_path}")
                print(f"   Skipping music mixing")
                music_path = None
            else:
                # Use ffmpeg to mix the two audio tracks
                # Voiceover will be at full volume, music at specified volume
                music_vol_adjusted = music_volume  # Use the provided music_volume parameter
                print(f"   Mixing with voiceover at 100% and music at {music_vol_adjusted*100:.0f}% volume")
                
                # Create a temporary output file to avoid writing to the same file we're reading from
                import uuid
                temp_output_path = os.path.join(temp_dir, f"mixed_audio_{uuid.uuid4().hex[:8]}_{timestamp}.mp3")
                
                # Ensure temp output is different from both input files
                while temp_output_path == voiceover_path or temp_output_path == music_path or temp_output_path == output_path:
                    temp_output_path = os.path.join(temp_dir, f"mixed_audio_{uuid.uuid4().hex[:8]}_{timestamp}.mp3")
                
                # Mix with explicit duration control - use voiceover duration as the limit
                # This ensures music stops when voiceover stops
                # Resample both to same rate and mix - ensure sample rates match for proper mixing
                # Voiceover is typically 24kHz mono, music is 44.1kHz stereo - resample both to 44.1kHz
                # amix will automatically handle mono+stereo mixing, but ensure both are at same sample rate
                # Add volume boost after mixing to prevent quiet audio (amix can reduce overall volume)
                # Mix music and narration together (delay will be applied when adding to video)
                filter_complex = (
                    f"[0:a]aresample=44100,volume=1.0[voice];"
                    f"[1:a]aresample=44100,volume={music_vol_adjusted}[music];"
                    f"[voice][music]amix=inputs=2:duration=first:dropout_transition=2,"
                    f"volume=2.0"  # Boost volume by 2x (6dB) after mixing to compensate for amix volume reduction
                )
                
                cmd = [
                    ffmpeg_path,
                    "-i", voiceover_path,
                    "-i", music_path,
                    "-filter_complex", filter_complex,
                    "-c:a", "libmp3lame",
                    "-b:a", "192k",
                    "-ar", "44100",  # Output sample rate
                    "-ac", "2",      # Stereo output
                ]
                
                # Explicitly limit to voiceover duration if we know it
                # This ensures music stops exactly when voiceover stops
                if voiceover_duration:
                    cmd.extend(["-t", str(voiceover_duration)])
                
                cmd.extend(["-y", temp_output_path])
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    # Verify the temp output file was created and has content
                    if os.path.exists(temp_output_path):
                        output_size = os.path.getsize(temp_output_path)
                        if output_size > 0:
                            print(f"   ✅ Mixed audio file created: {output_size / 1024:.1f} KB")
                            
                            # Verify the mixed file has reasonable size (should be larger than voiceover alone)
                            voiceover_size = os.path.getsize(voiceover_path) if os.path.exists(voiceover_path) else 0
                            if output_size < voiceover_size * 0.8:
                                print(f"   ⚠️  Warning: Mixed file seems too small ({output_size} vs voiceover {voiceover_size})")
                                print(f"   This might indicate mixing didn't work correctly")
                            
                            # Move temp file to final output path
                            import shutil
                            # Remove final output if it exists
                            if os.path.exists(output_path) and output_path != temp_output_path:
                                try:
                                    os.remove(output_path)
                                except Exception as e:
                                    print(f"   ⚠️  Warning: Could not remove existing output file: {e}")
                            
                            # Move temp file to final location
                            try:
                                shutil.move(temp_output_path, output_path)
                                print(f"✅ Voiceover and music mixed successfully: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
                            except Exception as e:
                                # Try copying instead
                                try:
                                    shutil.copy2(temp_output_path, output_path)
                                    print(f"✅ Voiceover and music mixed: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
                                except Exception as e2:
                                    # Use temp file as output
                                    output_path = temp_output_path
                                    print(f"✅ Using temp file as output: {output_path}")
                        else:
                            print(f"   ⚠️  Warning: Mixed audio file is empty!")
                            # Fallback: just use voiceover
                            import shutil
                            shutil.copy2(voiceover_path, output_path)
                            print(f"✅ Saved voiceover only (mixed file was empty): {output_path}")
                    else:
                        print(f"   ⚠️  Warning: Mixed audio file was not created!")
                        # Fallback: just use voiceover
                        import shutil
                        shutil.copy2(voiceover_path, output_path)
                        print(f"✅ Saved voiceover only (mixing failed): {output_path}")
                    
                    # Clean up temp files
                    if temp_output_path != output_path and os.path.exists(temp_output_path):
                        try:
                            os.remove(temp_output_path)
                        except:
                            pass
                    
                    # Clean up adjusted music if it was created
                    if music_path != voiceover_path and music_path != output_path:
                        try:
                            if os.path.exists(music_path):
                                os.remove(music_path)
                        except:
                            pass
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Audio mixing failed!")
                    print(f"   Return code: {e.returncode}")
                    if e.stdout:
                        print(f"   FFmpeg stdout: {e.stdout[:500]}")
                    if e.stderr:
                        print(f"   FFmpeg stderr: {e.stderr[:500]}")
                    print(f"   Full error: {e}")
                    # Clean up temp output if it exists
                    if 'temp_output_path' in locals() and os.path.exists(temp_output_path) and temp_output_path != output_path:
                        try:
                            os.remove(temp_output_path)
                        except:
                            pass
                    # Fallback: just use voiceover
                    import shutil
                    shutil.copy2(voiceover_path, output_path)
                    print(f"✅ Saved voiceover only (music mixing failed): {output_path}")
        else:
            # No music, just process voiceover
            print("Processing voiceover format...")
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            temp_output = os.path.join(temp_dir, f"voiceover_processed_{unique_id}_{timestamp}.mp3")
            
            # Ensure temp output is different from input and final output
            while temp_output == voiceover_path or temp_output == output_path:
                unique_id = str(uuid.uuid4())[:8]
                temp_output = os.path.join(temp_dir, f"voiceover_processed_{unique_id}_{timestamp}.mp3")
            
            cmd = [
                ffmpeg_path,
                "-i", voiceover_path,
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                "-y",
                temp_output
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                import shutil
                if os.path.exists(temp_output):
                    # Verify file has content
                    file_size = os.path.getsize(temp_output)
                    if file_size > 0:
                        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                        if os.path.exists(output_path) and output_path != temp_output:
                            try:
                                os.remove(output_path)
                            except:
                                pass
                        # Move temp file to final location
                        try:
                            shutil.move(temp_output, output_path)
                            print(f"✅ Audio file created: {output_path}")
                        except Exception as e:
                            # Try copying instead
                            try:
                                shutil.copy2(temp_output, output_path)
                                print(f"✅ Audio file created (copied): {output_path}")
                            except Exception as e2:
                                print(f"⚠️  Could not move/copy temp file: {e2}")
                                output_path = temp_output
                                print(f"✅ Using temp file as output: {output_path}")
                    else:
                        print(f"⚠️  Processed audio file is empty, using original")
                        import shutil
                        if voiceover_path != output_path:
                            shutil.copy2(voiceover_path, output_path)
                        print(f"✅ Saved voiceover: {output_path}")
                else:
                    print(f"⚠️  Processed audio file was not created, using original")
                    import shutil
                    if voiceover_path != output_path:
                        shutil.copy2(voiceover_path, output_path)
                    print(f"✅ Saved voiceover: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  FFmpeg error: {e.stderr}")
                import shutil
                if voiceover_path != output_path:
                    shutil.copy2(voiceover_path, output_path)
                print(f"✅ Saved voiceover: {output_path}")
        
        # Cleanup temp files (don't delete VIDEO_MUSIC.mp3 - it's a user-provided file)
        try:
            if voiceover_path and voiceover_path != output_path:
                os.remove(voiceover_path)
            # Don't delete VIDEO_MUSIC.mp3 - it's a user-provided file
            # Only delete temp music files if they were created (not VIDEO_MUSIC.mp3)
            if music_path and music_path != output_path:
                # Check if it's VIDEO_MUSIC.mp3 (user file) - don't delete
                if 'VIDEO_MUSIC' not in os.path.basename(music_path).upper():
                    os.remove(music_path)
        except Exception as cleanup_error:
            pass
        
        # Return output path and original voiceover backup path for later re-mixing
        return output_path, original_voiceover_backup if 'original_voiceover_backup' in locals() else None
        
    except Exception as e:
        raise Exception(f"Failed to generate voiceover: {e}")


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


def adjust_audio_duration(audio_path, target_duration, output_path=None, ffmpeg_path=None, method='speed'):
    """
    Adjust audio duration to match target duration.
    
    Args:
        audio_path: Path to the input audio file
        target_duration: Target duration in seconds
        output_path: Path to save the adjusted audio (default: overwrite input)
        ffmpeg_path: Path to ffmpeg executable
        method: Method to use ('speed' for speed up/slow down, 'pad' for padding/trimming)
        
    Returns:
        Path to the adjusted audio file
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise Exception("FFmpeg not found. Cannot adjust audio duration.")
    
    if output_path is None:
        base, ext = os.path.splitext(audio_path)
        output_path = f"{base}_adjusted{ext}"
    
    # Get current audio duration
    current_duration = get_media_duration(audio_path, ffmpeg_path)
    if current_duration is None:
        print("⚠️  Could not determine audio duration, using original audio")
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path
    
    duration_diff = target_duration - current_duration
    duration_ratio = target_duration / current_duration
    
    print(f"   Audio duration: {current_duration:.2f}s, Target: {target_duration:.2f}s, Difference: {duration_diff:+.2f}s")
    print(f"   Duration ratio: {duration_ratio:.3f}x")
    
    # Only use padding for very small differences AND if method is explicitly 'pad'
    # If method is 'speed', always use speed adjustment (even for small differences)
    if method == 'speed':
        # Use atempo filter to speed up or slow down
        # atempo range is 0.5 to 2.0 (half speed to double speed)
        if duration_ratio < 0.5:
            # Need to slow down too much, use multiple atempo filters
            # atempo can only do 0.5-2.0, so for ratios < 0.5, chain multiple
            tempo_filters = []
            remaining_ratio = duration_ratio
            while remaining_ratio < 0.5:
                tempo_filters.append("atempo=0.5")
                remaining_ratio *= 2
            if remaining_ratio != 1.0:
                tempo_filters.append(f"atempo={remaining_ratio}")
            filter_chain = ",".join(tempo_filters)
        elif duration_ratio > 2.0:
            # Need to speed up too much, use multiple atempo filters
            tempo_filters = []
            remaining_ratio = duration_ratio
            while remaining_ratio > 2.0:
                tempo_filters.append("atempo=2.0")
                remaining_ratio /= 2
            if remaining_ratio != 1.0:
                tempo_filters.append(f"atempo={remaining_ratio}")
            filter_chain = ",".join(tempo_filters)
        else:
            # Single atempo filter
            filter_chain = f"atempo={duration_ratio}"
        
        # Limit pitch correction to avoid too much distortion
        # For small adjustments, atempo is fine
        if abs(duration_ratio - 1.0) > 0.2:
            print(f"   Adjusting speed by {duration_ratio:.2f}x (this may slightly affect pitch)")
        else:
            print(f"   Adjusting speed by {duration_ratio:.2f}x (minimal pitch change)")
        
        # Don't use -t flag with atempo - let atempo handle the duration
        # The atempo filter will naturally change the duration
        cmd = [
            ffmpeg_path,
            "-i", audio_path,
            "-af", filter_chain,
            "-c:a", "libmp3lame",  # Preserve codec
            "-b:a", "192k",  # Preserve bitrate
            "-ac", "2",  # Preserve stereo
            "-y",
            output_path
        ]
        
        print(f"   Running FFmpeg command: {' '.join(cmd)}")
    else:
        # Method: pad or trim
        if duration_diff > 0:
            # Audio is shorter, add silence at the end
            # Use apad which preserves all channels and audio characteristics
            print(f"   Padding audio with {duration_diff:.2f}s of silence")
            # Get audio properties to preserve them
            # Use apad to add silence - it preserves all channels and audio characteristics
            # First, get the audio properties, then pad with silence
            # Use filter_complex to ensure proper channel handling
            cmd = [
                ffmpeg_path,
                "-i", audio_path,
                "-af", "apad",  # Add silence to end (preserves channels and sample rate)
                "-t", str(target_duration),  # Trim to exact target duration
                "-c:a", "libmp3lame",  # Preserve codec
                "-b:a", "192k",  # Preserve bitrate
                "-y",
                output_path
            ]
        else:
            # Audio is longer, trim it
            print(f"   Trimming audio by {abs(duration_diff):.2f}s")
            # Just trim, preserving all audio properties
            cmd = [
                ffmpeg_path,
                "-i", audio_path,
                "-t", str(target_duration),
                "-c:a", "copy",  # Copy audio stream without re-encoding to preserve quality
                "-y",
                output_path
            ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Verify the output duration
        adjusted_duration = get_media_duration(output_path, ffmpeg_path)
        if adjusted_duration:
            print(f"   ✅ Adjusted audio duration: {adjusted_duration:.2f}s (target: {target_duration:.2f}s)")
            # Check if adjustment was successful
            if method == 'speed' and abs(adjusted_duration - target_duration) > 0.5:
                print(f"   ⚠️  Warning: Speed adjustment may not have worked correctly. Duration is off by {abs(adjusted_duration - target_duration):.2f}s")
        
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Audio adjustment failed!")
        print(f"   Error: {e.stderr}")
        print(f"   Command: {' '.join(cmd)}")
        # Don't fall back to trim if speed adjustment was requested - raise error instead
        if method == 'speed':
            raise Exception(f"Speed adjustment failed: {e.stderr}")
        # Only fall back for pad/trim methods
        try:
            cmd = [
                ffmpeg_path,
                "-i", audio_path,
                "-t", str(target_duration),
                "-y",
                output_path
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path
        except:
            import shutil
            shutil.copy2(audio_path, output_path)
            return output_path


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
    
    print(f"Synchronizing mixed audio to video:")
    print(f"   Video duration: {video_duration:.2f}s")
    print(f"   Target audio duration: {target_duration:.2f}s (video + {voiceover_padding*2}s padding)")
    print(f"   Current audio duration: {current_duration:.2f}s")
    print(f"   Music will match video exactly (start/end together)")
    print(f"   Voiceover will have {voiceover_padding}s padding before and after video")
    
    duration_diff = target_duration - current_duration
    
    if abs(duration_diff) < 0.1:
        print(f"   ✅ Audio duration is already correct")
        import shutil
        shutil.copy2(mixed_audio_path, output_path)
        return output_path
    
    import tempfile
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    
    if duration_diff > 0:
        # Audio is shorter - pad with silence
        print(f"   Padding audio with {duration_diff:.2f}s of silence")
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
        print(f"   Trimming audio by {abs(duration_diff):.2f}s")
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
        final_duration = get_media_duration(output_path, ffmpeg_path)
        if final_duration:
            print(f"   ✅ Synchronized audio duration: {final_duration:.2f}s")
        return output_path
    except Exception as e:
        print(f"⚠️  Audio synchronization failed: {e}")
        import shutil
        shutil.copy2(mixed_audio_path, output_path)
        return output_path


def add_audio_to_video(video_path, audio_path, output_path=None, ffmpeg_path=None, remove_existing_audio=True, sync_duration=True):
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
                print(f"Synchronizing audio and video durations...")
                print(f"   Video duration: {video_duration:.2f}s")
                print(f"   Audio duration: {audio_duration:.2f}s")
                print(f"   Difference: {duration_diff:.2f}s ({percent_diff:.1f}%)")
                print(f"   Strategy: Center shorter media within longer, no more than 5% cut")
                
                import tempfile
                temp_dir = tempfile.gettempdir()
                timestamp = int(time.time())
                
                # Strategy: Center shorter media within longer, or adjust if difference > 5%
                # Never cut more than 5% of either media
                if audio_duration < video_duration:
                    # Audio is shorter - center it within video (no cutting)
                    audio_diff_percent = ((video_duration - audio_duration) / audio_duration) * 100
                    print(f"   Audio is shorter ({audio_diff_percent:.1f}% difference) - centering within video")
                    adjusted_audio_path = os.path.join(temp_dir, f"audio_centered_{timestamp}.mp3")
                    adjusted_audio_path = center_audio_in_duration(
                        audio_path=audio_path,
                        target_duration=video_duration,
                        output_path=adjusted_audio_path,
                        ffmpeg_path=ffmpeg_path
                    )
                elif video_duration < audio_duration:
                    # Video is shorter - extend video to match audio (no cutting)
                    video_diff_percent = ((audio_duration - video_duration) / video_duration) * 100
                    print(f"   Video is shorter ({video_diff_percent:.1f}% difference) - extending to match audio")
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
                            print(f"   ⚠️  WARNING: Adjusted audio is much smaller than input!")
                            print(f"   This might indicate audio processing lost content")
                    
                    print(f"✅ Audio synchronized to video duration")
                    
                except Exception as e:
                    print(f"⚠️  Audio synchronization failed: {e}")
                    print("   Using original audio (may be cut off or have silence)")
                    adjusted_audio_path = audio_path
            else:
                print(f"   ✅ Audio and video durations are synchronized ({duration_diff:.2f}s difference, {percent_diff:.1f}% - within 5% tolerance)")
        else:
            print("⚠️  Could not determine durations, skipping synchronization")
    
    # Step 2: Add audio to video with volume boost and 1-second delay
    # Use ffmpeg to add audio to video
    # Remove any existing audio and replace with new audio track
    # Apply volume boost to ensure audio is loud enough (compensate for any volume loss during encoding)
    # Add 1-second delay so audio starts 1 second after video starts
    if remove_existing_audio:
        # Map only video from first input, audio from second input (removes existing audio)
        # Apply volume boost and 1-second delay to ensure audio starts 1 second after video
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-i", adjusted_audio_path,
            "-c:v", "copy",      # Copy video stream without re-encoding
            "-af", "volume=1.5,adelay=1000|1000",  # Boost audio by 1.5x and delay by 1 second (1000ms)
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
        # Apply 1-second delay so audio starts 1 second after video
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-i", adjusted_audio_path,
            "-c:v", "copy",
            "-af", "adelay=1000|1000",  # Delay audio by 1 second (1000ms)
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
        print(f"✅ Audio added to video: {output_path}")
        
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


def generate_segment_scripts_from_overarching(
    overarching_script,
    num_segments,
    segment_duration,
    total_duration,
    api_key=None,
    model='gpt-4o',
    max_tokens=20000
):
    """
    Generate individual scripts for each segment from the overarching script.
    
    Args:
        overarching_script: The complete overarching script for the entire video
        num_segments: Number of segments to create scripts for
        segment_duration: Duration of each segment in seconds
        total_duration: Total video duration in seconds
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-4o')
        max_tokens: Maximum tokens for the response (default: 2500)
        
    Returns:
        List of individual segment scripts (one per segment)
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    segment_scripts = []
    
    for segment_id in range(1, num_segments + 1):
        start_time = (segment_id - 1) * segment_duration
        end_time = segment_id * segment_duration
        
        # Create prompt for this segment
        segment_prompt = f"""Extract segment {segment_id} ({start_time:.1f}-{end_time:.1f}s) from script for {segment_duration:.1f}s Sora video.

Script: {overarching_script[:800]}{'...' if len(overarching_script) > 800 else ''}

Provide ONLY the video generation prompt for this segment:"""
        
        try:
            print(f"Generating script for segment {segment_id}/{num_segments} (seconds {start_time:.1f}-{end_time:.1f})...")
            
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "Professional scriptwriter. Extract and refine script segments for video generation. Create engaging, well-paced narration."},
                    {"role": "user", "content": segment_prompt}
                ],
                max_output_tokens=max_tokens,
                temperature=1
            )
            
            segment_script = response.output_text.strip()
            # Clean up any labels or formatting that might have been added
            segment_script = segment_script.replace(f"Segment {segment_id}:", "").replace(f"Segment {segment_id}", "").strip()
            segment_scripts.append(segment_script)
            
            print(f"  ✅ Segment {segment_id} script generated ({len(segment_script)} characters)")
            
        except Exception as e:
            print(f"  ⚠️  Failed to generate script for segment {segment_id}: {e}")
            # Fallback: try to extract from overarching script manually
            # Split script by approximate time (rough estimate)
            script_words = overarching_script.split()
            words_per_segment = len(script_words) / num_segments
            start_word = int((segment_id - 1) * words_per_segment)
            end_word = int(segment_id * words_per_segment)
            fallback_script = " ".join(script_words[start_word:end_word])
            segment_scripts.append(fallback_script)
            print(f"  ⚠️  Using fallback extraction for segment {segment_id}")
    
    return segment_scripts


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


def generate_thumbnail_from_prompt(description, output_path=None, api_key=None):
    """
    Generate a thumbnail image from a video description using OpenAI API.
    
    Args:
        description: Description of the video (used to generate thumbnail prompt)
        output_path: Path to save the generated thumbnail (default: temp file)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        
    Returns:
        Path to generated thumbnail image file
    """
    prompt = thumbnail_prompt_template.format(description=description)
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
    This image will be used as the reference frame for Sora 2 video generation.
    
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
            max_output_tokens=200,
            temperature=0.7
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


def generate_still_image_prompt(script, context_segment, position, num_videos, api_key=None, model='gpt-5-2025-08-07', previous_segment_text=None, next_segment_text=None, reference_image_info=None):
    """
    Generate a DALL-E prompt for a still image based on script context.
    Uses the same style and approach as Sora prompt generation for consistency.
    
    Args:
        script: The full overarching script
        context_segment: The script text from the video segment before this still image
        position: Position of the still image (after which video number, or 0 for opening)
        num_videos: Total number of videos
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-5-2025-08-07' to match Sora prompts)
        previous_segment_text: Script text from the previous segment (for continuity)
        next_segment_text: Script text from the next segment (for forward continuity)
        reference_image_info: Dict with 'type' and 'description' of reference image (for consistency)
        
    Returns:
        String: DALL-E prompt for the still image
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    if position == 0:
        # Opening still image (test mode)
        position_text = "at the very beginning of the video (opening still image)"
        time_context = "0:00"
    else:
        position_text = f"after video segment {position} of {num_videos} total segments"
        # Estimate time position (assuming 12s segments + still images)
        time_context = f"approximately {position * 12 + ((position - 1) // 3) * 8}s"
    
    # Build context sections (similar to Sora prompt generation)
    context_sections = []
    
    # PRIMARY SCRIPT SEGMENT (the segment this still image follows)
    context_sections.append(f"""PRIMARY SCRIPT SEGMENT (the video segment this still image follows):
{context_segment}

This still image appears {position_text} ({time_context}) and should visually relate to the narration in this segment.""")
    
    # Full script context
    if script:
        context_sections.append(f"""FULL OVERARCHING SCRIPT (for narrative context and chronological flow):
{script}

This still image is part of a documentary-style video covering this complete narrative.""")
    
    # Previous segment context (for continuity)
    if previous_segment_text:
        context_sections.append(f"""PREVIOUS SEGMENT SCRIPT (for continuity):
{previous_segment_text}

The still image should maintain visual continuity with the previous segment's scene.""")
    
    # Next segment context (for forward continuity)
    if next_segment_text:
        context_sections.append(f"""NEXT SEGMENT SCRIPT (for forward continuity):
{next_segment_text}

The still image should lead naturally into what comes next in the narrative.""")
    
    # Reference image context (for consistency)
    if reference_image_info:
        ref_type = reference_image_info.get('type', 'subject')
        ref_desc = reference_image_info.get('description', 'the main visual element')
        if ref_type == 'character':
            context_sections.append(f"""═══════════════════════════════════════════════════════════════════════════════
CRITICAL - REFERENCE IMAGE FOR CHARACTER MATCHING:
A reference image will be provided showing the main character: {ref_desc}

ABSOLUTELY CRITICAL - EXACT CHARACTER MATCHING REQUIRED:
- The character in the generated still image MUST be THE EXACT SAME PERSON as shown in the reference image
- This is NOT a look-alike, similar person, or someone with matching features - it MUST be THE EXACT SAME INDIVIDUAL
- Every facial feature, body type, hair, clothing style, and physical characteristic must match EXACTLY
- The character's face, build, posture, and appearance must be IDENTICAL to the reference image
- When describing the character in your DALL-E prompt, you MUST emphasize that this is the EXACT SAME PERSON from the reference image
- Use phrases like "the exact same person as shown in the reference image", "identical to the reference character", "the precise individual from the reference image"
- Do NOT describe the character as "similar to" or "resembling" - it must be THE SAME PERSON
- This is the SINGLE MOST IMPORTANT requirement for character-based reference images
═══════════════════════════════════════════════════════════════════════════════""")
        else:
            context_sections.append(f"""REFERENCE IMAGE CONTEXT:
A reference image will be provided showing: {ref_desc}

IMPORTANT: The still image must maintain visual consistency with this reference image. The main visual elements, style, and atmosphere should align with the reference image.""")
    
    context_text = "\n\n".join(context_sections) if context_sections else ""
    
    # Build requirements (similar to Sora prompt requirements)
    visual_requirements = []
    # Add character matching requirement FIRST if reference image is character-based
    if reference_image_info and reference_image_info.get('type') == 'character':
        visual_requirements.append("- ═══════════════════════════════════════════════════════════════════════════════")
        visual_requirements.append("- SINGLE MOST IMPORTANT REQUIREMENT - EXACT CHARACTER MATCHING:")
        visual_requirements.append("- The character shown in this still image MUST be THE EXACT SAME PERSON as in the reference image")
        visual_requirements.append("- This is NOT a look-alike or similar person - it MUST be IDENTICAL to the reference image")
        visual_requirements.append("- Every facial feature, body type, hair, clothing, and physical characteristic must match EXACTLY")
        visual_requirements.append("- In your prompt, explicitly state that the character is 'the exact same person as shown in the reference image'")
        visual_requirements.append("- Use phrases like 'identical to the reference character', 'the precise individual from the reference image'")
        visual_requirements.append("- Do NOT use words like 'similar', 'resembling', 'looks like' - use 'is the same person', 'identical', 'exact match'")
        visual_requirements.append("- ═══════════════════════════════════════════════════════════════════════════════")
    visual_requirements.append("- Visuals directly relate to the PRIMARY SCRIPT SEGMENT above")
    visual_requirements.append("- Maintain continuity with previous segments (unless script indicates change)")
    visual_requirements.append("- Be visually striking and suitable for an 8-second contemplative moment with camera panning")
    visual_requirements.append("- Generate the most hyperrealistic, ultra-detailed, photorealistic image possible, as if photographed by a professional documentary photographer, with maximum detail, natural lighting, realistic textures, and lifelike quality. Make it look like a real photograph, not an illustration or artwork")
    visual_requirements.append("- Include detailed description of composition, lighting, mood, atmosphere")
    visual_requirements.append("- Be cinematic and documentary-style")
    visual_requirements.append("- Complies with OpenAI content policy: no violence, hate, adult content, illegal activity, copyrighted characters, or likenesses of real people")
    visual_requirements.append("- Uses generic, artistic, stylized representations only")
    visual_requirements.append("- Appropriate for a documentary-style video")
    
    requirements_text = "\n".join(visual_requirements)
    
    # Create prompt (similar structure to Sora prompt generation)
    conversion_prompt = f"""Create a detailed DALL-E prompt for a high-quality still image that will appear {position_text} ({time_context}).

{context_text}

Requirements:
{requirements_text}

CRITICAL: The still image prompt you create must directly relate to the narration in the PRIMARY SCRIPT SEGMENT above. The still image should visually represent or complement the story moment described in that segment. Match the style and tone of the Sora video prompts used for the video segments.
{f"═══════════════════════════════════════════════════════════════════════════════\nSINGLE MOST IMPORTANT REQUIREMENT - EXACT CHARACTER MATCHING:\nIf a character reference image is provided, the character in this still image MUST be THE EXACT SAME PERSON as shown in the reference image. This is NOT a look-alike - it MUST be IDENTICAL. Every feature must match EXACTLY. In your prompt, explicitly state the character is the exact same person from the reference image. Use phrases like 'identical to the reference character', 'the precise individual from the reference image'. Do NOT use 'similar' or 'resembling' - use 'is the same person', 'identical', 'exact match'.\n═══════════════════════════════════════════════════════════════════════════════\n" if (reference_image_info and reference_image_info.get('type') == 'character') else ""}
Provide ONLY the DALL-E prompt (no labels, no explanation, just the prompt text):"""
    
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": f"Professional video director and image prompt specialist. Create detailed cinematic DALL-E prompts for still images that match script narration and maintain visual consistency with video segments. CRITICAL: The still image must relate directly to the PRIMARY SCRIPT SEGMENT provided. Ensure visual continuity and narrative flow. Always provide complete, detailed prompts. CRITICAL: All images must be hyperrealistic and photorealistic, as if photographed by a professional documentary photographer - make them look like real photographs with natural lighting, realistic textures, and maximum detail. Never create prompts with likenesses of real people, celebrities, or historical figures. Always use generic, artistic, stylized representations, but make them appear completely realistic and photographic. {'ABSOLUTELY CRITICAL - CHARACTER REFERENCE IMAGE: When a character-based reference image is provided, the character in the generated still image MUST be THE EXACT SAME PERSON as shown in the reference image. This is NOT a look-alike or similar person - it MUST be IDENTICAL. Every facial feature, body type, hair, clothing, and physical characteristic must match EXACTLY. In your prompts, explicitly state that the character is the exact same person from the reference image. Use phrases like identical to the reference character, the precise individual from the reference image. Do NOT use words like similar, resembling, looks like - use is the same person, identical, exact match. This is the SINGLE MOST IMPORTANT requirement when a character reference image is provided.' if (reference_image_info and reference_image_info.get('type') == 'character') else ''}"},
                {"role": "user", "content": conversion_prompt}
            ],
            max_output_tokens=20000,
            temperature=1
        )
        
        image_prompt = response.output_text.strip()
        
        # Clean up any labels or formatting (similar to Sora prompt cleaning)
        labels_to_remove = [
            "DALL-E Prompt:",
            "DALL-E:",
            "Prompt:",
            "Image Prompt:",
            "Still Image Prompt:"
        ]
        
        for label in labels_to_remove:
            if image_prompt.startswith(label):
                image_prompt = image_prompt[len(label):].strip()
        
        # Additional cleanup
        image_prompt = image_prompt.lstrip(":- ").strip()
        
        # Sanitize for content policy
        image_prompt = sanitize_image_prompt(image_prompt)
        
        return image_prompt
        
    except Exception as e:
        print(f"⚠️  Failed to generate still image prompt: {e}")
        # Fallback: generic prompt based on context
        if context_segment and len(context_segment.strip()) > 0:
            return f"A hyperrealistic, photorealistic, high-quality still image, as if photographed by a professional documentary photographer, representing the story context: {context_segment[:200]}... Make it look like a real photograph with natural lighting, realistic textures, and maximum detail."
        else:
            return "A hyperrealistic, photorealistic, high-quality still image, as if photographed by a professional documentary photographer, suitable for a documentary-style video. Make it look like a real photograph with natural lighting, realistic textures, and maximum detail."


def analyze_script_for_still_images(script, segment_texts, target_num_stills, api_key=None, model='gpt-4o', reference_images=None):
    """
    Analyze script to identify which segments should be still images AND determine reference image usage for video segments.
    Multi-purpose function that:
    1. Identifies which segments should be still images (existing functionality)
    2. For each segment, determines: still image or video? If video, which reference image to use?
    
    Still images work well for: key moments, important visuals, transitions, dramatic pauses, etc.
    AVOIDS: action scenes, fights, battles, chases, fast-paced moments.
    
    Args:
        script: The full overarching script
        segment_texts: List of segment script texts (all segments, both video and still)
        target_num_stills: Target number of still images (approximately 1/3 of total segments)
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-4o')
        reference_images: List of reference image dictionaries with 'id', 'type', 'description' (from analyze_script_for_reference_images)
        
    Returns:
        Dictionary with:
        - 'still_image_segments': List of dictionaries with 'segment_id', 'segment_text', 'image_prompt', 'duration', 'reasoning'
        - 'segment_assignments': List of dictionaries, one per segment, with 'segment_id', 'type' ('still' or 'video'), 'reference_image_id' (or None)
    """
    # Standard still image duration (12 seconds per still image)
    STILL_IMAGE_DURATION = 12.0
    segment_duration = 12.0  # Each segment is 12 seconds
    num_segments = len(segment_texts)
    
    # Use target_num_stills directly (already calculated as approximately 1/3)
    ideal_num_stills = target_num_stills
    ideal_duration_per_still = STILL_IMAGE_DURATION
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    # Create segments info for analysis (include full text for better context)
    segments_info = []
    for i, seg_text in enumerate(segment_texts, 1):
        start_time = (i - 1) * segment_duration
        end_time = i * segment_duration
        segments_info.append({
            'segment_id': i,
            'start_time': start_time,
            'end_time': end_time,
            'text': seg_text  # Use full text for better analysis
        })
    
    # Build reference images context for the prompt
    reference_images_context = ""
    if reference_images and len(reference_images) > 0:
        ref_list = []
        for ref_img in reference_images:
            ref_id = ref_img.get('id', 'unknown')
            ref_desc = ref_img.get('description', '')
            ref_type = ref_img.get('type', 'subject')
            ref_list.append(f"- {ref_id}: {ref_desc} (type: {ref_type})")
        reference_images_context = f"""
═══════════════════════════════════════════════════════════════════════════════
CRITICAL - AVAILABLE REFERENCE IMAGES (MUST be used for video segments when applicable):
═══════════════════════════════════════════════════════════════════════════════
{chr(10).join(ref_list)}

REFERENCE IMAGE ASSIGNMENT RULES:
1. For EACH video segment, analyze if it features any of the reference images above
2. If a segment mentions, describes, or shows the subject/character from a reference image, you MUST assign that reference image's ID
3. Character-type reference images: Use when the segment features that character (even briefly or in description)
4. Subject-type reference images: Use when the segment features that location, object, or visual element
5. You should assign reference images to MOST video segments that relate to them - be generous, not conservative
6. Only use null if the segment truly has NO connection to any reference image
7. If multiple reference images could apply, choose the PRIMARY one that best matches the segment's focus

EXAMPLES:
- Segment mentions "Blackbeard" and ref_1 is "Blackbeard's face" → assign "ref_1"
- Segment describes "the ship" and ref_2 is "Blackbeard's ship" → assign "ref_2"
- Segment is about a battle location and ref_3 is "battlefield location" → assign "ref_3"
- Segment is about something completely unrelated → use null

CRITICAL: Reference images exist to maintain visual consistency. When in doubt, assign the reference image if there's ANY connection.
═══════════════════════════════════════════════════════════════════════════════
"""
    else:
        reference_images_context = """
NOTE: No reference images were identified for this video. All video segments will use null for reference_image_id.
"""
    
    analysis_prompt = f"""Analyze this script and perform TWO tasks:

TASK 1: Identify {ideal_num_stills} segments (out of {len(segment_texts)} total) where a high-quality still image with camera panning would be most effective.

TASK 2: For EACH of the {len(segment_texts)} segments, determine:
- Is it a still image or video?
- If video, which reference image should be used (from the available reference images below), or null if none?

{reference_images_context}

FULL SCRIPT (for complete context):
{script}

SEGMENTS (each is 12 seconds):
{chr(10).join([f"Segment {s['segment_id']} ({s['start_time']:.1f}s-{s['end_time']:.1f}s):\n{s['text']}\n" for s in segments_info])}

CRITICAL: AVOID placing still images during action scenes or fast-paced moments:

CRITICAL: AVOID placing still images during action scenes or fast-paced moments:
- NO still images during: fights, battles, chases, explosions, combat, violence, fast action
- NO still images during: rapid movement, running, jumping, dynamic action sequences
- NO still images during: intense dramatic moments that require motion
- AVOID segments with words like: fight, battle, attack, chase, run, jump, explode, crash, strike, clash, combat, conflict, etc.
- AVOID segments with [MUSICAL BREAK] or [VISUAL BREAK] (those already have pauses)

PREFER placing still images during:
- Contemplative moments: reflection, thought, observation, quiet scenes, peaceful moments
- Important visuals: landscapes, locations, objects, architecture, nature, scenery
- Transitions: between major story sections, scene changes, time shifts
- Descriptive moments: when narration describes a specific scene, location, or object in detail
- Dramatic pauses: moments of significance that benefit from a still, contemplative image
- Historical context: when explaining background, setting, or context
- Character moments: quiet character scenes (NOT action scenes)
- Establishing shots: when introducing a new location or setting

IMPORTANT: You have {len(segment_texts)} total segments. Select exactly {ideal_num_stills} segments for still images (approximately 1/3). The remaining {num_segments - ideal_num_stills} segments will be videos (approximately 2/3).
Each still image is 12 seconds long.

CRITICAL: Segment 1 (the opening segment) MUST ALWAYS be a video, NEVER a still image. The opening must be dynamic and engaging.

Full script: {script[:1000]}{'...' if len(script) > 1000 else ''}

Segments:
{chr(10).join([f"Segment {s['segment_id']} ({s['start_time']:.1f}s-{s['end_time']:.1f}s): {s['text']}" for s in segments_info])}

Output JSON object with TWO arrays:
{{
    "still_image_segments": [
        {{
            "segment_id": 1,
            "image_prompt": "Detailed DALL-E prompt for a hyperrealistic, photorealistic, high-quality still image matching this segment's narration, as if photographed by a professional documentary photographer. Make it look like a real photograph with natural lighting, realistic textures, and maximum detail. MUST comply with OpenAI content policy: no violence, hate, adult content, illegal activity, copyrighted characters, or likenesses of real people. Use generic, artistic representations only, but make them appear completely realistic and photographic.",
            "duration": 12.0,
            "reasoning": "Why this segment benefits from a still image (and why it's NOT an action scene)"
        }}
    ],
    "segment_assignments": [
        {{
            "segment_id": 1,
            "type": "still" or "video",
            "reference_image_id": "ref_1" or null
        }}
    ]
}}

Rules:
- still_image_segments: Select exactly {ideal_num_stills} segments (approximately 1/3 of {len(segment_texts)} total segments) for still images
- segment_assignments: Provide an entry for EVERY segment (1 through {len(segment_texts)})
- For each segment in segment_assignments:
  * type: "still" if it's a still image, "video" if it's a video
  * reference_image_id: The ID of the reference image to use (from available reference images above), or null if no reference image is needed
- ABSOLUTELY CRITICAL: Segment 1 (the opening segment) MUST be a video, NEVER a still image. Do NOT select segment 1 for still images.
- CRITICAL: Ensure there are MORE video segments than still images (you have {len(segment_texts)} segments total, select only {ideal_num_stills} for still images)
- Each still image is 12.0 seconds long
- CRITICAL: Skip any segments that contain action, fights, battles, or fast-paced scenes for still images
- For video segments, you MUST assign a reference image if the segment features, mentions, or describes the subject/character from any reference image
- Be GENEROUS with reference image assignments - if there's any connection between the segment and a reference image, assign it
- Only use null if the segment truly has NO connection to any reference image (e.g., generic narration, unrelated topic)
- CRITICAL: Most video segments should have a reference image assigned if reference images exist - they were created for visual consistency
- If a segment could match multiple reference images, choose the PRIMARY one that best matches the segment's main focus
- image_prompt MUST be safe, appropriate, and comply with OpenAI DALL-E content policies
- CRITICAL: NEVER include likenesses of real people, celebrities, or historical figures
- Avoid: violence, hate speech, adult content, illegal activities, real people, copyrighted characters
- Use: generic, artistic, educational, and appropriate visual descriptions

Provide ONLY valid JSON object:"""
    
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Video production assistant. Analyze scripts to identify optimal moments for still images with camera panning. CRITICAL: Avoid action scenes, fights, battles, chases, or any fast-paced moments - still images work best during contemplative, descriptive, or transitional moments. Create generic, artistic image prompts that comply with OpenAI content policies. CRITICAL: All images must be hyperrealistic and photorealistic, as if photographed by a professional documentary photographer - make them look like real photographs with natural lighting, realistic textures, and maximum detail. Never create prompts with likenesses of real people, celebrities, or historical figures. Always use generic, artistic, stylized representations, but make them appear completely realistic and photographic."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_output_tokens=20000,
            temperature=1
        )
        
        import json
        analysis_result = json.loads(response.output_text)
        
        # Extract still image segments
        still_image_segments_raw = analysis_result.get('still_image_segments', [])
        segment_assignments_raw = analysis_result.get('segment_assignments', [])
        
        # Validate and enrich still image segments with full segment text
        # CRITICAL: Segment 1 MUST be a video, never a still image
        validated_still_segments = []
        for seg_info in still_image_segments_raw:
            segment_id = seg_info.get('segment_id')
            if 1 <= segment_id <= len(segment_texts):
                # CRITICAL: Skip segment 1 - it must always be a video
                if segment_id == 1:
                    print(f"⚠️  Warning: Segment 1 was assigned as still image, but it MUST be a video. Removing from still images.")
                    continue
                
                # Get context for better image prompt generation
                # CRITICAL: segment_texts contains NARRATION-BASED segments (words actually spoken)
                # This ensures still images are generated based on what's being narrated, not the original script
                context_segment = segment_texts[segment_id - 1]  # NARRATION-BASED segment text
                
                # Use the AI-provided image prompt (it's already detailed and context-aware)
                # The AI generated this prompt based on the narration segment text above
                detailed_image_prompt = seg_info.get('image_prompt', '')
                
                validated_still_segments.append({
                    'segment_id': segment_id,
                    'segment_text': context_segment,  # NARRATION-BASED segment (from actual audio)
                    'image_prompt': detailed_image_prompt,  # Generated from narration segment
                    'duration': STILL_IMAGE_DURATION,  # Always 12.0 seconds
                    'reasoning': seg_info.get('reasoning', '')
                })
        
        # Validate and enrich segment assignments
        # CRITICAL: Segment 1 MUST be a video, never a still image
        validated_assignments = []
        for assignment in segment_assignments_raw:
            segment_id = assignment.get('segment_id')
            if 1 <= segment_id <= len(segment_texts):
                seg_type = assignment.get('type', 'video')
                
                # CRITICAL: Force segment 1 to be a video
                if segment_id == 1 and seg_type == 'still':
                    print(f"⚠️  Warning: Segment 1 was assigned as still image, but it MUST be a video. Changing to video.")
                    seg_type = 'video'
                
                ref_id = assignment.get('reference_image_id')
                # Validate reference_image_id exists in reference_images list
                if ref_id and reference_images:
                    # Check if this reference image ID actually exists
                    ref_ids = [ref.get('id') for ref in reference_images]
                    if ref_id not in ref_ids:
                        print(f"⚠️  Warning: Segment {segment_id} assigned invalid reference_image_id '{ref_id}', setting to None")
                        ref_id = None
                
                validated_assignments.append({
                    'segment_id': segment_id,
                    'type': seg_type,  # 'still' or 'video' (forced to 'video' for segment 1)
                    'reference_image_id': ref_id  # ID or null
                })
        
        # Ensure we have assignments for all segments
        if len(validated_assignments) < len(segment_texts):
            # Fill in missing segments as videos with no reference image
            existing_ids = {a['segment_id'] for a in validated_assignments}
            for i in range(1, len(segment_texts) + 1):
                if i not in existing_ids:
                    validated_assignments.append({
                        'segment_id': i,
                        'type': 'video',
                        'reference_image_id': None
                    })
        
        # CRITICAL: Ensure segment 1 is always a video (double-check)
        for assignment in validated_assignments:
            if assignment['segment_id'] == 1 and assignment['type'] == 'still':
                print(f"⚠️  Warning: Segment 1 was still marked as still image after validation. Forcing to video.")
                assignment['type'] = 'video'
        
        # Sort assignments by segment_id
        validated_assignments.sort(key=lambda x: x['segment_id'])
        
        # Validation: If reference images exist, warn if none are being used
        if reference_images and len(reference_images) > 0:
            video_segments_with_ref = [a for a in validated_assignments if a.get('type') == 'video' and a.get('reference_image_id')]
            if len(video_segments_with_ref) == 0:
                print(f"⚠️  WARNING: {len(reference_images)} reference image(s) were identified, but NONE are being assigned to video segments!")
                print(f"   This may indicate the AI is being too conservative. Reference images should be used for visual consistency.")
                print(f"   Available reference images: {[ref.get('id') for ref in reference_images]}")
            else:
                print(f"✅ {len(video_segments_with_ref)} video segment(s) assigned reference images out of {len([a for a in validated_assignments if a.get('type') == 'video'])} total video segments")
        
        return {
            'still_image_segments': validated_still_segments,
            'segment_assignments': validated_assignments
        }
        
    except Exception as e:
        print(f"⚠️  Script analysis for still images failed: {e}")
        # Return empty structure if analysis fails
        return {
            'still_image_segments': [],
            'segment_assignments': [{'segment_id': i, 'type': 'video', 'reference_image_id': None} for i in range(1, len(segment_texts) + 1)]
        }


def create_panning_video_from_image(image_path, output_path, duration, pan_direction='top_left_to_bottom_right', ffmpeg_path=None):
    """
    Create a video with camera panning over a still image using MoviePy (preferred) or ffmpeg fallback.
    Pans in a straight line from one corner to the opposite corner over the exact duration (always 12 seconds).
    No zoom - just a smooth linear pan across the image.
    
    Args:
        image_path: Path to the still image
        output_path: Path to save the output video
        duration: Duration of the video in seconds (should always be 12.0 for still images)
        pan_direction: Direction of pan - straight line corner to corner:
            'top_left_to_bottom_right': from top-left to bottom-right
            'top_right_to_bottom_left': from top-right to bottom-left
            'bottom_left_to_top_right': from bottom-left to top-right
            'bottom_right_to_top_left': from bottom-right to top-left
        ffmpeg_path: Path to ffmpeg executable (for fallback)
        
    Returns:
        Path to the generated video file
    """
    if not os.path.exists(image_path):
        raise Exception(f"Image file not found: {image_path}")
    
    # Output resolution
    output_w, output_h = 1280, 720
    
    # Try MoviePy first (much smoother panning with built-in interpolation)
    try:
        from moviepy import ImageClip
        
        print(f"🎬 Using MoviePy for smooth corner-to-corner panning (direction: {pan_direction})...")
        
        # Scale image larger than output to allow full corner-to-corner panning (2x for full diagonal)
        scale_factor = 2.0
        scaled_w = int(output_w * scale_factor)
        scaled_h = int(output_h * scale_factor)
        
        # Resize image using PIL first (MoviePy 2.x doesn't have resize method)
        from PIL import Image
        import numpy as np
        
        # Load and resize image with PIL
        pil_img = Image.open(image_path)
        pil_img_resized = pil_img.resize((scaled_w, scaled_h), Image.LANCZOS)
        
        # Convert to numpy array and create ImageClip
        img_array = np.array(pil_img_resized)
        img_clip = ImageClip(img_array)
        img_clip = img_clip.with_duration(duration)
        
        # Calculate maximum pan distances (full diagonal from corner to corner)
        # Image is 2x scaled, so we can pan from one corner to the opposite corner
        max_pan_w = scaled_w - output_w
        max_pan_h = scaled_h - output_h
        
        # Define corner positions for straight-line panning
        if pan_direction == 'top_left_to_bottom_right':
            # Pan from top-left corner to bottom-right corner (straight diagonal line)
            start_pos = (0, 0)
            end_pos = (max_pan_w, max_pan_h)
        elif pan_direction == 'top_right_to_bottom_left':
            # Pan from top-right corner to bottom-left corner (straight diagonal line)
            start_pos = (max_pan_w, 0)
            end_pos = (0, max_pan_h)
        elif pan_direction == 'bottom_left_to_top_right':
            # Pan from bottom-left corner to top-right corner (straight diagonal line)
            start_pos = (0, max_pan_h)
            end_pos = (max_pan_w, 0)
        elif pan_direction == 'bottom_right_to_top_left':
            # Pan from bottom-right corner to top-left corner (straight diagonal line)
            start_pos = (max_pan_w, max_pan_h)
            end_pos = (0, 0)
        else:
            # Default: top-left to bottom-right
            start_pos = (0, 0)
            end_pos = (max_pan_w, max_pan_h)
        
        # Use MoviePy's smooth linear interpolation for straight-line corner-to-corner panning
        # Show more of the image by using a larger crop that gets scaled down
        # This reduces zoom and shows more of the original image
        crop_zoom_factor = 0.85  # Crop 85% of output size (shows more of image, less zoom)
        crop_w = int(output_w / crop_zoom_factor)  # Larger crop = shows more of image
        crop_h = int(output_h / crop_zoom_factor)
        
        # Adjust pan distances to account for larger crop size
        # We need to pan less distance because we're showing more of the image
        adjusted_max_pan_w = max(0, scaled_w - crop_w)
        adjusted_max_pan_h = max(0, scaled_h - crop_h)
        
        # Reduce pan distance to make it slower (pan only 60% of the way instead of full corner-to-corner)
        # This makes the panning appear slower while maintaining the same direction
        pan_distance_factor = 0.6  # Pan only 60% of the maximum distance (slower movement)
        
        # Recalculate corner positions with reduced pan distances for slower movement
        if pan_direction == 'top_left_to_bottom_right':
            start_pos = (0, 0)
            end_pos = (int(adjusted_max_pan_w * pan_distance_factor), int(adjusted_max_pan_h * pan_distance_factor))
        elif pan_direction == 'top_right_to_bottom_left':
            start_pos = (adjusted_max_pan_w, 0)
            end_pos = (int(adjusted_max_pan_w * (1 - pan_distance_factor)), int(adjusted_max_pan_h * pan_distance_factor))
        elif pan_direction == 'bottom_left_to_top_right':
            start_pos = (0, adjusted_max_pan_h)
            end_pos = (int(adjusted_max_pan_w * pan_distance_factor), int(adjusted_max_pan_h * (1 - pan_distance_factor)))
        elif pan_direction == 'bottom_right_to_top_left':
            start_pos = (adjusted_max_pan_w, adjusted_max_pan_h)
            end_pos = (int(adjusted_max_pan_w * (1 - pan_distance_factor)), int(adjusted_max_pan_h * (1 - pan_distance_factor)))
        else:
            start_pos = (0, 0)
            end_pos = (int(adjusted_max_pan_w * pan_distance_factor), int(adjusted_max_pan_h * pan_distance_factor))
        
        def transform_func(get_frame, t):
            # Linear interpolation for straight-line movement (no easing)
            # Progress from 0.0 to 1.0 over the exact duration
            progress = t / duration
            
            # Linear interpolation: straight line from start to end corner
            current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
            
            # Calculate crop position (current position is top-left of crop)
            crop_x = int(current_x)
            crop_y = int(current_y)
            
            # Ensure crop stays within bounds
            crop_x = max(0, min(crop_x, scaled_w - crop_w))
            crop_y = max(0, min(crop_y, scaled_h - crop_h))
            
            # Get frame and crop
            frame = get_frame(t)
            
            # Crop a larger area (shows more of the image)
            cropped = frame[int(crop_y):int(crop_y)+crop_h, int(crop_x):int(crop_x)+crop_w]
            
            # Scale down the crop to output size (this shows more of the original image)
            from PIL import Image
            pil_crop = Image.fromarray(cropped)
            pil_crop_resized = pil_crop.resize((output_w, output_h), Image.LANCZOS)
            resized_array = np.array(pil_crop_resized)
            
            return resized_array
        
        # Create video clip with smooth straight-line animation
        # MoviePy 2.x uses transform instead of fl
        final_clip = img_clip.transform(transform_func)
        # Ensure exact duration (always 12 seconds for still images)
        final_clip = final_clip.with_fps(30).with_duration(duration)
        
        # Write video with high quality
        final_clip.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            preset='slow',
            bitrate='8000k',
            audio=False,
            logger=None  # Suppress MoviePy verbose output
        )
        
        print(f"✅ Smooth panning video created with MoviePy: {output_path}")
        return output_path
        
    except ImportError:
        # MoviePy not available, fall back to ffmpeg with improved method
        print(f"⚠️  MoviePy not available, using ffmpeg fallback (direction: {pan_direction})...")
        print(f"   Install MoviePy for smoother panning: pip install moviepy")
        
        if ffmpeg_path is None:
            ffmpeg_path = find_ffmpeg()
        
        if not ffmpeg_path:
            raise Exception("FFmpeg not found. Cannot create panning video from still image.")
        
        # Use crop filter with animated positions for corner-to-corner panning
        output_w, output_h = 1280, 720
        scale_w = int(output_w * 2.0)  # 2x scale for full corner-to-corner panning
        scale_h = int(output_h * 2.0)
        fps = 30
        
        # Show more of the image by using a larger crop that gets scaled down
        # This reduces zoom and shows more of the original image
        crop_zoom_factor = 0.85  # Crop 85% of output size (shows more of image, less zoom)
        crop_w = int(output_w / crop_zoom_factor)  # Larger crop = shows more of image
        crop_h = int(output_h / crop_zoom_factor)
        
        # Calculate maximum pan distances (adjusted for larger crop)
        # We need to pan less distance because we're showing more of the image
        adjusted_max_pan_w = max(0, scale_w - crop_w)
        adjusted_max_pan_h = max(0, scale_h - crop_h)
        
        # Reduce pan distance to make it slower (pan only 60% of the way instead of full corner-to-corner)
        # This makes the panning appear slower while maintaining the same direction
        pan_distance_factor = 0.6  # Pan only 60% of the maximum distance (slower movement)
        reduced_pan_w = int(adjusted_max_pan_w * pan_distance_factor)
        reduced_pan_h = int(adjusted_max_pan_h * pan_distance_factor)
        
        # Use crop filter with linear interpolation for straight-line corner-to-corner panning
        # Crop larger area and scale down to show more of the original image
        if pan_direction == 'top_left_to_bottom_right':
            # Pan from top-left to bottom-right (straight diagonal line, but slower - only 60% distance)
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'t*{reduced_pan_w}/{duration}':"
                f"'t*{reduced_pan_h}/{duration}',"
                f"scale={output_w}:{output_h}"
            )
        elif pan_direction == 'top_right_to_bottom_left':
            # Pan from top-right to bottom-left (straight diagonal line, but slower - only 60% distance)
            start_x = adjusted_max_pan_w
            end_x = int(adjusted_max_pan_w * (1 - pan_distance_factor))
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'{start_x}-t*{start_x-end_x}/{duration}':"
                f"'t*{reduced_pan_h}/{duration}',"
                f"scale={output_w}:{output_h}"
            )
        elif pan_direction == 'bottom_left_to_top_right':
            # Pan from bottom-left to top-right (straight diagonal line, but slower - only 60% distance)
            start_y = adjusted_max_pan_h
            end_y = int(adjusted_max_pan_h * (1 - pan_distance_factor))
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'t*{reduced_pan_w}/{duration}':"
                f"'{start_y}-t*{start_y-end_y}/{duration}',"
                f"scale={output_w}:{output_h}"
            )
        elif pan_direction == 'bottom_right_to_top_left':
            # Pan from bottom-right to top-left (straight diagonal line, but slower - only 60% distance)
            start_x = adjusted_max_pan_w
            start_y = adjusted_max_pan_h
            end_x = int(adjusted_max_pan_w * (1 - pan_distance_factor))
            end_y = int(adjusted_max_pan_h * (1 - pan_distance_factor))
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'{start_x}-t*{start_x-end_x}/{duration}':"
                f"'{start_y}-t*{start_y-end_y}/{duration}',"
                f"scale={output_w}:{output_h}"
            )
        else:
            # Default: top-left to bottom-right (straight diagonal line, but slower - only 60% distance)
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'t*{reduced_pan_w}/{duration}':"
                f"'t*{reduced_pan_h}/{duration}',"
                f"scale={output_w}:{output_h}"
            )
        
        cmd = [
            ffmpeg_path,
            "-loop", "1",
            "-i", image_path,
            "-vf", filter_complex,
            "-t", str(duration),
            "-r", str(fps),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-y",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Panning video created with ffmpeg: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            # Final fallback: static image
            print(f"⚠️  Panning failed, creating static image video: {e.stderr[:200]}")
            cmd_fallback = [
                ffmpeg_path,
                "-loop", "1",
                "-i", image_path,
                "-vf", f"scale={output_w}:{output_h}",
                "-t", str(duration),
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-y",
                output_path
            ]
            subprocess.run(cmd_fallback, capture_output=True, text=True, check=True)
            print(f"✅ Static image video created: {output_path}")
            return output_path


def start_video_generation_job(
    prompt,
    api_key=None,
    model='sora-2',
    resolution='1280x720',
    duration=8,
    reference_image_path=None
):
    """
    Start a video generation job and return the job ID (non-blocking).
    
    Args:
        prompt: Text prompt describing the video to generate
        api_key: OpenAI API key
        model: Model to use ('sora-2' or 'sora-2-pro')
        resolution: Video resolution
        duration: Video duration in seconds
        reference_image_path: Path to reference image (optional)
        
    Returns:
        Video job ID
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    
    try:
        params = {
            'model': model,
            'prompt': prompt
        }
        
        if duration:
            params['seconds'] = str(duration)
        if resolution:
            params['size'] = resolution
        
        if reference_image_path and os.path.exists(reference_image_path):
            # Verify file is readable
            try:
                with open(reference_image_path, 'rb') as test_file:
                    test_file.read(1)  # Try to read at least 1 byte
            except Exception as e:
                print(f"⚠️  Warning: Cannot read reference image file: {e}")
                reference_image_path = None
        
        if reference_image_path and os.path.exists(reference_image_path):
            try:
                # Try passing as file path first
                params['input_reference'] = reference_image_path
                response = client.videos.create(**params)
                print(f"✅ Reference image passed to Sora API: {os.path.basename(reference_image_path)}")
            except (TypeError, ValueError) as e:
                # If path doesn't work, try opening as file object
                try:
                    with open(reference_image_path, 'rb') as img_file:
                        params['input_reference'] = img_file
                        response = client.videos.create(**params)
                    print(f"✅ Reference image passed to Sora API (as file object): {os.path.basename(reference_image_path)}")
                except Exception as e2:
                    print(f"⚠️  Warning: Failed to pass reference image: {e2}")
                    print(f"   Continuing without reference image...")
                    # Remove reference image from params and try without it
                    params.pop('input_reference', None)
                    response = client.videos.create(**params)
        else:
            if reference_image_path:
                print(f"⚠️  Warning: Reference image path provided but file doesn't exist: {reference_image_path}")
            response = client.videos.create(**params)
        
        return response.id
    except TypeError as e:
        # If parameter error, try with just model and prompt
        try:
            response = client.videos.create(model=model, prompt=prompt)
            return response.id
        except Exception as e2:
            # Re-raise original exception to preserve error code and other attributes
            raise e2
    except Exception as e:
        # If reference image causes issues, try without it
        if reference_image_path:
            try:
                params_no_ref = {'model': model, 'prompt': prompt}
                if duration:
                    params_no_ref['seconds'] = str(duration)
                if resolution:
                    params_no_ref['size'] = resolution
                response = client.videos.create(**params_no_ref)
                return response.id
            except Exception as e2:
                # Re-raise original exception to preserve error code and other attributes
                raise e2
        else:
            # Re-raise original exception to preserve error code and other attributes
            raise e


def wait_for_video_completion(
    video_id,
    output_path,
    api_key=None,
    poll_interval=10,
    max_wait_time=600
):
    """
    Wait for a video generation job to complete and download the result.
    
    Args:
        video_id: Video job ID
        output_path: Path to save the output video
        api_key: OpenAI API key
        poll_interval: Seconds to wait between status checks
        max_wait_time: Maximum time to wait in seconds
        
    Returns:
        Path to generated video file
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    print(f"Polling job {video_id} for completion (checking every {poll_interval} seconds)...")
    start_time = time.time()
    last_status = None
    
    while True:
        elapsed_time = time.time() - start_time
        
        if elapsed_time > max_wait_time:
            raise TimeoutError(
                f"Video generation timed out after {max_wait_time} seconds. Job ID: {video_id}"
            )
        
        try:
            status_response = client.videos.retrieve(video_id)
            status = status_response.status
            
            if status != last_status:
                print(f"  Job {video_id}: Status: {status} (elapsed: {int(elapsed_time)}s)")
                last_status = status
            
            if status == 'completed':
                print(f"  ✅ Job {video_id} completed! Streaming video content...")
                stream_video_content(api_key, video_id, output_path)
                # Remove audio from Sora-generated video (we'll add our own voiceover)
                print(f"  Removing audio from Sora-generated video...")
                output_path = remove_audio_from_video(output_path, ffmpeg_path=find_ffmpeg())
                print(f"  ✅ Video saved (no audio): {output_path}")
                return output_path
                
            elif status == 'failed':
                error_obj = getattr(status_response, 'error', 'Unknown error')
                # Extract error message and code if available
                if isinstance(error_obj, dict):
                    error_msg = error_obj.get('message', str(error_obj))
                    error_code = error_obj.get('code', None)
                elif hasattr(error_obj, 'message') and hasattr(error_obj, 'code'):
                    error_msg = error_obj.message
                    error_code = error_obj.code
                else:
                    error_msg = str(error_obj)
                    error_code = None
                
                # Create exception with error code if available
                exception_msg = f"Video generation failed for job {video_id}: {error_msg}"
                exception = Exception(exception_msg)
                if error_code:
                    exception.code = error_code
                raise exception
            
            time.sleep(poll_interval)
            
        except Exception as e:
            if 'retrieve' in str(e).lower() or 'not found' in str(e).lower():
                print(f"  ⚠️  Warning: Could not retrieve job {video_id} status: {e}")
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
    duration=8,
    aspect_ratio='16:9',
    poll_interval=10,
    max_wait_time=600,
    reference_image_path=None
):
    """
    Generate a video from a text prompt using OpenAI Sora 2 API.
    
    Args:
        prompt: Text prompt describing the video to generate
        output_path: Path to save the output video (MP4)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        model: Model to use ('sora-2' or 'sora-2-pro', default: 'sora-2')
        resolution: Video resolution (default: '1280x720')
        duration: Video duration in seconds (default: 8, max varies by model)
        aspect_ratio: Aspect ratio (default: '16:9')
        poll_interval: Seconds to wait between status checks (default: 10)
        max_wait_time: Maximum time to wait for completion in seconds (default: 600)
        reference_image_path: Path to reference image to use as first frame (optional)
        
    Returns:
        Path to generated video file
    """
    """
    Generate a video from a text prompt using OpenAI Sora 2 API (blocking).
    This is a convenience function that combines start_video_generation_job and wait_for_video_completion.
    """
    print("Creating video generation job...")
    video_id = start_video_generation_job(
        prompt=prompt,
        api_key=api_key,
        model=model,
        resolution=resolution,
        duration=duration,
        reference_image_path=reference_image_path
    )
    print(f"✅ Video generation started! Job ID: {video_id}")
    
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


def stitch_videos(video_paths, output_path, ffmpeg_path=None):
    """
    Stitch multiple video files together into one video using ffmpeg.
    
    Args:
        video_paths: List of video file paths in order (should be sorted by segment ID)
        output_path: Path to save the stitched video
        ffmpeg_path: Path to ffmpeg executable (if None, will try to find it)
        
    Returns:
        Path to the stitched video file
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
                # Escape single quotes and backslashes for ffmpeg
                escaped_path = video_path.replace("'", "'\\''").replace("\\", "/")
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
        
        # Run ffmpeg to stitch
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Now apply fade in to first second and fade out to last second
        print(f"   Applying fade in (1s) and fade out (1s)...")
        video_duration = get_media_duration(temp_stitched, ffmpeg_path)
        if video_duration and video_duration > 2.0:  # Only apply fades if video is longer than 2 seconds
            # Apply fade in (first 1 second) and fade out (last 1 second)
            fade_out_start = max(0, video_duration - 1.0)
            cmd_fade = [
                ffmpeg_path,
                '-i', temp_stitched,
                '-vf', f'fade=t=in:st=0:d=1,fade=t=out:st={fade_out_start}:d=1',  # Fade in 1s, fade out 1s
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-y',
                output_path
            ]
            
            try:
                subprocess.run(
                    cmd_fade,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                print(f"   ✅ Fade in/out applied successfully")
                # Clean up temp file
                if os.path.exists(temp_stitched):
                    try:
                        os.remove(temp_stitched)
                    except:
                        pass
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️  Fade application failed: {e.stderr if e.stderr else 'Unknown error'}")
                print(f"   Using stitched video without fade effects")
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
        
        print(f"✅ Videos stitched successfully to: {output_path}")
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


def generate_srt_from_audio(audio_path, script, output_path=None, api_key=None, segment_duration=12.0):
    """
    Generate an SRT subtitle file with word-by-word timing using OpenAI Whisper API.
    Displays 1-2 words at a time as they are narrated, with words always side by side.
    Uses word-level timestamps for precise synchronization between captions and audio.
    
    Args:
        audio_path: Path to the audio file (voiceover) to analyze
        script: The narration script text (for reference/validation)
        output_path: Path to save the SRT file (default: temp file)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        segment_duration: Not used (kept for compatibility, word-level timing is used instead)
        
    Returns:
        Path to the generated SRT file, or None if generation fails
    """
    if not OPENAI_AVAILABLE:
        print("⚠️  OpenAI library not available, falling back to estimated timing")
        return None
    
    import tempfile
    
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        output_path = os.path.join(temp_dir, f"subtitles_{timestamp}.srt")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    if not api_key:
        print("⚠️  No OpenAI API key available, falling back to estimated timing")
        return None
    
    client = OpenAI(api_key=api_key)
    
    try:
        print("🎤 Transcribing audio with Whisper for word-level timestamps...")
        
        # Use Whisper API to get word-level timestamps
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",  # Get detailed JSON response
                timestamp_granularities=["word"]  # Request word-level timestamps
            )
        
        # Extract word-level timestamps from response
        words = []
        
        # Check if response is a dict (JSON) or object
        if isinstance(transcript, dict):
            # Handle dict response
            if 'words' in transcript and transcript['words']:
                words = transcript['words']
            elif 'segments' in transcript:
                # Extract words from segments
                for segment in transcript['segments']:
                    if 'words' in segment and segment['words']:
                        words.extend(segment['words'])
        else:
            # Handle object response
            if hasattr(transcript, 'words') and transcript.words:
                words = transcript.words
            elif hasattr(transcript, 'segments') and transcript.segments:
                # Fallback: extract words from segments
                for segment in transcript.segments:
                    if hasattr(segment, 'words') and segment.words:
                        words.extend(segment.words)
                    elif isinstance(segment, dict) and 'words' in segment:
                        words.extend(segment['words'])
        
        if not words:
            print("⚠️  No word-level timestamps available from Whisper")
            print(f"   Response type: {type(transcript)}")
            print(f"   Response keys/attrs: {dir(transcript) if not isinstance(transcript, dict) else list(transcript.keys())}")
            print("   Falling back to estimated timing")
            return None
        
        print(f"✅ Transcribed {len(words)} words with timestamps")
        
        # Format time as SRT format: HH:MM:SS,mmm
        def format_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        # Create word-by-word subtitles (1-2 words at a time, side by side)
        srt_content = []
        subtitle_index = 1
        
        # Parse words with their timestamps
        word_list = []
        for word_data in words:
            # Extract word and timing
            if isinstance(word_data, dict):
                word = word_data.get('word', '').strip()
                start = word_data.get('start', 0)
                end = word_data.get('end', start + 0.3)
            else:
                word = getattr(word_data, 'word', '').strip()
                start = getattr(word_data, 'start', 0)
                end = getattr(word_data, 'end', start + 0.3)
            
            if word:
                word_list.append({
                    'word': word,
                    'start': start,
                    'end': end
                })
        
        if not word_list:
            print("⚠️  No words found in transcription")
            return None
        
        # Generate subtitles with new requirements:
        # 1. Words appear at exact transcription timestamps (no delay)
        # 2. Each word stays on screen for exactly 0.75 seconds
        # 3. Words accumulate and grow horizontally (centered)
        # 4. Words are always side by side (never stacked)
        
        WORD_DISPLAY_DURATION = 0.75  # Each word stays on screen for 0.75 seconds
        
        # Create time intervals where subtitles need to be updated
        # Each word creates an interval: [start_time, start_time + 0.75]
        time_points = set()
        for word in word_list:
            time_points.add(word['start'])
            time_points.add(word['start'] + WORD_DISPLAY_DURATION)
        
        # Sort time points
        sorted_times = sorted(time_points)
        
        # For each time point, determine which words should be visible
        # A word is visible if: start_time <= current_time < start_time + 0.75
        for i, current_time in enumerate(sorted_times):
            # Find all words visible at this time
            visible_words = []
            for word in word_list:
                word_start = word['start']
                word_end = word_start + WORD_DISPLAY_DURATION
                # Word is visible if current_time is within its display window
                if word_start <= current_time < word_end:
                    visible_words.append(word)
            
            # Only create subtitle entry if there are visible words
            if visible_words:
                # Sort visible words by their start time to maintain order
                visible_words.sort(key=lambda w: w['start'])
                
                # Create text with words side by side (space-separated, single line)
                word_text = " ".join([w['word'] for w in visible_words])
                
                # Determine the time range for this subtitle entry
                # Start: current time point
                # End: next time point (or end of video if this is the last)
                start_time = current_time
                if i + 1 < len(sorted_times):
                    end_time = sorted_times[i + 1]
                else:
                    # Last time point - extend to the last word's end time
                    end_time = max([w['start'] + WORD_DISPLAY_DURATION for w in visible_words])
                
                # Create SRT entry (words always side by side, not stacked)
                srt_content.append(f"{subtitle_index}")
                srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
                srt_content.append(word_text)  # Single line, words side by side, centered
                srt_content.append("")  # Empty line between entries
                subtitle_index += 1
        
        # Write SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_content))
        
        print(f"✅ Generated SRT with {subtitle_index - 1} subtitle entries (words accumulate horizontally, 0.75s display duration)")
        return output_path
        
    except Exception as e:
        print(f"⚠️  Whisper transcription failed: {e}")
        print("   Falling back to estimated timing...")
        return None


def generate_srt_from_script(script, video_duration, output_path=None):
    """
    Generate an SRT subtitle file from a script, timing it based on video duration.
    This is a fallback method when audio-based timing is not available.
    Splits script into natural phrases/sentences and distributes them evenly.
    
    Args:
        script: The narration script text
        video_duration: Total video duration in seconds
        output_path: Path to save the SRT file (default: temp file)
        
    Returns:
        Path to the generated SRT file
    """
    import tempfile
    import re
    
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        output_path = os.path.join(temp_dir, f"subtitles_{timestamp}.srt")
    
    # Split script into sentences - preserve punctuation
    # Split on sentence-ending punctuation (. ! ?) but keep the punctuation with the sentence
    sentences = re.split(r'([.!?]+)', script)
    
    # Recombine sentences with their punctuation
    sentence_list = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences) and re.match(r'^[.!?]+$', sentences[i + 1].strip()):
            sentence += sentences[i + 1].strip()
            i += 2
        else:
            i += 1
        if sentence:
            sentence_list.append(sentence)
    
    # If no sentences found, split by commas or just use the whole script
    if not sentence_list:
        sentence_list = [p.strip() for p in re.split(r'[,;]', script) if p.strip()]
    if not sentence_list:
        sentence_list = [script]
    
    # Calculate timing for each sentence
    # Reserve 0.5 seconds at the start and end
    usable_duration = video_duration - 1.0
    if usable_duration <= 0:
        usable_duration = video_duration
    
    # Distribute sentences evenly, but allow for natural variation
    sentence_count = len(sentence_list)
    if sentence_count == 0:
        return None
    
    # Calculate timing based on natural speech patterns
    # Average reading speed: ~2.3 words/second, but with pauses (ellipses, dashes) it's slower
    words_per_second = 2.3  # Base reading speed
    
    # Format time as SRT format: HH:MM:SS,mmm
    def format_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    # Generate SRT content with new requirements:
    # 1. Words appear at estimated timestamps (no delay, based on reading speed)
    # 2. Each word stays on screen for exactly 0.75 seconds
    # 3. Words accumulate and grow horizontally (centered)
    # 4. Words are always side by side (never stacked)
    
    WORD_DISPLAY_DURATION = 0.75  # Each word stays on screen for 0.75 seconds
    
    # Build word list with estimated timestamps
    word_list = []
    current_time = 0.2  # Start slightly earlier (0.2 seconds in)
    
    for sentence in sentence_list:
        # Preserve punctuation - don't remove it
        sentence = sentence.strip()
        
        if not sentence:
            continue
        
        # Split sentence into words (preserving punctuation attached to words)
        words = re.findall(r'\S+', sentence)
        
        if not words:
            continue
        
        # Calculate timing for each word in this sentence
        word_count = len(words)
        pause_count = sentence.count('...') + sentence.count('..') + sentence.count('—') + sentence.count('-')
        pause_duration = pause_count * 0.5  # Each pause adds 0.5 seconds
        base_duration = word_count / words_per_second
        sentence_duration = base_duration + pause_duration
        
        # Apply bounds: minimum 0.8 seconds, maximum 4 seconds per sentence
        min_duration = 0.8
        max_duration = 4.0
        sentence_duration = min(max_duration, max(min_duration, sentence_duration))
        
        # Don't exceed video duration
        if current_time + sentence_duration > video_duration - 0.2:
            sentence_duration = max(0.3, video_duration - current_time - 0.2)
        
        # Calculate duration per word
        word_duration = sentence_duration / word_count if word_count > 0 else 0.4
        word_duration = min(0.7, max(0.35, word_duration))  # 0.35-0.7 seconds per word
        
        # Add words with estimated start times
        for word in words:
            if current_time >= video_duration - 0.2:
                break
            word_list.append({
                'word': word,
                'start': current_time,
                'end': current_time + word_duration
            })
            current_time += word_duration
    
    if not word_list:
        return None
    
    # Create time intervals where subtitles need to be updated (same logic as audio-based)
    time_points = set()
    for word in word_list:
        time_points.add(word['start'])
        time_points.add(word['start'] + WORD_DISPLAY_DURATION)
    
    # Sort time points
    sorted_times = sorted(time_points)
    
    # For each time point, determine which words should be visible
    srt_content = []
    subtitle_index = 1
    
    for i, current_time in enumerate(sorted_times):
        # Find all words visible at this time
        visible_words = []
        for word in word_list:
            word_start = word['start']
            word_end = word_start + WORD_DISPLAY_DURATION
            # Word is visible if current_time is within its display window
            if word_start <= current_time < word_end:
                visible_words.append(word)
        
        # Only create subtitle entry if there are visible words
        if visible_words:
            # Sort visible words by their start time to maintain order
            visible_words.sort(key=lambda w: w['start'])
            
            # Create text with words side by side (space-separated, single line)
            word_text = " ".join([w['word'] for w in visible_words])
            
            # Determine the time range for this subtitle entry
            start_time = current_time
            if i + 1 < len(sorted_times):
                end_time = sorted_times[i + 1]
            else:
                # Last time point - extend to the last word's end time
                end_time = max([w['start'] + WORD_DISPLAY_DURATION for w in visible_words])
            
            # Don't exceed video duration
            end_time = min(end_time, video_duration - 0.1)
            
            if start_time < end_time:
                # Create SRT entry (words always side by side, not stacked)
                srt_content.append(f"{subtitle_index}")
                srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
                srt_content.append(word_text)  # Single line, words side by side, centered
                srt_content.append("")  # Empty line between entries
                subtitle_index += 1
    
    # Write SRT file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_content))
        return output_path
    except Exception as e:
        print(f"⚠️  Failed to create SRT file: {e}")
        return None


def add_subtitles_to_video(video_path, script, video_duration, output_path=None, ffmpeg_path=None, audio_path=None, api_key=None):
    """
    Add styled subtitles to a video using FFmpeg.
    Creates an SRT file with word-level timestamps from audio (if available) or estimates from script.
    
    Args:
        video_path: Path to the input video
        script: The narration script text
        video_duration: Total video duration in seconds
        output_path: Path to save the output video (default: overwrite input)
        ffmpeg_path: Path to ffmpeg executable (default: auto-detect)
        audio_path: Path to the voiceover audio file for accurate word-level timing (optional)
        api_key: OpenAI API key for Whisper transcription (optional)
        
    Returns:
        Path to the output video with subtitles
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise Exception("FFmpeg not found. Cannot add subtitles to video.")
    
    if not script or len(script.strip()) == 0:
        print("⚠️  No script provided, skipping subtitle generation")
        return video_path
    
    # Clean script: remove musical break and visual break markers (these shouldn't appear in captions)
    import re
    cleaned_script = script
    # Remove [MUSICAL BREAK] and [VISUAL BREAK] markers (case-insensitive) and any text that follows them
    # This handles cases where the marker might have explanatory text after it
    cleaned_script = re.sub(r'\[MUSICAL\s+BREAK\][^\.!?\n\[\]]*[\.!?\n]?', '', cleaned_script, flags=re.IGNORECASE)
    cleaned_script = re.sub(r'\[VISUAL\s+BREAK\][^\.!?\n\[\]]*[\.!?\n]?', '', cleaned_script, flags=re.IGNORECASE)
    # Also remove any standalone markers (exact matches)
    cleaned_script = cleaned_script.replace('[MUSICAL BREAK]', '')
    cleaned_script = cleaned_script.replace('[VISUAL BREAK]', '')
    cleaned_script = cleaned_script.replace('[musical break]', '')
    cleaned_script = cleaned_script.replace('[visual break]', '')
    # Remove any phrases that might have been generated describing these breaks (without brackets)
    # This catches cases like "visual break look over castle" that might appear in transcriptions
    cleaned_script = re.sub(r'(?i)\b(visual\s+break|musical\s+break)\s+[^\.!?\n]*', '', cleaned_script)
    # Clean up any extra whitespace, newlines, or punctuation artifacts
    cleaned_script = re.sub(r'\s+', ' ', cleaned_script)
    cleaned_script = re.sub(r'\s*\.\s*\.\s*\.\s*', '...', cleaned_script)  # Normalize ellipses
    cleaned_script = cleaned_script.strip()
    
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        # If video already has "_with_subtitles" in the name, remove it first to avoid duplicates
        if "_with_subtitles" in base:
            # Remove the last occurrence of "_with_subtitles" and any trailing numbers
            base = base.rsplit("_with_subtitles", 1)[0]
        output_path = f"{base}_with_subtitles{ext}"
    
    import tempfile
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    srt_path = os.path.join(temp_dir, f"subtitles_{timestamp}.srt")
    
    try:
        # Try to generate SRT with word-level timestamps from audio first
        srt_path = None
        if audio_path and os.path.exists(audio_path):
            print("🎯 Attempting to generate captions with exact audio timing...")
            srt_path = generate_srt_from_audio(audio_path, cleaned_script, srt_path, api_key)
        
        # Fallback to estimated timing if audio-based generation failed
        if not srt_path or not os.path.exists(srt_path):
            print("📝 Using estimated timing from script (audio-based timing unavailable)...")
            srt_path = generate_srt_from_script(cleaned_script, video_duration, srt_path)
        
        if not srt_path or not os.path.exists(srt_path):
            print("⚠️  Failed to generate SRT file, skipping subtitles")
            return video_path
        
        # FFmpeg subtitle styling
        # Professional, clean appearance suitable for YouTube
        # Centered, horizontal growth as words accumulate
        subtitle_style = (
            "FontName=Segoe UI,"
            "FontSize=20,"
            "PrimaryColour=&H00F5F5F5,"  # Soft white (slightly off-white for better readability)
            "OutlineColour=&H00000000,"  # Black outline
            "BackColour=&H90000000,"  # More opaque black background for better contrast
            "Bold=0,"  # Not bold for cleaner, more professional look
            "Alignment=2,"  # Bottom center (1=bottom-left, 2=bottom-center, 3=bottom-right)
            "MarginV=20,"  # 20 pixels from bottom (ensures captions are at the bottom)
            "Outline=1.5,"  # 1.5 pixel outline (slightly thinner for elegance)
            "Shadow=0.3,"  # Subtle shadow
            "MarginL=10,"  # Left margin to prevent cutoff
            "MarginR=10"  # Right margin to prevent cutoff
        )
        
        # Escape SRT path for Windows (replace backslashes and escape special characters)
        srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        
        # Build FFmpeg command to burn subtitles
        # Use subtitles filter with force_style for consistent appearance
        # Ensure subtitles are always visible and don't get cut off
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-vf", f"subtitles='{srt_path_escaped}':force_style='{subtitle_style}'",
            "-c:a", "copy",  # Copy audio without re-encoding
            "-y",  # Overwrite output
            output_path
        ]
        
        # Run FFmpeg
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Subtitles added to video: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to add subtitles: {e.stderr if e.stderr else 'Unknown error'}")
        print("   Continuing without subtitles...")
        return video_path
    except Exception as e:
        print(f"⚠️  Error adding subtitles: {e}")
        print("   Continuing without subtitles...")
        return video_path
    finally:
        # Clean up SRT file
        if os.path.exists(srt_path):
            try:
                os.remove(srt_path)
            except:
                pass


def stream_video_content(api_key, video_id, filepath):
    """
    Stream video content from OpenAI API content endpoint to a local file.
    
    Args:
        api_key: OpenAI API key
        video_id: ID of the video to retrieve
        filepath: Local path to save the video
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Use the content endpoint to stream the MP4
        content_url = f"https://api.openai.com/v1/videos/{video_id}/content"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # Stream the video content
        response = requests.get(content_url, headers=headers, stream=True)
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
                        print(f"\rStreaming progress: {percent:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        
    except Exception as e:
        raise Exception(f"Failed to stream video content: {e}")


def generate_and_upload_sora(
    prompt,
    title,
    description='',
    tags=None,
    category_id='22',
    privacy_status='private',
    thumbnail_file=None,
    playlist_id=None,
    output_video_path=None,
    api_key=None,
    model='sora-2',
    resolution='1280x720',
    duration=8,
    aspect_ratio='16:9',
    poll_interval=10,
    max_wait_time=600,
    keep_video=False,
    upscale_to_1080p=True,
    test=False,
    skip_narration=False,
    skip_upload=False
):
    """
    Generate a video from a text prompt using Sora 2 and upload it to YouTube.
    
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
    
    # Create output folder for reference image and final video
    current_dir = os.getcwd()
    output_folder = os.path.join(current_dir, "video_output")
    
    # Archive workflow files before cleanup (save previous run's files)
    if os.path.exists(output_folder):
        print("\n" + "="*60)
        print("📦 Archiving previous workflow files...")
        print("="*60)
        archive_workflow_files()
        print("="*60 + "\n")
    
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
    if duration % 12 != 0:
        raise ValueError(f"Duration must be divisible by 12. Provided duration: {duration} seconds")
    
    # Fixed parameters
    SEGMENT_DURATION = 12.0  # Each segment is 12 seconds (video or still)
    
    # Calculate total number of segments (each segment is 12 seconds)
    num_segments = int(duration / 12)
    
    # Calculate number of still images (approximately 1/3 of segments)
    num_still_images = max(1, int(num_segments / 3))
    
    # Calculate number of video segments (approximately 2/3 of segments)
    num_videos = num_segments - num_still_images
    
    # Ensure we have at least 1 video segment
    if num_videos < 1:
        num_videos = 1
        num_still_images = num_segments - num_videos
    
    segment_duration = SEGMENT_DURATION
    
    print(f"Duration: {duration}s | Total segments: {num_segments} | Video segments: {num_videos} | Still images: {num_still_images}")
    
    # Step 0: Generate overarching script from video prompt (AI call)
    generated_script = None
    generated_segment_texts = []
    generated_video_prompts = []
    reference_image_info = None  # Initialize reference image info
    narration_audio_path = None  # Will be set in Step 0.1 (narration generation - MUST happen before video generation)
    original_voiceover_backup = None  # Will be set in Step 0.1 (narration generation)
    
    print("Step 0: Loading or generating overarching script...")
    # CRITICAL: If any API call fails before Sora 2 video generation, exit the program
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
            print("Generated script:")
        
        print("=" * 60)
        print(generated_script)
        print("=" * 60)
        
        # CRITICAL: Step 0.1 - Generate and save narration FIRST (before any video generation)
        # The assumption is that narration exists when video generation workflow begins
        print("\n" + "="*60)
        print("Step 0.1: Generating and saving narration (MUST complete before video generation)...")
        print("="*60 + "\n")
        
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
            # Generate narration now - this is the FINAL narration that will be used throughout
            try:
                # Use the standard narration path (not a temp file)
                current_dir = os.getcwd()
                narration_audio_path = os.path.join(current_dir, NARRATION_AUDIO_PATH)
                
                print(f"🎙️  Generating narration audio from script...")
                narration_audio_path, original_voiceover_backup = generate_voiceover_from_folder(
                    script=generated_script,
                    output_path=narration_audio_path,
                    narration_folder=None,
                    break_duration=1000,  # 1 second for breaks
                    music_volume=0.07  # 7% volume for background music
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
            # Generate it temporarily, but it won't be the final version
            try:
                import tempfile
                temp_dir = tempfile.gettempdir()
                timestamp = int(time.time())
                narration_audio_path = os.path.join(temp_dir, f"voiceover_segmentation_{timestamp}.mp3")
                
                print(f"⚠️  skip_narration=True, generating temporary narration for segmentation only...")
                narration_audio_path, _ = generate_voiceover_from_folder(
                    script=generated_script,
                    output_path=narration_audio_path,
                    narration_folder=None,
                    break_duration=1000,
                    music_volume=0.07
                )
                print(f"✅ Generated temporary narration for segmentation: {narration_audio_path}")
                print(f"   Note: Narration will be regenerated in Step 3")
            except Exception as e:
                print(f"⚠️  Failed to generate temporary narration for segmentation: {e}")
                print("   Falling back to rule-based segmentation...")
                narration_audio_path = None
        
        # Step 0.5: Segment script based on narration timing (or fallback to rule-based)
        print(f"\nStep 0.5: Segmenting script into {num_segments} segments...")
        
        if narration_audio_path and os.path.exists(narration_audio_path):
            print("   Using narration-based segmentation (12-second segments)...")
            # CRITICAL: This extracts segments from the ACTUAL NARRATION AUDIO using Whisper timestamps
            # The segments contain words that were actually spoken, not the original script text
            # These narration-based segments will be used for both video and image generation
            generated_segment_texts = segment_script_by_narration(
                script=generated_script,
                audio_path=narration_audio_path,
                segment_duration=segment_duration,
                api_key=api_key,
                expected_num_segments=num_segments  # Pass expected number to limit segments
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
        if generated_script:
            print("Step 0.52: Generating YouTube tags from script...")
            try:
                generated_tags = generate_tags_from_script(
                    script=generated_script,
                    video_prompt=prompt,
                    api_key=api_key,
                    model='gpt-4o'
                )
                
                # Combine user-provided tags with generated tags
                user_tags = tags if tags else []
                # Convert to list if it's not already
                if not isinstance(user_tags, list):
                    user_tags = [user_tags] if user_tags else []
                
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
                    print(f"✅ Generated {len(generated_tags)} tags from script:")
                    print(f"   {', '.join(generated_tags)}")
                print(f"   Final tags: {', '.join(tags)}")
            except Exception as e:
                print(f"⚠️  Failed to generate tags from script: {e}")
                print("   Using only user-provided tags..." if tags else "   Continuing without tags...")
        
        # Step 0.55: Analyze script for reference images (set of reference images needed)
        print("\n" + "="*60)
        print("Step 0.55: Analyzing script for reference image requirements...")
        
        reference_images = []
        if generated_script:
            reference_images = analyze_script_for_reference_images(
                script=generated_script,
                video_prompt=prompt,
                api_key=api_key,
                model='gpt-5-2025-08-07'
            )
            if len(reference_images) > 0:
                print(f"✅ Identified {len(reference_images)} reference image(s) needed:")
                for ref_img in reference_images:
                    ref_id = ref_img.get('id', 'unknown')
                    ref_desc = ref_img.get('description', '')
                    ref_type = ref_img.get('type', 'subject')
                    print(f"   - {ref_id}: {ref_desc} (type: {ref_type})")
            else:
                print(f"⚠️  No reference images identified for this video.")
                print(f"   This may be normal if the video doesn't require visual consistency (e.g., generic topics, abstract concepts)")
                print(f"   All video segments will be generated without reference images.")
        
        # Step 0.65: Analyze script for still image opportunities and segment assignments (MUST be before Sora prompt generation)
        print("\n" + "="*60)
        print("Step 0.65: Analyzing script for still image opportunities and segment assignments...")
        print("="*60 + "\n")
        
        still_image_segments = []
        segment_assignments = []
        try:
            # Debug: Check if reference images are being passed
            if reference_images and len(reference_images) > 0:
                print(f"   Passing {len(reference_images)} reference image(s) to segment assignment analysis:")
                for ref_img in reference_images:
                    print(f"      - {ref_img.get('id', 'unknown')}: {ref_img.get('description', '')}")
            else:
                print(f"   ⚠️  No reference images to pass - all segments will be assigned 'no ref'")
            
            # Pass the calculated number of still images and reference images
            # CRITICAL: generated_segment_texts contains NARRATION-BASED segments (words actually spoken)
            # These narration segments are used to generate both still image and video prompts
            analysis_result = analyze_script_for_still_images(
                script=generated_script,
                segment_texts=generated_segment_texts,  # NARRATION-BASED segments (from actual audio)
                target_num_stills=num_still_images,  # Pass calculated number (approximately 1/3 of segments)
                api_key=api_key,
                model='gpt-5-2025-08-07',  # Match Sora prompt model
                reference_images=reference_images  # Pass list of reference images
            )
            
            still_image_segments = analysis_result.get('still_image_segments', [])
            segment_assignments = analysis_result.get('segment_assignments', [])
            
            if still_image_segments:
                print(f"✅ Identified {len(still_image_segments)} still image position(s)")
                for seg_info in still_image_segments:
                    seg_id = seg_info.get('segment_id', 'unknown')
                    print(f"   - Still image at segment {seg_id} (12s)")
            
            if segment_assignments:
                print(f"✅ Segment assignments:")
                for assignment in segment_assignments:
                    seg_id = assignment.get('segment_id', 'unknown')
                    seg_type = assignment.get('type', 'video')
                    ref_id = assignment.get('reference_image_id')
                    ref_str = f" (ref: {ref_id})" if ref_id else " (no ref)"
                    print(f"   - Segment {seg_id}: {seg_type}{ref_str}")
            else:
                print("   No segment assignments identified")
        except Exception as e:
            print(f"⚠️  Still image analysis failed: {e}")
            still_image_segments = []
            segment_assignments = []
        
        # Step 0.6: Convert each segment text to Sora-2 video prompt (AI call per segment)
        # Now that we know where still images will be placed, we can calculate correct timing
        print("\n" + "="*60)
        print(f"Step 0.6: Converting segment texts to Sora-2 video prompts...")
        print("="*60 + "\n")
        
        # Note: We no longer pass reference_image_info here since we handle multiple reference images
        # per segment at the video generation level based on segment_assignments
        # CRITICAL: generated_segment_texts contains NARRATION-BASED segments (words actually spoken in audio)
        # These narration segments are used to generate Sora video prompts, ensuring videos match what's being narrated
        generated_video_prompts = generate_sora_prompts_from_segments(
            segment_texts=generated_segment_texts,  # NARRATION-BASED segments (from actual audio)
            segment_duration=segment_duration,
            total_duration=duration,
            overarching_script=generated_script,  # Pass full script for context and chronological flow
            reference_image_info=None,  # No longer using single reference image - handled per segment
            still_image_segments=still_image_segments,  # Pass still image positions for correct timing
            api_key=api_key,
            model='gpt-5-2025-08-07'
        )
        
        print(f"\nSora Prompts ({len(generated_video_prompts)} segments):")
        print("-" * 60)
        for i, vp in enumerate(generated_video_prompts, 1):
            print(f"\nSegment {i}: {vp}")
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: API call failed before Sora 2 video generation: {e}")
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
    # All video generation steps (reference images, still images, Sora videos) 
    # assume narration already exists
    # ============================================================================
    
    # Step 1: Generate reference images (multiple if needed)
    print("\n" + "="*60)
    print("Step 1: Generating reference images...")
    print("="*60 + "\n")
    print("📌 NOTE: Narration was already generated in Step 0.1 (before video generation)")
    if narration_audio_path and os.path.exists(narration_audio_path):
        print(f"   ✅ Narration ready: {os.path.basename(narration_audio_path)}")
    else:
        print(f"   ⚠️  WARNING: Narration path not found, but continuing with video generation...")
    
    reference_image_paths = {}  # Map reference_image_id -> file_path
    master_image_path = None  # Keep for backward compatibility (use first reference image)
    
    # CRITICAL: If API call fails before Sora 2 video generation, exit the program
    try:
        if reference_images and len(reference_images) > 0:
            print(f"Generating {len(reference_images)} reference image(s)...")
            timestamp = int(time.time())
            
            for i, ref_img in enumerate(reference_images):
                ref_id = ref_img.get('id', f'ref_{i+1}')
                ref_description = ref_img.get('description', '')
                image_prompt = ref_img.get('image_prompt', '')
                
                print(f"\n  Reference image {i+1}/{len(reference_images)}: {ref_id} - {ref_description}")
                
                # Generate image path
                ref_image_path = os.path.join(output_folder, f"reference_image_{ref_id}_{timestamp}.png")
                
                # Generate the image
                if image_prompt:
                    print(f"    Using AI-generated prompt...")
                    ref_image_path = generate_master_image_from_prompt(
                        image_prompt=image_prompt,
                        output_path=ref_image_path,
                        api_key=api_key,
                        resolution=resolution
                    )
                else:
                    print(f"    Using description-based generation...")
                    ref_image_path = generate_master_image_from_prompt(
                        description=ref_description,
                        output_path=ref_image_path,
                        api_key=api_key,
                        resolution=resolution
                    )
                
                reference_image_paths[ref_id] = ref_image_path
                print(f"    ✅ Generated: {ref_image_path}")
                
                # Set master_image_path to first reference image for backward compatibility
                if master_image_path is None:
                    master_image_path = ref_image_path
        else:
            # Fallback: generate single reference image from description
            print("No reference images identified, generating single reference image from description...")
            timestamp = int(time.time())
            master_image_path = os.path.join(output_folder, f"reference_image_{timestamp}.png")
            master_image_path = generate_master_image_from_prompt(
                description=description or prompt,
                output_path=master_image_path,
                api_key=api_key,
                resolution=resolution
            )
            reference_image_paths['ref_1'] = master_image_path
            print(f"✅ Master image generated: {master_image_path}")
        
        print(f"\n✅ All reference images generated: {len(reference_image_paths)} image(s)")
        for ref_id, path in reference_image_paths.items():
            print(f"   {ref_id}: {path}")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Reference image generation API call failed before Sora 2 video generation: {e}")
        print("   Exiting program as requested. All API calls must succeed before video generation.")
        import sys
        sys.exit(1)
    
    # Step 1.5: Generate still images and panning videos (if any identified)
    still_image_videos = {}  # Map segment_id -> video_path for still image panning videos
    still_image_segment_ids = set(seg['segment_id'] for seg in still_image_segments) if still_image_segments else set()
    
    if still_image_segments:
        print("\n" + "="*60)
        print(f"Step 1.5: Generating {len(still_image_segments)} still image(s) with panning...")
        print("="*60 + "\n")
        
        try:
            for seg_info in still_image_segments:
                segment_id = seg_info['segment_id']
                image_prompt = seg_info['image_prompt']
                still_duration = seg_info['duration']
                
                if segment_id == 0:
                    print(f"Generating opening still image (test mode)...")
                else:
                    print(f"Generating still image after video {segment_id}...")
                print(f"   Prompt: {image_prompt[:150]}...")
                print(f"   Duration: {still_duration:.1f}s")
                
                # Generate DALL-E image
                timestamp = int(time.time())
                still_image_path = os.path.join(output_folder, f"still_image_segment_{segment_id}_{timestamp}.png")
                
                # Sanitize prompt for content policy
                sanitized_prompt = sanitize_image_prompt(image_prompt)
                
                still_image_path = generate_image_from_prompt(
                    prompt=sanitized_prompt,
                    output_path=still_image_path,
                    api_key=api_key,
                    model='dall-e-3',
                )
                
                print(f"✅ Still image generated: {still_image_path}")
                
                # Create panning video from still image
                if segment_id == 0:
                    panning_video_path = os.path.join(output_folder, f"panning_video_opening_{timestamp}.mp4")
                else:
                    panning_video_path = os.path.join(output_folder, f"panning_video_segment_{segment_id}_{timestamp}.mp4")
                
                # Randomly choose corner-to-corner pan direction for variety
                import random
                pan_directions = ['top_left_to_bottom_right', 'top_right_to_bottom_left', 
                                 'bottom_left_to_top_right', 'bottom_right_to_top_left']
                pan_direction = random.choice(pan_directions)
                
                # Ensure still image panning is exactly 12 seconds
                panning_duration = 12.0  # Always exactly 12 seconds for still image panning
                panning_video_path = create_panning_video_from_image(
                    image_path=still_image_path,
                    output_path=panning_video_path,
                    duration=panning_duration,
                    pan_direction=pan_direction,
                    ffmpeg_path=find_ffmpeg()
                )
                
                still_image_videos[segment_id] = panning_video_path
                
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Still image generation failed: {e}")
            import sys
            sys.exit(1)
    
    # Step 2: Generate multiple videos with rate limiting
    print(f"Step 2: Generating {num_videos} video segment(s) using Sora 2...")
    
    # Each segment is 12 seconds
    video_segment_duration = 12
    
    # Create set of still image positions (after which video segment)
    still_image_segment_ids = set(seg['segment_id'] for seg in still_image_segments) if still_image_segments else set()
    
    # Use generated video prompts if available, otherwise use original prompt
    video_prompts_to_use = generated_video_prompts if generated_video_prompts else [prompt] * num_videos
    
    # Ensure we have the right number of prompts
    if len(video_prompts_to_use) < num_videos:
        # Pad with the last prompt or original prompt
        while len(video_prompts_to_use) < num_videos:
            video_prompts_to_use.append(video_prompts_to_use[-1] if video_prompts_to_use else prompt)
    elif len(video_prompts_to_use) > num_videos:
        # Take only the first num_videos
        video_prompts_to_use = video_prompts_to_use[:num_videos]
    
    # Rate limiting: 4 requests per minute = 15 seconds between requests
    rate_limit_delay = 15  # seconds
    
    generated_video_segments = []
    segment_video_paths = []
    video_path = None
    video_jobs = []  # List of (segment_id, video_id, output_path, prompt)
    
    try:
        # Step 2a: Start all video generation jobs 15 seconds apart (non-blocking)
        print("Step 2a: Starting video generation jobs...")
        
        # Create mapping from segment_id to assignment
        assignment_map = {}
        if segment_assignments:
            for assignment in segment_assignments:
                seg_id = assignment.get('segment_id')
                assignment_map[seg_id] = assignment
        
        # Filter to only video segments (skip still image segments)
        video_segment_ids = []
        for seg_id in range(1, num_segments + 1):
            assignment = assignment_map.get(seg_id, {'type': 'video', 'reference_image_id': None})
            if assignment.get('type') == 'video':
                video_segment_ids.append(seg_id)
        
        # Adjust num_videos to match actual video segments
        num_videos = len(video_segment_ids)
        print(f"Generating {num_videos} video segment(s) (out of {num_segments} total segments)")
        
        for video_idx, segment_id in enumerate(video_segment_ids, 1):
            # Get the prompt for this segment (use segment_id - 1 as index)
            if segment_id <= len(video_prompts_to_use):
                segment_prompt = video_prompts_to_use[segment_id - 1]
            else:
                segment_prompt = video_prompts_to_use[-1] if video_prompts_to_use else prompt
            
            # Validate prompt is not empty
            if not segment_prompt or len(segment_prompt.strip()) == 0:
                segment_prompt = prompt
            
            # Get assignment for this segment to determine reference image
            assignment = assignment_map.get(segment_id, {'type': 'video', 'reference_image_id': None})
            ref_image_id = assignment.get('reference_image_id')
            
            # Determine which reference image to use
            ref_image_to_use = None
            if ref_image_id and ref_image_id in reference_image_paths:
                ref_image_to_use = reference_image_paths[ref_image_id]
                print(f"Segment {segment_id}/{num_segments} (video {video_idx}/{num_videos}): Using reference image {ref_image_id}")
            elif master_image_path and os.path.exists(master_image_path):
                # Fallback to master image if no specific reference image assigned
                ref_image_to_use = master_image_path
                print(f"Segment {segment_id}/{num_segments} (video {video_idx}/{num_videos}): Using default master image")
            else:
                print(f"Segment {segment_id}/{num_segments} (video {video_idx}/{num_videos}): No reference image")
            
            print(f"  Prompt: {segment_prompt[:100]}...")
            
            # Create output path for this segment
            base, ext = os.path.splitext(output_video_path)
            segment_output_path = f"{base}_segment_{segment_id:03d}{ext}"
            
            # Start video generation job (non-blocking)
            try:
                video_segment_duration = 12
                
                video_id = start_video_generation_job(
                    prompt=segment_prompt,
                    api_key=api_key,
                    model=model,
                    resolution=resolution,
                    duration=video_segment_duration,
                    reference_image_path=ref_image_to_use
                )
                
                video_jobs.append({
                    'segment_id': segment_id,
                    'video_id': video_id,
                    'output_path': segment_output_path,
                    'prompt': segment_prompt,
                    'is_still_image': False
                })
                
            except Exception as e:
                print(f"❌ Failed to start segment {segment_id} job: {e}")
                raise
            
            # Rate limiting: wait before starting next job (except for the last segment)
            if segment_id < num_videos:
                time.sleep(rate_limit_delay)
        
        # Step 2b: Wait for all video generation jobs to complete with retry logic
        print(f"Step 2b: Waiting for {len(video_jobs)} video generation job(s) to complete...")
        
        for job in video_jobs:
            segment_id = job['segment_id']
            video_id = job['video_id']
            segment_output_path = job['output_path']
            segment_prompt = job['prompt']
            
            print(f"\n--- Processing Segment {segment_id} (Job {video_id}) ---")
            
            # Retry logic: try up to 3 times
            max_retries = 3
            segment_video_path = None
            last_error = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt == 1:
                        # First attempt: use the existing job
                        segment_video_path = wait_for_video_completion(
                            video_id=video_id,
                            output_path=segment_output_path,
                            api_key=api_key,
                            poll_interval=poll_interval,
                            max_wait_time=max_wait_time
                        )
                    else:
                        # Retry: start a new job
                        print(f"   Retry attempt {attempt}/{max_retries}: Starting new video generation job...")
                        # Get assignment for this segment to determine reference image for retry
                        assignment = assignment_map.get(segment_id, {'type': 'video', 'reference_image_id': None})
                        ref_image_id = assignment.get('reference_image_id')
                        retry_ref_image = None
                        if ref_image_id and ref_image_id in reference_image_paths:
                            retry_ref_image = reference_image_paths[ref_image_id]
                        elif master_image_path and os.path.exists(master_image_path):
                            retry_ref_image = master_image_path
                        
                        retry_video_id = start_video_generation_job(
                            prompt=segment_prompt,
                            api_key=api_key,
                            model=model,
                            resolution=resolution,
                            duration=12,
                            reference_image_path=retry_ref_image
                        )
                        print(f"   New job ID: {retry_video_id}")
                        
                        # Update output path for retry
                        base, ext = os.path.splitext(segment_output_path)
                        retry_output_path = f"{base}_retry{attempt}{ext}"
                        
                        segment_video_path = wait_for_video_completion(
                            video_id=retry_video_id,
                            output_path=retry_output_path,
                            api_key=api_key,
                            poll_interval=poll_interval,
                            max_wait_time=max_wait_time
                        )
                    
                    # Success - break out of retry loop
                    segment_video_paths.append(segment_video_path)
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
                    
                    # Check if this is a moderation_blocked error - don't retry these
                    is_moderation_blocked = False
                    if hasattr(e, 'code') and e.code == 'moderation_blocked':
                        is_moderation_blocked = True
                    elif 'moderation_blocked' in error_msg.lower():
                        is_moderation_blocked = True
                    elif hasattr(e, 'error') and hasattr(e.error, 'code') and e.error.code == 'moderation_blocked':
                        is_moderation_blocked = True
                    elif hasattr(e, 'error') and isinstance(e.error, dict) and e.error.get('code') == 'moderation_blocked':
                        is_moderation_blocked = True
                    
                    if is_moderation_blocked:
                        print(f"   🚫 Moderation blocked error detected - skipping retries and using fallback")
                        # Break out of retry loop immediately - will use fallback below
                        break
                    elif attempt < max_retries:
                        print(f"   ⚠️  Attempt {attempt} failed ({error_type}): {error_msg[:200]}")
                        print(f"   Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        # All retries exhausted - use still image fallback
                        print(f"   ❌ All {max_retries} attempts failed for segment {segment_id}")
                        print(f"   🔄 Using still image fallback for segment {segment_id}...")
                        
                        try:
                            # Get segment text for still image generation
                            segment_text = ""
                            if generated_segment_texts and segment_id <= len(generated_segment_texts):
                                segment_text = generated_segment_texts[segment_id - 1]
                            elif generated_script:
                                # Fallback: use a portion of the script
                                segment_text = generated_script[:500]
                            
                            # Generate still image prompt from segment text
                            fallback_image_prompt = generate_still_image_prompt(
                                script=generated_script if generated_script else "",
                                context_segment=segment_text,
                                position=segment_id,
                                num_videos=num_videos,
                                api_key=api_key,
                                model='gpt-5-2025-08-07',
                                previous_segment_text=generated_segment_texts[segment_id - 2] if segment_id > 1 and generated_segment_texts and segment_id - 2 < len(generated_segment_texts) else None,
                                next_segment_text=generated_segment_texts[segment_id] if segment_id <= len(generated_segment_texts) and generated_segment_texts else None,
                                reference_image_info=reference_image_info if 'reference_image_info' in locals() else None
                            )
                            
                            # Generate DALL-E image
                            timestamp = int(time.time())
                            fallback_image_path = os.path.join(output_folder, f"fallback_still_segment_{segment_id}_{timestamp}.png")
                            
                            # Sanitize prompt for content policy
                            sanitized_prompt = sanitize_image_prompt(fallback_image_prompt)
                            
                            fallback_image_path = generate_image_from_prompt(
                                prompt=sanitized_prompt,
                                output_path=fallback_image_path,
                                api_key=api_key,
                                model='dall-e-3',
                            )
                            
                            print(f"   ✅ Fallback still image generated: {fallback_image_path}")
                            
                            # Create panning video from still image (12 seconds)
                            fallback_video_path = os.path.join(output_folder, f"fallback_panning_segment_{segment_id}_{timestamp}.mp4")
                            
                            # Randomly choose pan direction
                            import random
                            pan_directions = ['top_left_to_bottom_right', 'top_right_to_bottom_left', 
                                             'bottom_left_to_top_right', 'bottom_right_to_top_left']
                            pan_direction = random.choice(pan_directions)
                            
                            # Create panning video matching the segment duration (to maintain synchronization)
                            # Use the same duration as the video segment would have been
                            fallback_video_duration = 12
                            fallback_video_path = create_panning_video_from_image(
                                image_path=fallback_image_path,
                                output_path=fallback_video_path,
                                duration=fallback_video_duration,
                                pan_direction=pan_direction,
                                ffmpeg_path=find_ffmpeg()
                            )
                            
                            print(f"   ✅ Fallback panning video created: {fallback_video_path}")
                            
                            # Add fallback video to segments
                            segment_video_paths.append(fallback_video_path)
                            generated_video_segments.append({
                                'segment_id': segment_id,
                                'prompt': segment_prompt,
                                'video_path': fallback_video_path,
                                'is_fallback': True
                            })
                            
                            print(f"   ✅ Segment {segment_id} fallback complete: {fallback_video_path}")
                            
                        except Exception as fallback_error:
                            print(f"   ❌ CRITICAL: Fallback still image generation also failed: {fallback_error}")
                            print(f"   Original error: {last_error}")
                            raise RuntimeError(f"Segment {segment_id} failed after {max_retries} retries and fallback generation failed: {fallback_error}")
            
            # If we exhausted retries and fallback also failed, the exception would have been raised
            if segment_video_path is None and segment_id not in [seg['segment_id'] for seg in generated_video_segments]:
                raise RuntimeError(f"Segment {segment_id} failed: {last_error}")
        
        # Step 2.1: Stitch all video segments together (including still image panning videos)
        # Use segment_assignments to determine order: videos and still images in sequence
        all_segment_paths = []
        
        # Create a mapping of segment_id -> video_path for Sora videos
        sora_video_map = {}
        for seg_info in generated_video_segments:
            sora_video_map[seg_info['segment_id']] = seg_info['video_path']
        
        # Create a mapping of segment_id -> still image video path
        still_image_map = {}
        for seg_id, still_path in still_image_videos.items():
            still_image_map[seg_id] = still_path
        
        # Create mapping from segment_id to assignment
        assignment_map = {}
        if segment_assignments:
            for assignment in segment_assignments:
                seg_id = assignment.get('segment_id')
                assignment_map[seg_id] = assignment
        
        # Combine segments in order based on segment_assignments
        for segment_id in range(1, num_segments + 1):
            assignment = assignment_map.get(segment_id, {'type': 'video', 'reference_image_id': None})
            seg_type = assignment.get('type', 'video')
            
            if seg_type == 'still':
                # Add still image
                if segment_id in still_image_map:
                    all_segment_paths.append(still_image_map[segment_id])
                    print(f"  Added still image for segment {segment_id}")
                else:
                    print(f"⚠️  Warning: Still image expected for segment {segment_id} but not found")
            elif seg_type == 'video':
                # Add video
                if segment_id in sora_video_map:
                    all_segment_paths.append(sora_video_map[segment_id])
                    print(f"  Added video for segment {segment_id}")
                else:
                    print(f"⚠️  Warning: Video expected for segment {segment_id} but not found")
        
        if not all_segment_paths:
            raise RuntimeError("No video segments were generated!")
        elif len(all_segment_paths) > 1:
            print(f"Step 2.1: Stitching {len(all_segment_paths)} video segments together...")
            
            # Create final stitched video path
            base, ext = os.path.splitext(output_video_path)
            stitched_video_path = f"{base}_stitched{ext}"
            
            video_path = stitch_videos(
                video_paths=all_segment_paths,
                output_path=stitched_video_path
            )
            
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
                        print(f"   Expected: {num_videos} video segments × 12s + {num_still_images} still images × 12s = {duration}s")
                    else:
                        print(f"✅ Stitched video duration matches input: {stitched_duration:.1f}s (target: {duration:.1f}s)")
        elif len(all_segment_paths) == 1:
            # Only one segment, no stitching needed
            video_path = all_segment_paths[0]
        else:
            # No segments generated (should not happen)
            raise RuntimeError("No video segments were generated!")
            
    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        # Clean up any generated segments on failure
        for seg in segment_video_paths:
            if os.path.exists(seg):
                try:
                    os.remove(seg)
                except:
                    pass
        raise
    
    if video_path is None:
        raise RuntimeError("Video generation completed but no video path was set")
    
    # Once Sora 2 videos are generated, always attempt to complete the video
    # Wrap all post-generation steps in error handling to ensure we try to complete
    try:
        # Step 2.5: Upscale video to 1080p if enabled
        original_video_path = video_path
        if upscale_to_1080p:
            try:
                print("Step 2.5: Upscaling video to 1080p...")
                
                ffmpeg_path = find_ffmpeg()
                
                if ffmpeg_path:
                    base, ext = os.path.splitext(video_path)
                    upscaled_path = f"{base}_1080p{ext}"
                    
                    video_path = upscale_video(
                        input_path=original_video_path,
                        output_path=upscaled_path,
                        target_resolution='1920x1080',
                        method='lanczos'
                    )
                    
                    # Clean up individual segment files if they exist (after upscaling)
                    if len(segment_video_paths) > 1:
                        for seg_path in segment_video_paths:
                            if os.path.exists(seg_path) and seg_path != original_video_path:
                                try:
                                    os.remove(seg_path)
                                except:
                                    pass
                        
                        if original_video_path != video_path and os.path.exists(original_video_path):
                            try:
                                os.remove(original_video_path)
                            except:
                                pass
                else:
                    print("⚠️  ffmpeg not found. Skipping upscaling.")
            except Exception as e:
                print(f"⚠️  Video upscaling failed: {e}")
                video_path = original_video_path
        
        # Step 2.6: Synchronize and add voiceover audio to video
        if not voiceover_audio_path:
            print("⚠️  No voiceover audio path available - skipping audio addition")
        elif not os.path.exists(voiceover_audio_path):
            print(f"⚠️  Voiceover audio file not found: {voiceover_audio_path} - skipping audio addition")
        elif not os.path.exists(video_path):
            print(f"⚠️  Video file not found: {video_path} - skipping audio addition")
        else:
            print("Step 2.6: Synchronizing and adding voiceover audio to video...")
            
            ffmpeg_path = find_ffmpeg()
            if not ffmpeg_path:
                print("⚠️  FFmpeg not found. Cannot synchronize audio.")
            else:
                video_duration = get_media_duration(video_path, ffmpeg_path)
                
                if video_duration:
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    timestamp = int(time.time())
                    
                    # CRITICAL: ALWAYS adjust narration duration to match video_duration - 2 seconds
                    # This must happen BEFORE mixing with music
                    print(f"\n   📏 Adjusting narration duration to match video...")
                    print(f"      Video duration: {video_duration:.2f}s")
                    
                    # Get original voiceover source (without music)
                    voiceover_source = None
                    if original_voiceover_backup and os.path.exists(original_voiceover_backup):
                        voiceover_source = original_voiceover_backup
                        print(f"      Using original voiceover from backup")
                    else:
                        # Fallback: try to find original voiceover in temp directory
                        import glob
                        temp_dir_check = tempfile.gettempdir()
                        original_pattern = os.path.join(temp_dir_check, "original_voiceover_*.mp3")
                        original_files = glob.glob(original_pattern)
                        if original_files:
                            voiceover_source = max(original_files, key=os.path.getmtime)
                            print(f"      Using original voiceover from temp directory")
                        else:
                            # Final fallback: use mixed audio (will extract narration if possible)
                            voiceover_source = voiceover_audio_path
                            print(f"      ⚠️  Using mixed audio as source (original not found)")
                    
                    # Adjust narration to exactly video_duration - 2 seconds
                    target_voiceover_duration = max(1.0, video_duration - 2.0)
                    voiceover_duration = get_media_duration(voiceover_source, ffmpeg_path)
                    
                    narration_was_adjusted = False
                    adjusted_voiceover_path = None
                    
                    if voiceover_duration:
                        print(f"      Current narration: {voiceover_duration:.2f}s")
                        print(f"      Target narration: {target_voiceover_duration:.2f}s (video - 2s)")
                        print(f"      Narration will start 1s after video, end 1s before video")
                        
                        if abs(voiceover_duration - target_voiceover_duration) > 0.1:
                            print(f"      Adjusting narration speed to match target duration...")
                            adjusted_voiceover_path = os.path.join(temp_dir, f"voiceover_adjusted_{timestamp}.mp3")
                            
                            # Use speed adjustment to match exact duration
                            try:
                                adjusted_voiceover = adjust_audio_duration(
                                    audio_path=voiceover_source,
                                    target_duration=target_voiceover_duration,
                                    output_path=adjusted_voiceover_path,
                                    ffmpeg_path=ffmpeg_path,
                                    method='speed'  # Use speed adjustment to preserve all content
                                )
                                
                                # Verify the adjustment worked
                                adjusted_duration = get_media_duration(adjusted_voiceover, ffmpeg_path)
                                if adjusted_duration and abs(adjusted_duration - target_voiceover_duration) < 0.5:
                                    voiceover_source = adjusted_voiceover
                                    narration_was_adjusted = True
                                    print(f"      ✅ Narration adjusted to {adjusted_duration:.2f}s (target: {target_voiceover_duration:.2f}s)")
                                else:
                                    print(f"      ⚠️  Speed adjustment may have failed. Adjusted duration: {adjusted_duration:.2f}s, expected: {target_voiceover_duration:.2f}s")
                                    print(f"      Using original narration")
                            except Exception as e:
                                print(f"      ❌ Speed adjustment failed: {e}")
                                print(f"      Using original narration")
                        else:
                            print(f"      ✅ Narration duration already matches target ({target_voiceover_duration:.2f}s)")
                    else:
                        print(f"      ⚠️  Could not determine narration duration, using original")
                    
                    # Now proceed with music mixing
                    # Re-mix audio with proper synchronization:
                    # 1. Narration is already adjusted to video_duration - 2 seconds
                    # 2. Sync music to video duration exactly
                    # 3. Re-mix with voiceover having padding
                    
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
                                        "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1",  # Fade in 1s, fade out 1s
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
                                        "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1",  # Fade in 1s, fade out 1s
                                        "-c:a", "libmp3lame",
                                        "-b:a", "192k",
                                        "-y",
                                        synced_music_path
                                    ]
                                
                                try:
                                    subprocess.run(cmd_music, capture_output=True, text=True, check=True)
                                    print(f"   ✅ Music synced to video duration: {video_duration:.2f}s")
                                    
                                    # Narration is already adjusted to video_duration - 2 seconds (done above before music sync)
                                    # Check if we're using mixed audio - if so, don't add music again!
                                    # IMPORTANT: If narration was speed-adjusted, we should NOT use the mixed audio path
                                    # Instead, we should mix the adjusted narration with fresh music
                                    using_mixed_audio_as_source = False
                                    if not narration_was_adjusted and voiceover_source == voiceover_audio_path:
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
                                                # Extend with silence or loop (simple: just pad with silence)
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
                                        # Now mix: voiceover (within video bounds) + music (synced to video)
                                        # Music starts at video start (0s), voiceover can start at 0s or up to voiceover_tolerance seconds after
                                        # Both should end at video_duration
                                        synced_audio_path = os.path.join(temp_dir, f"audio_resynced_{timestamp}.mp3")
                                        
                                        # Mix music and narration together
                                        # Delay will be applied when adding to video (in add_audio_to_video function)
                                        # Narration is already adjusted to video_duration - 2 seconds
                                        # Add volume boost after mixing to prevent quiet audio (amix can reduce overall volume)
                                        filter_complex = (
                                            f"[0:a]aresample=44100,volume=1.0[voice];"  # No delay here - will be applied when adding to video
                                            f"[1:a]aresample=44100,volume={0.07}[music];"  # 7% volume for background music
                                            f"[voice][music]amix=inputs=2:duration=longest:dropout_transition=2,"
                                            f"volume=2.0"  # Boost volume by 2x (6dB) after mixing to compensate for amix volume reduction
                                        )
                                        
                                        cmd_remix = [
                                            ffmpeg_path,
                                            "-i", voiceover_source,
                                            "-i", synced_music_path,
                                            "-filter_complex", filter_complex,
                                            "-t", str(video_duration),  # Total duration matches video exactly
                                            "-c:a", "libmp3lame",
                                            "-b:a", "192k",
                                            "-ar", "44100",
                                            "-ac", "2",
                                            "-y",
                                            synced_audio_path
                                        ]
                                        
                                        subprocess.run(cmd_remix, capture_output=True, text=True, check=True)
                                        voiceover_audio_path = synced_audio_path
                                        print(f"   ✅ Audio re-mixed: music synced to video ({video_duration:.2f}s)")
                                        print(f"   ✅ Narration will start 1s after video start (delay applied when adding to video)")
                                        print(f"   ✅ Narration will end 1s before video end (duration: {target_voiceover_duration:.2f}s)")
                                    
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
                                "-af", f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1",  # Fade in 1s, fade out 1s
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
                                print(f"   ⚠️  Music fade failed: {e}, using original music")
                                synced_music_path = music_source
                    else:
                        print(f"   ⚠️  Could not determine music duration")
                else:
                    print(f"   ⚠️  VIDEO_MUSIC.mp3 not found - cannot re-sync music separately")
                    print(f"   Using original mixed audio")
            
            try:
                base, ext = os.path.splitext(video_path)
                video_with_audio_path = f"{base}_with_audio{ext}"
                
                # Add audio to video - music is synced to video, voiceover has padding
                # Use sync_duration=False since we've already synced manually
                video_path = add_audio_to_video(
                    video_path=video_path,
                    audio_path=voiceover_audio_path,
                    output_path=video_with_audio_path,
                    ffmpeg_path=ffmpeg_path,
                    sync_duration=False  # Already synced above
                )
                print(f"✅ Video with voiceover audio: {video_path}")
                
                # Step 2.7: Add subtitles/captions to video
                if generated_script and video_duration:
                    print("\n" + "="*60)
                    print("Step 2.7: Adding subtitles/captions to video...")
                    print("="*60 + "\n")
                    try:
                        # Check if video already has subtitles - if so, use the original video without subtitles
                        # FFmpeg subtitles filter will add subtitles on top, so we need a clean video
                        base, ext = os.path.splitext(video_path)
                        if "_with_subtitles" in base:
                            # Video already has subtitles - find the original video without subtitles
                            original_base = base.rsplit("_with_subtitles", 1)[0]
                            original_video_path = f"{original_base}{ext}"
                            if os.path.exists(original_video_path):
                                print(f"   Using original video without subtitles: {os.path.basename(original_video_path)}")
                                video_path_for_subtitles = original_video_path
                            else:
                                # Original not found, use current video (will create new one)
                                video_path_for_subtitles = video_path
                        else:
                            video_path_for_subtitles = video_path
                        
                        # Create output path (will be handled by add_subtitles_to_video if None)
                        base_for_output, ext_for_output = os.path.splitext(video_path_for_subtitles)
                        if "_with_subtitles" in base_for_output:
                            base_for_output = base_for_output.rsplit("_with_subtitles", 1)[0]
                        video_with_subtitles_path = f"{base_for_output}_with_subtitles{ext_for_output}"
                        
                        # Use original voiceover backup for accurate word-level timing (pure voiceover, no music)
                        audio_for_subtitles = original_voiceover_backup if original_voiceover_backup and os.path.exists(original_voiceover_backup) else voiceover_audio_path
                        
                        video_with_subtitles = add_subtitles_to_video(
                            video_path=video_path_for_subtitles,
                            script=generated_script,
                            video_duration=video_duration,
                            output_path=video_with_subtitles_path,
                            ffmpeg_path=ffmpeg_path,
                            audio_path=audio_for_subtitles,
                            api_key=api_key
                        )
                        
                        if video_with_subtitles and os.path.exists(video_with_subtitles):
                            # Remove the video without subtitles (keep the one with subtitles)
                            video_without_subtitles = video_path
                            video_path = video_with_subtitles
                            print(f"✅ Video with subtitles: {video_path}")
                            
                            # Remove the video without subtitles
                            if video_without_subtitles != video_path and os.path.exists(video_without_subtitles):
                                try:
                                    os.remove(video_without_subtitles)
                                    print(f"  Removed video without subtitles: {os.path.basename(video_without_subtitles)}")
                                except Exception as e:
                                    print(f"  Warning: Could not remove {video_without_subtitles}: {e}")
                        else:
                            print("⚠️  Subtitle generation returned no output, keeping video without subtitles")
                    except Exception as e:
                        print(f"⚠️  Failed to add subtitles to video: {e}")
                        print("   Continuing with video without subtitles...")
                
                # Optionally remove the video without audio (only if we haven't already removed it)
                if original_video_path != video_path and os.path.exists(original_video_path):
                    # Check if this is the video without subtitles (already removed) or the original
                    if "_with_audio" not in original_video_path or os.path.exists(original_video_path):
                        try:
                            os.remove(original_video_path)
                            print(f"  Removed video without audio: {original_video_path}")
                        except Exception as e:
                            print(f"  Warning: Could not remove {original_video_path}: {e}")
            except Exception as e:
                print(f"⚠️  Failed to add audio to video: {e}")
                print("   Continuing with video without audio...")
        
        # Clean up individual segment files if upscaling was disabled or failed
        if not upscale_to_1080p and len(segment_video_paths) > 1:
            print("\nCleaning up individual segment files...")
            for seg_path in segment_video_paths:
                if os.path.exists(seg_path) and seg_path != video_path:
                    try:
                        os.remove(seg_path)
                        print(f"  Removed: {seg_path}")
                    except Exception as e:
                        print(f"  Warning: Could not remove {seg_path}: {e}")
        
        # Step 3: Generate thumbnail if not provided
    # COMMENTED OUT: Let YouTube auto-generate thumbnail
    # generated_thumbnail = None
    # if thumbnail_file is None:
    #     print("\n" + "="*60)
    #     print("Step 3: Generating thumbnail image...")
    #     print("="*60 + "\n")
    #     try:
    #         # Use the YouTube video description for thumbnail generation
    #         temp_dir = tempfile.gettempdir()
    #         timestamp = int(time.time())
    #         thumbnail_path = os.path.join(temp_dir, f"thumbnail_{timestamp}.png")
    #         
    #         generated_thumbnail = generate_thumbnail_from_prompt(
    #             description=description,
    #             output_path=thumbnail_path,
    #             api_key=api_key
    #         )
    #         thumbnail_file = generated_thumbnail
    #         print(f"✅ Thumbnail generated: {thumbnail_file}")
    #     except Exception as e:
    #         print(f"⚠️  Thumbnail generation failed: {e}")
    #         print("   Continuing without thumbnail...")
    #         thumbnail_file = None
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
            
            # Verify we're uploading the correct video (stitched if multiple segments)
            if len(segment_video_paths) > 1:
                if '_stitched' not in video_path and '_1080p' not in video_path:
                    # Check if video_path is actually a segment (shouldn't happen, but verify)
                    is_segment = any(seg_path == video_path for seg_path in segment_video_paths)
                    if is_segment:
                        raise RuntimeError(
                            f"ERROR: Attempting to upload individual segment instead of stitched video!\n"
                            f"Segment path: {video_path}\n"
                            f"This should not happen. Please check the stitching logic."
                        )
            
            try:
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
                if master_image_path and os.path.exists(master_image_path):
                    try:
                        os.remove(master_image_path)
                        print(f"Cleaned up master reference image: {master_image_path}")
                    except:
                        pass
                
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
    
        # Reference image and final video are saved in the output folder
        if master_image_path and os.path.exists(master_image_path):
            print(f"✅ Reference image saved: {master_image_path}")
            print(f"   (This image is used as reference for all Sora-generated scenes)")
        
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
        
        # Clean up all segment video files
        if 'segment_video_paths' in locals():
            for seg_path in segment_video_paths:
                if seg_path and os.path.exists(seg_path) and seg_path != video_path and seg_path != output_video_path:
                    try:
                        os.remove(seg_path)
                        cleaned_items.append(f"Segment Video: {os.path.basename(seg_path)}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not delete segment: {e}")
        
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
    
    except Exception as e:
        # Always attempt to complete even if errors occur after Sora 2 video generation
        print(f"\n⚠️  Error occurred during post-generation steps: {e}")
        print("   Attempting to complete video processing despite error...")
        import traceback
        traceback.print_exc()
        
        # Try to ensure we have a valid video path
        if video_path is None or not os.path.exists(video_path):
            # Try to find any generated video
            if 'segment_video_paths' in locals() and segment_video_paths:
                for seg_path in segment_video_paths:
                    if os.path.exists(seg_path):
                        video_path = seg_path
                        print(f"   Using segment video: {video_path}")
                        break
        
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
        return 8  # Default duration
    duration = int(duration)
    if duration > max_duration:
        print(f"⚠️  Warning: Duration {duration}s exceeds maximum of {max_duration}s (10 minutes).")
        print(f"   Capping duration to {max_duration}s to prevent excessive costs.")
        return max_duration
    if duration < 1:
        print(f"⚠️  Warning: Duration {duration}s is too short. Setting to minimum of 1 second.")
        return 1
    return duration


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate video using OpenAI Sora 2 and upload to YouTube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and upload with minimal parameters:
  python generate_and_upload_sora.py "A serene landscape" --title "My Video"
  
  # Full example:
  python generate_and_upload_sora.py "A cat playing piano" --title "Cat Piano" \\
    --description "AI generated video" --tags ai sora automation --privacy public \\
    --duration 10 --resolution 1920x1080 --keep-video

Environment Variables:
  OPENAI_API_KEY: Your OpenAI API key (required if not using --api-key)
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
        default='22',
        help='YouTube category ID (default: 22 - People & Blogs)'
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
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: Add an opening still image (12 seconds) at the beginning of the video'
    )
    
    # Video generation parameters
    parser.add_argument(
        '--output',
        help='Output video file path (default: temp file, deleted after upload)'
    )
    
    parser.add_argument(
        '--api-key',
        default=None,
        help='OpenAI API key (default: uses OPENAI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--model',
        choices=['sora-2', 'sora-2-pro'],
        default='sora-2',
        help='Sora model to use (default: sora-2)'
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
        help='Video duration in seconds (default: from config file, or 8 if no config, max: 600)'
    )
    
    # Removed --num-videos argument - now calculated automatically from duration
    # (12-second segments + 12-second still images after every 3 videos)
    
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
        duration = args.duration if args.duration else 8
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
            category_input = input("Category ID (default: 22 - People & Blogs, or press Enter to use default): ").strip()
        except (EOFError, KeyboardInterrupt):
            category_input = ""
        category_id = category_input if category_input else (args.category if args.category else '22')
        
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
        
        # Ask for test mode
        try:
            test_input = input("Test mode? Add opening still image? (y/n, default: n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            test_input = ""
        test_mode = test_input in ['y', 'yes']
        
        # Display all collected inputs for verification
        print("\n" + "="*60)
        print("📋 Collected Configuration Summary:")
        print("="*60)
        print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"  Title: {title}")
        print(f"  Description: {description[:100]}{'...' if len(description) > 100 else ''}")
        print(f"  Duration: {duration} seconds")
        print(f"  Tags: {tags if tags else 'None'}")
        print(f"  Privacy: {privacy_status}")
        print(f"  Category ID: {category_id}")
        print(f"  Model: {model}")
        print(f"  Resolution: {resolution}")
        print(f"  Thumbnail: {thumbnail_file if thumbnail_file else 'None'}")
        print(f"  Playlist ID: {playlist_id if playlist_id else 'None'}")
        print(f"  Test Mode: {test_mode}")
        print("="*60)
        
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
            'test_mode': test_mode
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
            print(f"\n✅ Script generation complete!")
            print(f"📝 Script saved to: {script_file}")
            print(f"💾 Configuration saved to: {CONFIG_FILE_PATH}")
            print(f"📌 Title: {title}")
            print(f"📌 Description: {description[:100]}{'...' if len(description) > 100 else ''}")
            print(f"\n💡 Next steps:")
            print(f"   1. Edit {script_file} if needed")
            print(f"   2. Run with --generate-narration-only to generate narration audio")
            print(f"   3. Run without flags to continue with video generation")
            return 0
        except Exception as e:
            print(f"\n❌ Error generating script: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Check if we should only generate the narration
    if args.generate_narration_only:
        # Narration generation only - load config if available, no questions asked
        config = load_config()
        if config:
            print(f"📋 Using saved configuration from: {CONFIG_FILE_PATH}")
        
        try:
            narration_file = generate_and_save_narration(
                script_file_path=SCRIPT_FILE_PATH,
                narration_audio_path=NARRATION_AUDIO_PATH,
                duration=None,  # Duration not needed for narration-only generation
                api_key=args.api_key
            )
            print(f"\n✅ Narration generation complete!")
            print(f"🎙️  Narration audio saved to: {narration_file}")
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
            print("="*60)
            print("⚠️  Running in non-interactive mode (no input available)")
            print("="*60)
            print("To run this script, you need to either:")
            print("1. Provide command-line arguments:")
            print('   python generate_and_upload_sora.py "your prompt" --title "Your Title"')
            print("2. Use --non-interactive flag with all required arguments")
            print("3. Run from a terminal where input is available")
            print("="*60)
            if not args.prompt or not args.title:
                print("\n❌ Error: Missing required arguments (prompt and title)")
                print("   Run with: --non-interactive --prompt '...' --title '...'")
                return 1
        
        # ============================================================
        # WORKFLOW: 5 Sequential Steps - Run one and exit
        # ============================================================
        
        # Step 1: Generate script
        print("\n" + "="*60)
        print("STEP 1: Generate Script")
        print("="*60)
        step1_input = 'n'
        if not args.non_interactive:
            try:
                step1_input = input("Generate script? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step1_input = 'n'
        
        if step1_input in ['y', 'yes']:
            # Archive workflow files before starting new script generation
            # This ensures narration_audio.mp3 and other files from previous run are saved
            print("\n" + "="*60)
            print("📦 Archiving previous workflow files...")
            print("="*60)
            archive_workflow_files()
            print("="*60 + "\n")
            # Always ask for all configuration inputs and save to config
            # Delete existing config file at the start (cleanup from last run)
            if os.path.exists(CONFIG_FILE_PATH):
                print(f"🧹 Deleting existing config file: {CONFIG_FILE_PATH}")
                try:
                    os.remove(CONFIG_FILE_PATH)
                except Exception as e:
                    print(f"⚠️  Warning: Could not delete config file: {e}")
            
            print("\n📋 Collecting video configuration (this will be saved for later steps)...")
            
            # Get prompt
            if not args.prompt:
                try:
                    print("Enter video prompt (text description of the video):")
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
            duration = args.duration if args.duration else 8
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
                category_input = input("Category ID (default: 22 - People & Blogs, or press Enter to use default): ").strip()
            except (EOFError, KeyboardInterrupt):
                category_input = ""
            category_id = category_input if category_input else (args.category if args.category else '22')
            
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
            
            # Ask for test mode
            try:
                test_input = input("Test mode? Add opening still image? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                test_input = ""
            test_mode = test_input in ['y', 'yes']
            
            # Display all collected inputs for verification
            print("\n" + "="*60)
            print("📋 Collected Configuration Summary:")
            print("="*60)
            print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"  Title: {title}")
            print(f"  Description: {description[:100]}{'...' if len(description) > 100 else ''}")
            print(f"  Duration: {duration} seconds")
            print(f"  Tags: {tags if tags else 'None'}")
            print(f"  Privacy: {privacy_status}")
            print(f"  Category ID: {category_id}")
            print(f"  Model: {model}")
            print(f"  Resolution: {resolution}")
            print(f"  Thumbnail: {thumbnail_file if thumbnail_file else 'None'}")
            print(f"  Playlist ID: {playlist_id if playlist_id else 'None'}")
            print(f"  Test Mode: {test_mode}")
            print("="*60)
            
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
                'test_mode': test_mode
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
                print(f"\n✅ STEP 1 COMPLETE: Script generation complete!")
                print(f"📝 Script saved to: {script_file}")
                print(f"💾 Configuration saved to: {CONFIG_FILE_PATH}")
                print(f"\n✅ Script execution complete. Exiting.")
                return 0
            except Exception as e:
                print(f"\n❌ Error generating script: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Step 2: Generate narration (generate narration audio)
        print("\n" + "="*60)
        print("STEP 2: Generate Narration")
        print("="*60)
        step2_input = 'n'
        if not args.non_interactive:
            try:
                step2_input = input("Generate narration? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step2_input = 'n'
        
        if step2_input in ['y', 'yes']:
            try:
                narration_file = generate_and_save_narration(
                    script_file_path=SCRIPT_FILE_PATH,
                    narration_audio_path=NARRATION_AUDIO_PATH,
                    duration=None,
                    api_key=args.api_key
                )
                print(f"\n✅ STEP 2 COMPLETE: Narration generation complete!")
                print(f"🎙️  Narration audio saved to: {narration_file}")
                print(f"\n✅ Script execution complete. Exiting.")
                return 0
            except Exception as e:
                print(f"\n❌ Error generating narration: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Step 3: Generate video (Sora video generation)
        print("\n" + "="*60)
        print("STEP 3: Generate Video")
        print("="*60)
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
            duration = args.duration if args.duration else config.get('duration', 8)
            
            if not prompt or not title:
                print("⚠️  Missing required configuration. Please generate script first.")
                return 1
            tags = config.get('tags')
            privacy_status = args.privacy if args.privacy != 'private' else config.get('privacy_status', 'private')
            category_id = args.category if args.category != '22' else config.get('category_id', '22')
            model = args.model if args.model != 'sora-2' else config.get('model', 'sora-2')
            resolution = args.resolution if args.resolution != '1920x1080' else config.get('resolution', '1920x1080')
            test_mode = config.get('test_mode', False)
            
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
                    test=test_mode,
                    skip_narration=True,  # Skip narration generation
                    skip_upload=True  # Skip YouTube upload
                )
                print(f"\n✅ STEP 2 COMPLETE: Video generation complete!")
                print(f"📹 Video saved to: {video_path}")
                print(f"\n✅ Script execution complete. Exiting.")
                return 0
            except Exception as e:
                print(f"\n❌ Error generating video: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Step 4: Add captions based on narration
        print("\n" + "="*60)
        print("STEP 4: Add Captions Based on Narration")
        print("="*60)
        step4_input = 'n'
        if not args.non_interactive:
            try:
                step4_input = input("Generate captions/subtitles? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step4_input = 'n'
        
        if step4_input in ['y', 'yes']:
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
            
            if not os.path.exists(NARRATION_AUDIO_PATH):
                print("❌ Error: Narration file not found. Please generate narration first.")
                return 1
            
            try:
                # Add narration, music, and captions to video
                # This function adds narration audio, background music, and captions based on the narration timing
                from add_music_to_video import add_narration_music_and_captions_to_video
                final_video_path = add_narration_music_and_captions_to_video(
                    video_path=video_path,
                    output_path=None,
                    api_key=args.api_key,
                    ffmpeg_path=None
                )
                video_path = final_video_path  # Update video_path to final version
                print(f"\n✅ STEP 4 COMPLETE: Captions added to video based on narration!")
                print(f"📹 Final video: {video_path}")
                print(f"\n✅ Script execution complete. Exiting.")
                return 0
            except Exception as e:
                print(f"\n❌ Error adding captions to video: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Step 5: Upload to YouTube
        print("\n" + "="*60)
        print("STEP 5: Upload to YouTube")
        print("="*60)
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
            category_id = args.category if args.category != '22' else config.get('category_id', '22')
            
            if not title:
                print("⚠️  Missing title. Please generate script first.")
                return 1
            
            thumbnail_file = find_thumbnail_file()
            if not thumbnail_file:
                thumbnail_file = args.thumbnail if args.thumbnail else None
            playlist_id = args.playlist if args.playlist else None
            
            try:
                from upload_video import upload_video
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
                print(f"\n✅ STEP 5 COMPLETE: Video uploaded to YouTube!")
                print(f"🎬 YouTube Video ID: {video_id}")
                print(f"🔗 Video URL: https://www.youtube.com/watch?v={video_id}")
                print(f"\n✅ Script execution complete. Exiting.")
                return 0
            except Exception as e:
                print(f"\n❌ Error uploading to YouTube: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # If we get here, user answered 'no' to all steps
        print("\n" + "="*60)
        print("No steps selected. Exiting.")
        print("="*60)
        return 0


if __name__ == '__main__':
    exit(main())

