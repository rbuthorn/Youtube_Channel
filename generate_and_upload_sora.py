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
# 2. segment_script_rule_based() - Segments the script into X segments using rules-based approach
# 3. generate_sora_prompts_from_segments() - Generates Sora 2 prompts from segments (separate API calls)
# This separation allows for better control, error handling, and modularity.


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
        voiceover_audio_path, original_voiceover_backup = generate_voiceover_with_music(
            script=script,
            output_path=narration_audio_path,
            api_key=api_key,
            voice='onyx',  # Deep, manly male voice
            music_style='cinematic',
            music_volume=0.10,  # 10% volume for background music
            duration=duration
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
    # Estimate: average reading speed is ~150-160 words per minute, or ~2.5 words per second
    # Narration should be shorter than video by at most 5 seconds (for opening/closing shots)
    # For {duration} seconds video, narration should be ~{duration - 3} seconds (3s buffer for shots)
    # Aim for approximately {int((duration - 3) * 2.3)} words (slightly conservative)
    narration_duration = max(1, duration - 3)  # At least 1 second, but typically 3 seconds shorter
    estimated_words = int(narration_duration * 2.3)
    
    script_prompt = f"""Create a {duration}-second documentary-style YouTube script (~{estimated_words} words) for: {video_prompt}

IMPORTANT: The narration should be approximately {narration_duration} seconds long (video is {duration} seconds). This allows for 2-3 second opening and closing shots with no narration.

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
  * Include strategic pauses marked with [MUSICAL BREAK] or [VISUAL BREAK] where narration stops
  * Place these breaks after dramatic moments, before transitions, or during visually stunning scenes
  * These breaks should be 2-4 seconds long - let the visuals and music tell part of the story
  * Flow: narration → pause/musical break → narration continues naturally
  * Use breaks to build tension, emphasize key moments, or transition between story sections
- Style: Informative yet engaging. Blend facts with storytelling. Use natural pauses (...), varied pacing, and smooth transitions
- Tone: Authoritative but accessible - like a knowledgeable expert sharing a fascinating story to someone who's never heard it before
- Be educational and explanatory - prioritize clarity and understanding over brevity
- Provide context continuously - don't just state facts, explain them
- Make it entertaining without sacrificing accuracy - facts should be compelling on their own
- One continuous script, ~{estimated_words} words

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
                {"role": "system", "content": "Expert documentary scriptwriter. Write informative, historically accurate scripts that tell complete stories in an engaging way. Structure content with hook, introduction, narrative, climax, conclusion, and impact. Blend factual accuracy with compelling storytelling - like BBC or National Geographic documentaries. Be authoritative yet accessible, informative yet entertaining. CRITICAL: Assume the viewer knows NOTHING about the topic. Provide extensive context, background, and explanations throughout. Explain who people are, what terms mean, when/where events occurred, why they happened, and the historical/cultural context. Don't assume prior knowledge - explain everything clearly and thoroughly. Include strategic musical breaks marked with [MUSICAL BREAK] or [VISUAL BREAK] where narration stops for 2-4 seconds to let visuals and music shine. These breaks should flow naturally - place them after dramatic moments, before transitions, or during visually stunning scenes. Cover the full story comprehensively with rich context and explanations. CRITICAL: Output ONLY the script text - dialogue/narration, [MUSICAL BREAK], and [VISUAL BREAK] markers only. NO labels, NO instructions, NO explanations. The output will be read directly by text-to-speech, so any extra text will be spoken as dialogue."},
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
            context_sections.append(f"""REFERENCE IMAGE CONTEXT:
A reference image will be provided showing the main character: {ref_desc}

IMPORTANT: The video must feature this same character consistently throughout. The character's appearance, style, and general features should match the reference image. Ensure the character in this segment matches the reference image description.""")
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
    visual_requirements.append("- MOST CRITICAL: The video MUST make perfect sense with what the script narration is saying during this segment. Every visual element must directly correspond to and support the narration.")
    visual_requirements.append("- Briefly include camera movement, angle, lighting, mood")
    visual_requirements.append("- Be concise and cinematic")
    # Maximum scenes/cuts per segment (but first and last must be continuous)
    if is_opening_segment or is_closing_segment:
        visual_requirements.append("- MOSTCRITICAL: This shot must be LONG and can have slow, smooth camera movement, but NO cuts or transitions. Hold the same scene throughout the entire segment. Let the visual breathe before narration begins.")
    else:
        visual_requirements.append("- MOST CRITICAL: The video MUST have at most ONE cut. There should be at the MOST one or two scenes in the video segment.")
    
    # Each segment (except the first and last) should start with a new scene/cut
        
    visual_requirements.append(f"- Optimized for a {segment_duration:.1f}s clip")
    
    requirements_text = "\n".join(visual_requirements)
    
    # Validate segment_text is provided and not empty
    if not segment_text or len(segment_text.strip()) == 0:
        raise ValueError(f"Segment {segment_id} text is empty! Cannot generate Sora prompt.")
    
    # Create prompt to convert segment script to Sora video prompt
    # CRITICAL: Put segment_text FIRST and make it the primary focus
    conversion_prompt = f"""Create a Sora-2 video prompt for segment {segment_id} ({start_time:.1f}-{end_time:.1f}s) of a {total_duration}s video.

═══════════════════════════════════════════════════════════════════════════════
MOST IMPORTANT: PRIMARY SCRIPT FOR THIS SEGMENT (THE VIDEO MUST MATCH THIS EXACTLY):
{segment_text}

CRITICAL: This is segment {segment_id} of {total_segments if total_segments else 'unknown'} segments. The script above is the EXACT narration text for seconds {start_time:.1f}-{end_time:.1f} of the video. Your video prompt must match THIS specific portion of the script.
CRITICAL: The prompt you generate should never tell sora 2 to generate diagrams or words. All of the shots must be a natural scene that could come legitimately from a camera.
CRITICAL:Do not ever describe the music or sound effects the video should have. Just describe the scene.

Context:
{context_text}

Requirements:
{requirements_text}

═══════════════════════════════════════════════════════════════════════════════
FINAL REMINDER: The video MUST make perfect sense with the script narration above.
If the script describes a main location, the video MUST show that location.
If the script mentions an action, the video MUST show that action.
The video segment MUST be 1 or 2 different scenes or shots.
Do not ever describe the music or sound effects the video should have. Just describe the scene.
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
                    {"role": "system", "content": "Professional Sora 2 Video Prompter. Prompt Sora 2 to create detailed cinematic prompts matching script narration. Always provide complete prompts."},
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
                        sora_prompt = f"Cinematic video scene matching the narration: {segment_text[:300]}"
                    else:
                        sora_prompt = f"Cinematic video scene for segment {segment_id} of the video"
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
                    sora_prompt = f"Cinematic video scene matching the narration: {segment_text[:300]}"
                else:
                    sora_prompt = f"Cinematic video scene for segment {segment_id} of the video"
                print(f"\n❌ FAILED: All {max_retries} attempts failed with errors ({error_type})")
                print(f"✅ Using fallback prompt based on segment text")
                return sora_prompt
    
    # Should never reach here, but just in case
    if segment_text and len(segment_text.strip()) > 0:
        return f"Cinematic video scene matching the narration: {segment_text[:300]}"
    else:
        return f"Cinematic video scene for segment {segment_id} of the video"


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
                    sora_prompt = f"Cinematic video scene matching the narration: {segment_text[:300]}"
                elif overarching_script and len(overarching_script.strip()) > 0:
                    sora_prompt = f"Cinematic video scene: {overarching_script[:300]}"
                else:
                    sora_prompt = f"Cinematic video scene for segment {i} with engaging visuals"
            
            sora_prompts.append(sora_prompt)
            print(f"  ✅ Segment {i} Sora prompt generated ({len(sora_prompt)} characters)")
        except Exception as e:
            print(f"  ⚠️  Failed to convert segment {i} to Sora prompt after retries: {e}")
            # Fallback: use a generic prompt based on the segment text
            if segment_text and len(segment_text.strip()) > 0:
                fallback_prompt = f"Cinematic video scene matching the narration: {segment_text[:300]}"
            else:
                # If segment text is also empty, use the overarching script or original prompt
                if overarching_script and len(overarching_script.strip()) > 0:
                    fallback_prompt = f"Cinematic video scene: {overarching_script[:300]}"
                else:
                    fallback_prompt = f"Cinematic video scene for segment {i} with engaging visuals"
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



def generate_voiceover_with_music(
    script,
    output_path=None,
    api_key=None,
    voice='echo',  # Deep, manly male voice
    music_style='cinematic',
    music_volume=0.10,  # 10% volume for background music
    duration=None,
    instructions="Use a very passionate and very exciting story telling style."): 
    """
    Generate voiceover audio from script using OpenAI TTS, add background music with dynamic swells, and combine them.
    
    Args:
        script: The script text to convert to voiceover
        output_path: Path to save the final audio file (default: temp file)
        api_key: OpenAI API key
        voice: TTS voice to use ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer') (default: 'onyx' - deep, manly male voice)
        music_style: Style of background music (default: 'cinematic', will be overridden by script analysis)
        music_volume: Base volume of background music relative to voiceover (0.0-1.0) (default: 0.10, 10%)
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
    
    # If difference is very small (< 0.1s), just pad/trim slightly
    if abs(duration_diff) < 0.1:
        print("   Duration difference is minimal, using padding/trimming")
        method = 'pad'
    
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
        
        cmd = [
            ffmpeg_path,
            "-i", audio_path,
            "-af", filter_chain,
            "-t", str(target_duration),  # Ensure exact duration
            "-c:a", "libmp3lame",  # Preserve codec
            "-b:a", "192k",  # Preserve bitrate
            "-ac", "2",  # Preserve stereo
            "-y",
            output_path
        ]
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
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Verify the output duration
        adjusted_duration = get_media_duration(output_path, ffmpeg_path)
        if adjusted_duration:
            print(f"   ✅ Adjusted audio duration: {adjusted_duration:.2f}s (target: {target_duration:.2f}s)")
        
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Audio adjustment failed: {e.stderr}")
        # Fallback: just trim or pad
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
    
    # Step 2: Add audio to video with volume boost
    # Use ffmpeg to add audio to video
    # Remove any existing audio and replace with new audio track
    # Apply volume boost to ensure audio is loud enough (compensate for any volume loss during encoding)
    if remove_existing_audio:
        # Map only video from first input, audio from second input (removes existing audio)
        # Apply volume boost to ensure audio is at proper level
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-i", adjusted_audio_path,
            "-c:v", "copy",      # Copy video stream without re-encoding
            "-af", "volume=1.5",  # Boost audio by 1.5x (3.5dB) to ensure it's loud enough
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
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-i", adjusted_audio_path,
            "-c:v", "copy",
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
            context_sections.append(f"""REFERENCE IMAGE CONTEXT:
A reference image will be provided showing the main character: {ref_desc}

IMPORTANT: The still image must feature this same character consistently. The character's appearance, style, and general features should match the reference image.""")
        else:
            context_sections.append(f"""REFERENCE IMAGE CONTEXT:
A reference image will be provided showing: {ref_desc}

IMPORTANT: The still image must maintain visual consistency with this reference image. The main visual elements, style, and atmosphere should align with the reference image.""")
    
    context_text = "\n\n".join(context_sections) if context_sections else ""
    
    # Build requirements (similar to Sora prompt requirements)
    visual_requirements = []
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

Provide ONLY the DALL-E prompt (no labels, no explanation, just the prompt text):"""
    
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Professional video director and image prompt specialist. Create detailed cinematic DALL-E prompts for still images that match script narration and maintain visual consistency with video segments. CRITICAL: The still image must relate directly to the PRIMARY SCRIPT SEGMENT provided. Ensure visual continuity and narrative flow. Always provide complete, detailed prompts. CRITICAL: All images must be hyperrealistic and photorealistic, as if photographed by a professional documentary photographer - make them look like real photographs with natural lighting, realistic textures, and maximum detail. Never create prompts with likenesses of real people, celebrities, or historical figures. Always use generic, artistic, stylized representations, but make them appear completely realistic and photographic."},
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


def calculate_still_image_positions(num_videos, segment_duration, total_duration, script, segment_texts, api_key=None, model='gpt-5-2025-08-07', test=False, reference_image_info=None):
    """
    Calculate positions for still images: after every 3 videos, a 12-second still image.
    If test=True, also adds an opening still image at the beginning.
    
    Pattern (normal): V1, V2, V3, Still(12s), V4, V5, V6, Still(12s), V7, V8, V9, Still(12s), ...
    Pattern (test): Still(12s), V1, V2, V3, Still(12s), V4, V5, V6, Still(12s), ...
    
    Args:
        num_videos: Total number of video segments
        segment_duration: Duration of each video segment in seconds
        total_duration: Total video duration in seconds
        script: The full overarching script (for generating image prompts)
        segment_texts: List of segment script texts
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-4o')
        test: If True, add an opening still image at the beginning
        
    Returns:
        List of dictionaries with 'segment_id' (0 for opening, or position after which video), 'image_prompt', 'duration' (12.0), 'reasoning'
    """
    STILL_IMAGE_DURATION = 12.0
    VIDEOS_PER_STILL = 3
    
    still_image_segments = []
    
    # If test mode, ALWAYS add opening still image (segment_id = 0 means at the beginning)
    # This happens regardless of number of videos or script content
    if test:
        try:
            # Get first segment text for opening still image context
            opening_context = segment_texts[0] if segment_texts and len(segment_texts) > 0 else (script[:500] if script else "")
            next_segment_text = segment_texts[1] if segment_texts and len(segment_texts) > 1 else None
            
            # Generate opening still image prompt based on the overall script
            opening_prompt = generate_still_image_prompt(
                script=script or "",  # Ensure script is not None
                context_segment=opening_context,  # Use first segment or beginning of script for opening
                position=0,  # Position 0 = opening
                num_videos=num_videos,
                api_key=api_key,
                model=model,
                previous_segment_text=None,  # No previous segment for opening
                next_segment_text=next_segment_text,  # Next segment for forward continuity
                reference_image_info=reference_image_info  # Reference image for consistency
            )
        except Exception as e:
            print(f"⚠️  Failed to generate opening still image prompt: {e}")
            # Fallback prompt for opening still image
            opening_prompt = f"A hyperrealistic, photorealistic, high-quality opening still image, as if photographed by a professional documentary photographer, representing the video topic: {script[:200] if script else 'documentary introduction'}. Make it look like a real photograph with natural lighting, realistic textures, and maximum detail."
        
        still_image_segments.append({
            'segment_id': 0,  # 0 = opening still image
            'image_prompt': opening_prompt,
            'duration': STILL_IMAGE_DURATION,
            'reasoning': "Opening still image (test mode)"
        })
    
    # Calculate still image positions: after videos 3, 6, 9, 12, ...
    still_image_positions = []
    for i in range(VIDEOS_PER_STILL, num_videos, VIDEOS_PER_STILL):
        still_image_positions.append(i)
    
    # Generate image prompts for each still image position
    for position in still_image_positions:
        # Get context from the script around this position
        # Use the segment text from the video before the still image
        if position > 0 and position <= len(segment_texts):
            context_segment = segment_texts[position - 1]  # Video segment before still image
            previous_segment_text = segment_texts[position - 2] if position > 1 and position - 2 < len(segment_texts) else None
            next_segment_text = segment_texts[position] if position < len(segment_texts) else None
        else:
            context_segment = segment_texts[0] if segment_texts else ""
            previous_segment_text = None
            next_segment_text = segment_texts[1] if segment_texts and len(segment_texts) > 1 else None
        
        # Generate image prompt based on context (matching Sora prompt style)
        image_prompt = generate_still_image_prompt(
            script=script,
            context_segment=context_segment,
            position=position,
            num_videos=num_videos,
            api_key=api_key,
            model=model,
            previous_segment_text=previous_segment_text,  # Previous segment for continuity
            next_segment_text=next_segment_text,  # Next segment for forward continuity
            reference_image_info=reference_image_info  # Reference image for consistency
        )
        
        still_image_segments.append({
            'segment_id': position,  # Position after which video (e.g., 3 = after video 3)
            'image_prompt': image_prompt,
            'duration': STILL_IMAGE_DURATION,
            'reasoning': f"Still image after video {position} (following pattern: after every {VIDEOS_PER_STILL} videos)"
        })
    
    # In test mode, we should always have at least the opening still image
    # Verify this is the case
    if test and not any(seg['segment_id'] == 0 for seg in still_image_segments):
        print("⚠️  Warning: Test mode enabled but opening still image was not added!")
        # Force add opening still image as fallback
        still_image_segments.insert(0, {
            'segment_id': 0,
            'image_prompt': f"A hyperrealistic, photorealistic, high-quality opening still image, as if photographed by a professional documentary photographer, representing the video topic. Make it look like a real photograph with natural lighting, realistic textures, and maximum detail.",
            'duration': STILL_IMAGE_DURATION,
            'reasoning': "Opening still image (test mode - fallback)"
        })
    
    return still_image_segments


def analyze_script_for_still_images(script, segment_texts, segment_duration, total_duration, target_num_stills=None, api_key=None, model='gpt-4o', test=False, reference_image_info=None):
    """
    Analyze script to identify where still images should be placed.
    Uses AI to avoid action scenes and find appropriate contemplative moments.
    
    Still images work well for: key moments, important visuals, transitions, dramatic pauses, etc.
    AVOIDS: action scenes, fights, battles, chases, fast-paced moments.
    
    Total still image duration should be approximately 1/3 (33%) of total video duration.
    Each still image should be 12.0 seconds long.
    
    Args:
        script: The full overarching script
        segment_texts: List of segment script texts
        segment_duration: Duration of each segment in seconds
        total_duration: Total video duration in seconds
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-4o')
        test: If True, add an opening still image at the beginning
        reference_image_info: Dict with reference image info for consistency
        
    Returns:
        List of dictionaries with 'segment_id' (0 for opening, or position after which video), 'segment_text', 'image_prompt', 'duration', 'reasoning'
    """
    # Standard still image duration (12 seconds per still image)
    STILL_IMAGE_DURATION = 12.0
    
    # Calculate target total still image duration (approximately 1/3 of video duration, max 1/3)
    # But we'll use 12 seconds per still image, so adjust number of stills accordingly
    target_total_still_duration = total_duration / 3.0  # Target 1/3 (33%) of total duration
    
    # Min and max duration per still image (for good viewing experience)
    # Note: We standardize to 12.0 seconds, but the analysis uses these for calculation
    MIN_STILL_DURATION = 12.0  # Standardized to 12 seconds
    MAX_STILL_DURATION = 12.0  # Standardized to 12 seconds
    
    # Calculate reasonable number of still images based on target duration (1/3 of total)
    # Target: 1/3 of total duration should be still images
    # Each still image is 12 seconds, so: ideal_num_stills = (total_duration / 3) / 12
    num_segments = len(segment_texts)
    
    # Calculate ideal number based on 1/3 target
    ideal_num_stills_by_duration = int(target_total_still_duration / STILL_IMAGE_DURATION)
    
    # Cap at 1/3 of segments to ensure we have more videos than still images
    # Never use more than 1/3 of segments for still images
    max_stills_by_segments = max(1, int(num_segments / 3))  # At most 1/3 of segments
    
    # Use the smaller of: (1) calculated from 1/3 duration target, (2) 1/3 of segments
    ideal_num_stills = min(ideal_num_stills_by_duration, max_stills_by_segments)
    
    # Ensure at least 1 still image for longer videos, but respect the 1/3 cap
    if total_duration >= 24:  # At least 24 seconds to warrant a still image
        ideal_num_stills = max(1, ideal_num_stills)
    
    # Final safety check: ensure we have more segments than still images
    ideal_num_stills = min(ideal_num_stills, max(1, num_segments - 1))
    
    # Ensure we don't exceed 1/3 of total duration
    max_still_duration_allowed = total_duration / 3.0
    max_stills_by_duration = int(max_still_duration_allowed / STILL_IMAGE_DURATION)
    ideal_num_stills = min(ideal_num_stills, max_stills_by_duration)
    
    # Use standard 12.0 seconds per still image
    ideal_duration_per_still = STILL_IMAGE_DURATION
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    # Create segments info for analysis
    segments_info = []
    for i, seg_text in enumerate(segment_texts, 1):
        start_time = (i - 1) * segment_duration
        end_time = i * segment_duration
        segments_info.append({
            'segment_id': i,
            'start_time': start_time,
            'end_time': end_time,
            'text': seg_text[:300]  # Truncate for analysis
        })
    
    analysis_prompt = f"""Analyze this script and identify {ideal_num_stills} segments (out of {len(segment_texts)} total) where a high-quality still image with camera panning would be most effective.

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

IMPORTANT: Total still image duration should be approximately {target_total_still_duration:.1f} seconds (about 1/3 of the {total_duration:.1f}s video). This should NOT exceed 1/3 of the total video duration.
Each still image should be between {MIN_STILL_DURATION:.1f} and {MAX_STILL_DURATION:.1f} seconds long.
Target duration per still image: approximately {ideal_duration_per_still:.1f} seconds.

Full script: {script[:1000]}{'...' if len(script) > 1000 else ''}

Segments:
{chr(10).join([f"Segment {s['segment_id']} ({s['start_time']:.1f}s-{s['end_time']:.1f}s): {s['text']}" for s in segments_info])}

Output JSON array:
[
    {{
        "segment_id": 1,
        "image_prompt": "Detailed DALL-E prompt for a hyperrealistic, photorealistic, high-quality still image matching this segment's narration, as if photographed by a professional documentary photographer. Make it look like a real photograph with natural lighting, realistic textures, and maximum detail. MUST comply with OpenAI content policy: no violence, hate, adult content, illegal activity, copyrighted characters, or likenesses of real people. Use generic, artistic representations only, but make them appear completely realistic and photographic.",
        "duration": {ideal_duration_per_still:.1f},
        "reasoning": "Why this segment benefits from a still image (and why it's NOT an action scene)"
    }}
]

Rules:
- Select exactly {ideal_num_stills} segments (to achieve approximately {target_total_still_duration:.1f}s total still image duration)
- CRITICAL: Ensure there are MORE video segments than still images (you have {len(segment_texts)} segments total, select only {ideal_num_stills} for still images)
- Each still image should be between {MIN_STILL_DURATION:.1f} and {MAX_STILL_DURATION:.1f} seconds long
- Aim for approximately {ideal_duration_per_still:.1f} seconds per still image
- Total duration of all still images should sum to approximately {target_total_still_duration:.1f} seconds (1/3 of total duration, but NOT more than 1/3)
- CRITICAL: Skip any segments that contain action, fights, battles, or fast-paced scenes
- image_prompt MUST be safe, appropriate, and comply with OpenAI DALL-E content policies
- CRITICAL: NEVER include likenesses of real people, celebrities, or historical figures
- Avoid: violence, hate speech, adult content, illegal activities, real people, copyrighted characters
- Use: generic, artistic, educational, and appropriate visual descriptions
- The image should directly relate to the segment's narration and story

Provide ONLY valid JSON array:"""
    
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
        still_image_segments = json.loads(response.output_text)
        
        # Validate and enrich with full segment text
        validated_segments = []
        for seg_info in still_image_segments:
            segment_id = seg_info.get('segment_id')
            if 1 <= segment_id <= len(segment_texts):
                # Clamp duration to min/max bounds
                raw_duration = float(seg_info.get('duration', ideal_duration_per_still))
                clamped_duration = min(max(MIN_STILL_DURATION, raw_duration), MAX_STILL_DURATION)
                
                # Get context for better image prompt generation
                context_segment = segment_texts[segment_id - 1]
                previous_segment_text = segment_texts[segment_id - 2] if segment_id > 1 and segment_id - 2 < len(segment_texts) else None
                next_segment_text = segment_texts[segment_id] if segment_id < len(segment_texts) else None
                
                # Use the AI-provided image prompt (it's already detailed and context-aware)
                detailed_image_prompt = seg_info.get('image_prompt', '')
                
                validated_segments.append({
                    'segment_id': segment_id,
                    'segment_text': context_segment,
                    'image_prompt': detailed_image_prompt,
                    'duration': STILL_IMAGE_DURATION,  # Always 12.0 seconds
                    'reasoning': seg_info.get('reasoning', '')
                })
        
        # All still images are standardized to 12.0 seconds
        # No need to adjust durations since they're all the same
        
        # If test mode, add opening still image if not already present
        if test:
            if not any(seg.get('segment_id') == 0 for seg in validated_segments):
                # Generate opening still image prompt
                try:
                    opening_prompt = generate_still_image_prompt(
                        script=script,
                        context_segment=segment_texts[0] if segment_texts else "",
                        position=0,
                        num_videos=len(segment_texts),
                        api_key=api_key,
                        model=model,
                        reference_image_info=reference_image_info
                    )
                except Exception as e:
                    print(f"⚠️  Failed to generate opening still image prompt: {e}")
                    opening_prompt = f"A hyperrealistic, photorealistic, high-quality opening still image, as if photographed by a professional documentary photographer, representing the video topic: {script[:200] if script else 'documentary introduction'}. Make it look like a real photograph with natural lighting, realistic textures, and maximum detail."
                
                validated_segments.insert(0, {
                    'segment_id': 0,
                    'segment_text': segment_texts[0] if segment_texts else "",
                    'image_prompt': opening_prompt,
                    'duration': 12.0,  # Opening still images are always 12 seconds
                    'reasoning': "Opening still image (test mode)"
                })
        
        return validated_segments
        
    except Exception as e:
        print(f"⚠️  Script analysis for still images failed: {e}")
        # If test mode and analysis failed, at least add opening still image
        if test:
            try:
                opening_prompt = generate_still_image_prompt(
                    script=script,
                    context_segment=segment_texts[0] if segment_texts else "",
                    position=0,
                    num_videos=len(segment_texts),
                    api_key=api_key,
                    model=model,
                    reference_image_info=reference_image_info
                )
                return [{
                    'segment_id': 0,
                    'segment_text': segment_texts[0] if segment_texts else "",
                    'image_prompt': opening_prompt,
                    'duration': 12.0,
                    'reasoning': "Opening still image (test mode - fallback)"
                }]
            except:
                pass
        return []  # Return empty list if analysis fails


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
        
        # Recalculate corner positions with adjusted pan distances
        if pan_direction == 'top_left_to_bottom_right':
            start_pos = (0, 0)
            end_pos = (adjusted_max_pan_w, adjusted_max_pan_h)
        elif pan_direction == 'top_right_to_bottom_left':
            start_pos = (adjusted_max_pan_w, 0)
            end_pos = (0, adjusted_max_pan_h)
        elif pan_direction == 'bottom_left_to_top_right':
            start_pos = (0, adjusted_max_pan_h)
            end_pos = (adjusted_max_pan_w, 0)
        elif pan_direction == 'bottom_right_to_top_left':
            start_pos = (adjusted_max_pan_w, adjusted_max_pan_h)
            end_pos = (0, 0)
        else:
            start_pos = (0, 0)
            end_pos = (adjusted_max_pan_w, adjusted_max_pan_h)
        
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
        
        # Use crop filter with linear interpolation for straight-line corner-to-corner panning
        # Crop larger area and scale down to show more of the original image
        if pan_direction == 'top_left_to_bottom_right':
            # Pan from top-left to bottom-right (straight diagonal line)
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'t*{adjusted_max_pan_w}/{duration}':"
                f"'t*{adjusted_max_pan_h}/{duration}',"
                f"scale={output_w}:{output_h}"
            )
        elif pan_direction == 'top_right_to_bottom_left':
            # Pan from top-right to bottom-left (straight diagonal line)
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'{adjusted_max_pan_w}-t*{adjusted_max_pan_w}/{duration}':"
                f"'t*{adjusted_max_pan_h}/{duration}',"
                f"scale={output_w}:{output_h}"
            )
        elif pan_direction == 'bottom_left_to_top_right':
            # Pan from bottom-left to top-right (straight diagonal line)
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'t*{adjusted_max_pan_w}/{duration}':"
                f"'{adjusted_max_pan_h}-t*{adjusted_max_pan_h}/{duration}',"
                f"scale={output_w}:{output_h}"
            )
        elif pan_direction == 'bottom_right_to_top_left':
            # Pan from bottom-right to top-left (straight diagonal line)
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'{adjusted_max_pan_w}-t*{adjusted_max_pan_w}/{duration}':"
                f"'{adjusted_max_pan_h}-t*{adjusted_max_pan_h}/{duration}',"
                f"scale={output_w}:{output_h}"
            )
        else:
            # Default: top-left to bottom-right (straight diagonal line)
            filter_complex = (
                f"scale={scale_w}:{scale_h},"
                f"crop={crop_w}:{crop_h}:"
                f"'t*{adjusted_max_pan_w}/{duration}':"
                f"'t*{adjusted_max_pan_h}/{duration}',"
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
            raise Exception(f"Failed to start video generation job: {e2}")
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
                raise Exception(f"Failed to start video generation job: {e2}")
        else:
            raise Exception(f"Failed to start video generation job: {e}")


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
                error_msg = getattr(status_response, 'error', 'Unknown error')
                raise Exception(f"Video generation failed for job {video_id}: {error_msg}")
            
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
        
        # Build ffmpeg command for concatenation
        cmd = [
            ffmpeg_path,
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # Copy streams without re-encoding (faster)
            '-y',  # Overwrite output file
            output_path
        ]
        
        # Run ffmpeg
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
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


def generate_srt_from_audio(audio_path, script, output_path=None, api_key=None):
    """
    Generate an SRT subtitle file with word-level timestamps using OpenAI Whisper API.
    This provides exact synchronization between captions and audio.
    
    Args:
        audio_path: Path to the audio file (voiceover) to analyze
        script: The narration script text (for reference/validation)
        output_path: Path to save the SRT file (default: temp file)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var or global)
        
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
        
        # Generate SRT content - split by sentences, preserve punctuation
        # First, identify sentence boundaries by checking for punctuation in words
        import re
        srt_content = []
        subtitle_index = 1
        
        # Process words and identify sentence endings
        word_list = []
        for word_data in words:
            # Extract word and timing
            if isinstance(word_data, dict):
                word = word_data.get('word', '').strip()
                start = word_data.get('start', 0)
                end = word_data.get('end', start + 0.3)  # Default 0.3s if end not provided
            else:
                # Handle object attributes
                word = getattr(word_data, 'word', '').strip()
                start = getattr(word_data, 'start', 0)
                end = getattr(word_data, 'end', start + 0.3)
            
            if not word:
                continue
            
            # Ensure minimum display time for readability (0.35 seconds)
            min_display_time = 0.35
            word_duration = end - start
            if word_duration < min_display_time:
                # Extend end time to meet minimum
                end = start + min_display_time
            
            # Check if this word ends a sentence (contains sentence-ending punctuation)
            is_sentence_end = bool(re.search(r'[.!?]+', word))
            
            word_list.append({'word': word, 'start': start, 'end': end, 'is_sentence_end': is_sentence_end})
        
        # Create subtitles: split by sentences, ensure sentence ends on its own line
        # Track the end time of the previous subtitle to ensure no overlap
        last_end_time = 0.0
        MIN_GAP = 0.02  # Minimum gap between subtitles (20ms) to prevent any overlap
        SENTENCE_GAP = 0.1  # Gap between sentences (100ms)
        
        if word_list:
            i = 0
            current_sentence_words = []
            
            while i < len(word_list):
                word_data = word_list[i]
                word = word_data['word']
                audio_start = word_data['start']
                audio_end = word_data['end']
                is_sentence_end = word_data['is_sentence_end']
                
                # Add word to current sentence
                current_sentence_words.append(word_data)
                
                # If this is the end of a sentence, process the entire sentence
                if is_sentence_end or i == len(word_list) - 1:
                    # Process all words in this sentence
                    j = 0
                    while j < len(current_sentence_words):
                        sentence_word_data = current_sentence_words[j]
                        sentence_word = sentence_word_data['word']
                        sentence_audio_start = sentence_word_data['start']
                        sentence_audio_end = sentence_word_data['end']
                        is_last_in_sentence = (j == len(current_sentence_words) - 1)
                        
                        # CRITICAL: Start time must be after previous subtitle ends (with gap)
                        subtitle_start = max(sentence_audio_start, last_end_time + MIN_GAP)
                        
                        # Ensure minimum display time
                        min_end = subtitle_start + 0.35
                        subtitle_end = max(sentence_audio_end, min_end)
                        
                        # If this is the last word in the sentence, it MUST be on its own line
                        if is_last_in_sentence:
                            # Last word of sentence - ensure it displays long enough
                            if subtitle_end - subtitle_start < 0.5:
                                subtitle_end = subtitle_start + 0.5
                            
                            srt_content.append(f"{subtitle_index}")
                            srt_content.append(f"{format_srt_time(subtitle_start)} --> {format_srt_time(subtitle_end)}")
                            srt_content.append(sentence_word)  # Includes punctuation
                            srt_content.append("")  # Empty line between entries
                            subtitle_index += 1
                            
                            # CRITICAL: Update last_end_time - next sentence MUST start after this ends + gap
                            last_end_time = subtitle_end + SENTENCE_GAP
                            j += 1
                        else:
                            # Not the last word - can group with next word in sentence (max 2 words)
                            next_sentence_word_data = current_sentence_words[j + 1] if j + 1 < len(current_sentence_words) else None
                            
                            if next_sentence_word_data and not next_sentence_word_data['is_sentence_end']:
                                # Group 2 words together (but not if next is sentence end)
                                next_word = next_sentence_word_data['word']
                                next_audio_start = next_sentence_word_data['start']
                                next_audio_end = next_sentence_word_data['end']
                                
                                # Group if next word starts within 0.2s of current word's end
                                if next_audio_start <= sentence_audio_end + 0.2:
                                    group_text = f"{sentence_word} {next_word}"
                                    group_start = subtitle_start
                                    group_end = max(subtitle_end, next_audio_end)
                                    # Ensure minimum display time for 2 words
                                    if group_end - group_start < 0.5:
                                        group_end = group_start + 0.5
                                    
                                    # Check if there's a word after this group
                                    next_next_word_data = current_sentence_words[j + 2] if j + 2 < len(current_sentence_words) else None
                                    if next_next_word_data:
                                        # Next subtitle must start after this one ends + gap
                                        next_subtitle_start = max(next_next_word_data['start'], group_end + MIN_GAP)
                                        # Ensure current subtitle ends before next starts
                                        if group_end >= next_subtitle_start:
                                            group_end = next_subtitle_start - MIN_GAP
                                            # But maintain minimum duration
                                            if group_end - group_start < 0.5:
                                                group_end = group_start + 0.5
                                    
                                    srt_content.append(f"{subtitle_index}")
                                    srt_content.append(f"{format_srt_time(group_start)} --> {format_srt_time(group_end)}")
                                    srt_content.append(group_text)
                                    srt_content.append("")  # Empty line between entries
                                    subtitle_index += 1
                                    
                                    # CRITICAL: Update last_end_time
                                    last_end_time = group_end
                                    
                                    # Skip next word since we've already included it
                                    j += 2
                                    continue
                            
                            # Single word subtitle (not last in sentence)
                            # CRITICAL: Check when next subtitle would start
                            if next_sentence_word_data:
                                # Next subtitle should start after current ends + gap
                                next_subtitle_start = max(next_sentence_word_data['start'], subtitle_end + MIN_GAP)
                                # Ensure current subtitle ends before next starts
                                if subtitle_end >= next_subtitle_start:
                                    subtitle_end = next_subtitle_start - MIN_GAP
                                    # But maintain minimum duration
                                    if subtitle_end - subtitle_start < 0.35:
                                        subtitle_end = subtitle_start + 0.35
                            
                            srt_content.append(f"{subtitle_index}")
                            srt_content.append(f"{format_srt_time(subtitle_start)} --> {format_srt_time(subtitle_end)}")
                            srt_content.append(sentence_word)
                            srt_content.append("")  # Empty line between entries
                            subtitle_index += 1
                            
                            # CRITICAL: Update last_end_time
                            last_end_time = subtitle_end
                            j += 1
                    
                    # Clear sentence words and move to next sentence
                    current_sentence_words = []
                
                i += 1
        
        # Write SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_content))
        
        print(f"✅ Generated SRT with {subtitle_index - 1} word-level timestamps")
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
    
    # Generate SRT content - split by sentences, ensure sentence ends on its own line
    srt_content = []
    current_time = 0.2  # Start slightly earlier (0.2 seconds in)
    subtitle_index = 1
    MIN_GAP = 0.01  # Minimum gap between subtitles (10ms) to prevent any overlap
    SENTENCE_GAP = 0.1  # Gap between sentences (100ms)
    last_end_time = 0.0
    
    for sentence in sentence_list:
        # Preserve punctuation - don't remove it
        sentence = sentence.strip()
        
        if not sentence:
            continue
        
        # Split sentence into words (preserving punctuation attached to words)
        # Use regex to split on whitespace but keep punctuation with words
        words = re.findall(r'\S+', sentence)
        
        if not words:
            continue
        
        # Check if last word ends with sentence-ending punctuation
        last_word = words[-1]
        is_sentence_end = bool(re.search(r'[.!?]+', last_word))
        
        # Calculate total duration for this sentence
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
        word_duration = sentence_duration / word_count
        word_duration = min(0.7, max(0.35, word_duration))  # 0.35-0.7 seconds per word
        
        # Process words in this sentence
        word_idx = 0
        while word_idx < len(words):
            if current_time >= video_duration - 0.2:
                break
            
            current_word = words[word_idx]
            is_last_word = (word_idx == len(words) - 1)
            
            # If this is the last word in the sentence, it MUST be on its own line
            if is_last_word:
                # Last word of sentence - ensure it displays long enough
                start_time = max(current_time, last_end_time + MIN_GAP)
                end_time = start_time + max(word_duration, 0.5)  # At least 0.5s for last word
                end_time = min(end_time, video_duration - 0.1)
                
                srt_content.append(f"{subtitle_index}")
                srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
                srt_content.append(current_word)  # Includes punctuation
                srt_content.append("")  # Empty line between entries
                subtitle_index += 1
                
                # CRITICAL: Update last_end_time - next sentence MUST start after this ends + gap
                last_end_time = end_time + SENTENCE_GAP
                current_time = last_end_time
                word_idx += 1
            else:
                # Not the last word - can group with next word (max 2 words)
                next_word = words[word_idx + 1] if word_idx + 1 < len(words) else None
                
                if next_word and word_idx + 1 < len(words) - 1:  # Don't group if next is last word
                    # Group 2 words together
                    group_text = f"{current_word} {next_word}"
                    start_time = max(current_time, last_end_time + MIN_GAP)
                    end_time = start_time + (word_duration * 1.8)  # Slightly longer for 2 words
                    end_time = min(end_time, video_duration - 0.1)
                    
                    srt_content.append(f"{subtitle_index}")
                    srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
                    srt_content.append(group_text)
                    srt_content.append("")  # Empty line between entries
                    subtitle_index += 1
                    
                    # Move to next subtitle start time (end of current + gap)
                    last_end_time = end_time
                    current_time = end_time + MIN_GAP
                    word_idx += 2  # Skip both words
                else:
                    # Single word subtitle (not last in sentence)
                    start_time = max(current_time, last_end_time + MIN_GAP)
                    end_time = start_time + word_duration
                    end_time = min(end_time, video_duration - 0.1)
                    
                    srt_content.append(f"{subtitle_index}")
                    srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
                    srt_content.append(current_word)
                    srt_content.append("")  # Empty line between entries
                    subtitle_index += 1
                    
                    # Move to next subtitle start time (end of current + gap)
                    last_end_time = end_time
                    current_time = end_time + MIN_GAP
                    word_idx += 1
    
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
        # Much lower on screen, refined typography
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
    
    # Calculate segment duration early (needed for display)
    # Fixed parameters
    VIDEO_SEGMENT_DURATION = 12.0  # Fixed 12-second video segments
    STILL_IMAGE_DURATION = 12.0  # Fixed 12-second still images
    VIDEOS_PER_STILL = 3  # Still image after every 3 videos
    
    # Calculate number of video segments and still images to match input duration exactly
    # Target: 1/3 of duration should be still images (max 1/3)
    # Formula: total_duration = (num_videos * 12) + (num_still * 12)
    # Where we want: num_still * 12 ≈ total_duration / 3 (but not more than 1/3)
    
    # If test mode, we need an extra still image at the beginning
    opening_still_duration = STILL_IMAGE_DURATION if test else 0
    
    # Calculate target still image duration (1/3 of total, but not more)
    target_still_duration = duration / 3.0
    max_still_duration = duration / 3.0  # Hard cap at 1/3
    
    # Calculate ideal number of still images based on 1/3 target
    ideal_num_stills = int(target_still_duration / STILL_IMAGE_DURATION)
    # Ensure we don't exceed 1/3
    max_stills_by_duration = int(max_still_duration / STILL_IMAGE_DURATION)
    ideal_num_stills = min(ideal_num_stills, max_stills_by_duration, 1)  # At least 1 if duration >= 24
    
    # Calculate remaining duration for video segments
    # Account for opening still if in test mode
    remaining_duration = duration - (ideal_num_stills * STILL_IMAGE_DURATION) - opening_still_duration
    
    # Calculate how many 12-second video segments fit in remaining duration
    num_videos = int(remaining_duration / VIDEO_SEGMENT_DURATION)
    
    # Recalculate total to ensure it matches input duration exactly
    total_still_duration = ideal_num_stills * STILL_IMAGE_DURATION
    total_video_duration = num_videos * VIDEO_SEGMENT_DURATION
    calculated_total = total_video_duration + total_still_duration + opening_still_duration
    
    # If we're under the target duration, try to add more still images or videos
    if calculated_total < duration:
        # First, try adding more still images (up to 1/3 limit)
        remaining_time = duration - calculated_total
        additional_stills = int(remaining_time / STILL_IMAGE_DURATION)
        max_additional = max_stills_by_duration - ideal_num_stills
        additional_stills = min(additional_stills, max_additional)
        
        if additional_stills > 0:
            ideal_num_stills += additional_stills
            total_still_duration = ideal_num_stills * STILL_IMAGE_DURATION
            remaining_duration = duration - total_still_duration - opening_still_duration
            num_videos = int(remaining_duration / VIDEO_SEGMENT_DURATION)
            total_video_duration = num_videos * VIDEO_SEGMENT_DURATION
            calculated_total = total_video_duration + total_still_duration + opening_still_duration
        
        # If still under, try adding one more video segment if it fits
        if calculated_total < duration:
            remaining_time = duration - calculated_total
            if remaining_time >= VIDEO_SEGMENT_DURATION:
                num_videos += 1
                total_video_duration = num_videos * VIDEO_SEGMENT_DURATION
                calculated_total = total_video_duration + total_still_duration + opening_still_duration
    
    # Final calculation with opening still
    num_still_images = ideal_num_stills
    if test:
        num_still_images += 1  # Add opening still image
        total_still_duration = num_still_images * STILL_IMAGE_DURATION
        calculated_total = total_video_duration + total_still_duration
    
    # Verify the calculation matches input duration (allow small rounding differences)
    if abs(calculated_total - duration) > 0.1:
        print(f"⚠️  Warning: Calculated total ({calculated_total:.1f}s) doesn't match input duration ({duration:.1f}s)")
        print(f"   Adjusting to match input duration...")
        # If we're over, we need to reduce still images or videos
        if calculated_total > duration:
            # Reduce still images first (they're less critical)
            while calculated_total > duration and num_still_images > 0:
                num_still_images -= 1
                total_still_duration = num_still_images * STILL_IMAGE_DURATION
                calculated_total = total_video_duration + total_still_duration
            # If still over, reduce video segments
            while calculated_total > duration and num_videos > 1:
                num_videos -= 1
                total_video_duration = num_videos * VIDEO_SEGMENT_DURATION
                calculated_total = total_video_duration + total_still_duration
        # If we're under, we can add padding or extend final segment (handled later)
    
    # Use fixed segment duration
    segment_duration = VIDEO_SEGMENT_DURATION
    
    print(f"Duration: {duration:.1f}s | Video segments: {num_videos} × {VIDEO_SEGMENT_DURATION}s = {total_video_duration:.1f}s | Still images: {num_still_images} × {STILL_IMAGE_DURATION}s = {total_still_duration:.1f}s | Total: {calculated_total:.1f}s")
    
    # Step 0: Generate overarching script from video prompt (AI call)
    generated_script = None
    generated_segment_texts = []
    generated_video_prompts = []
    reference_image_info = None  # Initialize reference image info
    
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
        
        # Step 0.5: Rule-based segmentation of the script
        print(f"Step 0.5: Segmenting script into {num_videos} segments...")
        
        generated_segment_texts = segment_script_rule_based(
            script=generated_script,
            num_segments=num_videos
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
        
        # Step 0.55: Analyze script for reference image (character or subject)
        print("\n" + "="*60)
        print("Step 0.55: Analyzing script for reference image requirements...")
        
        if generated_script:
            reference_image_info = analyze_script_for_reference_image(
                script=generated_script,
                video_prompt=prompt,
                api_key=api_key,
                model='gpt-5-2025-08-07'
            )
        
        # Step 0.65: Analyze script for still image opportunities (MUST be before Sora prompt generation)
        print("\n" + "="*60)
        print("Step 0.65: Analyzing script for still image opportunities...")
        print("="*60 + "\n")
        
        still_image_segments = []
        try:
            # Pass the calculated number of still images to ensure alignment with duration
            still_image_segments = analyze_script_for_still_images(
                script=generated_script,
                segment_texts=generated_segment_texts,
                segment_duration=segment_duration,
                total_duration=duration,
                target_num_stills=num_still_images,  # Pass calculated number to ensure alignment
                api_key=api_key,
                model='gpt-5-2025-08-07',  # Match Sora prompt model
                test=test,
                reference_image_info=reference_image_info  # Pass reference image for consistency
            )
            
            if still_image_segments:
                print(f"✅ Identified {len(still_image_segments)} still image position(s)")
                for seg_info in still_image_segments:
                    seg_id = seg_info.get('segment_id', 'unknown')
                    if seg_id == 0:
                        print(f"   - Opening still image (12s)")
                    else:
                        print(f"   - Still image after video segment {seg_id} (12s)")
            else:
                print("   No still images identified")
        except Exception as e:
            print(f"⚠️  Still image analysis failed: {e}")
            still_image_segments = []
        
        # Step 0.6: Convert each segment text to Sora-2 video prompt (AI call per segment)
        # Now that we know where still images will be placed, we can calculate correct timing
        print("\n" + "="*60)
        print(f"Step 0.6: Converting segment texts to Sora-2 video prompts...")
        print("="*60 + "\n")
        
        generated_video_prompts = generate_sora_prompts_from_segments(
            segment_texts=generated_segment_texts,
            segment_duration=segment_duration,
            total_duration=duration,
            overarching_script=generated_script,  # Pass full script for context and chronological flow
            reference_image_info=reference_image_info,  # Pass reference image info for context
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
    
    # Step 0.7: Load or generate voiceover with music from the overarching script
    # SKIP if skip_narration is True (narration will be generated in Step 3)
    voiceover_audio_path = None
    original_voiceover_backup = None
    if not skip_narration and generated_script:
        print("Step 0.7: Loading or generating voiceover with background music...")
        
        # Try to load narration audio from file first
        narration_file = load_narration_from_file()
        
        if narration_file:
            voiceover_audio_path = narration_file
            # Try to find the original voiceover backup (without music) if it exists
            backup_path = narration_file.replace('.mp3', '_original.mp3')
            if os.path.exists(backup_path):
                original_voiceover_backup = backup_path
        else:
            try:
                temp_dir = tempfile.gettempdir()
                timestamp = int(time.time())
                voiceover_audio_path = os.path.join(temp_dir, f"voiceover_{timestamp}.mp3")
                
                voiceover_audio_path, original_voiceover_backup = generate_voiceover_with_music(
                    script=generated_script,
                    output_path=voiceover_audio_path,
                    api_key=api_key,
                    voice='nova',  # Deep, manly male voice - options: 'alloy' (neutral), 'echo' (male, more British-sounding), 'fable' (male), 'onyx' (deep male), 'nova' (female), 'shimmer' (female)
                    music_style='cinematic',  # Can be: 'cinematic', 'ambient', 'upbeat', 'calm'
                    music_volume=0.10,  # 10% volume for background music
                    duration=duration
                )
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Voiceover generation failed: {e}")
                import sys
                sys.exit(1)
    elif skip_narration:
        print("⏭️  Step 0.7: Skipping narration generation (will be done in Step 3)")
    
    # Step 1: Generate master/reference image
    print("Step 1: Generating master reference image...")
    master_image_path = None
    # CRITICAL: If API call fails before Sora 2 video generation, exit the program
    try:
        # Save reference image to output folder
        timestamp = int(time.time())
        master_image_path = os.path.join(output_folder, f"reference_image_{timestamp}.png")
        
        # Use analyzed reference image prompt if available, otherwise fall back to description
        if reference_image_info and reference_image_info.get('image_prompt'):
            print("✅ Using analyzed reference image prompt (if branch)")
            image_prompt = reference_image_info['image_prompt']
            print(f"Using analyzed reference image prompt ({reference_image_info.get('type', 'subject')}):")
            print(f"  {image_prompt[:200]}...")
            master_image_path = generate_master_image_from_prompt(
                image_prompt=image_prompt,
                output_path=master_image_path,
                api_key=api_key,
                resolution=resolution
            )
        else:
            print("✅ Using description-based reference image generation (else branch)")
            # Fallback to description-based generation
            print(f"Using description-based reference image generation...")
            master_image_path = generate_master_image_from_prompt(
                description=description or prompt,
                output_path=master_image_path,
                api_key=api_key,
                resolution=resolution
            )
        print(f"✅ Master image generated: {master_image_path}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Master image generation API call failed before Sora 2 video generation: {e}")
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
    
    # segment_duration was calculated to ensure total video matches input duration
    # Formula: (num_videos * segment_duration) + total_still_duration = duration
    adjusted_segment_duration = segment_duration
    
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
        
        for segment_id in range(1, num_videos + 1):
            segment_prompt = video_prompts_to_use[segment_id - 1]
            
            # Validate prompt is not empty
            if not segment_prompt or len(segment_prompt.strip()) == 0:
                if segment_id > 1 and len(video_prompts_to_use) > 0:
                    segment_prompt = video_prompts_to_use[segment_id - 2] if segment_id > 1 else prompt
                else:
                    segment_prompt = prompt
            
            print(f"Segment {segment_id}/{num_videos}: {segment_prompt}")
            
            # Create output path for this segment
            base, ext = os.path.splitext(output_video_path)
            segment_output_path = f"{base}_segment_{segment_id:03d}{ext}"
            
            # Start video generation job (non-blocking)
            try:
                ref_image_to_use = None
                if master_image_path and os.path.exists(master_image_path):
                    ref_image_to_use = master_image_path
                
                video_segment_duration = int(adjusted_segment_duration) if adjusted_segment_duration != segment_duration else int(segment_duration)
                
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
                        retry_video_id = start_video_generation_job(
                            prompt=segment_prompt,
                            api_key=api_key,
                            model=model,
                            resolution=resolution,
                            duration=int(adjusted_segment_duration),
                            reference_image_path=master_image_path if master_image_path and os.path.exists(master_image_path) else None
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
                    
                    if attempt < max_retries:
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
                            fallback_video_duration = int(adjusted_segment_duration) if adjusted_segment_duration else int(segment_duration)
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
        # Pattern: V1, V2, V3, Still(12s), V4, V5, V6, Still(12s), V7, V8, V9, Still(12s), ...
        all_segment_paths = []
        
        # Create a mapping of segment_id -> video_path for Sora videos
        sora_video_map = {}
        for seg_info in generated_video_segments:
            sora_video_map[seg_info['segment_id']] = seg_info['video_path']
        
        # Create a mapping of position -> still image video path
        still_image_map = {}
        opening_still_path = None
        for seg_info in still_image_segments:
            position = seg_info['segment_id']  # Position: 0 = opening, or after which video (e.g., 3 = after video 3)
            if position == 0:
                # Opening still image (test mode)
                if 0 in still_image_videos:
                    opening_still_path = still_image_videos[0]
            else:
                if position in still_image_videos:
                    still_image_map[position] = still_image_videos[position]
        
        # If test mode, add opening still image first
        if opening_still_path:
            all_segment_paths.append(opening_still_path)
        
        # Combine in order: for each video segment, add it, then add still image if needed
        for segment_id in range(1, num_videos + 1):
            if segment_id in sora_video_map:
                all_segment_paths.append(sora_video_map[segment_id])
            else:
                print(f"⚠️  Warning: No video found for segment {segment_id}")
            
            if segment_id in still_image_map:
                all_segment_paths.append(still_image_map[segment_id])
        
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
                        print(f"   Expected: {num_videos} video segments × {VIDEO_SEGMENT_DURATION}s + {num_still_images} still images × {STILL_IMAGE_DURATION}s = {calculated_total:.1f}s")
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
                    
                    # Re-mix audio with proper synchronization:
                    # 1. Extract voiceover from mixed audio (we'll need to regenerate or extract it)
                    # 2. Sync music to video duration exactly
                    # 3. Re-mix with voiceover having padding
                    
                    # For now, we'll sync the mixed audio and use FFmpeg's audio positioning
                    # The music portion should align with video, voiceover can have padding
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    timestamp = int(time.time())
                    
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
                                    # Trim music
                                    cmd_music = [
                                        ffmpeg_path,
                                        "-i", music_source,
                                        "-t", str(video_duration),
                                        "-af", f"afade=t=out:st={max(0, video_duration-2)}:d=2",
                                        "-c:a", "libmp3lame",
                                        "-b:a", "192k",
                                        "-y",
                                        synced_music_path
                                    ]
                                else:
                                    # Loop music to extend
                                    loop_count = int((video_duration / music_duration) + 1)
                                    cmd_music = [
                                        ffmpeg_path,
                                        "-stream_loop", str(loop_count - 1),
                                        "-i", music_source,
                                        "-t", str(video_duration),
                                        "-af", f"afade=t=out:st={max(0, video_duration-2)}:d=2",
                                        "-c:a", "libmp3lame",
                                        "-b:a", "192k",
                                        "-y",
                                        synced_music_path
                                    ]
                                
                                try:
                                    subprocess.run(cmd_music, capture_output=True, text=True, check=True)
                                    print(f"   ✅ Music synced to video duration: {video_duration:.2f}s")
                                    
                                    # Try to get original voiceover from temp backup
                                    voiceover_source = None
                                    if original_voiceover_backup and os.path.exists(original_voiceover_backup):
                                        voiceover_source = original_voiceover_backup
                                        print(f"   Using original voiceover from backup for re-mixing")
                                    else:
                                        # Fallback: try to find original voiceover in temp directory
                                        import glob
                                        temp_dir_check = tempfile.gettempdir()
                                        original_pattern = os.path.join(temp_dir_check, "original_voiceover_*.mp3")
                                        original_files = glob.glob(original_pattern)
                                        if original_files:
                                            # Get most recent original voiceover
                                            voiceover_source = max(original_files, key=os.path.getmtime)
                                            print(f"   Using original voiceover from temp directory: {os.path.basename(voiceover_source)}")
                                    
                                    # Final fallback: use mixed audio (not ideal, but better than nothing)
                                    using_mixed_audio_as_source = False
                                    if not voiceover_source or not os.path.exists(voiceover_source):
                                        voiceover_source = voiceover_audio_path
                                        using_mixed_audio_as_source = True
                                        print(f"   ⚠️  Using mixed audio as voiceover source (original not found)")
                                    
                                    # Adjust voiceover to fit within video bounds (can start at/after video start, end at/before video end)
                                    voiceover_duration = get_media_duration(voiceover_source, ffmpeg_path)
                                    if voiceover_duration:
                                        # Voiceover should be <= video_duration (can be shorter)
                                        # Narration must be shorter than video by at most 5 seconds
                                        # It can start at video start or up to voiceover_tolerance seconds after
                                        # It can end at video end or up to 5 seconds before (for opening/closing shots)
                                        max_voiceover_duration = video_duration  # Can't be longer than video
                                        min_voiceover_duration = max(1.0, video_duration - 5.0)  # At most 5 seconds shorter than video
                                        
                                        target_voiceover_duration = min(max_voiceover_duration, max(min_voiceover_duration, voiceover_duration))
                                        
                                        if abs(voiceover_duration - target_voiceover_duration) > 0.1:
                                            print(f"   Adjusting voiceover duration: {voiceover_duration:.2f}s -> {target_voiceover_duration:.2f}s")
                                            adjusted_voiceover = os.path.join(temp_dir, f"voiceover_adjusted_{timestamp}.mp3")
                                            
                                            if voiceover_duration > target_voiceover_duration:
                                                # Trim voiceover
                                                cmd_voiceover = [
                                                    ffmpeg_path,
                                                    "-i", voiceover_source,
                                                    "-t", str(target_voiceover_duration),
                                                    "-c:a", "copy",
                                                    "-y",
                                                    adjusted_voiceover
                                                ]
                                            else:
                                                # Voiceover is shorter - that's fine, it can start after video start
                                                # Just copy it
                                                import shutil
                                                shutil.copy2(voiceover_source, adjusted_voiceover)
                                            
                                            try:
                                                if voiceover_duration > target_voiceover_duration:
                                                    subprocess.run(cmd_voiceover, capture_output=True, text=True, check=True)
                                                voiceover_source = adjusted_voiceover
                                            except Exception as e:
                                                print(f"   ⚠️  Voiceover adjustment failed: {e}")
                                                voiceover_source = voiceover_source  # Use original
                                        
                                        # Check if we're using mixed audio - if so, don't add music again!
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
                                            
                                            # Calculate voiceover start delay (can be 0 to voiceover_tolerance seconds)
                                            voiceover_start_delay = 0.0  # Start at video start by default
                                            if voiceover_duration < video_duration:
                                                # If voiceover is shorter, we can delay it slightly
                                                voiceover_start_delay = min(voiceover_tolerance, (video_duration - target_voiceover_duration) / 2)
                                            
                                            # Mix: music starts at 0s, voiceover starts at voiceover_start_delay
                                            # Add volume boost after mixing to prevent quiet audio (amix can reduce overall volume)
                                            filter_complex = (
                                                f"[0:a]aresample=44100,volume=1.0,adelay={int(voiceover_start_delay * 1000)}|{int(voiceover_start_delay * 1000)}[voice_delayed];"
                                                f"[1:a]aresample=44100,volume={0.10}[music];"  # 10% volume for background music
                                                f"[voice_delayed][music]amix=inputs=2:duration=longest:dropout_transition=2,"
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
                                            if voiceover_start_delay > 0:
                                                print(f"   ✅ Voiceover starts {voiceover_start_delay:.2f}s after video start")
                                            if target_voiceover_duration < video_duration:
                                                print(f"   ✅ Voiceover ends {video_duration - target_voiceover_duration:.2f}s before video end")
                                    
                                except Exception as e:
                                    print(f"   ⚠️  Music re-sync failed: {e}")
                                    print(f"   Using original mixed audio")
                        else:
                            print(f"   ✅ Music already matches video duration")
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
                        base, ext = os.path.splitext(video_path)
                        video_with_subtitles_path = f"{base}_with_subtitles{ext}"
                        
                        # Use original voiceover backup for accurate word-level timing (pure voiceover, no music)
                        audio_for_subtitles = original_voiceover_backup if original_voiceover_backup and os.path.exists(original_voiceover_backup) else voiceover_audio_path
                        
                        video_with_subtitles = add_subtitles_to_video(
                            video_path=video_path,
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
        
        # Step 2: Generate video (Sora video generation)
        print("\n" + "="*60)
        print("STEP 2: Generate Video")
        print("="*60)
        step2_input = 'n'
        if not args.non_interactive:
            try:
                step2_input = input("Generate video? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step2_input = 'n'
        
        if step2_input in ['y', 'yes']:
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
        
        # Step 3: Generate narration (generate narration audio)
        print("\n" + "="*60)
        print("STEP 3: Generate Narration")
        print("="*60)
        step3_input = 'n'
        if not args.non_interactive:
            try:
                step3_input = input("Generate narration audio? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                step3_input = 'n'
        
        if step3_input in ['y', 'yes']:
            try:
                narration_file = generate_and_save_narration(
                    script_file_path=SCRIPT_FILE_PATH,
                    narration_audio_path=NARRATION_AUDIO_PATH,
                    duration=None,
                    api_key=args.api_key
                )
                print(f"\n✅ STEP 3 COMPLETE: Narration generation complete!")
                print(f"🎙️  Narration audio saved to: {narration_file}")
                print(f"\n✅ Script execution complete. Exiting.")
                return 0
            except Exception as e:
                print(f"\n❌ Error generating narration: {e}")
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
                step4_input = input("Add captions based on narration? (y/n, default: n): ").strip().lower()
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

