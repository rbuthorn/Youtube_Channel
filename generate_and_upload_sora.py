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
import json
from pathlib import Path

# Audio processing imports (no speed adjustment - narration is used as-is)

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

def analyze_script_for_reference_image(script, video_prompt, api_key=None, model='gpt-5-2025-08-07'):
    """
    Analyze script to determine if there is a main character that needs a reference image.
    Returns ONLY the main character reference image (if one exists), or None.
    This function only looks for a single main character - no subject references, no multiple references.
    
    Args:
        script: The full script text
        video_prompt: The original video prompt/topic
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-5-2025-08-07')
        
    Returns:
        Dictionary with 'id', 'type' ('character' only), 'description', 'image_prompt', and 'reasoning'
        OR None if no main character is identified
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: openai")
    
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    client = OpenAI(api_key=api_key)
    
    analysis_prompt = f"""Analyze this script to determine if there is a SINGLE MAIN CHARACTER that needs a reference image for visual consistency.

Video topic: {video_prompt}

Script:
{script}

CRITICAL REQUIREMENTS:
- Identify ONLY the SINGLE MAIN CHARACTER (if one exists) that is the central focus of the video
- This must be a specific person/character that appears throughout the video and is the primary subject
- If there is NO clear main character, return null/empty
- Do NOT identify locations, objects, ships, buildings, or other subjects - ONLY characters
- Do NOT identify multiple characters - ONLY the single most important main character
- The character must be the central focus of the entire video, not just mentioned briefly

If a main character is identified, return a JSON object with this structure:
{{
    "id": "ref_1",
    "type": "character",
    "description": "Clear description of the main character (e.g., 'the main historical figure', 'the central person in the story')",
    "image_prompt": "Detailed DALL-E prompt for a hyperrealistic, photorealistic image of the main character, as if photographed by a professional documentary photographer. Generic enough to avoid copyright, specific enough for consistency. Make it look like a real photograph with natural lighting, realistic textures, and lifelike detail. MUST comply with OpenAI content policy: no violence, hate, adult content, illegal activity, copyrighted characters, or likenesses of real, living people. Use generic, artistic representations only, but make them appear completely realistic and photographic with maximum detail and photorealism.",
    "reasoning": "Why this is the main character that needs visual consistency"
}}

If NO main character is identified (video is about events, places, concepts without a central character), return:
null

Provide ONLY valid JSON object or null:"""
    
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Video production assistant. Analyze scripts to identify ONLY the single main character (if one exists) that needs a reference image for visual consistency. Return ONLY character references - never subject references like locations or objects. If no clear main character exists, return null. All images must be hyperrealistic and photorealistic, as if photographed by a professional documentary photographer."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_output_tokens=2000
        )
        
        import json
        import re
        result_text = response.output_text.strip()
        
        # Handle null or empty responses
        if not result_text or result_text.lower() in ['null', 'none', '']:
            return None
        
        # Try to extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1)
        else:
            # Try to find JSON object in the text (look for first { to last })
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
        
        # Try to parse JSON with error recovery
        reference_image = None
        try:
            reference_image = json.loads(result_text)
        except json.JSONDecodeError as json_error:
            # Try to fix common JSON issues
            print(f"   [WARNING] JSON parsing failed: {json_error}")
            print(f"   [INFO] Attempting to fix JSON issues...")
            
            # Try to fix unterminated strings by closing them at the end of the problematic line
            fixed_text = result_text
            error_msg = str(json_error)
            fixed_successfully = False
            
            # Extract line and column from error message
            line_match = re.search(r'line (\d+) column (\d+)', error_msg)
            if line_match:
                error_line = int(line_match.group(1))
                error_col = int(line_match.group(2))
                
                lines = fixed_text.split('\n')
                if error_line <= len(lines):
                    problem_line = lines[error_line - 1]
                    
                    # If it's an unterminated string, try to close it
                    if 'Unterminated string' in error_msg:
                        # Count quotes before the error column (accounting for escaped quotes)
                        before_error = problem_line[:error_col]
                        # Simple quote counting (may not handle all edge cases)
                        quote_count = before_error.count('"')
                        # Subtract escaped quotes (very basic - doesn't handle all cases)
                        escaped_quotes = before_error.count('\\"')
                        quote_count -= escaped_quotes
                        
                        # If odd number of quotes, string is not closed
                        if quote_count % 2 == 1:
                            # Try to find where the string should end
                            remaining = problem_line[error_col:]
                            # Look for comma, closing brace, or bracket
                            end_match = re.search(r'[,}\]]', remaining)
                            if end_match:
                                insert_pos = error_col + end_match.start()
                                fixed_line = problem_line[:insert_pos] + '"' + problem_line[insert_pos:]
                            else:
                                # Add quote at end of line
                                fixed_line = problem_line + '"'
                            
                            lines[error_line - 1] = fixed_line
                            fixed_text = '\n'.join(lines)
                            
                            # Try parsing again
                            try:
                                reference_image = json.loads(fixed_text)
                                print(f"   [OK] Fixed unterminated string and parsed JSON successfully")
                                fixed_successfully = True
                            except json.JSONDecodeError:
                                # If that didn't work, try a more aggressive fix
                                pass
            
            # If still failing, try to extract just the essential JSON structure
            if not fixed_successfully and (reference_image is None or not isinstance(reference_image, dict)):
                # Try to extract key-value pairs manually using regex
                try:
                    # Extract fields we need - use non-greedy matching and handle multiline strings
                    # For fields that might have unterminated strings, extract up to the next field or end
                    id_match = re.search(r'"id"\s*:\s*"([^"]*)"', fixed_text)
                    type_match = re.search(r'"type"\s*:\s*"([^"]*)"', fixed_text)
                    
                    # For description and image_prompt, they might be multiline or have unterminated strings
                    # Try to extract everything between the opening quote and the next field or closing brace
                    desc_match = re.search(r'"description"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', fixed_text, re.DOTALL)
                    if not desc_match:
                        # Fallback: extract up to next field or closing brace
                        desc_match = re.search(r'"description"\s*:\s*"([^"]*?)(?:"\s*[,}])', fixed_text, re.DOTALL)
                    
                    prompt_match = re.search(r'"image_prompt"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', fixed_text, re.DOTALL)
                    if not prompt_match:
                        # Fallback: extract up to next field or closing brace
                        prompt_match = re.search(r'"image_prompt"\s*:\s*"([^"]*?)(?:"\s*[,}])', fixed_text, re.DOTALL)
                    
                    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', fixed_text, re.DOTALL)
                    if not reasoning_match:
                        # Fallback: extract up to next field or closing brace
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*?)(?:"\s*[,}])', fixed_text, re.DOTALL)
                    
                    if type_match and type_match.group(1) == 'character':
                        # Build a minimal valid JSON object
                        reference_image = {
                            'id': id_match.group(1) if id_match else 'ref_1',
                            'type': 'character',
                            'description': desc_match.group(1) if desc_match else 'Main character',
                            'image_prompt': prompt_match.group(1) if prompt_match else '',
                            'reasoning': reasoning_match.group(1) if reasoning_match else ''
                        }
                        print(f"   [OK] Reconstructed JSON from extracted fields")
                    else:
                        # If we can't reconstruct, re-raise the original error
                        raise json_error
                except Exception as e:
                    # Last resort: log and re-raise
                    print(f"   [DEBUG] Could not fix JSON. Response preview (first 500 chars): {result_text[:500]}")
                    if len(result_text) > 500:
                        print(f"   [DEBUG] Response preview (last 300 chars): ...{result_text[-300:]}")
                    raise json_error
        
        # Validate it's a character type
        if isinstance(reference_image, dict):
            if reference_image.get('type') != 'character':
                # If it's not a character, return None
                return None
            # Ensure id exists
            if 'id' not in reference_image:
                reference_image['id'] = 'ref_1'
            return reference_image
        else:
            return None
        
    except Exception as e:
        print(f"⚠️  Script analysis for main character reference image failed: {e}")
        return None


def analyze_script_for_reference_images(script, video_prompt, api_key=None, model='gpt-4o'):
    """
    DEPRECATED: This function is no longer used. Use analyze_script_for_reference_image instead.
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
            "id": ref_img['id'],
            "type": "subject",
            "description": f"Visual representation of {video_prompt}",
            "image_prompt": f"The most hyperrealistic, ultra-detailed, photorealistic reference image possible representing {video_prompt}, as if photographed by a professional documentary photographer, suitable for video generation, with maximum detail, natural lighting, realistic textures, and lifelike quality. Make it look like a real photograph, not an illustration.",
            "reasoning": "Analysis failed, defaulting to general subject"
        }]

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
        
        # Archive music file if it exists
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
    
    # Clean dashes (already handled by clean_script_dashes, but ensure it's done here too)
    script = clean_script_dashes(script)
    
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
        try:
            os.remove(narration_audio_path)
        except Exception as e:
            print(f"⚠️  Could not delete previous narration file: {e}")
    
    # Generate narration
    print(f"Generating narration audio from script ({len(script)} chars)...")
    
    try:
        # Stitch narration files from folder (no speed adjustment - used as-is)
        voiceover_audio_path, _ = generate_voiceover_from_folder(
            script=script,
            output_path=narration_audio_path,
            narration_folder=None,  # Uses 'narration_segments' folder in current directory
            break_duration=1000,  # 1 second for breaks
            music_volume=0.07  # 7% volume for background music
        )
        
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
    
    print(f"Generating script ({duration}s)...")
    
    # Create a simplified prompt for script generation only
    # Calculate target character count: 750 characters per minute of video
    # Convert duration from seconds to minutes
    duration_minutes = duration / 60.0
    target_characters = int(duration_minutes * 750)
    
    script_prompt = f"""Create a {duration}-second documentary-style YouTube script (approximately {target_characters} characters) for: {video_prompt}

CRITICAL - THEME AND STORY CENTRALITY:
The video prompt above contains a CENTRAL THEME that must be the foundation of the entire script and story. This theme is the core concept, message, or focus that ties everything together. The script must:
- Make this theme the CENTRAL FOCUS of the entire narrative
- Weave the theme throughout every section of the script (hook, introduction, narrative, climax, conclusion, impact)
- Ensure every part of the story relates back to and reinforces this central theme
- Use the theme as the lens through which all events, characters, and concepts are presented
- Make it clear why this theme matters and how it connects all elements of the story
- The theme should be evident from the opening hook and remain central through the conclusion

IMPORTANT: The script should be approximately {target_characters} characters long (750 characters per minute of video). This allows for 2-3 second opening and closing shots with no narration.

CRITICAL ASSUMPTION: Assume the viewer knows NOTHING about this topic. Provide comprehensive context, background, and explanations throughout.

REQUIREMENTS:
- CENTRAL THEME: The theme detailed in the video prompt above must be the CENTRAL FOCUS of the entire script. Every section must relate back to and reinforce this theme. The theme should be evident from the opening hook and remain central through the conclusion.
- Tell the COMPLETE story - cover the full narrative from beginning to end, always connecting back to the central theme
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
- CRITICAL - NUMBERS: ALL numbers must be spelled out in words. For example: "1783" becomes "seventeen eighty three", "1945" becomes "nineteen forty five", "2024" becomes "twenty twenty four". Years should be split into two parts (e.g., "seventeen eighty three" not "one thousand seven hundred eighty three"). Single digit numbers should be spelled out (e.g., "three", "seven"). Never use numeric digits in the script.
- CRITICAL - NO DASHES: NEVER use dashes ("-", "--", or "---") anywhere in the script. Instead, use contextually appropriate alternatives:
  * Use commas (",") for pauses or separations (e.g., "Washington, a skilled general, led the army" not "Washington - a skilled general - led the army")
  * Use ellipses ("...") for dramatic pauses or trailing thoughts (e.g., "The battle raged on..." not "The battle raged on---")
  * Use natural phrasing to connect ideas without dashes (e.g., "He was brave, and his men followed" not "He was brave - his men followed")
  * Rewrite sentences to flow naturally without needing dashes
- One continuous script, approximately {target_characters} characters (750 characters per minute of video)

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
                {"role": "system", "content": "Expert documentary scriptwriter. Write informative, historically accurate scripts that tell complete stories in an engaging way. Structure content with hook, introduction, narrative, climax, conclusion, and impact. Blend factual accuracy with compelling storytelling - like BBC or National Geographic documentaries. Be authoritative yet accessible, informative yet entertaining. CRITICAL - THEME CENTRALITY: The video prompt provided by the user contains a CENTRAL THEME that must be the foundation of the entire script. This theme is the core concept, message, or focus that ties everything together. You MUST make this theme the CENTRAL FOCUS throughout the entire narrative - from the opening hook through the conclusion. Every section of the script (hook, introduction, narrative, climax, conclusion, impact) must weave the theme throughout and ensure every part of the story relates back to and reinforces this central theme. Use the theme as the lens through which all events, characters, and concepts are presented. Make it clear why this theme matters and how it connects all elements of the story. The theme should be evident from the opening and remain central throughout. CRITICAL: Assume the viewer knows NOTHING about the topic. Provide extensive context, background, and explanations throughout. Explain who people are, what terms mean, when/where events occurred, why they happened, and the historical/cultural context. Don't assume prior knowledge - explain everything clearly and thoroughly. CRITICAL - NUMBERS: ALL numbers must be spelled out in words. Years should be split into two parts (e.g., 'seventeen eighty three' for 1783, 'nineteen forty five' for 1945). Single digit numbers should be spelled out (e.g., 'three', 'seven'). Never use numeric digits in the script. CRITICAL - NO DASHES: NEVER use dashes ('-', '--', or '---') anywhere in the script. Use commas for pauses/separations, ellipses for dramatic pauses, or natural phrasing to connect ideas. Rewrite sentences to flow naturally without dashes. CRITICAL BREAK REQUIREMENT: You MUST include a [MUSICAL BREAK] or [VISUAL BREAK] after no more than every 2000 characters of narration text. Count characters carefully and ensure breaks occur regularly to maintain pacing. Include strategic musical breaks marked with [MUSICAL BREAK] or [VISUAL BREAK] where narration stops for 2-4 seconds to let visuals and music shine. These breaks should flow naturally - place them after dramatic moments, before transitions, or during visually stunning scenes, but ALWAYS ensure they occur at least every 2000 characters. Cover the full story comprehensively with rich context and explanations. CRITICAL: Output ONLY the script text - dialogue/narration, [MUSICAL BREAK], and [VISUAL BREAK] markers only. NO labels, NO instructions, NO explanations. The output will be read directly by text-to-speech, so any extra text will be spoken as dialogue."},
                {"role": "user", "content": script_prompt}
            ],
            max_output_tokens=max_tokens,
            temperature=1
        )
        
        script = response.output_text.strip()
        
        # Clean script to ensure it only contains dialogue, [MUSICAL BREAK], and [VISUAL BREAK]
        script = clean_script_for_tts(script)
        
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


def segment_script_by_narration(script, audio_path, segment_duration=12.0, api_key=None, expected_num_segments=None):
    """
    Segment a script into segments based on narration audio timing.
    Uses Whisper API to get word-level timestamps and groups words into 12-second segments.
    This ensures segments align with actual narration timing rather than word count.
    
    IMPORTANT: The audio_path should be the final narration audio (narration_audio.mp3)
    that matches the target duration from video_config.json. This ensures word-level timestamps
    are accurate for Sora prompt generation.
    
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
        
        print(f"✅ Segmented into {len(segments)} segments ({total_duration:.1f}s audio)")
        
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
    
    # Build context sections (no previous/next segments to avoid confusion)
    context_parts = []
    
    if reference_image_info:
        ref_type = reference_image_info.get('type', 'subject')
        ref_desc = reference_image_info.get('description', 'the main visual element')
        if ref_type == 'character':
            context_parts.append(f"CHARACTER REFERENCE: {ref_desc} - Character MUST be IDENTICAL to reference image (exact same person, not similar)")
        else:
            context_parts.append(f"Reference image: {ref_desc} - Maintain visual consistency")
    
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
    
    if reference_image_info and reference_image_info.get('type') == 'character':
        requirements_parts.append("CHARACTER: Must be IDENTICAL to reference (exact same person)")
    
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
    
    # Create concise prompt with segment_text as primary focus
    # Truncate full script to first 1000 chars for brief context only
    script_preview = overarching_script[:1000] + "..." if len(overarching_script) > 1000 else overarching_script
    
    character_ref_note = ""
    if reference_image_info and reference_image_info.get('type') == 'character':
        character_ref_note = "\nCHARACTER REFERENCE: Character MUST be IDENTICAL to reference image (exact same person, use 'identical', 'exact match', NOT 'similar' or 'resembling')."
    
    conversion_prompt = f"""Generate a Sora-2 video prompt for segment {segment_id} ({start_time:.1f}-{end_time:.1f}s).

═══════════════════════════════════════════════════════════════════════════════
PRIMARY FOCUS - NARRATION FOR THIS SEGMENT:
"{segment_text}"

YOUR TASK: Create a photorealistic video that PERFECTLY matches this narration.
- If narration mentions a location → show that location
- If narration mentions a person → show that person  
- If narration describes an action → show that action
- If narration mentions an object → show that object
- Video must make perfect sense when watched with this narration

{key_phrase_instructions if key_phrase_instructions else ""}
{character_ref_note}

Brief context (full script preview): {script_preview}
{("Additional context:\n" + context_text) if context_text else ""}

Requirements:
{requirements_text}

Provide ONLY the Sora-2 prompt (no labels, no explanations):"""
    
    # Retry logic: try up to 3 times to get a valid prompt
    max_retries = 3
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                print(f"Processing segment {segment_id}...")
            
            # Build concise system prompt
            system_prompt = "Create Sora-2 video prompts matching narration. Videos must be PHOTOREALISTIC documentary-style (real-life footage, natural lighting, authentic). No artistic/stylized/animated styles."
            if reference_image_info and reference_image_info.get('type') == 'character':
                system_prompt += " For character references: character MUST be IDENTICAL to reference (exact same person, use 'identical'/'exact match', NOT 'similar')."
            
            # Call Responses API
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
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
                    print(f"⚠️  All attempts failed, using fallback prompt")
                    return sora_prompt
            
            print(f"✅ Segment {segment_id} prompt generated")
            return sora_prompt
            
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
                    sora_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
                else:
                    sora_prompt = f"Photorealistic documentary-style video scene for segment {segment_id}, as if filmed by a professional camera with natural lighting and realistic textures"
                print(f"⚠️  All attempts failed, using fallback prompt")
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
    
        print(f"Processing {len(segment_texts)} segments...")
    
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
        
        print(f"Converting segment {i}/{len(segment_texts)} ({start_time:.1f}s-{end_time:.1f}s)...")
        
        # Verify this segment is different from previous segments
        if i > 1:
            prev_segment_text = segment_texts[i-2]  # Previous segment (i-2 because i is 1-indexed)
            if segment_text == prev_segment_text:
                raise ValueError(f"Segment {i} text is identical to segment {i-1}! Segmentation may have failed.")
        
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
                if segment_text and len(segment_text.strip()) > 0:
                    sora_prompt = f"Photorealistic documentary-style video scene matching the narration, as if filmed by a professional camera with natural lighting and realistic textures: {segment_text[:300]}"
                elif overarching_script and len(overarching_script.strip()) > 0:
                    sora_prompt = f"Photorealistic documentary-style video scene, as if filmed by a professional camera with natural lighting and realistic textures: {overarching_script[:300]}"
                else:
                    sora_prompt = f"Photorealistic documentary-style video scene for segment {i}, as if filmed by a professional camera with natural lighting and realistic textures"
            
            sora_prompts.append(sora_prompt)
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
            sora_prompts.append(fallback_prompt)
            print(f"  ⚠️  Using fallback prompt for segment {i}")
    
    # Final validation: ensure we have the correct number of prompts
    if len(sora_prompts) != len(segment_texts):
        raise ValueError(f"Mismatch: Generated {len(sora_prompts)} prompts but expected {len(segment_texts)} segments!")
    
    print(f"✅ Generated {len(sora_prompts)} Sora prompts")
    
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
        print(f"Looking for narration files...")
        
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
        
        # Step 2: Split script by break markers to determine where breaks should go
        print("Analyzing script for break markers...")
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
        
        print(f"✅ Found {len(break_positions)} break markers")
        
        # Step 2.5: Check for CTA_AUDIO file
        cta_audio_path = None
        # Check in narration folder first, then current directory
        possible_cta_paths = [
            os.path.join(narration_folder, "CTA_AUDIO.mp3"),
            os.path.join(narration_folder, "cta_audio.mp3"),
            os.path.join(narration_folder, "CTA_AUDIO.MP3"),
            os.path.join(os.getcwd(), "CTA_AUDIO.mp3"),
            os.path.join(os.getcwd(), "cta_audio.mp3"),
            os.path.join(os.getcwd(), "CTA_AUDIO.MP3"),
        ]
        
        for cta_path in possible_cta_paths:
            if os.path.exists(cta_path):
                cta_audio_path = cta_path
                print(f"✅ Found CTA_AUDIO file: {cta_audio_path}")
                break
        
        if not cta_audio_path:
            print("ℹ️  CTA_AUDIO.mp3 not found - skipping CTA and stitching narration segments normally")
        
        # Step 3: Stitch narration files together with breaks
        print("Stitching narration files...")
        
        # First pass: Load all segments and find the loudest one to use as reference
        segments = []
        max_volume = float('-inf')
        for narration_file in narration_files:
            segment_audio = AudioSegment.from_file(narration_file['path'])
            segments.append(segment_audio)
            # Get volume in dBFS (decibels relative to full scale)
            segment_volume = segment_audio.dBFS
            if segment_volume != float('-inf') and segment_volume > max_volume:
                max_volume = segment_volume
        
        # Second pass: Normalize all segments to match the loudest segment's volume
        # This ensures we maintain the maximum volume level from the originals
        normalized_segments = []
        for segment in segments:
            if segment.dBFS != float('-inf') and max_volume != float('-inf'):
                volume_diff = max_volume - segment.dBFS
                if volume_diff > 0:  # Only boost quieter segments, don't reduce louder ones
                    segment = segment.apply_gain(volume_diff)
            normalized_segments.append(segment)
        
        # Stitch normalized segments together
        final_audio = AudioSegment.empty()
        for i, segment_audio in enumerate(normalized_segments):
            final_audio += segment_audio
            
            # Insert CTA_AUDIO after narration_0 (before narration_1)
            if i == 0 and cta_audio_path and len(normalized_segments) > 1:
                print(f"   Inserting CTA_AUDIO after narration_0...")
                try:
                    cta_audio = AudioSegment.from_file(cta_audio_path)
                    final_audio += cta_audio
                    print(f"   ✅ Added CTA_AUDIO ({len(cta_audio) / 1000:.1f}s)")
                except Exception as e:
                    print(f"   ⚠️  Failed to load CTA_AUDIO: {e}")
                    print("   Continuing without CTA audio...")
            
            if i < len(normalized_segments) - 1:
                silence = AudioSegment.silent(duration=break_duration)
                final_audio += silence
        
        # Save voiceover-only file (before mixing with music)
        voiceover_only_path = os.path.join(temp_dir, f"voiceover_only_{timestamp}.mp3")
        final_audio.export(voiceover_only_path, format='mp3', bitrate='192k')
        print(f"✅ Stitched narration ({len(final_audio) / 1000:.1f}s)")
        
        # Step 4: Mix with music if available
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
        raise Exception(f"Failed to generate voiceover from folder: {e}")


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


def analyze_script_for_still_images(script, segment_texts, target_num_stills, api_key=None, model='gpt-4o', has_character_reference=False):
    """
    Analyze script to identify which segments should be still images AND determine if video segments need the character reference image.
    Multi-purpose function that:
    1. Identifies which segments should be still images (existing functionality)
    2. For each video segment, determines: does it need the main character reference image?
    
    Still images work well for: key moments, important visuals, transitions, dramatic pauses, etc.
    AVOIDS: action scenes, fights, battles, chases, fast-paced moments.
    
    Args:
        script: The full overarching script
        segment_texts: List of segment script texts (all segments, both video and still)
        target_num_stills: Target number of still images (approximately 1/3 of total segments)
        api_key: OpenAI API key
        model: Model to use (default: 'gpt-4o')
        has_character_reference: Boolean indicating if a main character reference image exists
        
    Returns:
        Dictionary with:
        - 'still_image_segments': List of dictionaries with 'segment_id', 'segment_text', 'image_prompt', 'duration', 'reasoning'
        - 'segment_assignments': List of dictionaries, one per segment, with 'segment_id', 'type' ('still' or 'video'), 'needs_character_ref' (boolean)
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
    
    # Build character reference context for the prompt
    character_reference_context = ""
    if has_character_reference:
        character_reference_context = """
═══════════════════════════════════════════════════════════════════════════════
CRITICAL - MAIN CHARACTER REFERENCE IMAGE AVAILABLE:
═══════════════════════════════════════════════════════════════════════════════
A main character reference image exists for this video. This reference image shows the main character that the video focuses on.

CHARACTER REFERENCE ASSIGNMENT RULES:
1. For EACH video segment, analyze if it features, mentions, or describes the main character
2. If a segment features the main character (even briefly or in description), set needs_character_ref to true
3. Be GENEROUS with assignments - if there's ANY connection to the main character, set needs_character_ref to true
4. Only set needs_character_ref to false if the segment truly has NO connection to the main character (e.g., generic narration, unrelated topic, establishing shots without the character)

EXAMPLES:
- Segment mentions the main character's name or actions → needs_character_ref: true
- Segment describes the main character or their story → needs_character_ref: true
- Segment is about something completely unrelated to the main character → needs_character_ref: false

CRITICAL: The reference image exists to maintain visual consistency of the main character. When in doubt, set needs_character_ref to true if there's ANY connection.
═══════════════════════════════════════════════════════════════════════════════
"""
    else:
        character_reference_context = """
NOTE: No main character reference image exists for this video. All video segments will have needs_character_ref set to false.
"""
    
    analysis_prompt = f"""Analyze this script and perform TWO tasks:

TASK 1: Identify {ideal_num_stills} segments (out of {len(segment_texts)} total) where a high-quality still image with camera panning would be most effective.

TASK 2: For EACH of the {len(segment_texts)} segments, determine:
- Is it a still image or video?
- If video, does it need the main character reference image? (true if segment features/mentions the main character, false otherwise)

{character_reference_context}

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

CRITICAL CONSTRAINTS (MUST FOLLOW):
1. FIRST THREE AND LAST SEGMENTS MUST BE VIDEOS: Segments 1, 2, and 3 (first three) and segment {len(segment_texts)} (last) MUST be type "video" - this overrides any other decision. Never assign still images to the first three segments or the last segment.
2. DISTRIBUTION: Approximately 1/3 of segments should be still images ({ideal_num_stills} out of {len(segment_texts)}), and 2/3 should be videos ({num_segments - ideal_num_stills} out of {len(segment_texts)}).
3. NO MORE THAN 2 STILL IMAGES IN A ROW: Never have more than 2 consecutive still image segments. If you need to place still images, ensure there are video segments between groups of still images.

IMPORTANT: You have {len(segment_texts)} total segments. Select exactly {ideal_num_stills} segments for still images (approximately 1/3). The remaining {num_segments - ideal_num_stills} segments will be videos (approximately 2/3).
Each still image is 12 seconds long.

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
            "needs_character_ref": true or false
        }}
    ]
}}

Rules:
- still_image_segments: Select exactly {ideal_num_stills} segments (approximately 1/3 of {len(segment_texts)} total segments) for still images
- segment_assignments: Provide an entry for EVERY segment (1 through {len(segment_texts)})
- For each segment in segment_assignments:
  * type: "still" if it's a still image, "video" if it's a video
  * needs_character_ref: true if the segment features/mentions the main character (only applies if character reference exists), false otherwise
- CRITICAL CONSTRAINT 1: Segments 1, 2, and 3 (first three) and segment {len(segment_texts)} (last) MUST be type "video" - NEVER assign still images to the first three segments or the last segment
- CRITICAL CONSTRAINT 2: Approximately 1/3 still images ({ideal_num_stills}), 2/3 videos ({num_segments - ideal_num_stills})
- CRITICAL CONSTRAINT 3: Never have more than 2 consecutive still image segments - ensure video segments break up any groups of still images
- CRITICAL: Ensure there are MORE video segments than still images (you have {len(segment_texts)} segments total, select only {ideal_num_stills} for still images)
- Each still image is 12.0 seconds long
- CRITICAL: Skip any segments that contain action, fights, battles, or fast-paced scenes for still images
- For video segments, set needs_character_ref to true if the segment features, mentions, or describes the main character
- Be GENEROUS with character reference assignments - if there's any connection to the main character, set needs_character_ref to true
- Only set needs_character_ref to false if the segment truly has NO connection to the main character (e.g., generic narration, unrelated topic, establishing shots without the character)
- CRITICAL: Most video segments should have needs_character_ref set to true if a character reference exists and the segment relates to the main character
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
        validated_still_segments = []
        for seg_info in still_image_segments_raw:
            segment_id = seg_info.get('segment_id')
            if 1 <= segment_id <= len(segment_texts):
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
        validated_assignments = []
        for assignment in segment_assignments_raw:
            segment_id = assignment.get('segment_id')
            if 1 <= segment_id <= len(segment_texts):
                needs_ref = assignment.get('needs_character_ref', False)
                # Only set to true if character reference exists
                if not has_character_reference:
                    needs_ref = False
                
                validated_assignments.append({
                    'segment_id': segment_id,
                    'type': assignment.get('type', 'video'),  # 'still' or 'video'
                    'needs_character_ref': bool(needs_ref)  # Boolean
                })
        
        # Ensure we have assignments for all segments
        if len(validated_assignments) < len(segment_texts):
            # Fill in missing segments as videos with no character reference
            existing_ids = {a['segment_id'] for a in validated_assignments}
            for i in range(1, len(segment_texts) + 1):
                if i not in existing_ids:
                    validated_assignments.append({
                        'segment_id': i,
                        'type': 'video',
                        'needs_character_ref': False
                    })
        
        # Sort assignments by segment_id
        validated_assignments.sort(key=lambda x: x['segment_id'])
        
        # ENFORCE CONSTRAINTS: Override AI output to ensure constraints are met
        print("\n🔧 Enforcing constraints on segment assignments...")
        
        # CONSTRAINT 1: First three segments and last segment MUST be videos
        if len(validated_assignments) > 0:
            # Create a mapping by segment_id for easier lookup
            assignment_by_id = {a.get('segment_id'): a for a in validated_assignments}
            
            # Enforce first three segments are videos
            for seg_id in [1, 2, 3]:
                if seg_id <= len(segment_texts) and seg_id in assignment_by_id:
                    segment = assignment_by_id[seg_id]
                    if segment.get('type') == 'still':
                        print(f"   ⚠️  Overriding: Segment {seg_id} (first three) changed from 'still' to 'video' (constraint)")
                        segment['type'] = 'video'
                        # Remove from still_image_segments if present
                        validated_still_segments = [s for s in validated_still_segments if s.get('segment_id') != seg_id]
            
            # Enforce last segment is video
            last_segment_id = len(segment_texts)
            if last_segment_id in assignment_by_id:
                last_segment = assignment_by_id[last_segment_id]
                if last_segment.get('type') == 'still':
                    print(f"   ⚠️  Overriding: Segment {last_segment_id} (last) changed from 'still' to 'video' (constraint)")
                    last_segment['type'] = 'video'
                    # Remove from still_image_segments if present
                    validated_still_segments = [s for s in validated_still_segments if s.get('segment_id') != last_segment_id]
        
        # CONSTRAINT 3: Never more than 2 still images in a row
        consecutive_stills = 0
        for assignment in validated_assignments:
            if assignment.get('type') == 'still':
                consecutive_stills += 1
                if consecutive_stills > 2:
                    # Convert this segment to video
                    print(f"   ⚠️  Overriding: Segment {assignment.get('segment_id')} changed from 'still' to 'video' (max 2 consecutive stills)")
                    assignment['type'] = 'video'
                    # Remove from still_image_segments if present
                    validated_still_segments = [s for s in validated_still_segments if s.get('segment_id') != assignment.get('segment_id')]
                    consecutive_stills = 0  # Reset counter
            else:
                consecutive_stills = 0  # Reset counter when we hit a video
        
        # Update still_image_segments list to match validated_assignments
        still_segment_ids = {a['segment_id'] for a in validated_assignments if a.get('type') == 'still'}
        validated_still_segments = [s for s in validated_still_segments if s.get('segment_id') in still_segment_ids]
        
        # CONSTRAINT 2: Ensure approximately 1/3 still, 2/3 video distribution
        current_stills = len([a for a in validated_assignments if a.get('type') == 'still'])
        target_stills = ideal_num_stills
        
        if current_stills != target_stills:
            print(f"   ⚠️  Still image count mismatch: {current_stills} stills, target is {target_stills}")
            if current_stills < target_stills:
                # Need to add more still images (but respect constraints)
                needed = target_stills - current_stills
                print(f"   📝 Need to add {needed} more still image(s) (respecting constraints)")
                # Find segments that can be converted to still (not first/last, not creating >2 consecutive)
                candidates = []
                for assignment in validated_assignments:
                    seg_id = assignment.get('segment_id')
                    if (assignment.get('type') == 'video' and 
                        seg_id != 1 and seg_id != len(segment_texts) and
                        seg_id not in still_segment_ids):
                        # Check if converting this would create >2 consecutive stills
                        # Count consecutive stills immediately before and after this segment
                        before_stills = 0
                        after_stills = 0
                        # Count consecutive stills before (going backwards from this segment)
                        for a in reversed(validated_assignments):
                            if a.get('segment_id') < seg_id:
                                if a.get('type') == 'still':
                                    before_stills += 1
                                else:
                                    break  # Stop counting when we hit a video
                        # Count consecutive stills after (going forwards from this segment)
                        for a in validated_assignments:
                            if a.get('segment_id') > seg_id:
                                if a.get('type') == 'still':
                                    after_stills += 1
                                else:
                                    break  # Stop counting when we hit a video
                        
                        # Can convert if: before_stills + 1 (this segment) + after_stills <= 2
                        if (before_stills + 1 + after_stills) <= 2:
                            candidates.append(assignment)
                
                # Convert candidates to still images (up to needed amount)
                for assignment in candidates[:needed]:
                    seg_id = assignment.get('segment_id')
                    print(f"   📝 Converting segment {seg_id} to still image to meet target")
                    assignment['type'] = 'still'
                    # Add to still_image_segments if not already present
                    if seg_id not in still_segment_ids:
                        context_segment = segment_texts[seg_id - 1]
                        validated_still_segments.append({
                            'segment_id': seg_id,
                            'segment_text': context_segment,
                            'image_prompt': f"A hyperrealistic, photorealistic, high-quality still image, as if photographed by a professional documentary photographer, representing: {context_segment[:200]}... Make it look like a real photograph with natural lighting, realistic textures, and maximum detail.",
                            'duration': STILL_IMAGE_DURATION,
                            'reasoning': 'Added to meet target distribution (1/3 still, 2/3 video)'
                        })
            elif current_stills > target_stills:
                # Need to remove some still images (but respect constraints)
                excess = current_stills - target_stills
                print(f"   📝 Need to remove {excess} still image(s) to meet target")
                # Remove still images that aren't constrained (not first/last, not breaking consecutive rule)
                still_assignments = [a for a in validated_assignments if a.get('type') == 'still']
                for assignment in still_assignments[:excess]:
                    seg_id = assignment.get('segment_id')
                    if seg_id != 1 and seg_id != len(segment_texts):
                        print(f"   📝 Converting segment {seg_id} from still to video to meet target")
                        assignment['type'] = 'video'
                        validated_still_segments = [s for s in validated_still_segments if s.get('segment_id') != seg_id]
        
        # Final count verification
        final_stills = len([a for a in validated_assignments if a.get('type') == 'still'])
        final_videos = len([a for a in validated_assignments if a.get('type') == 'video'])
        print(f"   ✅ Final distribution: {final_stills} still images, {final_videos} videos (target: {target_stills} stills, {len(segment_texts) - target_stills} videos)")
        
        # Validation: If character reference exists, warn if none are being used
        if has_character_reference:
            video_segments_with_ref = [a for a in validated_assignments if a.get('type') == 'video' and a.get('needs_character_ref')]
            if len(video_segments_with_ref) == 0:
                print(f"⚠️  WARNING: Character reference image exists, but NONE of the video segments are using it!")
                print(f"   This may indicate the AI is being too conservative. Character reference should be used for visual consistency.")
            else:
                print(f"✅ {len(video_segments_with_ref)} video segment(s) will use character reference out of {len([a for a in validated_assignments if a.get('type') == 'video'])} total video segments")
        
        return {
            'still_image_segments': validated_still_segments,
            'segment_assignments': validated_assignments
        }
        
    except Exception as e:
        print(f"⚠️  Script analysis for still images failed: {e}")
        # Return empty structure if analysis fails
        return {
            'still_image_segments': [],
            'segment_assignments': [{'segment_id': i, 'type': 'video', 'needs_character_ref': False} for i in range(1, len(segment_texts) + 1)]
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


def save_segment_metadata(output_folder, segment_id_to_prompt, generated_video_segments, 
                          still_image_videos, segment_assignments, generated_segment_texts,
                          generated_script, num_segments, num_videos, num_still_images,
                          character_reference_image_path, output_video_path):
    """
    Save segment metadata to a JSON file for later use in regeneration and stitching.
    
    Args:
        output_folder: Folder where metadata will be saved
        segment_id_to_prompt: Mapping from segment_id to Sora prompt
        generated_video_segments: List of dicts with segment_id, prompt, video_path
        still_image_videos: Dict mapping segment_id to still image video path
        segment_assignments: List of segment assignment dicts
        generated_segment_texts: List of segment text strings
        generated_script: Full script text
        num_segments: Total number of segments
        num_videos: Number of video segments
        num_still_images: Number of still image segments
        character_reference_image_path: Path to character reference image (if any)
        output_video_path: Base output video path
    """
    metadata = {
        'segment_id_to_prompt': segment_id_to_prompt,
        'generated_video_segments': generated_video_segments,
        'still_image_videos': still_image_videos,
        'segment_assignments': segment_assignments,
        'generated_segment_texts': generated_segment_texts,
        'generated_script': generated_script,
        'num_segments': num_segments,
        'num_videos': num_videos,
        'num_still_images': num_still_images,
        'character_reference_image_path': character_reference_image_path,
        'output_video_path': output_video_path
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


def ensure_audio_on_video(video_path, ffmpeg_path=None, narration_audio_path=None, music_volume=0.07):
    """
    Ensure audio (narration + music) is added to a video file.
    This function automatically finds narration and music files and adds them to the video.
    
    Args:
        video_path: Path to the video file
        ffmpeg_path: Path to ffmpeg executable (if None, will try to find it)
        narration_audio_path: Path to narration audio (if None, will try to find it)
        music_volume: Volume level for background music (default: 0.07 = 7%)
        
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
        result_path = add_audio_to_video(
            video_path=video_path,
            audio_path=final_audio_path,
            output_path=video_with_audio_path,
            ffmpeg_path=ffmpeg_path,
            sync_duration=False  # Audio already synced
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


def stitch_videos(video_paths, output_path, ffmpeg_path=None, upscale_to_1080p=False):
    """
    Stitch multiple video files together into one video using ffmpeg.
    
    Args:
        video_paths: List of video file paths in order (should be sorted by segment ID)
        output_path: Path to save the stitched video
        ffmpeg_path: Path to ffmpeg executable (if None, will try to find it)
        upscale_to_1080p: If True, upscale the stitched video to 1080p using lanczos algorithm
        
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
            ensure_audio_on_video(output_path, ffmpeg_path=ffmpeg_path)
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
# def generate_srt_from_audio(audio_path, script, output_path=None, api_key=None, segment_duration=12.0):
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


def stitch_all_segments(
    generated_video_segments,
    still_image_videos,
    segment_assignments,
    num_segments,
    output_video_path,
    duration,
    upscale_to_1080p=False
):
    """
    Stitch all video segments and still image panning videos together into final video.
    
    Args:
        generated_video_segments: List of dicts with segment_id, prompt, video_path
        still_image_videos: Dict mapping segment_id to still image video path
        segment_assignments: List of segment assignment dicts
        num_segments: Total number of segments
        output_video_path: Base output video path
        duration: Expected total duration
        upscale_to_1080p: If True, upscale the stitched video to 1080p using lanczos algorithm
        
    Returns:
        Path to stitched (and optionally upscaled) video file
    """
    # Create a mapping of segment_id -> video_path for Sora videos
    # Ensure segment_id is an integer for consistent matching
    sora_video_map = {}
    for seg_info in generated_video_segments:
        seg_id = int(seg_info['segment_id']) if seg_info.get('segment_id') is not None else None
        if seg_id is not None:
            sora_video_map[seg_id] = seg_info['video_path']
    
    # Create a mapping of segment_id -> still image video path
    # Ensure segment_id is an integer for consistent matching
    still_image_map = {}
    for seg_id, still_path in still_image_videos.items():
        # Convert segment_id to int if it's a string
        seg_id_int = int(seg_id) if isinstance(seg_id, str) else seg_id
        if still_path and os.path.exists(still_path):
            still_image_map[seg_id_int] = still_path
        else:
            print(f"⚠️  Warning: Still image video file not found or invalid: segment_id={seg_id}, path={still_path}")
    
    # Create mapping from segment_id to assignment
    # Ensure segment_id is an integer for consistent matching
    assignment_map = {}
    if segment_assignments:
        for assignment in segment_assignments:
            seg_id = assignment.get('segment_id')
            if seg_id is not None:
                seg_id_int = int(seg_id) if isinstance(seg_id, str) else seg_id
                assignment_map[seg_id_int] = assignment
    
    # Combine segments in order based on segment_assignments
    all_segment_paths = []
    print(f"\nBuilding segment order (total segments: {num_segments}):")
    print(f"  Sora video segments available: {len(sora_video_map)}")
    print(f"  Still image segments available: {len(still_image_map)}")
    print(f"  Segment assignments: {len(assignment_map)}")
    
    for segment_id in range(1, num_segments + 1):
        assignment = assignment_map.get(segment_id, {'type': 'video', 'needs_character_ref': False})
        seg_type = assignment.get('type', 'video')
        
        if seg_type == 'still':
            # Add still image panning video
            if segment_id in still_image_map:
                still_path = still_image_map[segment_id]
                if os.path.exists(still_path):
                    all_segment_paths.append(still_path)
                    print(f"  [{segment_id}] Added still image: {os.path.basename(still_path)}")
                else:
                    print(f"⚠️  Warning: Still image video file not found for segment {segment_id}: {still_path}")
            else:
                print(f"⚠️  Warning: Still image expected for segment {segment_id} but not found in still_image_map")
                print(f"     Available still image IDs: {sorted(still_image_map.keys())}")
        elif seg_type == 'video':
            # Add Sora video
            if segment_id in sora_video_map:
                video_path = sora_video_map[segment_id]
                if os.path.exists(video_path):
                    all_segment_paths.append(video_path)
                    print(f"  [{segment_id}] Added video: {os.path.basename(video_path)}")
                else:
                    print(f"⚠️  Warning: Video file not found for segment {segment_id}: {video_path}")
            else:
                print(f"⚠️  Warning: Video expected for segment {segment_id} but not found in sora_video_map")
                print(f"     Available video segment IDs: {sorted(sora_video_map.keys())}")
        else:
            print(f"⚠️  Warning: Unknown segment type '{seg_type}' for segment {segment_id}")
    
    print(f"\nTotal segments to stitch: {len(all_segment_paths)}")
    if len(all_segment_paths) != num_segments:
        print(f"⚠️  Warning: Expected {num_segments} segments but only {len(all_segment_paths)} segments found")
    
    if not all_segment_paths:
        # CRITICAL: Never stop execution - create emergency placeholder video
        print(f"⚠️  WARNING: No video segments were generated!")
        print(f"   🔄 Creating emergency placeholder video to continue...")
        try:
            timestamp = int(time.time())
            output_folder = os.path.dirname(output_video_path) if os.path.dirname(output_video_path) else "video_output"
            emergency_video_path = os.path.join(output_folder, f"emergency_placeholder_all_{timestamp}.mp4")
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
                all_segment_paths = [emergency_video_path]
                video_path = emergency_video_path
                print(f"   ✅ Emergency placeholder video created: {emergency_video_path}")
            else:
                raise RuntimeError("FFmpeg not available - cannot create emergency placeholder")
        except Exception as emergency_error:
            print(f"   ❌ CRITICAL: Emergency placeholder creation failed: {emergency_error}")
            raise RuntimeError("No video segments were generated and emergency placeholder failed")
    elif len(all_segment_paths) > 1:
        print(f"Stitching {len(all_segment_paths)} video segments together...")
        
        # Create final stitched video path
        base, ext = os.path.splitext(output_video_path)
        stitched_video_path = f"{base}_stitched{ext}"
        
        try:
            video_path = stitch_videos(
                video_paths=all_segment_paths,
                output_path=stitched_video_path,
                upscale_to_1080p=upscale_to_1080p
            )
        except Exception as stitch_error:
            print(f"⚠️  WARNING: Video stitching failed: {stitch_error}")
            print(f"   🔄 Attempting to use first segment as fallback...")
            if all_segment_paths and os.path.exists(all_segment_paths[0]):
                video_path = all_segment_paths[0]
                print(f"   ✅ Using first segment as fallback: {video_path}")
            else:
                raise RuntimeError(f"Video stitching failed and no fallback available: {stitch_error}")
        
        # Verify stitched video exists and has content
        if not os.path.exists(video_path):
            print(f"⚠️  WARNING: Stitched video was not created: {video_path}")
            if all_segment_paths and os.path.exists(all_segment_paths[0]):
                video_path = all_segment_paths[0]
                print(f"   ✅ Using first segment as fallback: {video_path}")
            else:
                raise RuntimeError(f"Stitched video was not created and no fallback segments available: {video_path}")
        
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            print(f"⚠️  WARNING: Stitched video is empty: {video_path}")
            if all_segment_paths and os.path.exists(all_segment_paths[0]):
                video_path = all_segment_paths[0]
                print(f"   ✅ Using first segment as fallback: {video_path}")
            else:
                raise RuntimeError(f"Stitched video is empty and no fallback segments available: {video_path}")
        
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
                ensure_audio_on_video(video_path, ffmpeg_path=ffmpeg_path)
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
    character_reference_image_path,
    generated_segment_texts,
    generated_script,
    reference_image_info,
    api_key,
    model,
    resolution,
    poll_interval,
    max_wait_time,
    segments_to_regenerate=None
):
    """
    Generate video segments using Sora 2. Can generate all segments or regenerate specific ones.
    
    Args:
        segment_id_to_prompt: Mapping from segment_id to Sora prompt
        segment_assignments: List of segment assignment dicts
        num_segments: Total number of segments
        num_videos: Number of video segments
        output_folder: Folder where videos will be saved
        output_video_path: Base output video path
        character_reference_image_path: Path to character reference image (if any)
        generated_segment_texts: List of segment text strings
        generated_script: Full script text
        reference_image_info: Reference image info dict
        api_key: OpenAI API key
        model: Sora model to use
        resolution: Video resolution
        poll_interval: Polling interval for status checks
        max_wait_time: Maximum wait time for generation
        segments_to_regenerate: Optional list of segment IDs to regenerate. If None, generates all.
        
    Returns:
        List of dicts with segment_id, prompt, video_path
    """
    # Create mapping from segment_id to assignment
    assignment_map = {}
    if segment_assignments:
        for assignment in segment_assignments:
            seg_id = assignment.get('segment_id')
            assignment_map[seg_id] = assignment
    
    # Filter to only video segments (skip still image segments)
    video_segment_ids = []
    for seg_id in range(1, num_segments + 1):
        assignment = assignment_map.get(seg_id, {'type': 'video', 'needs_character_ref': False})
        if assignment.get('type') == 'video':
            video_segment_ids.append(seg_id)
    
    # If segments_to_regenerate is specified, filter to only those segments
    if segments_to_regenerate is not None:
        video_segment_ids = [seg_id for seg_id in video_segment_ids if seg_id in segments_to_regenerate]
        print(f"Regenerating {len(video_segment_ids)} specific video segment(s): {video_segment_ids}")
    else:
        print(f"Generating {len(video_segment_ids)} video segment(s) (out of {num_segments} total segments)")
    
    # Rate limiting: 4 requests per minute = 15 seconds between requests
    rate_limit_delay = 15  # seconds
    
    generated_video_segments = []
    video_jobs = []
    
    try:
        # Step 2a: Start all video generation jobs 15 seconds apart (non-blocking)
        print("Starting video generation jobs...")
        
        for video_idx, segment_id in enumerate(video_segment_ids, 1):
            # Get the prompt for this segment using the mapping
            if segment_id_to_prompt and segment_id in segment_id_to_prompt:
                segment_prompt = segment_id_to_prompt[segment_id]
            else:
                # Fallback: use a default prompt
                segment_prompt = "A cinematic scene"
            
            # Validate prompt is not empty
            if not segment_prompt or len(segment_prompt.strip()) == 0:
                segment_prompt = "A cinematic scene"
            
            # Get assignment for this segment to determine if character reference is needed
            assignment = assignment_map.get(segment_id, {'type': 'video', 'needs_character_ref': False})
            needs_character_ref = assignment.get('needs_character_ref', False)
            
            # Determine which reference image to use
            ref_image_to_use = None
            if needs_character_ref and character_reference_image_path and os.path.exists(character_reference_image_path):
                ref_image_to_use = character_reference_image_path
                print(f"Segment {segment_id}/{num_segments} (video {video_idx}/{len(video_segment_ids)}): Using character reference image")
            else:
                print(f"Segment {segment_id}/{num_segments} (video {video_idx}/{len(video_segment_ids)}): No reference image")
            
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
                print(f"   🔄 Will use fallback for segment {segment_id} after other segments complete...")
                # Don't raise - mark this segment for fallback handling later
                video_jobs.append({
                    'segment_id': segment_id,
                    'video_id': None,  # Mark as failed
                    'output_path': None,
                    'prompt': segment_prompt,
                    'is_still_image': False,
                    'start_failed': True,
                    'start_error': e
                })
            
            # Rate limiting: wait before starting next job (except for the last segment)
            if video_idx < len(video_segment_ids):
                time.sleep(rate_limit_delay)
        
        # Step 2b: Wait for all video generation jobs to complete with retry logic
        print(f"Waiting for {len(video_jobs)} video generation job(s) to complete...")
        
        for job in video_jobs:
            segment_id = job['segment_id']
            video_id = job['video_id']
            segment_output_path = job['output_path']
            segment_prompt = job['prompt']
            start_failed = job.get('start_failed', False)
            
            print(f"\n--- Processing Segment {segment_id} (Job {video_id if video_id else 'FAILED TO START'}) ---")
            
            # If job start failed, skip retry loop and go straight to fallback
            if start_failed:
                print(f"   ⚠️  Segment {segment_id} failed to start - using fallback immediately")
                segment_video_path = None
                last_error = job.get('start_error', Exception("Failed to start video generation job"))
            else:
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
                            # Get assignment for this segment to determine if character reference is needed for retry
                            assignment = assignment_map.get(segment_id, {'type': 'video', 'needs_character_ref': False})
                            needs_character_ref = assignment.get('needs_character_ref', False)
                            retry_ref_image = None
                            if needs_character_ref and character_reference_image_path and os.path.exists(character_reference_image_path):
                                retry_ref_image = character_reference_image_path
                            
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
                            break
                        elif attempt < max_retries:
                            print(f"   ⚠️  Attempt {attempt} failed ({error_type}): {error_msg[:200]}")
                            print(f"   Retrying in 5 seconds...")
                            time.sleep(5)
                        else:
                            print(f"   ❌ All {max_retries} attempts failed for segment {segment_id}")
                            print(f"   🔄 Will use still image fallback for segment {segment_id}...")
            
            # CRITICAL: If video generation failed (for any reason), ALWAYS use fallback
            if segment_video_path is None and segment_id not in [seg['segment_id'] for seg in generated_video_segments]:
                print(f"   🔄 Using still image fallback for segment {segment_id}...")
                
                try:
                    # Get segment text for still image generation
                    segment_text = ""
                    if generated_segment_texts and segment_id <= len(generated_segment_texts):
                        segment_text = generated_segment_texts[segment_id - 1]
                    elif generated_script:
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
                        reference_image_info=reference_image_info if reference_image_info else None
                    )
                    
                    # Generate DALL-E image with fallback handling
                    timestamp = int(time.time())
                    fallback_image_path = os.path.join(output_folder, f"fallback_still_segment_{segment_id}_{timestamp}.png")
                    
                    try:
                        sanitized_prompt = sanitize_image_prompt(fallback_image_prompt)
                        
                        fallback_image_path = generate_image_from_prompt(
                            prompt=sanitized_prompt,
                            output_path=fallback_image_path,
                            api_key=api_key,
                            model='dall-e-3',
                        )
                        
                        print(f"   ✅ Fallback still image generated: {fallback_image_path}")
                        
                    except Exception as image_gen_error:
                        print(f"   ⚠️  Fallback still image generation failed: {image_gen_error}")
                        print(f"   🔄 Using reference image as fallback...")
                        
                        # FALLBACK 1: Try to use character reference image
                        fallback_image_found = False
                        if character_reference_image_path and os.path.exists(character_reference_image_path):
                            fallback_image_path = character_reference_image_path
                            print(f"   ✅ Using character reference image as fallback: {character_reference_image_path}")
                            fallback_image_found = True
                        
                        # FALLBACK 2: Create a simple placeholder image
                        if not fallback_image_found:
                            print(f"   🔄 No reference images available - creating emergency placeholder...")
                            try:
                                ffmpeg_path = find_ffmpeg()
                                if ffmpeg_path:
                                    placeholder_path = os.path.join(output_folder, f"placeholder_fallback_segment_{segment_id}_{timestamp}.png")
                                    cmd_placeholder = [
                                        ffmpeg_path,
                                        "-f", "lavfi",
                                        "-i", "color=c=0x2a2a3e:s=1536x1024:d=1",
                                        "-frames:v", "1",
                                        "-y",
                                        placeholder_path
                                    ]
                                    subprocess.run(cmd_placeholder, capture_output=True, text=True, check=True)
                                    fallback_image_path = placeholder_path
                                    print(f"   ✅ Emergency placeholder image created: {placeholder_path}")
                                else:
                                    raise Exception("FFmpeg not available for placeholder creation")
                            except Exception as placeholder_error:
                                print(f"   ❌ CRITICAL: Even placeholder creation failed: {placeholder_error}")
                                raise
                    
                    # CRITICAL: Always create panning video, even if using fallback image
                    if fallback_image_path and os.path.exists(fallback_image_path):
                        try:
                            fallback_video_path = os.path.join(output_folder, f"fallback_panning_segment_{segment_id}_{timestamp}.mp4")
                            
                            import random
                            pan_directions = ['top_left_to_bottom_right', 'top_right_to_bottom_left', 
                                             'bottom_left_to_top_right', 'bottom_right_to_top_left']
                            pan_direction = random.choice(pan_directions)
                            
                            fallback_video_duration = 12
                            fallback_video_path = create_panning_video_from_image(
                                image_path=fallback_image_path,
                                output_path=fallback_video_path,
                                duration=fallback_video_duration,
                                pan_direction=pan_direction,
                                ffmpeg_path=find_ffmpeg()
                            )
                            
                            print(f"   ✅ Fallback panning video created: {fallback_video_path}")
                            
                            generated_video_segments.append({
                                'segment_id': segment_id,
                                'prompt': segment_prompt,
                                'video_path': fallback_video_path,
                                'is_fallback': True
                            })
                            
                            print(f"   ✅ Segment {segment_id} fallback complete: {fallback_video_path}")
                            
                        except Exception as panning_error:
                            print(f"   ⚠️  Panning video creation failed: {panning_error}")
                            raise
                    else:
                        raise Exception(f"Fallback image path invalid: {fallback_image_path}")
                    
                except Exception as fallback_error:
                    # CRITICAL: Even if fallback fails, we MUST continue - use a final emergency fallback
                    print(f"   ❌ CRITICAL: Fallback still image generation failed: {fallback_error}")
                    print(f"   Original error: {last_error}")
                    print(f"   🔄 Using emergency placeholder video for segment {segment_id}...")
                    
                    try:
                        timestamp = int(time.time())
                        emergency_video_path = os.path.join(output_folder, f"emergency_placeholder_segment_{segment_id}_{timestamp}.mp4")
                        
                        ffmpeg_path = find_ffmpeg()
                        if ffmpeg_path:
                            cmd_emergency = [
                                ffmpeg_path,
                                "-f", "lavfi",
                                "-i", "color=c=0x1a1a2e:s=1280x720:d=12",
                                "-c:v", "libx264",
                                "-pix_fmt", "yuv420p",
                                "-y",
                                emergency_video_path
                            ]
                            subprocess.run(cmd_emergency, capture_output=True, text=True, check=True)
                            
                            generated_video_segments.append({
                                'segment_id': segment_id,
                                'prompt': segment_prompt,
                                'video_path': emergency_video_path,
                                'is_fallback': True,
                                'is_emergency': True
                            })
                            
                            print(f"   ✅ Emergency placeholder video created: {emergency_video_path}")
                            print(f"   ⚠️  WARNING: Segment {segment_id} using emergency placeholder due to all fallbacks failing")
                        else:
                            print(f"   ❌ FFmpeg not available - cannot create emergency placeholder")
                            print(f"   ⚠️  WARNING: Segment {segment_id} will be skipped - continuing with remaining segments")
                    except Exception as emergency_error:
                        print(f"   ❌ CRITICAL: Even emergency fallback failed: {emergency_error}")
                        print(f"   ⚠️  WARNING: Segment {segment_id} cannot be generated - continuing with remaining segments")
    
    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - return what we have so far
    
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
    duration=8,
    aspect_ratio='16:9',
    poll_interval=10,
    max_wait_time=600,
    keep_video=False,
    upscale_to_1080p=True,
    test=False,
    skip_narration=False,
    skip_upload=False,
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
    segment_id_to_prompt = {}  # Mapping from segment_id to prompt (only for video segments)
    reference_image_info = None  # Initialize reference image info
    narration_audio_path = None  # Will be set in Step 0.1 (narration generation - MUST happen before video generation)
    original_voiceover_backup = None  # Will be set in Step 0.1 (narration generation)
    generated_video_segments = []  # Will be populated in Step 1
    still_image_videos = {}  # Will be populated in Step 1
    segment_assignments = []  # Will be populated in Step 0.5
    
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
        print("Step 0.1: Generating narration...")
        
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
            # IMPORTANT: narration_audio_path is the STRETCHED/SHRUNK version (if time-stretching was applied)
            # This ensures word-level timestamps match the final audio duration from video_config.json
            print(f"   📍 Using audio file: {narration_audio_path} (this is the stretched/shrunk version if time-stretching was applied)")
            generated_segment_texts = segment_script_by_narration(
                script=generated_script,
                audio_path=narration_audio_path,  # This is the stretched version that matches config duration
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
        
        # Step 0.55: Analyze script for main character reference image (0 or 1 only)
        print("\n" + "="*60)
        print("Step 0.55: Analyzing script for main character reference image...")
        
        reference_image_info = None
        if generated_script:
            reference_image_info = analyze_script_for_reference_image(
                script=generated_script,
                video_prompt=prompt,
                api_key=api_key,
                model='gpt-5-2025-08-07'
            )
            if reference_image_info:
                ref_type = reference_image_info.get('type', 'subject')
                ref_desc = reference_image_info.get('description', '')
                
                # Only generate reference image if it's a character (main character)
                if ref_type == 'character':
                    print(f"✅ Main character identified: {ref_desc}")
                    print(f"   Reference image will be generated in Step 1")
                else:
                    print(f"⚠️  No main character identified (type: {ref_type})")
                    print(f"   No reference image will be generated")
                    reference_image_info = None  # Don't generate if not a character
            else:
                print(f"⚠️  No main character identified for this video.")
                print(f"   This is normal if the video doesn't focus on a specific character")
                print(f"   All video segments will be generated without reference images.")
        
        # Step 0.65: Analyze script for still image opportunities and segment assignments (MUST be before Sora prompt generation)
        print("\n" + "="*60)
        print("Step 0.65: Analyzing script for still image opportunities and segment assignments...")
        print("="*60 + "\n")
        
        still_image_segments = []
        segment_assignments = []
        try:
            # Debug: Check if character reference image exists
            has_character_ref = reference_image_info is not None and reference_image_info.get('type') == 'character'
            if has_character_ref:
                print(f"   Main character reference image will be available for segments that need it")
            else:
                print(f"   ⚠️  No character reference image - all segments will be generated without reference")
            
            # Pass the calculated number of still images and character reference info
            # CRITICAL: generated_segment_texts contains NARRATION-BASED segments (words actually spoken)
            # These narration segments are used to generate both still image and video prompts
            analysis_result = analyze_script_for_still_images(
                script=generated_script,
                segment_texts=generated_segment_texts,  # NARRATION-BASED segments (from actual audio)
                target_num_stills=num_still_images,  # Pass calculated number (approximately 1/3 of segments)
                api_key=api_key,
                model='gpt-5-2025-08-07',  # Match Sora prompt model
                has_character_reference=has_character_ref  # Pass boolean indicating if character ref exists
            )
            
            still_image_segments = analysis_result.get('still_image_segments', [])
            segment_assignments = analysis_result.get('segment_assignments', [])
            
            # RULES-BASED: Ensure first three segments and last segment are always videos (not still images)
            num_segments_total = len(generated_segment_texts)
            first_three_segment_ids = [1, 2, 3]  # First three segments must be videos
            last_segment_id = num_segments_total
            
            # Remove first three segments from still_image_segments if present
            for seg_id in first_three_segment_ids:
                if seg_id <= num_segments_total:
                    still_image_segments = [seg for seg in still_image_segments if seg.get('segment_id') != seg_id]
            
            # Remove last segment from still_image_segments if present
            still_image_segments = [seg for seg in still_image_segments if seg.get('segment_id') != last_segment_id]
            
            # Update segment_assignments to ensure first three and last are videos
            for assignment in segment_assignments:
                seg_id = assignment.get('segment_id')
                if seg_id in first_three_segment_ids or seg_id == last_segment_id:
                    if assignment.get('type') == 'still':
                        if seg_id in first_three_segment_ids:
                            print(f"   ⚠️  Rules-based override: Segment {seg_id} changed from 'still' to 'video' (first three segments must be video)")
                        else:
                            print(f"   ⚠️  Rules-based override: Segment {seg_id} changed from 'still' to 'video' (last segment must be video)")
                        assignment['type'] = 'video'
            
            if still_image_segments:
                print(f"✅ Identified {len(still_image_segments)} still image position(s) (after rules-based filtering)")
                for seg_info in still_image_segments:
                    seg_id = seg_info.get('segment_id', 'unknown')
                    print(f"   - Still image at segment {seg_id} (12s)")
            else:
                print(f"✅ No still images after rules-based filtering (first three and last segments must be videos)")
            
            if segment_assignments:
                print(f"✅ Segment assignments:")
                for assignment in segment_assignments:
                    seg_id = assignment.get('segment_id', 'unknown')
                    seg_type = assignment.get('type', 'video')
                    needs_ref = assignment.get('needs_character_ref', False)
                    ref_str = " (needs character ref)" if needs_ref else " (no ref)"
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
        
        # CRITICAL: Only generate prompts for VIDEO segments, not still image segments
        # This ensures we don't waste API calls generating prompts for segments that will be still images
        # Create a list of video segment texts and their corresponding segment IDs
        video_segment_texts = []
        video_segment_id_map = {}  # Maps index in video_segment_texts to actual segment_id
        
        # Create mapping from segment_id to assignment for quick lookup
        assignment_map_for_prompts = {}
        if segment_assignments:
            for assgn in segment_assignments:
                seg_id = assgn.get('segment_id')
                if seg_id:
                    assignment_map_for_prompts[seg_id] = assgn
        
        for seg_id in range(1, len(generated_segment_texts) + 1):
            # Check if this segment is a video segment (not a still image)
            assignment = assignment_map_for_prompts.get(seg_id)
            
            # Default to 'video' if no assignment found
            seg_type = assignment.get('type', 'video') if assignment else 'video'
            
            if seg_type == 'video':
                video_segment_texts.append(generated_segment_texts[seg_id - 1])
                video_segment_id_map[len(video_segment_texts) - 1] = seg_id
        
        print(f"   Generating prompts for {len(video_segment_texts)} video segment(s) (skipping {len(generated_segment_texts) - len(video_segment_texts)} still image segment(s))")
        
        # Note: We no longer pass reference_image_info here since we handle multiple reference images
        # per segment at the video generation level based on segment_assignments
        # CRITICAL: video_segment_texts contains only VIDEO segments (still images are filtered out)
        # These narration segments are used to generate Sora video prompts, ensuring videos match what's being narrated
        generated_video_prompts = generate_sora_prompts_from_segments(
            segment_texts=video_segment_texts,  # Only VIDEO segments (still images filtered out)
            segment_duration=segment_duration,
            total_duration=duration,
            overarching_script=generated_script,  # Pass full script for context and chronological flow
            reference_image_info=None,  # No longer using single reference image - handled per segment
            still_image_segments=still_image_segments,  # Pass still image positions for correct timing
            api_key=api_key,
            model='gpt-5-2025-08-07'
        )
        
        # Create a mapping from actual segment_id to prompt
        # This ensures we can look up the correct prompt for each video segment
        segment_id_to_prompt = {}
        for video_idx, segment_id in video_segment_id_map.items():
            if video_idx < len(generated_video_prompts):
                segment_id_to_prompt[segment_id] = generated_video_prompts[video_idx]
        
        print(f"\nSora Prompts ({len(generated_video_prompts)} video segments):")
        print("-" * 60)
        for video_idx, (segment_id, prompt) in enumerate(sorted(segment_id_to_prompt.items()), 1):
            print(f"\nVideo Segment {video_idx} (Segment ID {segment_id}): {prompt[:100]}...")
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
    
    character_reference_image_path = None  # Single character reference image path (or None)
    
    # CRITICAL: If API call fails before Sora 2 video generation, exit the program
    try:
        # Only generate reference image if main character was identified
        if reference_image_info and reference_image_info.get('type') == 'character':
            ref_description = reference_image_info.get('description', '')
            image_prompt = reference_image_info.get('image_prompt', '')
            
            print(f"Generating main character reference image: {ref_description}")
            timestamp = int(time.time())
            
            # Generate image path
            character_reference_image_path = os.path.join(output_folder, f"character_reference_image_{timestamp}.png")
            
            # Generate the image
            if image_prompt:
                print(f"    Using AI-generated prompt...")
                character_reference_image_path = generate_master_image_from_prompt(
                    image_prompt=image_prompt,
                    output_path=character_reference_image_path,
                    api_key=api_key,
                    resolution=resolution
                )
            else:
                print(f"    Using description-based generation...")
                character_reference_image_path = generate_master_image_from_prompt(
                    description=ref_description,
                    output_path=character_reference_image_path,
                    api_key=api_key,
                    resolution=resolution
                )
            
            print(f"    ✅ Generated: {character_reference_image_path}")
        else:
            print("No main character identified - skipping reference image generation")
        
        if character_reference_image_path:
            print(f"\n✅ Character reference image generated: {character_reference_image_path}")
        else:
            print(f"\n✅ No character reference image (video doesn't focus on a main character)")
            
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
        
        # CRITICAL: Each still image must have a fallback - execution must NEVER stop
        for seg_info in still_image_segments:
            segment_id = seg_info['segment_id']
            image_prompt = seg_info['image_prompt']
            still_duration = seg_info['duration']
            still_image_path = None
            fallback_used = False
            
            if segment_id == 0:
                print(f"Generating opening still image (test mode)...")
            else:
                print(f"Generating still image after video {segment_id}...")
            print(f"   Prompt: {image_prompt[:150]}...")
            print(f"   Duration: {still_duration:.1f}s")
            
            # Try to generate DALL-E image
            timestamp = int(time.time())
            still_image_path = os.path.join(output_folder, f"still_image_segment_{segment_id}_{timestamp}.png")
            
            try:
                # Sanitize prompt for content policy
                sanitized_prompt = sanitize_image_prompt(image_prompt)
                
                still_image_path = generate_image_from_prompt(
                    prompt=sanitized_prompt,
                    output_path=still_image_path,
                    api_key=api_key,
                    model='dall-e-3',
                )
                
                print(f"✅ Still image generated: {still_image_path}")
                
            except Exception as image_error:
                print(f"   ⚠️  Still image generation failed: {image_error}")
                print(f"   🔄 Using fallback reference image for segment {segment_id}...")
                fallback_used = True
                
                # FALLBACK 1: Try to use character reference image
                fallback_image_found = False
                if character_reference_image_path and os.path.exists(character_reference_image_path):
                    still_image_path = character_reference_image_path
                    print(f"   ✅ Using character reference image as fallback: {character_reference_image_path}")
                    fallback_image_found = True
                
                # FALLBACK 3: Create a simple placeholder image
                if not fallback_image_found:
                    print(f"   🔄 No reference images available - creating emergency placeholder...")
                    try:
                        # Create a simple colored image using FFmpeg
                        ffmpeg_path = find_ffmpeg()
                        if ffmpeg_path:
                            placeholder_path = os.path.join(output_folder, f"placeholder_still_segment_{segment_id}_{timestamp}.png")
                            cmd_placeholder = [
                                ffmpeg_path,
                                "-f", "lavfi",
                                "-i", "color=c=0x2a2a3e:s=1536x1024:d=1",
                                "-frames:v", "1",
                                "-y",
                                placeholder_path
                            ]
                            subprocess.run(cmd_placeholder, capture_output=True, text=True, check=True)
                            still_image_path = placeholder_path
                            print(f"   ✅ Emergency placeholder image created: {placeholder_path}")
                        else:
                            raise Exception("FFmpeg not available for placeholder creation")
                    except Exception as placeholder_error:
                        print(f"   ❌ CRITICAL: Even placeholder creation failed: {placeholder_error}")
                        print(f"   ⚠️  WARNING: Cannot create still image for segment {segment_id} - will skip this still image")
                        # Skip this still image but continue with others
                        continue
            
            # CRITICAL: Always create panning video, even if using fallback image
            if still_image_path and os.path.exists(still_image_path):
                try:
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
                    if fallback_used:
                        print(f"   ✅ Fallback panning video created: {panning_video_path}")
                    else:
                        print(f"✅ Panning video created: {panning_video_path}")
                        
                except Exception as panning_error:
                    print(f"   ⚠️  Panning video creation failed: {panning_error}")
                    print(f"   🔄 Creating emergency static video placeholder...")
                    try:
                        # Last resort: create a static video (no panning)
                        ffmpeg_path = find_ffmpeg()
                        if ffmpeg_path:
                            emergency_video_path = os.path.join(output_folder, f"emergency_static_segment_{segment_id}_{timestamp}.mp4")
                            cmd_emergency = [
                                ffmpeg_path,
                                "-loop", "1",
                                "-i", still_image_path,
                                "-t", "12",
                                "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
                                "-c:v", "libx264",
                                "-pix_fmt", "yuv420p",
                                "-y",
                                emergency_video_path
                            ]
                            subprocess.run(cmd_emergency, capture_output=True, text=True, check=True)
                            still_image_videos[segment_id] = emergency_video_path
                            print(f"   ✅ Emergency static video created: {emergency_video_path}")
                        else:
                            print(f"   ❌ FFmpeg not available - cannot create emergency video")
                            print(f"   ⚠️  WARNING: Skipping still image segment {segment_id}")
                    except Exception as emergency_error:
                        print(f"   ❌ CRITICAL: Even emergency video creation failed: {emergency_error}")
                        print(f"   ⚠️  WARNING: Skipping still image segment {segment_id} - continuing with remaining segments")
            else:
                print(f"   ⚠️  WARNING: Still image path invalid or missing: {still_image_path}")
                print(f"   ⚠️  WARNING: Skipping still image segment {segment_id} - continuing with remaining segments")
    
    # Step 2: Generate multiple videos with rate limiting
    print(f"Step 2: Generating {num_videos} video segment(s) using Sora 2...")
    
    # Each segment is 12 seconds
    video_segment_duration = 12
    
    # Create set of still image positions (after which video segment)
    still_image_segment_ids = set(seg['segment_id'] for seg in still_image_segments) if still_image_segments else set()
    
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
            character_reference_image_path=character_reference_image_path,
            generated_segment_texts=generated_segment_texts,
            generated_script=generated_script,
            reference_image_info=reference_image_info,
            api_key=api_key,
            model=model,
            resolution=resolution,
            poll_interval=poll_interval,
            max_wait_time=max_wait_time,
            segments_to_regenerate=None  # Generate all
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
            character_reference_image_path=character_reference_image_path,
            output_video_path=output_video_path
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
            still_image_videos = metadata.get('still_image_videos', {})
            # Filter out still image videos that don't exist
            still_image_videos = {k: v for k, v in still_image_videos.items() if os.path.exists(v)}
            
            # Use metadata values if available, otherwise keep values from Step 0
            segment_assignments = metadata.get('segment_assignments', segment_assignments)
            generated_segment_texts = metadata.get('generated_segment_texts', generated_segment_texts)
            generated_script = metadata.get('generated_script', generated_script)
            num_segments = metadata.get('num_segments', num_segments)
            num_videos = metadata.get('num_videos', num_videos)
            num_still_images = metadata.get('num_still_images', num_still_images)
            character_reference_image_path = metadata.get('character_reference_image_path', character_reference_image_path)
            segment_id_to_prompt = metadata.get('segment_id_to_prompt', segment_id_to_prompt)
            
            print(f"   [OK] Loaded {len(generated_video_segments)} video segment(s) and {len(still_image_videos)} still image(s) from metadata")
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
                character_reference_image_path=character_reference_image_path,
                generated_segment_texts=generated_segment_texts,
                generated_script=generated_script,
                reference_image_info=reference_image_info,
                api_key=api_key,
                model=model,
                resolution=resolution,
                poll_interval=poll_interval,
                max_wait_time=max_wait_time,
                segments_to_regenerate=None  # Generate all
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
                character_reference_image_path=character_reference_image_path,
                output_video_path=output_video_path
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
        print(f"  Still image segments: {len(still_image_videos)}")
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
        
        if still_image_videos:
            print(f"\nStill image segment details:")
            for seg_id, video_path in sorted(still_image_videos.items()):
                exists = os.path.exists(video_path) if video_path else False
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
                    upscale_to_1080p=upscale_to_1080p
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
                    character_reference_image_path=character_reference_image_path,
                    generated_segment_texts=generated_segment_texts,
                    generated_script=generated_script,
                    reference_image_info=reference_image_info,
                    api_key=api_key,
                    model=model,
                    resolution=resolution,
                    poll_interval=poll_interval,
                    max_wait_time=max_wait_time,
                    segments_to_regenerate=segments_to_regenerate
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
                    character_reference_image_path=character_reference_image_path,
                    output_video_path=output_video_path
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
                            upscale_to_1080p=upscale_to_1080p
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
                upscale_to_1080p=upscale_to_1080p
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
        
        # Add audio to stitched video (mix narration + music at 7%)
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
                # 3. Re-mix with voiceover at 7% music volume
                
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
                                        # Mix: voiceover + music (7% volume)
                                        synced_audio_path = os.path.join(temp_dir, f"audio_resynced_{timestamp}.mp3")
                                        
                                        # Mix music and narration together at 7% music volume
                                        filter_complex = (
                                            f"[0:a]aresample=44100,volume=1.0[voice];"
                                            f"[1:a]aresample=44100,volume={0.07}[music];"  # 7% volume for background music
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
                                        print(f"   ✅ Audio mixed: narration + music (7% volume) synced to video ({video_duration:.2f}s)")
                                    
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
                        
                        # Add audio to video
                        video_path = add_audio_to_video(
                            video_path=video_path,
                            audio_path=voiceover_audio_path,
                            output_path=video_with_audio_path,
                            ffmpeg_path=ffmpeg_path,
                            sync_duration=False  # Already synced above
                        )
                        print(f"✅ Video with mixed audio (narration + 7% music): {video_path}")
                        
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
    
        # Character reference image and final video are saved in the output folder
        if character_reference_image_path and os.path.exists(character_reference_image_path):
            print(f"✅ Character reference image saved: {character_reference_image_path}")
            print(f"   (This image is used as reference for segments featuring the main character)")
        
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
        
        # Ask for test mode
        try:
            test_input = input("Test mode? Add opening still image? (y/n, default: n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            test_input = ""
        test_mode = test_input in ['y', 'yes']
        
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
            print(f"✅ Script generation complete")
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
        
        try:
            narration_file = generate_and_save_narration(
                script_file_path=SCRIPT_FILE_PATH,
                narration_audio_path=NARRATION_AUDIO_PATH,
                duration=None,  # Duration not needed for narration-only generation
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
            
            # Ask for test mode
            try:
                test_input = input("Test mode? Add opening still image? (y/n, default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                test_input = ""
            test_mode = test_input in ['y', 'yes']
            
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
            try:
                narration_file = generate_and_save_narration(
                    script_file_path=SCRIPT_FILE_PATH,
                    narration_audio_path=NARRATION_AUDIO_PATH,
                    duration=None,
                    api_key=args.api_key,
                )
                print(f"✅ Step 2 complete: Narration saved")
                return 0
            except Exception as e:
                print(f"\n❌ Error generating narration: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Step 3: Generate video (Sora video generation)
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
            duration = args.duration if args.duration else config.get('duration', 8)
            
            if not prompt or not title:
                print("⚠️  Missing required configuration. Please generate script first.")
                return 1
            tags = config.get('tags')
            privacy_status = args.privacy if args.privacy != 'private' else config.get('privacy_status', 'private')
            category_id = args.category if args.category != '27' else config.get('category_id', '27')
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
                still_image_videos = metadata.get('still_image_videos', {})
                segment_assignments = metadata.get('segment_assignments', [])
                num_segments = metadata.get('num_segments', 0)
                segment_id_to_prompt = metadata.get('segment_id_to_prompt', {})
                generated_segment_texts = metadata.get('generated_segment_texts', [])
                generated_script = metadata.get('generated_script', '')
                num_videos = metadata.get('num_videos', 0)
                num_still_images = metadata.get('num_still_images', 0)
                character_reference_image_path = metadata.get('character_reference_image_path', None)
                output_video_path = metadata.get('output_video_path', os.path.join(output_folder, 'sora_video.mp4'))
                duration = config.get('duration', 60)
                
                # Determine upscale_to_1080p from command-line argument
                upscale_to_1080p = not args.no_upscale
                
                # Show available segments
                print(f"\nAvailable segments:")
                print(f"  Video segments: {len(generated_video_segments)}")
                print(f"  Still image segments: {len(still_image_videos)}")
                print(f"  Total segments: {num_segments}")
                
                if generated_video_segments:
                    print(f"\nVideo segment details:")
                    for seg in sorted(generated_video_segments, key=lambda x: x.get('segment_id', 0)):
                        seg_id = seg.get('segment_id', '?')
                        video_path = seg.get('video_path', 'unknown')
                        exists = os.path.exists(video_path) if video_path != 'unknown' else False
                        status = "[EXISTS]" if exists else "[MISSING]"
                        print(f"  Segment {seg_id}: {os.path.basename(video_path)} {status}")
                
                if still_image_videos:
                    print(f"\nStill image segment details:")
                    for seg_id, video_path in sorted(still_image_videos.items()):
                        exists = os.path.exists(video_path) if video_path else False
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
                        upscale_to_1080p=upscale_to_1080p
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
                        # Get reference_image_info if character reference exists
                        reference_image_info = None
                        if character_reference_image_path and os.path.exists(character_reference_image_path):
                            reference_image_info = {
                                'type': 'character',
                                'image_path': character_reference_image_path
                            }
                        
                        # Regenerate specific segments
                        print(f"\nRegenerating segments: {segments_to_regenerate}")
                        regenerated_segments = generate_video_segments(
                            segment_id_to_prompt=segment_id_to_prompt,
                            segment_assignments=segment_assignments,
                            num_segments=num_segments,
                            num_videos=num_videos,
                            output_folder=output_folder,
                            output_video_path=output_video_path,
                            character_reference_image_path=character_reference_image_path,
                            generated_segment_texts=generated_segment_texts,
                            generated_script=generated_script,
                            reference_image_info=reference_image_info,
                            api_key=args.api_key,
                            model=config.get('model', 'sora-2'),
                            resolution=config.get('resolution', '1280x720'),
                            poll_interval=args.poll_interval,
                            max_wait_time=args.max_wait,
                            segments_to_regenerate=segments_to_regenerate
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
                            character_reference_image_path=character_reference_image_path,
                            output_video_path=output_video_path
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
                                upscale_to_1080p=upscale_to_1080p
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
        #         from add_music_to_video import add_narration_music_and_captions_to_video
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
                from upload_video import upload_video
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

