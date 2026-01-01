"""
Standalone script to add narration, music, and captions to a Sora-generated video.
Uses the same exact implementation as generate_and_upload_sora.py.

Usage:
    python add_music_to_video.py [video_path]
    
If no video_path is provided, it will look for the latest video in video_output directory.
"""

import os
import sys
import subprocess
import shutil
import glob
import time
import argparse
import tempfile

# Import functions from the main script
try:
    from generate_and_upload_sora import (
        find_ffmpeg,
        get_media_duration,
        load_script_from_file,
        load_narration_from_file,
        add_audio_to_video,
        add_subtitles_to_video,
        SCRIPT_FILE_PATH,
        NARRATION_AUDIO_PATH,
        OPENAI_API_KEY
    )
except ImportError as e:
    print(f"Error importing from generate_and_upload_sora.py: {e}")
    print("Make sure generate_and_upload_sora.py is in the same directory")
    sys.exit(1)


def find_latest_video(video_output_dir="video_output"):
    """
    Find the latest video file in the video_output directory.
    
    Args:
        video_output_dir: Directory to search for videos
        
    Returns:
        Path to the latest video file, or None if not found
    """
    if not os.path.exists(video_output_dir):
        return None
    
    # Look for video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        pattern = os.path.join(video_output_dir, f"*{ext}")
        video_files.extend(glob.glob(pattern))
    
    if not video_files:
        return None
    
    # Sort by modification time, most recent first
    video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return video_files[0]


def add_narration_music_and_captions_to_video(video_path, output_path=None, api_key=None, ffmpeg_path=None):
    """
    Add narration, music, and captions to a video using the same implementation
    as generate_and_upload_sora.py.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the output video (default: video_path with _final suffix)
        api_key: OpenAI API key (optional, uses default if not provided)
        ffmpeg_path: Path to ffmpeg executable (optional, auto-detects if not provided)
        
    Returns:
        Path to the output video file
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    
    if not ffmpeg_path:
        raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
    
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")
    
    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_final{ext}"
    
    print("="*60)
    print("🎵 Adding Narration, Music, and Captions to Video")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    print("="*60)
    
    # Get video duration
    video_duration = get_media_duration(video_path, ffmpeg_path)
    if video_duration is None:
        raise Exception("Could not determine video duration")
    
    print(f"\n📹 Video duration: {video_duration:.2f}s")
    
    # Step 1: Load script
    print("\n📝 Step 1: Loading script...")
    generated_script = load_script_from_file()
    if not generated_script:
        print("⚠️  No script file found. Captions will not be added.")
        print(f"   Expected script file: {SCRIPT_FILE_PATH}")
    else:
        print(f"✅ Script loaded: {len(generated_script)} characters")
    
    # Step 2: Load narration audio
    print("\n🎙️  Step 2: Loading narration audio...")
    voiceover_audio_path = load_narration_from_file()
    original_voiceover_backup = None
    
    if voiceover_audio_path:
        print(f"✅ Narration loaded: {voiceover_audio_path}")
        # Try to find the original voiceover backup (without music) if it exists
        backup_path = voiceover_audio_path.replace('.mp3', '_original.mp3')
        if os.path.exists(backup_path):
            original_voiceover_backup = backup_path
            print(f"✅ Found original voiceover backup (for captions): {backup_path}")
    else:
        print(f"⚠️  No narration audio file found.")
        print(f"   Expected narration file: {NARRATION_AUDIO_PATH}")
        print(f"   Audio will not be added to video.")
    
    # Step 3: Synchronize and add audio to video (same logic as main script)
    if voiceover_audio_path and os.path.exists(voiceover_audio_path):
        print("\n🔊 Step 3: Synchronizing and adding audio to video...")
        
        video_duration = get_media_duration(video_path, ffmpeg_path)
        
        if video_duration:
            # Re-mix audio with proper synchronization (same as main script)
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
                                original_pattern = os.path.join(temp_dir, "original_voiceover_*.mp3")
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
                            
                            # Adjust voiceover to fit within video bounds
                            voiceover_duration = get_media_duration(voiceover_source, ffmpeg_path)
                            if voiceover_duration:
                                max_voiceover_duration = video_duration
                                min_voiceover_duration = max(1.0, video_duration - 5.0)
                                
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
                                        # Voiceover is shorter - just copy it
                                        import shutil
                                        shutil.copy2(voiceover_source, adjusted_voiceover)
                                    
                                    try:
                                        if voiceover_duration > target_voiceover_duration:
                                            subprocess.run(cmd_voiceover, capture_output=True, text=True, check=True)
                                        voiceover_source = adjusted_voiceover
                                    except Exception as e:
                                        print(f"   ⚠️  Voiceover adjustment failed: {e}")
                                        voiceover_source = voiceover_source
                                
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
                                    # Now mix: voiceover (within video bounds) + music (synced to video)
                                    synced_audio_path = os.path.join(temp_dir, f"audio_resynced_{timestamp}.mp3")
                                    
                                    # Calculate voiceover start delay
                                    voiceover_start_delay = 0.0
                                    if voiceover_duration < video_duration:
                                        voiceover_start_delay = min(voiceover_tolerance, (video_duration - target_voiceover_duration) / 2)
                                    
                                    # Mix: music starts at 0s, voiceover starts at voiceover_start_delay
                                    filter_complex = (
                                        f"[0:a]aresample=44100,volume=1.0,adelay={int(voiceover_start_delay * 1000)}|{int(voiceover_start_delay * 1000)}[voice_delayed];"
                                        f"[1:a]aresample=44100,volume={0.15}[music];"
                                        f"[voice_delayed][music]amix=inputs=2:duration=longest:dropout_transition=2,"
                                        f"volume=2.0"
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
                                    print(f"   ✅ Audio re-mixed: music synced to video ({video_duration:.2f}s)")
                            
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
        
        # Add audio to video using the same function as main script
        try:
            base, ext = os.path.splitext(video_path)
            video_with_audio_path = f"{base}_with_audio{ext}"
            
            # Use sync_duration=False since we've already synced manually
            video_path = add_audio_to_video(
                video_path=video_path,
                audio_path=voiceover_audio_path,
                output_path=video_with_audio_path,
                ffmpeg_path=ffmpeg_path,
                sync_duration=False  # Already synced above
            )
            print(f"✅ Video with audio: {video_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to add audio to video: {e}")
            print("   Continuing without audio...")
    
    # Step 4: Add subtitles/captions to video
    if generated_script and video_duration:
        print("\n📝 Step 4: Adding subtitles/captions to video...")
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
                    except Exception as e:
                        print(f"  Warning: Could not remove {video_without_subtitles}: {e}")
            else:
                print("⚠️  Subtitle generation returned no output, keeping video without subtitles")
        except Exception as e:
            print(f"⚠️  Failed to add subtitles to video: {e}")
            print("   Continuing with video without subtitles...")
    
    # Move final video to output path if different
    if video_path != output_path:
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(video_path, output_path)
            video_path = output_path
            print(f"✅ Final video saved to: {output_path}")
        except Exception as e:
            print(f"⚠️  Could not move video to output path: {e}")
            print(f"   Video is at: {video_path}")
    
    print(f"\n✅ Complete! Final video: {video_path}")
    
    # Show file sizes
    if os.path.exists(video_path):
        output_size = os.path.getsize(video_path) / (1024*1024)
        print(f"   Final size: {output_size:.2f} MB")
    
    return video_path


def main():
    parser = argparse.ArgumentParser(
        description='Add narration, music, and captions to a Sora-generated video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_music_to_video.py
    (Finds latest video in video_output directory)
  
  python add_music_to_video.py video.mp4
    (Uses specified video file)
  
  python add_music_to_video.py video.mp4 --output final_video.mp4
    (Specifies custom output path)
        """
    )
    
    parser.add_argument(
        'video_path',
        nargs='?',
        help='Path to the video file (if not provided, finds latest in video_output directory)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output video path (default: input_video_final.mp4)'
    )
    
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (optional, uses default if not provided)'
    )
    
    args = parser.parse_args()
    
    # Find video file
    if args.video_path:
        video_path = args.video_path
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            sys.exit(1)
    else:
        print("🔍 Looking for latest video in video_output directory...")
        video_path = find_latest_video()
        if not video_path:
            print("❌ Error: No video file found in video_output directory")
            print("   Please provide a video path as an argument")
            sys.exit(1)
        print(f"✅ Found: {video_path}")
    
    try:
        # Add narration, music, and captions to video
        output_path = add_narration_music_and_captions_to_video(
            video_path=video_path,
            output_path=args.output,
            api_key=args.api_key
        )
        
        print("\n" + "="*60)
        print("🎉 Success!")
        print("="*60)
        print(f"Output video: {output_path}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
