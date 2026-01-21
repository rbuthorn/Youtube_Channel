# Complete Workflow Documentation
## OpenAI Sora 2 Video Generation and YouTube Upload Pipeline

This document provides a detailed breakdown of every step and substep in the video generation and upload workflow.

---

## Overview

The workflow generates a documentary-style YouTube video from a text prompt using:
- **OpenAI GPT** for script generation and prompt creation
- **OpenAI Sora 2** for video generation
- **OpenAI DALL-E** for reference and still images
- **OpenAI Whisper** for audio transcription and segmentation
- **FFmpeg** for video/audio processing
- **YouTube API** for video upload

**Total Duration**: Must be divisible by 12 seconds (each segment is 12 seconds)
**Segment Distribution**: ~2/3 video segments, ~1/3 still image segments with panning

---

## Pre-Workflow Setup

### Initialization
1. **Archive Previous Run**
   - Check if `video_output/` folder exists
   - Archive previous workflow files if present
   - Delete old output folder

2. **Create Fresh Output Folder**
   - Create new `video_output/` directory
   - Set output video path (timestamped filename)

3. **Validate Duration**
   - Ensure duration is divisible by 12
   - Calculate total segments: `num_segments = duration / 12`
   - Calculate video segments: `num_videos = num_segments * 2/3` (rounded)
   - Calculate still images: `num_still_images = num_segments - num_videos`

---

## STEP 0: Script Generation and Preparation

### Step 0.0: Load or Generate Script
**Purpose**: Create the overarching documentary script

**Substeps**:
1. Try to load script from `overarching_script.txt`
2. If not found, call `generate_script_from_prompt()`:
   - **AI Model**: `gpt-5-2025-08-07`
   - **System Prompt**: "Expert documentary scriptwriter. Write informative, historically accurate scripts..."
   - **User Prompt**: Creates a documentary-style script with:
     - Target character count: `duration_minutes * 750 characters`
     - Complete story structure (hook, introduction, narrative, climax, conclusion, impact)
     - Extensive context and background (assumes viewer knows nothing)
     - `[MUSICAL BREAK]` or `[VISUAL BREAK]` markers every 2000 characters
   - **Output**: Raw script text (no labels, just narration and break markers)

### Step 0.1: Generate Narration Audio
**Purpose**: Create the final narration audio BEFORE video generation

**Substeps**:
1. Try to load existing narration from `narration_audio.mp3`
2. If not found and `skip_narration=False`:
   - Call `generate_voiceover_from_folder()`:
     - Convert script to speech using text-to-speech API
     - Split into segments at `[MUSICAL BREAK]` and `[VISUAL BREAK]` markers
     - Add 1-second breaks between segments
     - Mix with background music at 7% volume
     - Save as `narration_audio.mp3`
     - Save original voiceover (without music) as backup
3. If `skip_narration=True`:
   - Generate temporary narration for segmentation only
   - Will be regenerated later in Step 3

**Critical**: Narration MUST be generated before video generation to ensure proper segmentation.

### Step 0.5: Segment Script
**Purpose**: Divide script into 12-second segments based on actual narration timing

**Substeps**:
1. **If narration audio exists**:
   - Call `segment_script_by_narration()`:
     - Use OpenAI Whisper API to transcribe audio with word-level timestamps
     - Divide into 12-second segments based on actual spoken words
     - Extract text for each segment from transcription
     - **Result**: Narration-based segments (words actually spoken)
2. **If narration audio missing**:
   - Fallback to `segment_script_rule_based()`:
     - Divide script by word count evenly
     - Less accurate but works without audio

**Output**: List of segment texts (one per 12-second segment)

### Step 0.52: Generate YouTube Tags
**Purpose**: Create SEO-optimized tags from script

**Substeps**:
1. Call `generate_tags_from_script()`:
   - **AI Model**: `gpt-4o`
   - **System Prompt**: "You are a YouTube SEO expert..."
   - **User Prompt**: Analyzes script and generates exactly 5 tags (1-3 words each)
2. Combine with user-provided tags
3. Remove duplicates (case-insensitive)

**Output**: Final list of unique tags

### Step 0.55: Analyze Script for Reference Images
**Purpose**: Determine what reference images are needed for visual consistency

**Substeps**:
1. Call `analyze_script_for_reference_images()`:
   - **AI Model**: `gpt-5-2025-08-07`
   - **System Prompt**: "Video production assistant. Analyze scripts to determine what set of reference images..."
   - **User Prompt**: Analyzes script to identify:
     - Characters (specific people/entities that appear throughout)
     - Subjects (locations, objects, visual elements)
   - **Output**: JSON array with:
     - `id`: Reference image ID (e.g., "ref_1")
     - `type`: "character" or "subject"
     - `description`: What the reference represents
     - `image_prompt`: Detailed DALL-E prompt for the image
     - `reasoning`: Why this reference is needed

**Output**: List of reference image definitions (0-4 images typically)

### Step 0.65: Analyze Script for Still Images
**Purpose**: Identify optimal positions for still images with camera panning

**Substeps**:
1. Call `analyze_script_for_still_images()`:
   - **AI Model**: `gpt-5-2025-08-07`
   - **System Prompt**: "Video production assistant. Analyze scripts to identify optimal moments for still images..."
   - **User Prompt**: Analyzes narration-based segments to:
     - Identify contemplative, descriptive, or transitional moments
     - Avoid action scenes, fights, battles, or fast-paced moments
     - Assign reference images to segments (if applicable)
   - **Output**: JSON with:
     - `still_image_segments`: List of segment IDs where still images should appear
     - `segment_assignments`: For each segment, indicates:
       - `segment_id`: Segment number
       - `type`: "video" or "still"
       - `reference_image_id`: Which reference image to use (if any)

**Output**: 
- List of still image positions
- Segment assignment map (video vs still, reference image assignments)

### Step 0.6: Convert Segments to Sora Prompts
**Purpose**: Generate detailed Sora 2 video prompts for each video segment

**Substeps**:
1. Filter segments to only video segments (skip still image segments)
2. For each video segment, call `convert_segment_to_sora_prompt()`:
   - **AI Model**: `gpt-5-2025-08-07`
   - **System Prompt**: "Professional Sora 2 Video Prompter. Prompt Sora 2 to create detailed cinematic prompts..."
   - **User Prompt**: Converts segment text to Sora prompt with:
     - Full script context for narrative flow
     - Previous segment prompt (for visual continuity)
     - Next segment text (for forward context)
     - Reference image info (if assigned)
     - Still image positions (for correct timing calculations)
     - Key phrase extraction (identifies main visual focus)
     - Shot structure (1 or 2 shots, with duration requirements)
     - **Critical Requirements**:
       - Must be PHOTOREALISTIC (looks like real documentary footage)
       - Must match script narration exactly
       - No quick cuts (minimum shot duration enforced)
       - Character matching (if reference image provided, must be identical)
   - **Retry Logic**: Up to 3 attempts if generation fails

**Output**: List of Sora 2 prompts (one per video segment)

### Step 0.7: Verify Narration Ready
**Purpose**: Confirm narration is available for video processing

**Substeps**:
1. Check if narration from Step 0.1 exists
2. If missing, try to load from file
3. If still missing and `skip_narration=False`, exit with error
4. If `skip_narration=True`, note that narration will be regenerated later

---

## STEP 1: Reference Image Generation

### Step 1.0: Generate Reference Images
**Purpose**: Create reference images for visual consistency across video segments

**Substeps**:
1. **If reference images were identified in Step 0.55**:
   - For each reference image:
     - Extract `image_prompt` or `description`
     - Call `generate_master_image_from_prompt()`:
       - **API**: OpenAI DALL-E 3
       - **Resolution**: Matches video resolution (default 1280x720)
       - **Prompt**: Uses AI-generated prompt or description-based template
     - Save as `reference_image_{ref_id}_{timestamp}.png`
     - Store in `reference_image_paths` dictionary
2. **If no reference images identified**:
   - Generate single fallback reference image from video prompt/description
   - Save as `reference_image_{timestamp}.png`

**Output**: Dictionary mapping reference image IDs to file paths

**Critical**: If API call fails, program exits (must succeed before Sora generation).

---

## STEP 1.5: Still Image Generation

### Step 1.5.0: Generate Still Images and Panning Videos
**Purpose**: Create still images with camera panning for contemplative moments

**Substeps**:
For each still image segment identified in Step 0.65:

1. **Generate Still Image**:
   - Extract `image_prompt` from segment info
   - Sanitize prompt for content policy compliance
   - Call `generate_image_from_prompt()`:
     - **API**: OpenAI DALL-E 3
     - **Model**: `dall-e-3`
   - Save as `still_image_segment_{segment_id}_{timestamp}.png`
   
2. **Fallback Handling** (if image generation fails):
   - **Fallback 1**: Use reference image (if available)
   - **Fallback 2**: Use master image (if available)
   - **Fallback 3**: Create emergency placeholder using FFmpeg (solid color image)

3. **Create Panning Video**:
   - Call `create_panning_video_from_image()`:
     - **Duration**: Exactly 12 seconds
     - **Pan Direction**: Randomly chosen (top-left to bottom-right, etc.)
     - **Method**: FFmpeg with zoompan filter
   - Save as `panning_video_segment_{segment_id}_{timestamp}.mp4`
   
4. **Panning Video Fallback** (if panning fails):
   - Create static video (no panning) using FFmpeg
   - Same duration (12 seconds)

**Output**: Dictionary mapping segment IDs to panning video paths

**Critical**: Execution NEVER stops - always uses fallbacks if generation fails.

---

## STEP 2: Video Generation

### Step 2.0: Start Video Generation Jobs
**Purpose**: Initiate Sora 2 video generation for all video segments

**Substeps**:
1. Filter segments to only video segments (using `segment_assignments`)
2. For each video segment:
   - Get Sora prompt from `segment_id_to_prompt` mapping
   - Get reference image assignment from `segment_assignments`
   - Determine which reference image to use:
     - If segment has `reference_image_id`, use that reference image
     - Else, use master image (first reference image)
     - Else, no reference image
   - Call `start_video_generation_job()`:
     - **API**: OpenAI Sora 2
     - **Model**: `sora-2` or `sora-2-pro`
     - **Duration**: 12 seconds
     - **Resolution**: 1280x720 (default)
     - **Reference Image**: Path to assigned reference image (if any)
   - Store job info: `(segment_id, video_id, output_path, prompt)`
   - **Rate Limiting**: Wait 15 seconds between job starts (4 requests/minute limit)

**Output**: List of video generation jobs (non-blocking)

### Step 2.1: Wait for Video Completion
**Purpose**: Poll Sora 2 API until all videos are generated

**Substeps**:
For each video job:

1. **First Attempt**:
   - Call `wait_for_video_completion()`:
     - Poll Sora 2 API every `poll_interval` seconds (default 10s)
     - Check job status until complete or failed
     - Download video when ready
     - Save to `{base}_segment_{segment_id:03d}.mp4`

2. **Retry Logic** (if first attempt fails):
   - Up to 3 total attempts
   - **Exception**: Skip retries for `moderation_blocked` errors
   - For retries:
     - Start new video generation job with same prompt/reference image
     - Wait for completion
     - Save with `_retry{attempt}` suffix

3. **Fallback Handling** (if all retries fail):
   - Generate still image fallback:
     - Create still image prompt from segment text
     - Generate DALL-E image (with fallback chain: ref image → master → placeholder)
     - Create 12-second panning video from image
   - If still image fallback fails:
     - Create emergency placeholder video (solid color, 12 seconds)

**Output**: List of video file paths (one per segment)

**Critical**: Execution NEVER stops - always uses fallbacks if video generation fails.

### Step 2.2: Stitch Video Segments
**Purpose**: Combine all video segments and still image panning videos into final video

**Substeps**:
1. **Build Segment Order**:
   - Use `segment_assignments` to determine order
   - For each segment ID (1 to num_segments):
     - If type is "still": Add panning video from `still_image_videos`
     - If type is "video": Add Sora video from `generated_video_segments`

2. **Stitch Videos**:
   - If multiple segments:
     - Call `stitch_videos()`:
       - Use FFmpeg to concatenate all segments
       - Maintains video quality and codec
       - Output: `{base}_stitched.mp4`
   - If single segment:
     - Use that segment directly (no stitching needed)

3. **Fallback Handling**:
   - If stitching fails:
     - Try using first segment as fallback
     - If that fails, create emergency placeholder video

4. **Validation**:
   - Verify stitched video exists and has content
   - Check duration matches expected duration
   - Warn if duration mismatch > 0.5 seconds

**Output**: Single stitched video file

### Step 2.5: Upscale Video
**Purpose**: Increase video resolution from 720p to 1080p

**Substeps**:
1. If `upscale_to_1080p=True`:
   - Call `upscale_video()`:
     - **Method**: Lanczos upscaling algorithm
     - **Target**: 1920x1080
     - **Tool**: FFmpeg
   - Save as `{base}_1080p.mp4`
   - Clean up original 720p video and individual segments
2. If upscaling fails:
   - Use original 720p video (continue with warning)

**Output**: Upscaled 1080p video (or original if upscaling disabled/failed)

### Step 2.6: Add Audio to Video
**Purpose**: Synchronize and mix narration with background music, then add to video

**Substeps**:
1. **Get Video Duration**:
   - Use FFmpeg to get exact video duration

2. **Prepare Narration Audio**:
   - Use original voiceover backup (without music) if available
   - Else, use mixed audio (will extract narration if possible)
   - Narration is used as-is (no speed adjustment)

3. **Sync Music to Video**:
   - Load `VIDEO_MUSIC.mp3` from current directory
   - If music duration ≠ video duration:
     - If music longer: Trim to video duration with fade in/out (1s each)
     - If music shorter: Loop to extend to video duration with fade in/out
   - Save as `music_synced_{timestamp}.mp3`

4. **Mix Audio**:
   - If using original voiceover (no music):
     - Mix narration + synced music:
       - Narration: 100% volume
       - Music: 7% volume
       - Apply 2x volume boost after mixing (compensate for amix reduction)
     - Save as `audio_resynced_{timestamp}.mp3`
   - If using mixed audio (already has music):
     - Just sync duration (trim/extend) without re-adding music
     - Save as `audio_synced_no_remix_{timestamp}.mp3`

5. **Add Audio to Video**:
   - Call `add_audio_to_video()`:
     - Replace video audio track with mixed audio
     - Maintain video quality
   - Save as `{base}_with_audio.mp4`

**Output**: Video with synchronized narration and background music

### Step 2.7: Add Subtitles
**Purpose**: Add word-level subtitles synchronized with narration

**Substeps**:
1. **Prepare Video**:
   - If video already has subtitles, find original without subtitles
   - Else, use current video

2. **Generate Subtitles**:
   - Call `add_subtitles_to_video()`:
     - Use `generated_script` for subtitle text
     - Use `narration_audio_path` for word-level timing (Whisper transcription)
     - Create SRT subtitle file
     - Burn subtitles into video using FFmpeg subtitles filter
   - Save as `{base}_with_subtitles.mp4`

3. **Cleanup**:
   - Remove video without subtitles (keep subtitled version)

**Output**: Final video with burned-in subtitles

---

## STEP 3: Thumbnail Generation

### Step 3.0: Generate or Use Thumbnail
**Purpose**: Create or use thumbnail for YouTube upload

**Substeps**:
1. If `thumbnail_file` provided:
   - Use provided thumbnail
2. Else:
   - **SKIPPED**: Use YouTube auto-generated thumbnail
   - (Thumbnail generation code exists but is currently skipped)

**Output**: Thumbnail file path (or None for auto-generated)

---

## STEP 4: YouTube Upload

### Step 4.0: Upload Video to YouTube
**Purpose**: Upload final video to YouTube

**Substeps**:
1. If `skip_upload=False`:
   - Verify uploading stitched video (not individual segment)
   - Call `upload_video()`:
     - **Title**: User-provided title
     - **Description**: User-provided description
     - **Tags**: Combined user + generated tags
     - **Category**: Default '22' (People & Blogs)
     - **Privacy**: 'private', 'public', or 'unlisted'
     - **Thumbnail**: Provided thumbnail or auto-generated
     - **Playlist**: Optional playlist ID
   - Handle upload errors:
     - Clean up temporary files if upload fails
     - Re-raise exception
2. If `skip_upload=True`:
   - Skip upload (video saved locally)

**Output**: YouTube video ID (or video path if upload skipped)

---

## Post-Workflow Cleanup

### File Management
1. **Move Final Video**:
   - Move final video to `output_video_path` in `video_output/` folder
   - Ensure output folder exists

2. **Clean Up Temporary Files**:
   - Individual segment videos (if multiple segments)
   - Intermediate videos (`_stitched`, `_1080p`, `_with_audio` versions)
   - Temporary audio files in temp directory
   - Temporary video files in temp directory
   - Audio review folders
   - Generated thumbnails (if auto-generated)

3. **Preserve Important Files**:
   - Final video (in `video_output/`)
   - Reference images (in `video_output/`)
   - Script file (`overarching_script.txt`)
   - Narration audio (`narration_audio.mp3`)

---

## Error Handling Philosophy

### Critical Points (Exit on Failure)
- **Before Sora Generation**: Any API call failure before video generation causes program exit
  - Script generation (Step 0.0)
  - Narration generation (Step 0.1) - if `skip_narration=False`
  - Reference image generation (Step 1.0)

### Non-Critical Points (Continue with Fallbacks)
- **During Sora Generation**: Always use fallbacks, never stop execution
  - Video generation failures → Still image fallback → Emergency placeholder
  - Still image generation failures → Reference image fallback → Placeholder
  - Panning video failures → Static video fallback
  - Stitching failures → Use first segment or emergency placeholder

- **Post-Generation**: Attempt to complete despite errors
  - Upscaling failures → Use original resolution
  - Audio addition failures → Continue without audio
  - Subtitle failures → Continue without subtitles
  - Upload failures → Save video locally

---

## Key Files and Outputs

### Input Files
- `VIDEO_MUSIC.mp3`: Background music file (must be in current directory)
- `overarching_script.txt`: Pre-generated script (optional)
- `narration_audio.mp3`: Pre-generated narration (optional)
- `thumbnail_file.png`: Custom thumbnail (optional)

### Output Files (in `video_output/`)
- `sora_video_{timestamp}_with_audio_with_subtitles.mp4`: Final video
- `reference_image_{ref_id}_{timestamp}.png`: Reference images
- `still_image_segment_{segment_id}_{timestamp}.png`: Still images
- `panning_video_segment_{segment_id}_{timestamp}.mp4`: Panning videos
- `{base}_segment_{segment_id:03d}.mp4`: Individual video segments

### Configuration Files
- `overarching_script.txt`: Generated script
- `narration_audio.mp3`: Generated narration
- `video_config.json`: Video configuration (duration, etc.)

---

## API Usage Summary

### OpenAI API Calls
1. **GPT-5** (Script Generation): 1 call
2. **GPT-5** (Reference Image Analysis): 1 call
3. **GPT-5** (Still Image Analysis): 1 call
4. **GPT-5** (Sora Prompt Generation): N calls (one per video segment)
5. **GPT-5** (Still Image Prompt Generation): M calls (one per still image)
6. **GPT-4o** (Tag Generation): 1 call
7. **GPT-4o-mini** (Key Phrase Extraction): N calls (one per video segment)
8. **Whisper** (Audio Transcription): 1 call (for segmentation)
9. **DALL-E 3** (Reference Images): R calls (one per reference image)
10. **DALL-E 3** (Still Images): M calls (one per still image)
11. **Sora 2** (Video Generation): N calls (one per video segment)

**Total**: Approximately 3 + 2N + 2M + R API calls per video

Where:
- N = Number of video segments (~2/3 of total segments)
- M = Number of still images (~1/3 of total segments)
- R = Number of reference images (0-4, typically 1-2)

---

## Rate Limiting

### Sora 2 API
- **Limit**: 4 requests per minute
- **Implementation**: 15-second delay between video generation job starts
- **Impact**: For a 60-second video (5 segments), job starts take ~60 seconds

### Other APIs
- No explicit rate limiting (handled by OpenAI's default limits)
- Retry logic implemented for transient failures

---

## Duration and Segment Calculations

### Example: 60-Second Video
- **Total Segments**: 60 / 12 = 5 segments
- **Video Segments**: 5 * 2/3 ≈ 3 segments (36 seconds)
- **Still Images**: 5 - 3 = 2 segments (24 seconds)
- **Total Duration**: 36s + 24s = 60 seconds ✓

### Example: 120-Second Video
- **Total Segments**: 120 / 12 = 10 segments
- **Video Segments**: 10 * 2/3 ≈ 7 segments (84 seconds)
- **Still Images**: 10 - 7 = 3 segments (36 seconds)
- **Total Duration**: 84s + 36s = 120 seconds ✓

---

## Notes and Considerations

1. **Narration Timing**: Narration is generated BEFORE video generation to ensure accurate segmentation based on actual spoken words.

2. **Visual Consistency**: Reference images ensure characters/objects look identical across all video segments.

3. **Photorealism**: All prompts emphasize photorealistic, documentary-style footage (not artistic or stylized).

4. **Fallback Strategy**: Multiple fallback layers ensure execution never stops, even if individual components fail.

5. **Segment Synchronization**: Still images add 12 seconds before subsequent segments, which is accounted for in timing calculations.

6. **Music Mixing**: Background music is synced to exact video duration with fade in/out, mixed at 7% volume.

7. **Subtitle Accuracy**: Word-level timing from Whisper ensures subtitles match narration exactly.

---

## End of Workflow

Upon completion, the workflow outputs:
- ✅ Final video file (with audio and subtitles)
- ✅ YouTube video ID (if uploaded)
- ✅ Reference images (for reference)
- ✅ Script and narration files (for future use)

All files are saved in the `video_output/` folder, with temporary files cleaned up automatically.

