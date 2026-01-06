"""
YouTube Video Upload Script
Uploads videos to YouTube using the YouTube Data API v3 with configurable parameters.
"""

import os
import argparse
import tempfile
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not installed. Thumbnail optimization will be limited.")
    print("Install with: pip install Pillow")


# YouTube API scopes
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'


def get_authenticated_service(client_secrets_file='client_secrets.json'):
    """
    Authenticate and return a YouTube API service object.
    Automatically refreshes the token if it's expired or close to expiring.
    
    Args:
        client_secrets_file: Path to OAuth 2.0 client secrets JSON file
        
    Returns:
        YouTube API service object
    """
    creds = None
    token_updated = False
    
    # Load existing credentials if available
    if os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing token: {e}")
            print("   Will request new authorization...")
            creds = None
    
    # Check if credentials need refresh
    needs_refresh = False
    if creds:
        if not creds.valid:
            # Token is invalid (expired or revoked)
            needs_refresh = True
            print("🔄 Token is invalid, refreshing...")
        elif creds.expired:
            # Token is expired
            needs_refresh = True
            print("🔄 Token is expired, refreshing...")
        elif creds.expiry:
            # Check if token is close to expiring (within 1 hour)
            time_until_expiry = creds.expiry - datetime.utcnow()
            if time_until_expiry < timedelta(hours=1):
                needs_refresh = True
                hours_left = time_until_expiry.total_seconds() / 3600
                print(f"🔄 Token expires in {hours_left:.1f} hours, refreshing proactively...")
    
    # Refresh token if needed
    if needs_refresh:
        if creds and creds.refresh_token:
            try:
                creds.refresh(Request())
                token_updated = True
                print("✅ Token refreshed successfully")
            except Exception as e:
                print(f"⚠️  Token refresh failed: {e}")
                print("   Will request new authorization...")
                creds = None
        else:
            print("⚠️  No refresh token available, will request new authorization...")
            creds = None
    
    # If there are no valid credentials, request authorization
    if not creds or not creds.valid:
        if not os.path.exists(client_secrets_file):
            raise FileNotFoundError(
                f"Client secrets file not found: {client_secrets_file}\n"
                f"Please download your OAuth 2.0 credentials from Google Cloud Console\n"
                f"and save them as '{client_secrets_file}'"
            )
        
        print("🔐 Requesting new authorization...")
        flow = InstalledAppFlow.from_client_secrets_file(
            client_secrets_file, SCOPES)
        creds = flow.run_local_server(port=0)
        token_updated = True
        print("✅ New authorization granted")
    
    # Save credentials if they were updated
    if token_updated or not os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
            print("💾 Token saved to token.pickle")
        except Exception as e:
            print(f"⚠️  Warning: Could not save token: {e}")
    
    return build(API_SERVICE_NAME, API_VERSION, credentials=creds)


def optimize_thumbnail_for_youtube(thumbnail_file, max_size_mb=2, target_size=(1280, 720)):
    """
    Optimize thumbnail image for YouTube upload.
    
    YouTube requirements:
    - Maximum file size: 2MB
    - Recommended dimensions: 1280x720 pixels (16:9 aspect ratio)
    - Minimum width: 640 pixels
    - Maximum width: 1280 pixels
    - Formats: JPG, GIF (non-animated), BMP, or PNG
    
    Args:
        thumbnail_file: Path to the thumbnail image file
        max_size_mb: Maximum file size in MB (default: 2MB)
        target_size: Target dimensions as (width, height) tuple (default: 1280x720)
        
    Returns:
        Path to optimized thumbnail file (may be same file if no optimization needed,
        or a new temp file if optimization was required)
    """
    if not os.path.exists(thumbnail_file):
        raise FileNotFoundError(f"Thumbnail file not found: {thumbnail_file}")
    
    # Check file size
    file_size_mb = os.path.getsize(thumbnail_file) / (1024 * 1024)
    
    if not PIL_AVAILABLE:
        # Without PIL, we can only check file size
        if file_size_mb > max_size_mb:
            raise ValueError(
                f"Thumbnail file is too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB). "
                f"Please resize/compress the image manually or install Pillow: pip install Pillow"
            )
        return thumbnail_file
    
    # Get image dimensions
    try:
        with Image.open(thumbnail_file) as img:
            original_size = img.size
            original_format = img.format
    except Exception as e:
        raise ValueError(f"Could not open thumbnail image: {e}")
    
    # Check if optimization is needed
    needs_resize = original_size != target_size
    needs_compress = file_size_mb > max_size_mb
    
    # If no optimization needed, return original
    if not needs_resize and not needs_compress:
        print(f"✅ Thumbnail is already optimized: {original_size[0]}x{original_size[1]}, {file_size_mb:.2f}MB")
        return thumbnail_file
    
    print(f"🔄 Optimizing thumbnail...")
    print(f"   Original: {original_size[0]}x{original_size[1]}, {file_size_mb:.2f}MB")
    
    # Create optimized version
    try:
        with Image.open(thumbnail_file) as img:
            # Convert RGBA to RGB if needed (for JPG compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparent images
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target dimensions (maintain aspect ratio, then crop if needed)
            # Use LANCZOS resampling (high quality), fallback to ANTIALIAS for older PIL versions
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            img.thumbnail(target_size, resample)
            
            # If image is smaller than target, create a new image with target size and paste centered
            if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
                # Create new image with target size and black background
                new_img = Image.new('RGB', target_size, (0, 0, 0))
                # Calculate position to center the image
                x_offset = (target_size[0] - img.size[0]) // 2
                y_offset = (target_size[1] - img.size[1]) // 2
                new_img.paste(img, (x_offset, y_offset))
                img = new_img
            else:
                # Crop to exact target size (center crop)
                left = (img.size[0] - target_size[0]) // 2
                top = (img.size[1] - target_size[1]) // 2
                right = left + target_size[0]
                bottom = top + target_size[1]
                img = img.crop((left, top, right, bottom))
            
            # Save to temp file with compression
            temp_dir = tempfile.gettempdir()
            temp_thumbnail = os.path.join(temp_dir, f"youtube_thumbnail_{os.path.basename(thumbnail_file)}")
            
            # Try different quality levels to get under 2MB
            quality = 95
            max_size_bytes = max_size_mb * 1024 * 1024
            
            while quality >= 50:
                img.save(temp_thumbnail, 'JPEG', quality=quality, optimize=True)
                temp_size = os.path.getsize(temp_thumbnail)
                
                if temp_size <= max_size_bytes:
                    print(f"   ✅ Optimized: {img.size[0]}x{img.size[1]}, {temp_size / (1024 * 1024):.2f}MB (quality: {quality})")
                    return temp_thumbnail
                
                quality -= 5
            
            # Check final size after loop (shouldn't happen, but just in case)
            final_check_size = os.path.getsize(temp_thumbnail)
            if final_check_size <= max_size_bytes:
                print(f"   ✅ Optimized: {img.size[0]}x{img.size[1]}, {final_check_size / (1024 * 1024):.2f}MB (quality: 50)")
                return temp_thumbnail
            
            # If still too large, try more aggressive compression
            if final_check_size > max_size_bytes:
                # Resize smaller and try again
                smaller_size = (int(target_size[0] * 0.9), int(target_size[1] * 0.9))
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                img.thumbnail(smaller_size, resample)
                
                # Create new image with target size
                new_img = Image.new('RGB', target_size, (0, 0, 0))
                x_offset = (target_size[0] - img.size[0]) // 2
                y_offset = (target_size[1] - img.size[1]) // 2
                new_img.paste(img, (x_offset, y_offset))
                img = new_img
                
                img.save(temp_thumbnail, 'JPEG', quality=85, optimize=True)
                final_size = os.path.getsize(temp_thumbnail)
                
                if final_size <= max_size_bytes:
                    print(f"   ✅ Optimized: {img.size[0]}x{img.size[1]}, {final_size / (1024 * 1024):.2f}MB")
                    return temp_thumbnail
                else:
                    raise ValueError(
                        f"Could not compress thumbnail below {max_size_mb}MB. "
                        f"Final size: {final_size / (1024 * 1024):.2f}MB. "
                        f"Please manually compress the image."
                    )
            
            return temp_thumbnail
            
    except Exception as e:
        raise ValueError(f"Failed to optimize thumbnail: {e}")


def upload_video(
    video_file,
    title,
    description='',
    tags=None,
    category_id='22',  # Default: People & Blogs
    privacy_status='private',  # 'private', 'public', 'unlisted'
    thumbnail_file=None,
    playlist_id=None,
    client_secrets_file='client_secrets.json'
):
    """
    Upload a video to YouTube.
    
    Args:
        video_file: Path to the video file to upload
        title: Video title (required)
        description: Video description
        tags: List of tags for the video
        category_id: YouTube video category ID (default: 22 - People & Blogs)
        privacy_status: Privacy status ('private', 'public', 'unlisted')
        thumbnail_file: Optional path to thumbnail image file
        playlist_id: Optional YouTube playlist ID to add video to
        client_secrets_file: Path to OAuth 2.0 client secrets JSON file
        
    Returns:
        Video ID of the uploaded video
    """
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")
    
    youtube = get_authenticated_service(client_secrets_file)
    
    # Build the video metadata
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
    
    # Create media file upload object
    media = MediaFileUpload(
        video_file,
        chunksize=-1,
        resumable=True,
        mimetype='video/*'
    )
    
    # Insert video
    insert_request = youtube.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media
    )
    
    # Upload video
    print(f"Uploading video: {title}")
    response = None
    error = None
    retry = 0
    
    while response is None:
        try:
            status, response = insert_request.next_chunk()
            if response is not None:
                if 'id' in response:
                    video_id = response['id']
                    print(f"Video uploaded successfully! Video ID: {video_id}")
                    print(f"Video URL: https://www.youtube.com/watch?v={video_id}")
                else:
                    print("Upload failed: Unexpected response")
                    return None
            else:
                if status:
                    print(f"Upload progress: {int(status.progress() * 100)}%")
        except Exception as e:
            error = e
            if retry > 3:
                print(f"Upload failed after retries: {error}")
                return None
            retry += 1
            print(f"Error occurred, retrying ({retry}/3)...")
    
    # Upload thumbnail if provided
    if thumbnail_file and os.path.exists(thumbnail_file):
        try:
            # Optimize thumbnail for YouTube before uploading
            print("🖼️  Optimizing thumbnail for YouTube...")
            optimized_thumbnail = optimize_thumbnail_for_youtube(thumbnail_file)
            
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(optimized_thumbnail)
            ).execute()
            print(f"✅ Thumbnail uploaded successfully!")
            
            # Clean up temp file if it was created
            if optimized_thumbnail != thumbnail_file and os.path.exists(optimized_thumbnail):
                try:
                    os.remove(optimized_thumbnail)
                except:
                    pass
        except Exception as e:
            print(f"⚠️  Warning: Failed to upload thumbnail: {e}")
            import traceback
            traceback.print_exc()
    
    # Add to playlist if provided
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


def main():
    """Main function to handle command-line arguments and interactive prompts for video upload."""
    parser = argparse.ArgumentParser(
        description='Upload a video to YouTube with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (will prompt for missing info):
  python upload_video.py
  
  # Command-line mode:
  python upload_video.py video.mp4 --title "My Video" --privacy public
  
  # Full example with all parameters:
  python upload_video.py video.mp4 --title "My Video" --description "Description" \\
    --tags ai automation youtube --privacy public --category 22 --thumbnail thumb.jpg
        """
    )
    
    parser.add_argument(
        'video_file',
        nargs='?',
        help='Path to the video file to upload'
    )
    
    parser.add_argument(
        '--title',
        help='Video title'
    )
    
    parser.add_argument(
        '--description',
        default='',
        help='Video description'
    )
    
    parser.add_argument(
        '--tags',
        nargs='+',
        help='Video tags (space-separated)'
    )
    
    parser.add_argument(
        '--category',
        default='22',
        help='YouTube category ID (default: 22 - People & Blogs). Common: 1=Film, 10=Music, 20=Gaming, 22=People & Blogs, 24=Entertainment, 27=Education, 28=Science & Technology'
    )
    
    parser.add_argument(
        '--privacy',
        choices=['private', 'public', 'unlisted'],
        default='private',
        help='Privacy status: private, public, or unlisted (default: private)'
    )
    
    parser.add_argument(
        '--thumbnail',
        help='Path to thumbnail image file (JPG, PNG, GIF, or BMP)'
    )
    
    parser.add_argument(
        '--playlist',
        help='YouTube playlist ID to add video to'
    )
    
    parser.add_argument(
        '--client-secrets',
        default='client_secrets.json',
        help='Path to OAuth 2.0 client secrets file (default: client_secrets.json)'
    )
    
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Disable interactive prompts (fail if required parameters missing)'
    )
    
    args = parser.parse_args()
    
    # Store client secrets file path (used in both modes)
    client_secrets_file = args.client_secrets
    
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
            print('   python upload_video.py video.mp4 --title "Your Title"')
            print("2. Use --non-interactive flag with all required arguments")
            print("3. Run from a terminal where input is available")
            print("="*60)
            if not args.video_file or not args.title:
                print("\n❌ Error: Missing required arguments (video_file and title)")
                print("   Run with: --non-interactive video.mp4 --title '...'")
                return 1
        video_file = args.video_file
        title = args.title
        
        # Prompt for video file if not provided
        if not video_file:
            try:
                video_file = input("Enter path to video file: ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Error: Video file path is required!")
                print("   Hint: Run with --non-interactive and provide video_file, or use command-line arguments")
                return 1
            if not video_file:
                print("❌ Error: Video file path is required!")
                return 1
        
        # Prompt for title if not provided
        if not title:
            try:
                title = input("Enter video title: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Error: Video title is required!")
                print("   Hint: Run with --non-interactive and provide --title, or use command-line arguments")
                return 1
            if not title:
                print("❌ Error: Video title is required!")
                return 1
        
        # Prompt for description if not provided
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
        
        # Prompt for tags if not provided
        if not args.tags:
            try:
                tags_input = input("Enter tags (comma or space-separated, or press Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                tags_input = ""
            if tags_input:
                tags = [tag.strip() for tag in tags_input.replace(',', ' ').split() if tag.strip()]
            else:
                tags = None
        else:
            tags = args.tags
        
        # Prompt for privacy if using default
        if args.privacy == 'private':
            try:
                privacy_input = input("Privacy status [private/public/unlisted] (default: private): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                privacy_input = ""
            if privacy_input in ['private', 'public', 'unlisted']:
                privacy_status = privacy_input
            else:
                privacy_status = 'private'
        else:
            privacy_status = args.privacy
        
        # Prompt for category
        try:
            category_input = input("Category ID (default: 22 - People & Blogs, or press Enter to use default): ").strip()
        except (EOFError, KeyboardInterrupt):
            category_input = ""
        category_id = category_input if category_input else args.category
        
        # Prompt for thumbnail
        if not args.thumbnail:
            try:
                thumbnail_input = input("Thumbnail image path (optional, press Enter to skip): ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                thumbnail_input = ""
            thumbnail_file = thumbnail_input if thumbnail_input else None
        else:
            thumbnail_file = args.thumbnail
        
        # Prompt for playlist
        if not args.playlist:
            try:
                playlist_input = input("Playlist ID (optional, press Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                playlist_input = ""
            playlist_id = playlist_input if playlist_input else None
        else:
            playlist_id = args.playlist
        
    else:
        # Non-interactive mode: use provided arguments or fail
        if not args.video_file:
            print("❌ Error: video_file is required in non-interactive mode!")
            return 1
        if not args.title:
            print("❌ Error: --title is required in non-interactive mode!")
            return 1
        
        video_file = args.video_file
        title = args.title
        description = args.description
        tags = args.tags
        privacy_status = args.privacy
        category_id = args.category
        thumbnail_file = args.thumbnail
        playlist_id = args.playlist
    
    # Validate video file exists
    if not os.path.exists(video_file):
        print(f"❌ Error: Video file not found: {video_file}")
        return 1
    
    # Display upload summary
    print("\n" + "="*60)
    print("📹 YouTube Video Upload")
    print("="*60)
    print(f"Video File: {video_file}")
    print(f"Title: {title}")
    print(f"Description: {description[:50]}{'...' if len(description) > 50 else ''}")
    print(f"Tags: {', '.join(tags) if tags else 'None'}")
    print(f"Privacy: {privacy_status}")
    print(f"Category: {category_id}")
    if thumbnail_file:
        print(f"Thumbnail: {thumbnail_file}")
    if playlist_id:
        print(f"Playlist ID: {playlist_id}")
    print("="*60 + "\n")
    
    try:
        video_id = upload_video(
            video_file=video_file,
            title=title,
            description=description,
            tags=tags,
            category_id=category_id,
            privacy_status=privacy_status,
            thumbnail_file=thumbnail_file,
            playlist_id=playlist_id,
            client_secrets_file=client_secrets_file
        )
    
        if video_id:
            print("\n" + "="*60)
            print("✅ Upload completed successfully!")
            print("="*60)
            print(f"Video ID: {video_id}")
            print(f"Video URL: https://www.youtube.com/watch?v={video_id}")
            print("="*60)
            return 0
        else:
            print("\n❌ Upload failed!")
            return 1
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

