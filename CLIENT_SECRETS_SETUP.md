# Client Secrets JSON Setup Guide

## What is client_secrets.json?

The `client_secrets.json` file contains your OAuth 2.0 credentials from Google Cloud Console. This file is required to authenticate with the YouTube Data API.

## How to Get Your Credentials

1. **Go to Google Cloud Console**
   - Visit: https://console.cloud.google.com/

2. **Create or Select a Project**
   - Click on the project dropdown at the top
   - Create a new project or select an existing one
   - Note your Project ID

3. **Enable YouTube Data API v3**
   - Go to "APIs & Services" > "Library"
   - Search for "YouTube Data API v3"
   - Click on it and press "Enable"

4. **Create OAuth 2.0 Credentials**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - If prompted, configure the OAuth consent screen first:
     - Choose "External" (unless you have a Google Workspace)
     - Fill in required fields (App name, User support email, Developer contact)
     - Add scopes: `https://www.googleapis.com/auth/youtube.upload`
     - Add test users (your email) if needed
   - For Application type, select "Desktop app"
   - Give it a name (e.g., "YouTube Uploader")
   - Click "Create"

5. **Download the Credentials**
   - After creating, you'll see a popup with your Client ID and Client Secret
   - Click "Download JSON" button
   - OR copy the values manually

6. **Save the File**
   - Rename the downloaded file to `client_secrets.json`
   - Place it in the same directory as `upload_video.py`
   - OR copy the values from the template below

## File Structure

The `client_secrets.json` file should have this structure:

```json
{
  "installed": {
    "client_id": "123456789-abcdefghijklmnop.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "GOCSPX-abcdefghijklmnopqrstuvwxyz",
    "redirect_uris": ["http://localhost"]
  }
}
```

## Important Notes

- **Keep this file secret!** Never commit it to version control (it's already in `.gitignore`)
- The `client_id` will look like: `xxxxx.apps.googleusercontent.com`
- The `client_secret` will look like: `GOCSPX-xxxxx`
- The `project_id` is your Google Cloud project ID
- The other fields (`auth_uri`, `token_uri`, etc.) are standard and usually don't need to change

## Quick Setup

1. Copy `client_secrets_template.json` to `client_secrets.json`
2. Replace the placeholder values with your actual credentials from Google Cloud Console
3. Save the file
4. Run `python upload_video.py` - it will open a browser for authentication on first run

## Troubleshooting

- **"Invalid client"**: Check that your Client ID and Secret are correct
- **"Redirect URI mismatch"**: Make sure `redirect_uris` includes `http://localhost`
- **"API not enabled"**: Make sure YouTube Data API v3 is enabled in your project
- **"Access blocked"**: Check OAuth consent screen configuration and add your email as a test user

