import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# This is the ID of the main folder you created ("Fruit App Uploads")
# You can get this from the URL of the folder in Google Drive.
# It's better to set this as a Heroku config var: heroku config:set DRIVE_PARENT_FOLDER_ID=...
PARENT_FOLDER_ID = os.environ.get('DRIVE_PARENT_FOLDER_ID', 'YOUR_MAIN_FOLDER_ID_HERE')

# In-memory cache to store folder IDs so we don't look them up every time.
FOLDER_ID_CACHE = {}

def authenticate():
    """Authenticates with the Google Drive API using service account credentials."""
    creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    if not creds_json_str:
        raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set.")
    
    creds_info = json.loads(creds_json_str)
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=['https://www.googleapis.com/auth/drive'])
    service = build('drive', 'v3', credentials=creds)
    return service

def find_or_create_folder(service, folder_name):
    """Finds a folder by name, or creates it if it doesn't exist. Returns the folder ID."""
    # Check cache first
    if folder_name in FOLDER_ID_CACHE:
        return FOLDER_ID_CACHE[folder_name]

    # Search for the folder
    query = f"name = '{folder_name}' and '{PARENT_FOLDER_ID}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])

    if files:
        # Folder exists, cache and return its ID
        folder_id = files[0].get('id')
        FOLDER_ID_CACHE[folder_name] = folder_id
        return folder_id
    else:
        # Folder doesn't exist, create it
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [PARENT_FOLDER_ID]
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
        FOLDER_ID_CACHE[folder_name] = folder_id
        return folder_id

def upload_image(service, image_path, image_filename, folder_id):
    """Uploads an image file to a specific folder in Google Drive."""
    file_metadata = {
        'name': image_filename,
        'parents': [folder_id]
    }
    media = MediaFileUpload(image_path, mimetype='image/jpeg')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"File ID: {file.get('id')} uploaded successfully.")
    return file.get('id')