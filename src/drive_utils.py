# src/drive_utils.py
import os
import json
import mimetypes # ✅ 1. ADD THIS IMPORT
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

PARENT_FOLDER_ID = os.environ.get('DRIVE_PARENT_FOLDER_ID', 'YOUR_MAIN_FOLDER_ID_HERE')
FOLDER_ID_CACHE = {}

def authenticate():
    creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    if not creds_json_str:
        raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set.")
    
    creds_info = json.loads(creds_json_str)
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=['https://www.googleapis.com/auth/drive'])
    service = build('drive', 'v3', credentials=creds)
    return service

def find_or_create_folder(service, folder_name):
    if folder_name in FOLDER_ID_CACHE:
        return FOLDER_ID_CACHE[folder_name]

    query = f"name = '{folder_name}' and '{PARENT_FOLDER_ID}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])

    if files:
        folder_id = files[0].get('id')
        FOLDER_ID_CACHE[folder_name] = folder_id
        return folder_id
    else:
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
    
    # ✅ 2. THIS IS THE FIX - AUTOMATICALLY DETECT THE FILE TYPE
    mimetype, _ = mimetypes.guess_type(image_path)
    if mimetype is None: # Default to a generic binary type if detection fails
        mimetype = 'application/octet-stream'

    media = MediaFileUpload(image_path, mimetype=mimetype)
    # ✅ --- END OF FIX ---
    
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"File ID: {file.get('id')} uploaded successfully.")
    return file.get('id')