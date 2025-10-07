# src/drive_utils.py
import os
import json
import mimetypes
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

PARENT_FOLDER_ID = os.environ.get('DRIVE_PARENT_FOLDER_ID') # No default value needed now
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

    if not PARENT_FOLDER_ID:
        raise ValueError("DRIVE_PARENT_FOLDER_ID environment variable not set.")

    query = f"name = '{folder_name}' and '{PARENT_FOLDER_ID}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    
    # ✅ ADD supportsAllDrives=True TO THE SEARCH
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)', supportsAllDrives=True).execute()
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
        # ✅ ADD supportsAllDrives=True TO THE FOLDER CREATION
        folder = service.files().create(body=file_metadata, fields='id', supportsAllDrives=True).execute()
        folder_id = folder.get('id')
        FOLDER_ID_CACHE[folder_name] = folder_id
        return folder_id

def upload_image(service, image_path, image_filename, folder_id):
    file_metadata = {
        'name': image_filename,
        'parents': [folder_id]
    }
    
    mimetype, _ = mimetypes.guess_type(image_path)
    if mimetype is None:
        mimetype = 'application/octet-stream'

    media = MediaFileUpload(image_path, mimetype=mimetype)
    
    # ✅ ADD supportsAllDrives=True TO THE FILE UPLOAD
    file = service.files().create(
        body=file_metadata, 
        media_body=media, 
        fields='id', 
        supportsAllDrives=True
    ).execute()

    print(f"File ID: {file.get('id')} uploaded successfully.")
    return file.get('id')