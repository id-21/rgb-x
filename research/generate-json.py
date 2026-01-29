import json
import os
import base64
import mimetypes
import re

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILENAME = '/Users/ishan-aiworkspace/Documents/Daga PDF All Design Assets/TexturesForViz2D/all-file-paths.txt'
OUTPUT_FILENAME = '/Users/ishan-aiworkspace/Documents/Daga PDF All Design Assets/TexturesForViz2D/textureAssets.json'
FILE_ROOT = '/Users/ishan-aiworkspace/Documents/Daga PDF All Design Assets/TexturesForViz2D/'

# The prefix you want for your web app
WEB_PATH_PREFIX = '/local-textures/'

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_base64_image(file_path):
    """
    Reads a file from disk and returns a base64 data URI.
    Returns None if file is not found or cannot be read.
    """
    # Clean the path for the OS (handle potential windows/unix path differences)
    # The list has "./Thumbnails...", os.path.normpath fixes slashes
    clean_path = FILE_ROOT + os.path.normpath(file_path)
    # clean_path = file_path
    
    if not os.path.exists(clean_path):
        print(f"Warning: Thumbnail file not found at: {clean_path}")
        return None

    try:
        with open(clean_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Guess mime type based on extension (usually image/jpeg)
        mime_type, _ = mimetypes.guess_type(clean_path)
        if not mime_type:
            mime_type = 'image/jpeg' # Default fallback
            
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error reading {clean_path}: {e}")
        return None

# ---------------------------------------------------------
# SCRIPT
# ---------------------------------------------------------

def parse_and_generate_json():
    # Dictionaries to store the paths found
    # Key = "SubFolder/Filename.jpg" (e.g., "Energie/30185.jpg")
    # Value = The full original line from the text file
    textures_map = {}
    thumbnails_map = {}

    print(f"Reading {INPUT_FILENAME}...")

    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILENAME}'. Please create this file with your paths.")
        return
    
    # 1. Parse the file list
    for line in lines:
        # Strip quotes and whitespace
        raw_path = line.strip().strip('"').strip("'")
        
        if not raw_path or raw_path.endswith('/'): 
            continue 

        # Normalize path separators to forward slashes for internal logic
        path = raw_path.replace('\\', '/')

        # Identify Textures
        if '/Texture File/' in path:
            parts = path.split('/Texture File/')
            if len(parts) > 1:
                relative_key = parts[1] 
                textures_map[relative_key] = raw_path

        # Identify Thumbnails
        elif '/Thumbnails/' in path:
            parts = path.split('/Thumbnails/')
            if len(parts) > 1:
                relative_key = parts[1]
                thumbnails_map[relative_key] = raw_path

    print(f"Found {len(textures_map)} textures and {len(thumbnails_map)} thumbnails.")
    print("Processing images and converting to Base64 (this may take a moment)...")

    # 2. Match and Build JSON
    assets_list = []
    current_id = 1

    for relative_key, original_texture_path in textures_map.items():
        
        filename = os.path.basename(relative_key)
        name, _ = os.path.splitext(filename)
        
        # Construct web path
        # Assuming relative_key is "Energie/30185.jpg"
        texture_web_path = f"{WEB_PATH_PREFIX}{relative_key}"

        # Generate Base64 Placeholder from Thumbnail
        placeholder_data = None
        
        if relative_key in thumbnails_map:
            thumbnail_disk_path = thumbnails_map[relative_key]
            # Convert the actual file on disk to base64
            placeholder_data = get_base64_image(thumbnail_disk_path)
        
        # Build Object
        asset_obj = {
            "id": current_id,
            "name": name,
            "filename": filename,
            "localPath": texture_web_path,
            "cloudUrl": None,
            "placeholder": placeholder_data 
        }

        assets_list.append(asset_obj)
        current_id += 1

        if current_id % 100 == 0:
            print(f"Processed {current_id} items...")

    # 3. Write to JSON
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(assets_list, f, indent=2)

    print(f"Done! Saved {len(assets_list)} assets to {OUTPUT_FILENAME}")

    
if __name__ == "__main__":
    parse_and_generate_json()