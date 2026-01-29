import os
from PIL import Image

# CONFIGURATION
SOURCE_ROOT = "/Users/ishan-aiworkspace/Documents/Daga PDF All Design Assets/TexturesForViz2D/Texture File"
DEST_ROOT = "/Users/ishan-aiworkspace/Documents/Daga PDF All Design Assets/TexturesForViz2D/Thumbnails"
THUMBNAIL_SIZE = (200, 200)

def create_thumbnails():
    # Counter to track progress
    processed_count = 0
    error_count = 0

    print(f"Starting thumbnail generation...")
    print(f"Source: {SOURCE_ROOT}")
    print(f"Destination: {DEST_ROOT}")
    print("-" * 40)

    # Walk through the entire directory tree
    for root, dirs, files in os.walk(SOURCE_ROOT):
        
        # calculate the relative path (e.g., "Energie" or "Forbidden City")
        rel_path = os.path.relpath(root, SOURCE_ROOT)
        
        # create the corresponding folder in the destination
        target_dir = os.path.join(DEST_ROOT, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                
                # Build full file paths
                source_path = os.path.join(root, filename)
                dest_path = os.path.join(target_dir, filename)

                try:
                    with Image.open(source_path) as img:
                        # Convert to RGB (in case of PNG transparency issues when saving as JPG)
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        
                        # Create the thumbnail (Maintains Aspect Ratio)
                        # This ensures the image fits WITHIN 200x200 without stretching/distortion.
                        img.thumbnail(THUMBNAIL_SIZE)
                        
                        # Save the new file
                        img.save(dest_path, "JPEG", quality=85)
                        
                        processed_count += 1
                        print(f"[OK] {filename}")

                except Exception as e:
                    print(f"[ERROR] Could not process {filename}: {e}")
                    error_count += 1

    print("-" * 40)
    print(f"Done! Processed {processed_count} images.")
    if error_count > 0:
        print(f"Encountered {error_count} errors.")

if __name__ == "__main__":
    create_thumbnails()