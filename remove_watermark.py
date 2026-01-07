
import os
import sys
import numpy as np
from PIL import Image

def get_watermark_info(width, height):
    is_large = width > 1024 and height > 1024
    size = 96 if is_large else 48
    margin = 64 if is_large else 32
    
    return {
        'size': size,
        'x': width - margin - size,
        'y': height - margin - size,
        'width': size,
        'height': size
    }

def calculate_alpha_map(bg_image):
    # Ensure image is RGB
    bg_image = bg_image.convert('RGB')
    data = np.array(bg_image, dtype=np.float32)
    
    # Normalize max channel to 0-1
    # equivalent to: Math.max(r, g, b) / 255.0
    alpha_map = np.max(data, axis=2) / 255.0
    return alpha_map

def remove_watermark_logic(image_path, assets_dir):
    try:
        # Open with context manager to ensure file handle is closed
        with Image.open(image_path) as ref_img:
            img = ref_img.convert('RGB')
            
        width, height = img.size
        
        info = get_watermark_info(width, height)
        size = info['size']
        
        bg_filename = f"bg_{size}.png"
        bg_path = os.path.join(assets_dir, bg_filename)
        
        if not os.path.exists(bg_path):
            print(f"Error: Background asset not found at {bg_path}")
            return None

        with Image.open(bg_path) as ref_bg:
            bg_img = ref_bg.copy()
            
        alpha_map = calculate_alpha_map(bg_img)
        
        # Convert main image to numpy
        img_data = np.array(img, dtype=np.float32)
        
        # Define constants
        ALPHA_THRESHOLD = 0.002
        MAX_ALPHA = 0.99
        LOGO_VALUE = 255.0
        
        # Extract region of interest (ROI)
        x, y, w, h = info['x'], info['y'], info['width'], info['height']
        
        # roi shape: (h, w, 3)
        roi = img_data[y:y+h, x:x+w]
        
        # alpha_map shape: (size, size) -> (h, w)
        # We need to broadcast alpha_map to (h, w, 3) or handle channel-wise
        
        # Iterate or vectorize. Vectorization is better in numpy.
        
        # Filter based on threshold
        # mask where alpha >= ALPHA_THRESHOLD
        mask = alpha_map >= ALPHA_THRESHOLD
        
        # Clamp alpha
        effective_alpha = np.minimum(alpha_map, MAX_ALPHA)
        
        # We process only where mask is True to avoid divide by zero or useless ops
        # However, for vectorized, we can just compute everything and apply only where mask is True.
        # But (1 - alpha) could be 0 if alpha is 1 (though capped at 0.99), so no div by zero.
        
        # Formula: original = (watermarked - alpha * LOGO_VALUE) / (1.0 - alpha)
        
        # Prepare alpha for broadcasting: (h, w, 1)
        alpha_expanded = effective_alpha[:, :, np.newaxis]
        
        # Original calculation
        restored_roi = (roi - alpha_expanded * LOGO_VALUE) / (1.0 - alpha_expanded)
        
        # Clamp and round
        restored_roi = np.clip(np.round(restored_roi), 0, 255)
        
        # Apply changes only where alpha threshold is met
        # We can construct the final ROI by combining original ROI (where mask False) and Restored ROI (where mask True)
        
        mask_expanded = mask[:, :, np.newaxis]
        final_roi = np.where(mask_expanded, restored_roi, roi)
        
        # Place back into image
        img_data[y:y+h, x:x+w] = final_roi
        
        # Save result - Overwrite original
        result_img = Image.fromarray(img_data.astype(np.uint8))
        
        # Overwrite the original file
        # We use the same path
        output_path = image_path
        
        # Save with quality options if needed, but defaults are usually fine for PNG/JPG
        result_img.save(output_path)
        print(f"Processed and Overwritten: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

if __name__ == "__main__":
    import time

    def get_resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.dirname(os.path.abspath(__file__))

        return os.path.join(base_path, relative_path)

    # Allow asset lookup logic to use this helper
    # We need to wrap logic to pass the correct asset path
    # But remove_watermark_logic takes assets_dir.
    # We can just pass the result of get_resource_path('.') as assets_dir?
    # No, get_resource_path('bg_48.png') returns full path.
    # remove_watermark_logic does: os.path.join(assets_dir, bg_filename)
    # So we should pass the base dir.
    
    ASSETS_DIR = get_resource_path('.')

    def start_monitoring():
        # Cross-platform way to get Downloads folder
        # Windows default:
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        
        print(f"Monitoring {downloads_path} for Gemini images...")
        print("Press Ctrl+C to exit.")
        
        seen_files = set()
        try:
            # Initial scan to avoid processing old files
            if os.path.exists(downloads_path):
                seen_files = set(os.listdir(downloads_path))
        except Exception as e:
            print(f"Error accessing downloads: {e}")
            return

        while True:
            try:
                time.sleep(2)
                current_files = set(os.listdir(downloads_path))
                new_files = current_files - seen_files
                
                for filename in new_files:
                    if filename.startswith("Gemini_Generated_Image") and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        full_path = os.path.join(downloads_path, filename)
                        
                        # Wait a moment for file write to complete
                        time.sleep(1)
                        
                        print(f"Detected: {filename}")
                        remove_watermark_logic(full_path, ASSETS_DIR)
                
                seen_files = current_files
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                time.sleep(5)

    if len(sys.argv) < 2:
        # No arguments -> Monitor Mode
        start_monitoring()
    else:
        # Arguments provided -> Single File Mode
        img_path = sys.argv[1]
        remove_watermark_logic(img_path, ASSETS_DIR)
