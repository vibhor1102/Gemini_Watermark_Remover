import os
import sys
import ctypes
import numpy as np
import threading
import logging
import subprocess
import time
from PIL import Image
import pystray # pip install pystray
import configparser
import argparse

# --- Logging Setup ---
APP_NAME = "GeminiWatermarkRemover"

def get_app_data_dir():
    """Get the application data directory."""
    app_data = os.getenv('APPDATA')
    if not app_data:
        app_data = os.path.expanduser("~")
    
    app_dir = os.path.join(app_data, APP_NAME)
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
    return app_dir

def setup_logging():
    """Configure logging to file and console."""
    log_dir = os.path.join(get_app_data_dir(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "app.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

def setup_config():
    """Setup and load configuration."""
    app_dir = get_app_data_dir()
    config_file = os.path.join(app_dir, "config.ini")
    
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        config['Settings'] = {
            '; Change this path if your downloads folder is in a different location': '',
            '; Example: C:\\Users\\YourName\\MyDownloads': '',
            'downloads_dir': ''
        }
        # We need to write comments manually because ConfigParser doesn't persist them well by default
        # But for simplicity in this script, we'll write a raw string if creating fresh
        with open(config_file, 'w') as f:
            f.write("[Settings]\n")
            f.write("; Change this path if your downloads folder is in a different location\n")
            f.write("; Example: C:\\Users\\YourName\\MyDownloads\n")
            f.write("downloads_dir =\n")
    
    config.read(config_file)
    return config, config_file

def enforce_single_instance():
    # Create a named mutex to ensure only one instance runs
    # This works for both script and compiled EXE, and independent of visibility
    mutex_name = "Global\\GeminiWatermarkRemover_Instance_Mutex_v2" 
    
    # CreateMutexW returns a handle. We must keep this handle alive for the process duration.
    # If the mutex already exists, GetLastError returns ERROR_ALREADY_EXISTS (183).
    kernel32 = ctypes.windll.kernel32
    mutex = kernel32.CreateMutexW(None, False, mutex_name)
    
    last_error = kernel32.GetLastError()
    if last_error == 183:
        # Already running
        print("Another instance is already running. Exiting.")
        logging.warning("Another instance is already running. Exiting.")
        sys.exit(0)
    
    return mutex

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
        logging.info(f"Processing: {image_path}")
        # Open with context manager to ensure file handle is closed
        with Image.open(image_path) as ref_img:
            img = ref_img.convert('RGB')
            
        width, height = img.size
        
        info = get_watermark_info(width, height)
        size = info['size']
        
        bg_filename = f"bg_{size}.png"
        bg_path = os.path.join(assets_dir, bg_filename)
        
        if not os.path.exists(bg_path):
            logging.error(f"Error: Background asset not found at {bg_path}")
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
        logging.info(f"Processed and Overwritten: {output_path}")
        print(f"Processed and Overwritten: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        print(f"Error processing {image_path}: {e}")
        return None

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

def start_monitoring(assets_dir, stop_event, downloads_path=None):
    # Determine downloads path
    if not downloads_path:
        # Windows default:
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    
    logging.info(f"Monitoring {downloads_path} for Gemini images...")
    print(f"Monitoring {downloads_path} for Gemini images...")
    
    seen_files = set()
    try:
        # Initial scan to avoid processing old files
        if os.path.exists(downloads_path):
            seen_files = set(os.listdir(downloads_path))
    except Exception as e:
        logging.error(f"Error accessing downloads: {e}")
        return

    while not stop_event.is_set():
        try:
            # Check every 2 seconds
            if stop_event.wait(2):
                break
                
            current_files = set(os.listdir(downloads_path))
            new_files = current_files - seen_files
            
            for filename in new_files:
                if filename.startswith("Gemini_Generated_Image") and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    full_path = os.path.join(downloads_path, filename)
                    
                    # Wait a moment for file write to complete
                    time.sleep(1)
                    
                    logging.info(f"Detected: {filename}")
                    print(f"Detected: {filename}")
                    remove_watermark_logic(full_path, assets_dir)
            
            seen_files = current_files
        except Exception as e:
            logging.error(f"Error in monitor loop: {e}")
            time.sleep(5)

def restart_program(icon):
    logging.info("Restarting application...")
    icon.stop()
    python = sys.executable
    os.execl(python, python, *sys.argv)

def show_logs():
    log_dir = os.path.join(get_app_data_dir(), "logs")
    log_file = os.path.join(log_dir, "app.log")
    if os.path.exists(log_file):
        os.startfile(log_file) # Opens file in default editor/viewer
    elif os.path.exists(log_dir):
        os.startfile(log_dir)

def edit_config():
    config_file = os.path.join(get_app_data_dir(), "config.ini")
    if os.path.exists(config_file):
        os.startfile(config_file)

def quit_program(icon, stop_event):
    logging.info("Quitting application...")
    stop_event.set()
    icon.stop()

if __name__ == "__main__":
    _instance_mutex = enforce_single_instance()
    log_file_path = setup_logging()
    
    ASSETS_DIR = get_resource_path('.')
    
    # Parse Arguments
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument('input_file', nargs='?', help='Process a single file and exit')
    parser.add_argument('--downloads-dir', help='Custom downloads directory to monitor')
    args = parser.parse_args()

    # Load Config
    config, config_file_path = setup_config()
    
    # Determine Downloads Directory Priority:
    # 1. CLI Argument
    # 2. Config File
    # 3. Default (handled in start_monitoring if None passed)
    
    monitor_dir = None
    if args.downloads_dir:
        monitor_dir = args.downloads_dir
    elif config.has_option('Settings', 'downloads_dir'):
        cfg_dir = config.get('Settings', 'downloads_dir').strip()
        if cfg_dir:
            monitor_dir = cfg_dir

    # Check arguments for single file mode
    if args.input_file and os.path.isfile(args.input_file):
        img_path = args.input_file
        remove_watermark_logic(img_path, ASSETS_DIR)
        sys.exit(0)
    
    # Background Monitor Mode
    stop_event = threading.Event()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=start_monitoring, args=(ASSETS_DIR, stop_event, monitor_dir))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # System Tray
    try:
        icon_path = get_resource_path("bg_48.png")
        if not os.path.exists(icon_path):
             # Fallback if image not found, though it should be there
             # pystray requires an image. We can make a simple one or fail.
             logging.error(f"Icon not found at {icon_path}")
             # Create a simple red image
             image = Image.new('RGB', (64, 64), color = 'red')
        else:
            image = Image.open(icon_path)
            
        menu = pystray.Menu(
            pystray.MenuItem("Show Logs", show_logs),
            pystray.MenuItem("Edit Config", edit_config),
            pystray.MenuItem("Restart", restart_program),
            pystray.MenuItem("Quit", lambda icon, item: quit_program(icon, stop_event))
        )
        
        icon = pystray.Icon("GeminiWatermarkRemover", image, "Gemini Watermark Remover", menu)
        
        logging.info("Starting System Tray Icon...")
        icon.run()
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        stop_event.set()
        sys.exit(1)

