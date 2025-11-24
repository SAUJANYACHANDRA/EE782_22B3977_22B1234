import os
import glob
import numpy as np
import cv2
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory containing the original high-resolution sketches (e.g., 767x816)
INPUT_SKETCH_DIR = "./original_sketch" 

# Directory containing the corresponding .3pts files
INPUT_3PTS_DIR = "./sketch_points" 

# Directory where the final aligned, clean images will be saved (e.g., 200x250 size)
OUTPUT_SKETCH_DIR = "./sketches_aligned_final_200x250" 

# Target size for the clean, aligned output image. (NOW NON-SQUARE: W x H)
TARGET_CROP_WIDTH = 200
TARGET_CROP_HEIGHT = 250

# Standard coordinates for the final output canvas (normalized for a 200x250 image)
# Placement optimized for the taller canvas to capture the full head without neck.
TARGET_POINTS = np.array([
    # X-coordinates centered on 200 width, Y-coordinates optimized for 250 height
    [65, TARGET_CROP_HEIGHT * 0.45],   # Left Eye Center (Y=87.5)
    [135, TARGET_CROP_HEIGHT * 0.45],  # Right Eye Center
    [100, TARGET_CROP_HEIGHT * 0.70]   # Mouth/Nose Center (Y=132.5)
], dtype=np.float32)

# Fixed geometric assumptions for documentation (not used for warping calculation)
ORIGINAL_WIDTH = 767
ORIGINAL_HEIGHT = 816
CONTENT_WIDTH = 560 # User-specified content width 

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_3pts_file(filepath):
    """Loads 3 landmark points from a .3pts file."""
    try:
        points = np.loadtxt(filepath, dtype=np.float32)
        if points.shape != (3, 2):
            print(f"  Warning: {filepath} has incorrect shape: {points.shape}")
            return None
        return points
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None

def align_and_crop(image_path, pts_filepath, output_dir):
    """
    Performs direct warping from original coordinates to the final aligned output.
    """
    filename = os.path.basename(image_path)
    base_id = os.path.splitext(filename)[0]
    
    # 1. Load Data
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Skipping {filename}: Could not read image.")
        return

    # Ensure 3 channels are used
    if original_img.shape[2] == 4:
        original_img = original_img[:, :, :3]

    source_points = load_3pts_file(pts_filepath)
    if source_points is None:
        return

    # ----------------------------------------------------------------------
    # STEP 1: FACIAL LANDMARK ALIGNMENT (DIRECT WARPING)
    # ----------------------------------------------------------------------
    
    # Calculate the Affine Matrix using the original source points 
    # and the standardized target points.
    M = cv2.getAffineTransform(source_points, TARGET_POINTS)

    # Apply the Affine Transformation (Warp) directly to the original image
    # The output size is the new TARGET_CROP_WIDTH x TARGET_CROP_HEIGHT.
    aligned_img = cv2.warpAffine(original_img, M, (TARGET_CROP_WIDTH, TARGET_CROP_HEIGHT))
    
    # ----------------------------------------------------------------------
    # STEP 2: SAVE FINAL ALIGNED IMAGE
    # ----------------------------------------------------------------------
    
    output_path = os.path.join(output_dir, base_id + "_aligned.jpg")
    cv2.imwrite(output_path, aligned_img)
    
    print(f"Processed: {filename} -> Saved to {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    if not os.path.exists(INPUT_SKETCH_DIR) or not os.path.exists(INPUT_3PTS_DIR):
        print(f"Error: Required directories not found. Check paths:")
        print(f"  Sketches: {INPUT_SKETCH_DIR}")
        print(f"  3PTS: {INPUT_3PTS_DIR}")
        exit()
        
    os.makedirs(OUTPUT_SKETCH_DIR, exist_ok=True)
    
    sketch_files = glob.glob(os.path.join(INPUT_SKETCH_DIR, "*.jpg"))
    if not sketch_files:
        print(f"Error: No .jpg files found in {INPUT_SKETCH_DIR}")
        exit()

    print(f"Found {len(sketch_files)} sketches. Starting direct alignment...")

    processed_count = 0
    for image_path in sketch_files:
        base_id = os.path.splitext(os.path.basename(image_path))[0]
        pts_filepath = os.path.join(INPUT_3PTS_DIR, base_id + ".3pts")
        
        # Robust check for .3pts file (handles various naming conventions)
        if not os.path.exists(pts_filepath):
             # Extract the numeric ID part (e.g., from '00013fa010-930831.jpg' -> '00013')
             parts = base_id.split('.')[0].split('-')[0].split('_')[0]
             numeric_id = ''.join(filter(str.isdigit, parts))
             pts_filepath_fallback = os.path.join(INPUT_3PTS_DIR, numeric_id + ".3pts")
             
             if not os.path.exists(pts_filepath_fallback):
                print(f"Skipping {base_id}: Corresponding .3pts file not found.")
                continue
             pts_filepath = pts_filepath_fallback
        
        align_and_crop(image_path, pts_filepath, OUTPUT_SKETCH_DIR)
        processed_count += 1

    print("\n----------------------------------------------------------------------")
    print(f"Alignment script finished. {processed_count} files successfully processed.")
    print("----------------------------------------------------------------------")
    print(f"Next Step: Update SKETCH_PATH in your model training script to: '{OUTPUT_SKETCH_DIR}'")
