#00_02_clean_road_mask.py
#00_02_clean_road_mask.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from s3_handler import S3Handler

# --- CLI argument ---
parser = argparse.ArgumentParser()
parser.add_argument("connection_id", help="Connection ID for this processing run")
parser.add_argument("--visualize", action="store_true", help="Show debug plots")
args = parser.parse_args()

# --- Step 1: Load previous road mask ---
input_path = "final_road_mask.png"
output_path = "final_road_mask_cleaned.png"

if not os.path.exists(input_path):
    print(f"❌ Required file not found: {input_path}")
    sys.exit(1)

try:
    # Initialize S3 handler with connection_id from CLI argument
    s3_handler = S3Handler(connection_id=args.connection_id)
    
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"❌ Failed to load image: {input_path}")
        sys.exit(1)
        
    binary = (mask > 127).astype(np.uint8)

    # --- Step 2: Morphological cleanup ---
    # Remove small artifacts
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill small gaps
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Step 3: Save cleaned mask locally ---
    cv2.imwrite(output_path, closed * 255)
    print(f"✅ Cleaned mask saved locally: {output_path}")

    # --- Step 4: Upload to S3 ---
    print("\n🌐 UPLOADING TO S3...")
    print("-" * 40)
    
    # Define S3 key with connection_id folder structure
    s3_key = f"outputs/{s3_handler.connection_id}/{output_path}"
    
    # Upload the file to S3
    s3_handler.upload_to_s3(output_path, s3_key)
    
    print(f"✅ Road mask cleaning and upload completed!")

    # --- Step 5: Optional Visualization ---
    if args.visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(binary * 255, cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("Opened (denoise)")
        plt.imshow(opened * 255, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Closed (fill gaps)")
        plt.imshow(closed * 255, cmap="gray")

        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"❌ Error processing road mask: {str(e)}")
    sys.exit(1)