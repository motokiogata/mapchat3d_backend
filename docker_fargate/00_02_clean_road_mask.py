import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

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

    # --- Step 3: Save cleaned mask ---
    cv2.imwrite(output_path, closed * 255)
    print(f"✅ Cleaned mask saved: {output_path}")

    # --- Step 4: Optional Visualization ---
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