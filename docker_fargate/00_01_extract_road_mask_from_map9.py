# 00_01_extract_road_mask_from_map9.py

import cv2
import numpy as np
from skimage import measure
import os
import sys

def main():
    input_path = "roadmap.png"

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    try:
        # --- Step 1: Load and grayscale ---
        img = cv2.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Step 2: Extract road mask ---
        road_mask = (gray > 100) & (gray < 200)

        # === Step 2.5: Remove white labels/icons inside road ===
        white_labels = measure.label(road_mask)
        white_props = measure.regionprops(white_labels)

        label_removal_mask = np.zeros_like(road_mask, dtype=bool)
        for p in white_props:
            area = p.area
            major = p.major_axis_length
            minor = p.minor_axis_length + 1e-5
            ratio = major / minor
            if area < 3000 and ratio < 3.0:
                label_removal_mask[white_labels == p.label] = True

        road_mask[label_removal_mask] = False

        # --- Step 3: Invert mask to find dark holes ---
        inverse = ~road_mask
        labels = measure.label(inverse)
        props = measure.regionprops(labels)

        holes_to_remove = np.zeros_like(road_mask, dtype=bool)
        for p in props:
            area = p.area
            major = p.major_axis_length
            minor = p.minor_axis_length + 1e-5
            ratio = major / minor
            if area < 12000 and ratio < 4.0:
                holes_to_remove[labels == p.label] = True

        # --- Step 4: Subtract bugs from road mask ---
        clean_road_mask = road_mask.copy()
        clean_road_mask[holes_to_remove] = True

        # --- Step 5: Save final mask ---
        save_mask = (clean_road_mask.astype(np.uint8)) * 255
        cv2.imwrite("final_road_mask.png", save_mask)
        print("✅ Saved: final_road_mask.png")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
