# 00_01_extract_road_mask_from_map9.py

import cv2
import numpy as np
from skimage import measure
import os
import sys
from s3_handler import S3Handler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_crosswalks_and_white_areas(img, config=None):
    """
    Remove white crosswalk stripes and isolated white areas from road image
    
    Args:
        img: OpenCV image (BGR format)
        config: Optional configuration dictionary (uses sensible defaults if None)
    
    Returns:
        Cleaned image with crosswalks replaced by road color
    """
    # Simple defaults - no configuration needed
    white_threshold = 240
    min_area = 20
    max_area = 8000
    road_color = (102, 102, 102)  # BGR format - matches 0x666666
    
    # Override defaults if config provided (optional)
    if config:
        white_threshold = config.get('white_threshold', white_threshold)
        min_area = config.get('min_area', min_area)
        max_area = config.get('max_area', max_area)
        road_color = config.get('road_color', road_color)
    
    logger.info("Starting crosswalk removal...")
    
    # Work on a copy
    cleaned_img = img.copy()
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    
    # Find white areas
    white_mask = cv2.inRange(gray, white_threshold, 255)
    
    # Clean up noise
    kernel = np.ones((3,3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    # Remove crosswalk-like areas
    crosswalk_areas_removed = 0
    height, width = cleaned_img.shape[:2]
    
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        center_x, center_y = centroids[i]
        
        # Simple filters for crosswalk detection
        is_crosswalk = False
        
        # Size filter
        if min_area < area < max_area:
            is_crosswalk = True
            
        # Aspect ratio filter (rectangular shapes)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if area > 50 and 1.5 < aspect_ratio < 10:
            is_crosswalk = True
            
        # Position filter (center areas of image)
        if 0.2 * width < center_x < 0.8 * width and 0.2 * height < center_y < 0.8 * height:
            if area > 30:
                is_crosswalk = True
        
        # Replace with road color
        if is_crosswalk:
            component_mask = (labels == i).astype(np.uint8) * 255
            cleaned_img[component_mask == 255] = road_color
            crosswalk_areas_removed += 1
    
    logger.info(f"✅ Removed {crosswalk_areas_removed} crosswalk/white areas")
    return cleaned_img

def main():
    input_path = "roadmap.png"
    output_filename = "final_road_mask.png"

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    try:
        # Initialize S3 handler
        s3_handler = S3Handler()
        
        # --- Step 0: Remove crosswalks ---
        print("🧹 CLEANING CROSSWALKS...")
        print("-" * 30)
        
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Remove crosswalks (uses sensible defaults)
        cleaned_img = remove_crosswalks_and_white_areas(img)
        
        # --- Step 1: Convert to grayscale ---
        print("\n🔍 EXTRACTING ROAD MASK...")
        print("-" * 30)
        
        gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)

        # --- Step 2: Extract road mask ---
        road_mask = (gray > 100) & (gray < 200)

        # --- Step 2.5: Remove white labels/icons ---
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

        # --- Step 3: Fill holes ---
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

        # --- Step 4: Create final mask ---
        clean_road_mask = road_mask.copy()
        clean_road_mask[holes_to_remove] = True

        # --- Step 5: Save and upload ---
        save_mask = (clean_road_mask.astype(np.uint8)) * 255
        cv2.imwrite(output_filename, save_mask)
        print(f"✅ Saved locally: {output_filename}")

        print("\n🌐 UPLOADING TO S3...")
        print("-" * 20)
        
        s3_key = f"outputs/{s3_handler.connection_id}/{output_filename}"
        s3_handler.upload_to_s3(output_filename, s3_key)
        
        print(f"✅ Road mask processing completed!")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()