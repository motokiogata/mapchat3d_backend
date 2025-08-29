# 00_01_extract_road_mask_from_map9.py

import cv2
import numpy as np
from skimage import measure
import os
import sys
from s3_handler import S3Handler
import logging

# Set up logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_crosswalk_stripes_advanced(img, mask_region, stats_info):
    """
    Advanced crosswalk stripe detection using multiple methods
    """
    x, y, w, h = stats_info
    
    # Extract the region of interest
    roi = img[y:y+h, x:x+w]
    roi_mask = mask_region[y:y+h, x:x+w]
    
    if roi.size == 0:
        return False
    
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Edge-based stripe detection
    edges = cv2.Canny(roi_gray, 50, 150)
    
    # Look for parallel lines (characteristic of crosswalks)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                           minLineLength=min(w,h)//4, maxLineGap=10)
    
    parallel_lines = 0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        # Count lines with similar angles (parallel lines)
        angles = np.array(angles)
        for angle in angles:
            similar_angles = np.sum(np.abs(angles - angle) < 15)  # Within 15 degrees
            parallel_lines = max(parallel_lines, similar_angles)
    
    # Method 2: Frequency domain analysis for regular patterns
    # Apply FFT to detect repetitive patterns
    if w > 20 and h > 20:
        # Sample middle row/column for pattern analysis
        mid_row = roi_gray[h//2, :]
        mid_col = roi_gray[:, w//2]
        
        # Check for periodic patterns in intensity
        row_fft = np.fft.fft(mid_row)
        col_fft = np.fft.fft(mid_col)
        
        # Look for strong frequency components (indicating regular stripes)
        row_power = np.abs(row_fft[1:len(row_fft)//2])  # Exclude DC component
        col_power = np.abs(col_fft[1:len(col_fft)//2])
        
        max_row_power = np.max(row_power) if len(row_power) > 0 else 0
        max_col_power = np.max(col_power) if len(col_power) > 0 else 0
        
        # Strong periodic pattern indicates crosswalk
        has_pattern = max_row_power > np.mean(row_power) * 3 or max_col_power > np.mean(col_power) * 3
    else:
        has_pattern = False
    
    # Method 3: Variance-based stripe detection
    variance_threshold = 1000  # Adjust based on your images
    
    # Check row-wise variance (horizontal stripes)
    row_variances = []
    for i in range(0, h, max(1, h//10)):  # Sample every 10th row
        if i < h:
            row_var = np.var(roi_gray[i, :])
            row_variances.append(row_var)
    
    # Check column-wise variance (vertical stripes)  
    col_variances = []
    for j in range(0, w, max(1, w//10)):  # Sample every 10th column
        if j < w:
            col_var = np.var(roi_gray[:, j])
            col_variances.append(col_var)
    
    high_variance_rows = np.sum(np.array(row_variances) > variance_threshold)
    high_variance_cols = np.sum(np.array(col_variances) > variance_threshold)
    
    # Decision logic: Multiple indicators suggest crosswalk
    crosswalk_score = 0
    
    if parallel_lines >= 3:  # At least 3 parallel lines
        crosswalk_score += 2
    
    if has_pattern:  # Strong periodic pattern in FFT
        crosswalk_score += 2
        
    if high_variance_rows >= 3 or high_variance_cols >= 3:  # High variance in multiple rows/cols
        crosswalk_score += 1
    
    # Additional size and aspect ratio checks
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    area = w * h
    
    if 2.0 < aspect_ratio < 8.0 and 500 < area < 50000:  # Reasonable crosswalk dimensions
        crosswalk_score += 1
    
    return crosswalk_score >= 3  # Require at least 3 positive indicators

def remove_crosswalks_and_white_areas_enhanced(img, config=None):
    """
    Enhanced crosswalk removal with better stripe detection
    """
    white_threshold = 200  # Lowered to catch more crosswalk whites
    road_color = (102, 102, 102)  # BGR format
    
    if config:
        white_threshold = config.get('white_threshold', white_threshold)
        road_color = config.get('road_color', road_color)
    
    logger.info("Starting enhanced crosswalk removal...")
    
    cleaned_img = img.copy()
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Direct crosswalk pattern detection (NEW)
    crosswalk_mask = detect_crosswalk_patterns_directly(cleaned_img)
    
    # Method 2: Enhanced white area analysis
    white_mask = cv2.inRange(gray, white_threshold, 255)
    
    # Slightly more aggressive morphology to connect stripe segments
    kernel = np.ones((3,3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    removal_mask = np.zeros_like(gray, dtype=np.uint8)
    crosswalk_areas_removed = 0
    height, width = cleaned_img.shape[:2]
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        center_x, center_y = centroids[i]
        
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # Use enhanced stripe detection
        component_mask = (labels == i).astype(np.uint8) * 255
        is_crosswalk = detect_crosswalk_stripes_advanced(img, component_mask, (x, y, w, h))
        
        # Fallback to original logic for obvious cases
        if not is_crosswalk:
            if 100 < area < 20000 and 1.5 < aspect_ratio < 12.0:
                # Additional check for crosswalk-like areas
                center_distance_from_edge = min(center_x, center_y, width - center_x, height - center_y)
                if center_distance_from_edge > 50:  # Not too close to edges
                    is_crosswalk = True
        
        if is_crosswalk:
            removal_mask = cv2.bitwise_or(removal_mask, component_mask)
            crosswalk_areas_removed += 1
            logger.info(f"Removed crosswalk component: Area={area}, AspectRatio={aspect_ratio:.1f}")
    
    # Combine with direct pattern detection
    removal_mask = cv2.bitwise_or(removal_mask, crosswalk_mask)
    
    # Apply aggressive dilation to remove fragments
    kernel_aggressive = np.ones((5,5), np.uint8)  # Increased kernel size
    removal_mask_dilated = cv2.dilate(removal_mask, kernel_aggressive, iterations=3)  # More iterations
    
    # Apply removal
    cleaned_img[removal_mask_dilated == 255] = road_color
    
    logger.info(f"✅ Enhanced removal: {crosswalk_areas_removed} areas + direct pattern detection")
    return cleaned_img

def detect_crosswalk_patterns_directly(img):
    """
    Direct crosswalk pattern detection using template matching and morphology
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create crosswalk detection mask
    crosswalk_mask = np.zeros_like(gray, dtype=np.uint8)
    
    # Method 1: Morphological operations to detect stripe patterns
    # Horizontal crosswalk detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    horizontal_detected = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, horizontal_kernel)
    
    # Vertical crosswalk detection  
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    vertical_detected = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, vertical_kernel)
    
    # Combine both directions
    stripe_pattern = cv2.bitwise_or(horizontal_detected, vertical_detected)
    
    # Threshold to get strong stripe patterns
    _, stripe_mask = cv2.threshold(stripe_pattern, 30, 255, cv2.THRESH_BINARY)
    
    # Clean up the mask
    cleanup_kernel = np.ones((3,3), np.uint8)
    stripe_mask = cv2.morphologyEx(stripe_mask, cv2.MORPH_OPEN, cleanup_kernel, iterations=1)
    stripe_mask = cv2.morphologyEx(stripe_mask, cv2.MORPH_CLOSE, cleanup_kernel, iterations=2)
    
    return stripe_mask

def detect_crosswalk_in_edge_area(img, component_mask, stats_info, labels, label_id):
    """
    Detect if an edge-touching area contains crosswalk stripes
    Only for areas that touch edges (buildings/sidewalks)
    """
    x, y, w, h = stats_info
    
    # Only check reasonably sized rectangular areas
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    area = w * h
    
    # Must be crosswalk-sized and rectangular
    if not (500 < area < 20000 and 2.0 < aspect_ratio < 8.0):
        return False
    
    # Extract the component region
    component_region = (labels == label_id).astype(np.uint8) * 255
    
    # Look for stripe patterns by checking intensity variations
    roi = img[y:max(1, y):y+h, x:max(1, x):x+w]
    roi_mask = component_region[y:max(1, y):y+h, x:max(1, x):x+w]
    
    if roi.size == 0:
        return False
    
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Method: Look for regular alternating patterns (crosswalk stripes)
    stripe_detected = False
    
    # Check horizontal stripes (sample multiple rows)
    for row_offset in [h//4, h//2, 3*h//4]:
        if row_offset < roi_gray.shape[0]:
            row = roi_gray[row_offset, :]
            if len(row) > 10:
                # Look for alternating bright/dark pattern
                diffs = np.abs(np.diff(row.astype(np.int16)))
                strong_changes = np.sum(diffs > 100)  # Strong brightness changes
                if strong_changes > len(row) * 0.2:  # At least 20% of pixels have strong changes
                    stripe_detected = True
                    break
    
    # Check vertical stripes if horizontal didn't work
    if not stripe_detected:
        for col_offset in [w//4, w//2, 3*w//4]:
            if col_offset < roi_gray.shape[1]:
                col = roi_gray[:, col_offset]
                if len(col) > 10:
                    diffs = np.abs(np.diff(col.astype(np.int16)))
                    strong_changes = np.sum(diffs > 100)
                    if strong_changes > len(col) * 0.2:
                        stripe_detected = True
                        break
    
    return stripe_detected

def remove_crosswalks_and_white_areas(img, config=None):
    """
    Remove white crosswalk stripes and isolated white areas from road image
    Now includes detection of crosswalks within building-connected areas
    
    Args:
        img: OpenCV image (BGR format)
        config: Optional configuration dictionary (uses sensible defaults if None)
    
    Returns:
        Cleaned image with crosswalks replaced by road color
    """
    # Simple defaults - no configuration needed
    white_threshold = 240
    road_color = (102, 102, 102)  # BGR format - matches 0x666666
    
    # Override defaults if config provided (optional)
    if config:
        white_threshold = config.get('white_threshold', white_threshold)
        road_color = config.get('road_color', road_color)
    
    logger.info("Starting crosswalk removal...")
    
    # Work on a copy
    cleaned_img = img.copy()
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    
    # Find white areas
    white_mask = cv2.inRange(gray, white_threshold, 255)
    
    # Clean up noise but preserve crosswalk stripes
    kernel = np.ones((2,2), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    crosswalk_areas_removed = 0
    height, width = cleaned_img.shape[:2]
    
    logger.info(f"Found {num_labels-1} white components to analyze...")
    
    # Create mask for areas to remove with padding
    removal_mask = np.zeros_like(gray, dtype=np.uint8)
    
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        center_x, center_y = centroids[i]
        
        # Check if white area touches image edges
        touches_edge = (x <= 1 or y <= 1 or 
                       (x + w) >= width-1 or 
                       (y + h) >= height-1)
        
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        is_crosswalk = False
        reason = ""
        
        # ORIGINAL LOGIC for isolated areas (non-edge-touching)
        if not touches_edge:
            if 15 < area < 15000:
                is_crosswalk = True
                reason = "size_filter"
            
            if aspect_ratio > 2.0 and area > 50:
                is_crosswalk = True
                reason = "stripe_filter"
                
            if 0.5 < aspect_ratio < 2.0 and 100 < area < 5000:
                is_crosswalk = True
                reason = "square_filter"
            
            if area > 200 and aspect_ratio > 1.2:
                if (0.1 * width < center_x < 0.9 * width and 
                    0.1 * height < center_y < 0.9 * height):
                    is_crosswalk = True
                    reason = "large_crosswalk"
        
        # BONUS: Check edge-touching areas for crosswalk patterns
        elif touches_edge:
            has_crosswalk_pattern = detect_crosswalk_in_edge_area(img, labels == i, (x, y, w, h), labels, i)
            if has_crosswalk_pattern:
                is_crosswalk = True
                reason = "edge_crosswalk_stripes"
                logger.info(f"BONUS: Found crosswalk stripes in building-connected area!")
        
        # Apply removal
        if is_crosswalk:
            logger.info(f"Component {i}: REMOVE ({reason}) - Area:{area}, Ratio:{aspect_ratio:.1f}")
            
            # Add to removal mask
            component_mask = (labels == i).astype(np.uint8) * 255
            removal_mask = cv2.bitwise_or(removal_mask, component_mask)
            crosswalk_areas_removed += 1
        else:
            if touches_edge:
                logger.debug(f"Component {i}: SKIP (touches edge) - Area:{area}, Ratio:{aspect_ratio:.1f}")
            else:
                logger.debug(f"Component {i}: KEEP - Area:{area}, Ratio:{aspect_ratio:.1f}")
    
    # AGGRESSIVE REMOVAL: Apply morphological dilation to remove fragments
    kernel_aggressive = np.ones((3,3), np.uint8)
    removal_mask_dilated = cv2.dilate(removal_mask, kernel_aggressive, iterations=2)
    
    # Apply aggressive removal
    cleaned_img[removal_mask_dilated == 255] = road_color
    
    logger.info(f"✅ Removed {crosswalk_areas_removed} crosswalk/white areas (with aggressive padding)")
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
        
        # Remove crosswalks with improved detection (uses sensible defaults)
        cleaned_img = remove_crosswalks_and_white_areas_enhanced(img)
        
        # --- Step 1: Convert to grayscale ---
        print("\n🔍 EXTRACTING ROAD MASK...")
        print("-" * 30)
        
        gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)

        # --- Step 2: Extract road mask ---
        road_mask = (gray > 100) & (gray < 200)

        # --- Step 2.5: Remove white labels/icons inside road ---
        print("🔧 Removing remaining white labels/icons...")
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

        # --- Step 3: IMPROVED hole filling (avoiding boundaries) ---
        print("🕳️ Finding and filling holes (avoiding boundaries)...")
        inverse = ~road_mask
        labels = measure.label(inverse)
        props = measure.regionprops(labels)

        holes_to_remove = np.zeros_like(road_mask, dtype=bool)
        height, width = road_mask.shape

        for p in props:
            area = p.area
            major = p.major_axis_length
            minor = p.minor_axis_length + 1e-5
            ratio = major / minor
            
            # Get bounding box of the hole
            bbox = p.bbox  # (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = bbox
            
            # Check if hole touches image boundaries (if so, don't fill it)
            touches_boundary = (min_row <= 2 or min_col <= 2 or 
                               max_row >= height-2 or max_col >= width-2)
            
            # Only fill small holes that don't touch boundaries
            if not touches_boundary and area < 8000 and ratio < 4.0:  # Reduced from 12000 to 8000
                holes_to_remove[labels == p.label] = True
                logger.info(f"Filling hole: area={area}, ratio={ratio:.1f}")
            elif touches_boundary:
                logger.debug(f"Skipping hole (touches boundary): area={area}")
            else:
                logger.debug(f"Skipping hole (too large): area={area}")

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
        print(f"   - Enhanced crosswalk removal (including building-connected crosswalks)")
        print(f"   - Boundary-aware hole filling")
        print(f"   - Aggressive fragment removal")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()