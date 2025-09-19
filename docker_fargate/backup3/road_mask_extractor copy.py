#road_mask_extractor.py
import cv2
import numpy as np
import sys
import os
from skimage import measure

def remove_crosswalks_and_white_areas(img):
    """
    Remove white crosswalk stripes and isolated white areas from road image
    """
    print("🧹 Removing crosswalks and white areas...")
    
    white_threshold = 240
    road_color = (102, 102, 102)  # BGR format
    
    cleaned_img = img.copy()
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    
    # DEBUG: Save original grayscale
    cv2.imwrite("debug_00_original_gray.png", gray)
    print("💾 Saved: debug_00_original_gray.png")
    
    # Find white areas
    white_mask = cv2.inRange(gray, white_threshold, 255)
    
    # DEBUG: Save white mask before processing
    cv2.imwrite("debug_01_white_mask_raw.png", white_mask)
    print("💾 Saved: debug_01_white_mask_raw.png")
    
    # Clean up noise but preserve crosswalk stripes
    kernel = np.ones((2,2), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # DEBUG: Save white mask after morphology
    cv2.imwrite("debug_02_white_mask_morph.png", white_mask)
    print("💾 Saved: debug_02_white_mask_morph.png")
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    # DEBUG: Create visualization of detected components
    debug_components = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    crosswalk_areas_removed = 0
    height, width = cleaned_img.shape[:2]
    
    print(f"🔍 Found {num_labels-1} white components to analyze...")
    
    # Create mask for areas to remove with padding
    removal_mask = np.zeros_like(gray, dtype=np.uint8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        center_x, center_y = centroids[i]
        
        # Check if white area touches image edges (skip if it does)
        touches_edge = (x <= 1 or y <= 1 or 
                       (x + w) >= width-1 or 
                       (y + h) >= height-1)
        
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # DEBUG: Draw bounding box and info for each component
        color = (0, 255, 0) if not touches_edge else (0, 0, 255)
        cv2.rectangle(debug_components, (x, y), (x+w, y+h), color, 1)
        cv2.putText(debug_components, f"A:{area}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv2.putText(debug_components, f"R:{aspect_ratio:.1f}", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        if touches_edge:
            print(f"   Component {i}: SKIP (touches edge) - Area:{area}, Ratio:{aspect_ratio:.1f}")
            continue
        
        is_crosswalk = False
        reason = ""
        
        # More aggressive crosswalk detection
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
        
        # DEBUG: Print decision
        if is_crosswalk:
            print(f"   Component {i}: REMOVE ({reason}) - Area:{area}, Ratio:{aspect_ratio:.1f}")
            cv2.rectangle(debug_components, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(debug_components, "REMOVE", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Add to removal mask with padding for aggressive removal
            component_mask = (labels == i).astype(np.uint8) * 255
            removal_mask = cv2.bitwise_or(removal_mask, component_mask)
            crosswalk_areas_removed += 1
        else:
            print(f"   Component {i}: KEEP - Area:{area}, Ratio:{aspect_ratio:.1f}")
            cv2.rectangle(debug_components, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(debug_components, "KEEP", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # AGGRESSIVE REMOVAL: Apply morphological dilation to remove fragments
    kernel_aggressive = np.ones((3,3), np.uint8)
    removal_mask_dilated = cv2.dilate(removal_mask, kernel_aggressive, iterations=2)
    
    # DEBUG: Save removal masks
    cv2.imwrite("debug_02a_removal_mask.png", removal_mask)
    cv2.imwrite("debug_02b_removal_mask_dilated.png", removal_mask_dilated)
    print("💾 Saved: debug_02a_removal_mask.png and debug_02b_removal_mask_dilated.png")
    
    # Apply aggressive removal
    cleaned_img[removal_mask_dilated == 255] = road_color
    
    # DEBUG: Save component analysis and final result
    cv2.imwrite("debug_03_components_analysis.png", debug_components)
    cv2.imwrite("debug_04_final_cleaned.png", cleaned_img)
    cv2.imwrite("debug_05_difference.png", cv2.absdiff(img, cleaned_img))
    print("💾 Saved: debug_03, debug_04, debug_05")
    
    print(f"✅ Removed {crosswalk_areas_removed} crosswalk/white areas (with aggressive padding)")
    return cleaned_img

def is_road_color(img, tolerance=15):
    """
    Detect actual road colors in RGB space more precisely
    Road colors are blue-gray like #a9b8c8 (169,184,200) or #aab9c9 (170,185,201)
    """
    print("🛣️ Detecting road colors with RGB analysis...")
    
    # Convert BGR to RGB for analysis
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
    
    # Define typical road color characteristics
    # Road colors: blue-gray with B > G > R pattern
    road_mask = (
        # Blue channel should be highest
        (b >= g) & (b >= r) &
        # Green should be higher than red (blue-gray characteristic)
        (g >= r) &
        # Range check for typical road colors
        (r >= 160) & (r <= 190) &  # Red component
        (g >= 175) & (g <= 210) &  # Green component  
        (b >= 190) & (b <= 220) &  # Blue component (highest)
        # Additional check: not too bright (avoid light backgrounds)
        ((r.astype(np.int16) + g.astype(np.int16) + b.astype(np.int16)) < 600)
    )
    
    # DEBUG: Save road color detection
    debug_road_rgb = np.zeros_like(rgb_img)
    debug_road_rgb[road_mask] = rgb_img[road_mask]
    debug_road_bgr = cv2.cvtColor(debug_road_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_06a_road_color_detection.png", debug_road_bgr)
    
    # Also show what we're excluding
    non_road_mask = ~road_mask
    debug_excluded = np.zeros_like(rgb_img)
    debug_excluded[non_road_mask] = rgb_img[non_road_mask]
    debug_excluded_bgr = cv2.cvtColor(debug_excluded, cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_06b_excluded_colors.png", debug_excluded_bgr)
    
    print(f"   🎯 Found {np.sum(road_mask):,} road pixels")
    print(f"   ❌ Excluded {np.sum(non_road_mask):,} non-road pixels")
    
    return road_mask

def remove_obstacles_and_clean_roads_advanced(road_mask_binary, connection_id):
    """
    Enhanced road mask cleaning: removes obstacles, fills holes, and smooths boundaries
    Adapted from local version with docker-style debugging
    """
    print("🚧 Starting enhanced road mask cleaning with boundary smoothing...")
    
    # Input is already a binary mask (white = road, black = background)
    binary = road_mask_binary.copy()
    
    # DEBUG: Save initial state
    cv2.imwrite(f"{connection_id}_debug_10_initial_binary.png", binary)
    print("💾 Saved: debug_10_initial_binary.png")
    
    print("Step 1: Initial cleaning - removing small objects...")
    # 1. Morphological opening to remove small objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # DEBUG: Save after opening
    cv2.imwrite(f"{connection_id}_debug_11_after_opening.png", opened)
    print("💾 Saved: debug_11_after_opening.png")
    
    # 2. Remove small connected components (noise)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    min_area = 300
    
    cleaned = np.zeros_like(binary)
    removed_components = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
        else:
            removed_components += 1
    
    print(f"   🗑️ Removed {removed_components} small components (< {min_area} pixels)")
    
    # DEBUG: Save after component filtering
    cv2.imwrite(f"{connection_id}_debug_12_after_component_filter.png", cleaned)
    print("💾 Saved: debug_12_after_component_filter.png")
    
    print("Step 2: Filling holes and obstacles in roads...")
    # 3. Fill holes in white regions (roads)
    filled = cleaned.copy()
    
    # Find contours of white regions (roads)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled_holes = 0
    # Fill holes in each road region
    for contour in contours:
        area = cv2.contourArea(contour)
        # If this road region is large enough, fill its holes
        if area > 1000:  # Only fill holes in significant road areas
            # Create a mask for this road region
            mask = np.zeros_like(cleaned)
            cv2.fillPoly(mask, [contour], 255)
            filled = cv2.bitwise_or(filled, mask)
            filled_holes += 1
    
    print(f"   🕳️ Filled holes in {filled_holes} road regions")
    
    # DEBUG: Save after hole filling
    cv2.imwrite(f"{connection_id}_debug_13_after_hole_filling.png", filled)
    print("💾 Saved: debug_13_after_hole_filling.png")
    
    print("Step 3: Morphological hole filling...")
    # 4. Use closing to fill gaps and holes
    # Fill horizontal gaps (crosswalks and obstacles)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed_h = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_horizontal)
    
    cv2.imwrite(f"{connection_id}_debug_14a_horizontal_close.png", closed_h)
    print("💾 Saved: debug_14a_horizontal_close.png")
    
    # Fill vertical gaps
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed_v = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kernel_vertical)
    
    cv2.imwrite(f"{connection_id}_debug_14b_vertical_close.png", closed_v)
    print("💾 Saved: debug_14b_vertical_close.png")
    
    # Fill small circular holes (obstacles like cars, poles)
    kernel_circular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed_circular = cv2.morphologyEx(closed_v, cv2.MORPH_CLOSE, kernel_circular)
    
    cv2.imwrite(f"{connection_id}_debug_14c_circular_close.png", closed_circular)
    print("💾 Saved: debug_14c_circular_close.png")
    
    print("Step 4: Initial smoothing...")
    # 5. Initial smoothing
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smoothed = cv2.morphologyEx(closed_circular, cv2.MORPH_CLOSE, kernel_smooth)
    
    cv2.imwrite(f"{connection_id}_debug_15_initial_smoothing.png", smoothed)
    print("💾 Saved: debug_15_initial_smoothing.png")
    
    print("Step 5: Advanced boundary smoothing...")
    # 6. Advanced boundary smoothing techniques
    
    # Method 1: Gaussian blur + re-threshold for smoother edges
    blurred = cv2.GaussianBlur(smoothed, (5, 5), 1.5)
    _, re_thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(f"{connection_id}_debug_16a_gaussian_smooth.png", re_thresholded)
    print("💾 Saved: debug_16a_gaussian_smooth.png")
    
    # Method 2: Additional morphological smoothing with circular kernel
    kernel_round = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    morph_smoothed = cv2.morphologyEx(re_thresholded, cv2.MORPH_OPEN, kernel_round)
    morph_smoothed = cv2.morphologyEx(morph_smoothed, cv2.MORPH_CLOSE, kernel_round)
    
    cv2.imwrite(f"{connection_id}_debug_16b_morph_smooth.png", morph_smoothed)
    print("💾 Saved: debug_16b_morph_smooth.png")
    
    # Method 3: Contour approximation for ultra-smooth boundaries
    contours, _ = cv2.findContours(morph_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create final smoothed result
    final_result = np.zeros_like(binary)
    
    processed_contours = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        # Only process significant contours
        if area > 500:
            # Approximate contour to reduce jaggedness
            epsilon = 0.002 * cv2.arcLength(contour, True)  # Adjust this value for more/less smoothing
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Fill the smoothed contour
            cv2.fillPoly(final_result, [smoothed_contour], 255)
            processed_contours += 1
    
    print(f"   ✨ Smoothed {processed_contours} contours")
    
    cv2.imwrite(f"{connection_id}_debug_17_contour_smoothed.png", final_result)
    print("💾 Saved: debug_17_contour_smoothed.png")
    
    print("Step 6: Final boundary refinement...")
    # 7. Final refinement - light smoothing to eliminate any remaining roughness
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_result = cv2.morphologyEx(final_result, cv2.MORPH_CLOSE, kernel_final)
    final_result = cv2.morphologyEx(final_result, cv2.MORPH_OPEN, kernel_final)
    
    cv2.imwrite(f"{connection_id}_debug_18_final_morph.png", final_result)
    print("💾 Saved: debug_18_final_morph.png")
    
    # Optional: One more light Gaussian blur + threshold for ultra-smooth edges
    final_blurred = cv2.GaussianBlur(final_result, (3, 3), 0.8)
    _, ultra_smooth = cv2.threshold(final_blurred, 127, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(f"{connection_id}_debug_19_ultra_smooth.png", ultra_smooth)
    print("💾 Saved: debug_19_ultra_smooth.png")
    
    print("✅ Enhanced road mask cleaning completed with ultra-smooth boundaries! 🛣️✨")
    return ultra_smooth

def main():
    if len(sys.argv) < 2:
        print("❌ connection_id argument is required")
        sys.exit(1)
    
    connection_id = sys.argv[1]
    
    # Input: Use the cleaned roadmap from map_cleaner.py
    input_path = f"{connection_id}_cleaned_roadmap.png"
    
    # Output: Final road mask
    output_path = f"{connection_id}_road_mask.png"
    
    print(f"🛣️ ROAD MASK EXTRACTOR for connection_id: {connection_id}")
    print(f"📍 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        print(f"   Make sure map_cleaner.py has run first!")
        sys.exit(1)
    
    try:
        # --- Step 0: Load cleaned image ---
        print("📂 Loading cleaned image...")
        img = cv2.imread(input_path)
        if img is None:
            print(f"❌ Error: Could not load image {input_path}")
            sys.exit(1)
        
        cv2.imwrite(f"{connection_id}_debug_original.png", img)
        
        # --- Step 0.5: Remove remaining crosswalks ---
        cleaned_img = remove_crosswalks_and_white_areas(img)
        cv2.imwrite(f"{connection_id}_roadmap_double_cleaned.png", cleaned_img)
        print(f"💾 Saved double-cleaned roadmap: {connection_id}_roadmap_double_cleaned.png")
        
        # --- Step 1: Extract road mask with PRECISE road color detection ---
        print("\n🔍 Extracting road mask with precise color detection...")
        road_mask = is_road_color(cleaned_img)
        
        cv2.imwrite(f"{connection_id}_debug_06_initial_road_mask.png", (road_mask.astype(np.uint8)) * 255)
        
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
        
        cv2.imwrite(f"{connection_id}_debug_07_after_label_removal.png", (road_mask.astype(np.uint8)) * 255)
        
        # --- Step 3: IMPROVED hole filling (fix the boundary issue) ---
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
                print(f"   Filling hole: area={area}, ratio={ratio:.1f}")
            elif touches_boundary:
                print(f"   Skipping hole (touches boundary): area={area}")
            else:
                print(f"   Skipping hole (too large): area={area}")
        
        cv2.imwrite(f"{connection_id}_debug_08_holes_to_remove_fixed.png", (holes_to_remove.astype(np.uint8)) * 255)
        
        # --- Step 4: Create intermediate mask ---
        clean_road_mask = road_mask.copy()
        clean_road_mask[holes_to_remove] = True
        
        # Convert to binary format for advanced cleaning
        intermediate_binary = (clean_road_mask.astype(np.uint8)) * 255
        cv2.imwrite(f"{connection_id}_debug_09_before_advanced_cleaning.png", intermediate_binary)
        print("💾 Saved: debug_09_before_advanced_cleaning.png")
        
        # --- Step 5: Apply advanced cleaning from local version ---
        print("\n🚀 Applying advanced road mask cleaning...")
        final_cleaned_mask = remove_obstacles_and_clean_roads_advanced(intermediate_binary, connection_id)
        
        # --- Step 6: Save final result ---
        cv2.imwrite(output_path, final_cleaned_mask)
        print(f"✅ Saved: {output_path}")
        
        # Also save without connection_id prefix for compatibility with existing scripts
        cv2.imwrite("final_road_mask.png", final_cleaned_mask)
        print("✅ Saved: final_road_mask.png (for compatibility)")
        
        # Save the ultra-clean version as well
        cv2.imwrite("final_road_mask_cleaned.png", final_cleaned_mask)
        print("✅ Saved: final_road_mask_cleaned.png (ultra-clean version)")
        
        print(f"\n📊 SUMMARY:")
        print(f"   - Fixed road color detection to exclude #f1f3f4")
        print(f"   - Added precise RGB-based road detection")
        print(f"   - Applied advanced obstacle removal and boundary smoothing")
        print(f"   - Enhanced hole filling with morphological operations")
        print(f"   - Ultra-smooth boundary processing with contour approximation")
        print(f"   - Check debug files 10-19 to see advanced cleaning steps")
        print(f"   - Road colors: blue-gray pattern (B≥G≥R) in range 160-220")
        print(f"🏆 ADVANCED ROAD MASK EXTRACTION COMPLETE!")
        
    except Exception as e:
        print(f"💥 Error during road mask extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()