#road_mask_cleaner.py

import cv2
import numpy as np
import sys
import os

def remove_obstacles_and_clean_roads_advanced(img_path, output_path, connection_id):
    """
    Enhanced road mask cleaning: removes obstacles, fills holes, and smooths boundaries
    Adapted from local version with docker-style debugging and error handling
    """
    print("🚀 Starting enhanced road mask cleaning with boundary smoothing...")
    
    # Load your road mask (white = road, black = background)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Error: Could not load image from {img_path}")
        return None
    
    # Save original for debugging
    cv2.imwrite(f"{connection_id}_debug_cleaning_00_original.png", img)
    print("💾 Saved: debug_cleaning_00_original.png")
    
    # Threshold just in case (ensure it's binary)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # DEBUG: Save binary version
    cv2.imwrite(f"{connection_id}_debug_cleaning_01_binary.png", binary)
    print("💾 Saved: debug_cleaning_01_binary.png")
    
    print("🔧 Step 1: Initial cleaning - removing small objects...")
    # 1. Morphological opening to remove small objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # DEBUG: Save after opening
    cv2.imwrite(f"{connection_id}_debug_cleaning_02_opened.png", opened)
    print("💾 Saved: debug_cleaning_02_opened.png")
    
    # 2. Remove small connected components (noise)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    min_area = 300
    
    cleaned = np.zeros_like(binary)
    removed_components = 0
    kept_components = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
            kept_components += 1
        else:
            removed_components += 1
    
    print(f"   ✅ Removed {removed_components} small components (< {min_area} pixels)")
    print(f"   ✅ Kept {kept_components} large components")
    
    # DEBUG: Save after component filtering
    cv2.imwrite(f"{connection_id}_debug_cleaning_03_components_filtered.png", cleaned)
    print("💾 Saved: debug_cleaning_03_components_filtered.png")
    
    print("🕳️ Step 2: Filling holes and obstacles in roads...")
    # 3. Fill holes in white regions (roads)
    filled = cleaned.copy()
    
    # Find contours of white regions (roads)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled_regions = 0
    # Fill holes in each road region
    for contour in contours:
        area = cv2.contourArea(contour)
        # If this road region is large enough, fill its holes
        if area > 1000:  # Only fill holes in significant road areas
            # Create a mask for this road region
            mask = np.zeros_like(cleaned)
            cv2.fillPoly(mask, [contour], 255)
            filled = cv2.bitwise_or(filled, mask)
            filled_regions += 1
    
    print(f"   🕳️ Filled holes in {filled_regions} road regions")
    
    # DEBUG: Save after hole filling
    cv2.imwrite(f"{connection_id}_debug_cleaning_04_holes_filled.png", filled)
    print("💾 Saved: debug_cleaning_04_holes_filled.png")
    
    print("🦓 Step 3: Morphological hole filling...")
    # 4. Use closing to fill gaps and holes
    # Fill horizontal gaps (crosswalks and obstacles)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed_h = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_horizontal)
    
    # DEBUG: Save horizontal closing
    cv2.imwrite(f"{connection_id}_debug_cleaning_05a_horizontal_close.png", closed_h)
    print("💾 Saved: debug_cleaning_05a_horizontal_close.png")
    
    # Fill vertical gaps
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed_v = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kernel_vertical)
    
    # DEBUG: Save vertical closing
    cv2.imwrite(f"{connection_id}_debug_cleaning_05b_vertical_close.png", closed_v)
    print("💾 Saved: debug_cleaning_05b_vertical_close.png")
    
    # Fill small circular holes (obstacles like cars, poles)
    kernel_circular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed_circular = cv2.morphologyEx(closed_v, cv2.MORPH_CLOSE, kernel_circular)
    
    # DEBUG: Save circular closing
    cv2.imwrite(f"{connection_id}_debug_cleaning_05c_circular_close.png", closed_circular)
    print("💾 Saved: debug_cleaning_05c_circular_close.png")
    
    print("✨ Step 4: Initial smoothing...")
    # 5. Initial smoothing
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smoothed = cv2.morphologyEx(closed_circular, cv2.MORPH_CLOSE, kernel_smooth)
    
    # DEBUG: Save initial smoothing
    cv2.imwrite(f"{connection_id}_debug_cleaning_06_initial_smooth.png", smoothed)
    print("💾 Saved: debug_cleaning_06_initial_smooth.png")
    
    print("🎨 Step 5: Advanced boundary smoothing...")
    # 6. Advanced boundary smoothing techniques
    
    # Method 1: Gaussian blur + re-threshold for smoother edges
    blurred = cv2.GaussianBlur(smoothed, (5, 5), 1.5)
    _, re_thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # DEBUG: Save Gaussian smoothing
    cv2.imwrite(f"{connection_id}_debug_cleaning_07a_gaussian_smooth.png", re_thresholded)
    print("💾 Saved: debug_cleaning_07a_gaussian_smooth.png")
    
    # Method 2: Additional morphological smoothing with circular kernel
    kernel_round = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    morph_smoothed = cv2.morphologyEx(re_thresholded, cv2.MORPH_OPEN, kernel_round)
    morph_smoothed = cv2.morphologyEx(morph_smoothed, cv2.MORPH_CLOSE, kernel_round)
    
    # DEBUG: Save morphological smoothing
    cv2.imwrite(f"{connection_id}_debug_cleaning_07b_morph_smooth.png", morph_smoothed)
    print("💾 Saved: debug_cleaning_07b_morph_smooth.png")
    
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
    
    print(f"   📐 Processed {processed_contours} contours for smoothing")
    
    # DEBUG: Save contour smoothing
    cv2.imwrite(f"{connection_id}_debug_cleaning_08_contour_smooth.png", final_result)
    print("💾 Saved: debug_cleaning_08_contour_smooth.png")
    
    print("🏁 Step 6: Final boundary refinement...")
    # 7. Final refinement - light smoothing to eliminate any remaining roughness
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_result = cv2.morphologyEx(final_result, cv2.MORPH_CLOSE, kernel_final)
    final_result = cv2.morphologyEx(final_result, cv2.MORPH_OPEN, kernel_final)
    
    # DEBUG: Save final morphology
    cv2.imwrite(f"{connection_id}_debug_cleaning_09a_final_morph.png", final_result)
    print("💾 Saved: debug_cleaning_09a_final_morph.png")
    
    # Optional: One more light Gaussian blur + threshold for ultra-smooth edges
    final_blurred = cv2.GaussianBlur(final_result, (3, 3), 0.8)
    _, ultra_smooth = cv2.threshold(final_blurred, 127, 255, cv2.THRESH_BINARY)
    
    # DEBUG: Save ultra-smooth result
    cv2.imwrite(f"{connection_id}_debug_cleaning_09b_ultra_smooth.png", ultra_smooth)
    print("💾 Saved: debug_cleaning_09b_ultra_smooth.png")
    
    # Save the result
    cv2.imwrite(output_path, ultra_smooth)
    print(f"✅ Ultra-smooth road mask saved to: {output_path}")
    
    return ultra_smooth

def remove_crosswalk_gaps_and_clean_roads(img_path, output_path, connection_id):
    """
    Complete road mask cleaning: removes small objects AND fills crosswalk gaps
    Enhanced version with better debugging
    """
    print("🔧 Using comprehensive crosswalk gap filling method...")
    
    # Load your road mask (white = road, black = background)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Error: Could not load image from {img_path}")
        return None
    
    # Threshold just in case (ensure it's binary)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    print("🔧 Step 1: Removing small objects...")
    # 1. Morphological opening to remove small objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 2. Remove small connected components (e.g., noise)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    min_area = 300  # Tune this as needed
    
    cleaned = np.zeros_like(binary)
    kept_components = 0
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
            kept_components += 1
    
    removed_components = (num_labels - 1) - kept_components
    print(f"   ✅ Removed {removed_components} small components")
    print(f"   ✅ Kept {kept_components} large components")
    
    print("🦓 Step 2: Filling crosswalk gaps...")
    # 3. Fill crosswalk gaps using morphological closing
    
    # Fill horizontal gaps (most crosswalk stripes are horizontal)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))  # Wide vertical kernel
    closed_h = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_horizontal)
    
    # Fill vertical gaps (for vertical crosswalk stripes)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))    # Wide horizontal kernel
    closed_v = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kernel_vertical)
    
    # 4. Final smoothing to remove any remaining small gaps
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_result = cv2.morphologyEx(closed_v, cv2.MORPH_CLOSE, kernel_smooth)
    
    print("🧹 Step 3: Final cleanup...")
    # 5. Optional: One more connected component analysis to remove any artifacts created
    num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(final_result, connectivity=8)
    min_area_final = 500  # Slightly larger threshold for final cleanup
    
    final_cleaned = np.zeros_like(binary)
    final_components = 0
    for i in range(1, num_labels_final):
        if stats_final[i, cv2.CC_STAT_AREA] >= min_area_final:
            final_cleaned[labels_final == i] = 255
            final_components += 1
    
    print(f"   ✅ Kept {final_components} road components after final cleanup")
    
    # Save the result
    cv2.imwrite(output_path, final_cleaned)
    print(f"✅ Comprehensive cleaned road mask saved to: {output_path}")
    
    return final_cleaned

def simple_crosswalk_fill(img_path, output_path, connection_id):
    """
    Simpler version focusing mainly on filling crosswalk gaps
    Enhanced with better error reporting
    """
    print("🔧 Using simple crosswalk fill method...")
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Could not load image from {img_path}")
        return None
        
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Basic cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    min_area = 300
    
    cleaned = np.zeros_like(binary)
    kept_components = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
            kept_components += 1
    
    print(f"   ✅ Kept {kept_components} components after basic cleaning")
    
    # Add crosswalk gap filling
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))  # Fill horizontal gaps
    final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_fill)
    
    cv2.imwrite(output_path, final)
    print(f"✅ Simple cleaned road mask saved to: {output_path}")
    return final

def main():
    if len(sys.argv) < 2:
        print("❌ connection_id argument is required")
        sys.exit(1)
    
    connection_id = sys.argv[1]
    
    # Input: Road mask from road_mask_extractor.py
    input_path = "final_road_mask.png"  # Generic name for compatibility
    
    # Outputs: Both connection-specific and generic names
    output_path = f"{connection_id}_final_road_mask_cleaned.png"
    output_generic = "final_road_mask_cleaned.png"  # For compatibility with next steps
    
    print(f"🧹 ROAD MASK CLEANER for connection_id: {connection_id}")
    print(f"📍 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print(f"📁 Generic output: {output_generic}")
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        print(f"   Make sure road_mask_extractor.py has run first!")
        sys.exit(1)
    
    try:
        print("🚀 Starting advanced road mask cleaning...")
        
        # Save debug version of original
        original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if original is not None:
            cv2.imwrite(f"{connection_id}_debug_before_cleaning.png", original)
            print(f"💾 Saved debug original: {connection_id}_debug_before_cleaning.png")
        
        # Use the ADVANCED version first (from local version)
        print("📋 Trying advanced enhanced cleaning method...")
        result = remove_obstacles_and_clean_roads_advanced(input_path, output_path, connection_id)
        
        if result is not None:
            # Also save generic version for pipeline compatibility
            cv2.imwrite(output_generic, result)
            
            # Save debug comparison
            if original is not None:
                combined = np.hstack([original, result])
                cv2.imwrite(f"{connection_id}_debug_before_after_cleaning.png", combined)
                print(f"💾 Saved before/after comparison: {connection_id}_debug_before_after_cleaning.png")
            
            # Calculate detailed stats
            original_pixels = np.sum(original == 255) if original is not None else 0
            cleaned_pixels = np.sum(result == 255)
            
            print(f"\n📊 ADVANCED CLEANING SUMMARY:")
            if original is not None:
                print(f"   🔵 Original road pixels: {original_pixels:,}")
                print(f"   🟢 Cleaned road pixels: {cleaned_pixels:,}")
                pixel_change = cleaned_pixels - original_pixels
                if pixel_change > 0:
                    print(f"   ⬆️  Pixel change: +{pixel_change:,} (gaps filled)")
                    improvement = (pixel_change / original_pixels) * 100
                    print(f"   📈 Road coverage improved by: {improvement:.1f}%")
                elif pixel_change < 0:
                    print(f"   ⬇️  Pixel change: {pixel_change:,} (noise removed)")
                    cleanup = abs(pixel_change / original_pixels) * 100
                    print(f"   🧹 Noise reduction: {cleanup:.1f}%")
                else:
                    print(f"   ➡️  Pixel change: {pixel_change:,} (perfectly balanced)")
            else:
                print(f"   🟢 Final road pixels: {cleaned_pixels:,}")
            
            print(f"   ✨ Applied ultra-smooth boundary processing")
            print(f"   🕳️ Filled obstacles and holes in road surface")
            print(f"   📐 Used contour approximation for smooth boundaries")
            print(f"🏆 ADVANCED ROAD MASK CLEANING COMPLETE!")
            
        else:
            print("❌ Advanced cleaning failed, trying comprehensive method...")
            result = remove_crosswalk_gaps_and_clean_roads(input_path, output_path, connection_id)
            if result is not None:
                cv2.imwrite(output_generic, result)
                print("✅ Comprehensive cleaning completed successfully")
            else:
                print("❌ Comprehensive cleaning failed, trying simple method...")
                result = simple_crosswalk_fill(input_path, output_path, connection_id)
                if result is not None:
                    cv2.imwrite(output_generic, result)
                    print("✅ Simple cleaning completed successfully")
                else:
                    print("❌ All cleaning methods failed!")
                    sys.exit(1)
                
    except Exception as e:
        print(f"💥 Error during road mask cleaning: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback to comprehensive method
        print("🔄 Attempting fallback to comprehensive cleaning method...")
        try:
            result = remove_crosswalk_gaps_and_clean_roads(input_path, output_path, connection_id)
            if result is not None:
                cv2.imwrite(output_generic, result)
                print("✅ Comprehensive fallback cleaning completed successfully")
            else:
                # Final fallback to simple method
                print("🔄 Attempting final fallback to simple cleaning method...")
                result = simple_crosswalk_fill(input_path, output_path, connection_id)
                if result is not None:
                    cv2.imwrite(output_generic, result)
                    print("✅ Simple fallback cleaning completed successfully")
                else:
                    sys.exit(1)
        except Exception as e2:
            print(f"💥 All fallback methods failed: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    main()