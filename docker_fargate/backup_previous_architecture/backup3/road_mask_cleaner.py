#road_mask_cleaner.py

import cv2
import numpy as np
import sys
import os

def remove_crosswalk_gaps_and_clean_roads(img_path, output_path):
    """
    Complete road mask cleaning: removes small objects AND fills crosswalk gaps
    """
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
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    
    removed_components = num_labels - 1 - np.sum([1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area])
    print(f"   ✅ Removed {removed_components} small components")
    
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
    print(f"✅ Cleaned road mask saved to: {output_path}")
    
    return final_cleaned

def simple_crosswalk_fill(img_path, output_path):
    """
    Simpler version focusing mainly on filling crosswalk gaps
    """
    print("🔧 Using simple crosswalk fill method...")
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Could not load image from {img_path}")
        return None
        
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Your original cleaning
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
        print("🚀 Starting road mask cleaning...")
        
        # Save debug version of original
        original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if original is not None:
            cv2.imwrite(f"{connection_id}_debug_before_cleaning.png", original)
            print(f"💾 Saved debug original: {connection_id}_debug_before_cleaning.png")
        
        # Use the comprehensive version first
        print("📋 Trying comprehensive cleaning method...")
        result = remove_crosswalk_gaps_and_clean_roads(input_path, output_path)
        
        if result is not None:
            # Also save generic version for pipeline compatibility
            cv2.imwrite(output_generic, result)
            
            # Save debug comparison
            if original is not None:
                combined = np.hstack([original, result])
                cv2.imwrite(f"{connection_id}_debug_before_after_cleaning.png", combined)
                print(f"💾 Saved before/after comparison: {connection_id}_debug_before_after_cleaning.png")
            
            # Calculate some stats
            original_pixels = np.sum(original == 255) if original is not None else 0
            cleaned_pixels = np.sum(result == 255)
            
            print(f"\n📊 CLEANING SUMMARY:")
            if original is not None:
                print(f"   🔵 Original road pixels: {original_pixels:,}")
                print(f"   🟢 Cleaned road pixels: {cleaned_pixels:,}")
                pixel_change = cleaned_pixels - original_pixels
                if pixel_change > 0:
                    print(f"   ⬆️  Pixel change: +{pixel_change:,} (gaps filled)")
                elif pixel_change < 0:
                    print(f"   ⬇️  Pixel change: {pixel_change:,} (noise removed)")
                else:
                    print(f"   ➡️  Pixel change: {pixel_change:,} (no change)")
            else:
                print(f"   🟢 Final road pixels: {cleaned_pixels:,}")
            
            print(f"🏆 ROAD MASK CLEANING COMPLETE!")
            
        else:
            print("❌ Comprehensive cleaning failed, trying simple method...")
            result = simple_crosswalk_fill(input_path, output_path)
            if result is not None:
                cv2.imwrite(output_generic, result)
                print("✅ Simple cleaning completed successfully")
            else:
                print("❌ Both cleaning methods failed!")
                sys.exit(1)
                
    except Exception as e:
        print(f"💥 Error during road mask cleaning: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback to simple method
        print("🔄 Attempting fallback to simple cleaning method...")
        try:
            result = simple_crosswalk_fill(input_path, output_path)
            if result is not None:
                cv2.imwrite(output_generic, result)
                print("✅ Fallback cleaning completed successfully")
            else:
                sys.exit(1)
        except Exception as e2:
            print(f"💥 Fallback method also failed: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    main()