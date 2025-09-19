import cv2
import numpy as np
import sys
import os
from skimage import measure

def is_road_color(img, tolerance=15):
    """
    Detect actual road colors in RGB space more precisely
    IMPORTANT: Explicitly exclude #f8f7f7 (248,247,247)
    """
    print("🛣️ Detecting road colors with RGB analysis...")
    
    # Convert BGR to RGB for analysis
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
    
    # DEBUG: First, let's find where #f8f7f7 pixels are
    problem_color_mask = (
        (r >= 247) & (r <= 249) &
        (g >= 246) & (g <= 248) &
        (b >= 246) & (b <= 248)
    )
    
    problem_count = np.sum(problem_color_mask)
    if problem_count > 0:
        print(f"   ⚠️ Found {problem_count:,} pixels of #f8f7f7 color in the image!")
    
    # Your ORIGINAL road detection logic
    road_mask_original = (
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
    
    # EXPLICITLY exclude the problematic color and similar very light grays
    exclude_mask = (
        # Exclude #f8f7f7 and nearby colors
        ((r >= 245) & (g >= 245) & (b >= 245)) |
        # Also exclude any very light grays
        ((r + g + b) >= 735)  # Sum of 245+245+245
    )
    
    # Final road mask: original logic AND NOT excluded colors
    road_mask = road_mask_original & (~exclude_mask)
    
    print(f"   🎯 Found {np.sum(road_mask):,} road pixels")
    print(f"   ✅ #f8f7f7 pixels in final road mask: {np.sum(road_mask & problem_color_mask)} (should be 0)")
    
    return road_mask, problem_color_mask

def smart_hole_filling(road_mask, original_img, problem_color_mask, connection_id):
    """
    Fill holes in road mask, but AVOID filling areas that contain light colors like #F8F7F7
    """
    print("🧠 Smart hole filling (avoiding #F8F7F7 areas)...")
    
    # Convert original image to RGB for analysis
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
    
    # Define "light color areas" that should NEVER be filled as holes
    light_color_mask = (
        # #F8F7F7 and similar
        problem_color_mask |
        # Any very light colors
        ((r >= 240) & (g >= 240) & (b >= 240)) |
        # Light beige/cream colors (common for buildings)
        ((r >= 240) & (g >= 235) & (b >= 230)) |
        # Light gray areas
        ((r + g + b) >= 720)
    )
    
    inverse = ~road_mask
    labels = measure.label(inverse)
    props = measure.regionprops(labels)
    
    holes_to_remove = np.zeros_like(road_mask, dtype=bool)
    height, width = road_mask.shape
    
    holes_filled = 0
    holes_skipped_light_color = 0
    holes_skipped_boundary = 0
    holes_skipped_too_large = 0
    
    for p in props:
        area = p.area
        major = p.major_axis_length
        minor = p.minor_axis_length + 1e-5
        ratio = major / minor
        
        # Get the pixels of this hole
        hole_mask = (labels == p.label)
        
        # Check if this hole contains light colors
        hole_has_light_colors = np.sum(light_color_mask & hole_mask) > 0
        light_pixels_in_hole = np.sum(light_color_mask & hole_mask)
        light_percentage = (light_pixels_in_hole / area) * 100 if area > 0 else 0
        
        # Get bounding box
        bbox = p.bbox
        min_row, min_col, max_row, max_col = bbox
        touches_boundary = (min_row <= 2 or min_col <= 2 or 
                           max_row >= height-2 or max_col >= width-2)
        
        # SMART DECISION LOGIC
        should_fill = True
        skip_reason = ""
        
        # Rule 1: Don't fill if it touches boundaries
        if touches_boundary:
            should_fill = False
            skip_reason = "touches_boundary"
            holes_skipped_boundary += 1
        
        # Rule 2: Don't fill if it's too large
        elif area >= 8000 or ratio >= 4.0:
            should_fill = False
            skip_reason = "too_large"
            holes_skipped_too_large += 1
        
        # Rule 3: DON'T fill if it contains significant light colors
        elif hole_has_light_colors and (light_percentage > 10 or light_pixels_in_hole > 50):
            should_fill = False
            skip_reason = f"contains_light_colors({light_percentage:.1f}%)"
            holes_skipped_light_color += 1
        
        if should_fill:
            holes_to_remove[hole_mask] = True
            holes_filled += 1
            print(f"   ✅ Filling hole: area={area}, ratio={ratio:.1f}")
        else:
            print(f"   ❌ Skipping hole ({skip_reason}): area={area}, ratio={ratio:.1f}, light_pixels={light_pixels_in_hole}")
    
    print(f"📊 HOLE FILLING SUMMARY:")
    print(f"   ✅ Holes filled: {holes_filled}")
    print(f"   ❌ Skipped (boundary): {holes_skipped_boundary}")
    print(f"   ❌ Skipped (too large): {holes_skipped_too_large}")
    print(f"   🚫 Skipped (light colors): {holes_skipped_light_color}")
    
    # Save debug image
    cv2.imwrite(f"{connection_id}_debug_smart_holes_to_remove.png", (holes_to_remove.astype(np.uint8)) * 255)
    
    return holes_to_remove

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
        
        # --- Step 1: Extract road mask with PRECISE road color detection ---
        print("\n🔍 Extracting road mask with precise color detection...")
        road_mask, problem_color_mask = is_road_color(img)
        
        cv2.imwrite(f"{connection_id}_debug_initial_road_mask.png", (road_mask.astype(np.uint8)) * 255)
        
        # --- Step 2: SMART hole filling that avoids #F8F7F7 areas ---
        holes_to_remove = smart_hole_filling(road_mask, img, problem_color_mask, connection_id)
        
        # --- Step 3: Create final mask ---
        clean_road_mask = road_mask.copy()
        clean_road_mask[holes_to_remove] = True
        
        # --- Step 4: Final verification ---
        f8f7f7_in_final = np.sum(clean_road_mask & problem_color_mask)
        print(f"\n🎯 FINAL VERIFICATION:")
        print(f"   #F8F7F7 pixels in final road mask: {f8f7f7_in_final:,} (should be 0!)")
        
        # --- Step 5: Save final result ---
        save_mask = (clean_road_mask.astype(np.uint8)) * 255
        cv2.imwrite(output_path, save_mask)
        print(f"✅ Saved: {output_path}")
        
        # Also save without connection_id prefix for compatibility with existing scripts
        cv2.imwrite("final_road_mask.png", save_mask)
        print("✅ Saved: final_road_mask.png (for compatibility)")
        
        if f8f7f7_in_final == 0:
            print("🎉 SUCCESS! No #F8F7F7 pixels were converted to road!")
        else:
            print(f"⚠️ Still {f8f7f7_in_final:,} #F8F7F7 pixels in road mask - need to adjust light color detection")
        
        print(f"\n📊 SUMMARY:")
        print(f"   - Used precise RGB-based road detection")
        print(f"   - Applied smart hole filling that avoids light colors") 
        print(f"   - Road colors: blue-gray pattern (B≥G≥R) in range 160-220")
        print(f"   - Excluded #F8F7F7 and similar light colors")
        print(f"🏆 SIMPLIFIED ROAD MASK EXTRACTION COMPLETE!")
        
    except Exception as e:
        print(f"💥 Error during road mask extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()