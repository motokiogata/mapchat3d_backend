#map_cleaner.py
import os
import numpy as np
from PIL import Image
from scipy import ndimage
import sys

# 🎯 EXACT COLORS
PERFECT_ROAD_COLOR = np.array([170, 185, 201], dtype=np.uint8)           # Blue-gray road
PERFECT_PEDESTRIAN_COLOR = np.array([248, 245, 239], dtype=np.uint8)     # Light cream pedestrian
CROSSWALK_COLOR = np.array([223, 230, 238], dtype=np.uint8)              # Pure crosswalk gray
UNDERGROUND_PLUS_CROSSWALK_COLOR = np.array([233, 183, 185], dtype=np.uint8)  # Underground + crosswalk layered
UNDERGROUND_PLUS_PEDESTRIAN_COLOR = np.array([238, 183, 185], dtype=np.uint8)  # Underground + pedestrian layered

def ensure_rgb_image(img_array):
    """Ensure image is in RGB format"""
    if len(img_array.shape) == 2:
        return np.stack([img_array, img_array, img_array], axis=2)
    elif len(img_array.shape) == 3:
        if img_array.shape[2] == 1:
            return np.repeat(img_array, 3, axis=2)
        elif img_array.shape[2] == 4:
            return img_array[:, :, :3]
        elif img_array.shape[2] == 3:
            return img_array
    return img_array

def detect_underground_plus_crosswalk(img_array, tolerance=8):
    """Detect underground+crosswalk layered color RGB(233,183,185)"""
    print(f"🔍 Detecting underground+crosswalk layered areas RGB{tuple(UNDERGROUND_PLUS_CROSSWALK_COLOR)}...")
    
    img_array = ensure_rgb_image(img_array)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Detect the specific layered color
    layered_mask = (
        (abs(r - UNDERGROUND_PLUS_CROSSWALK_COLOR[0]) <= tolerance) &
        (abs(g - UNDERGROUND_PLUS_CROSSWALK_COLOR[1]) <= tolerance) &
        (abs(b - UNDERGROUND_PLUS_CROSSWALK_COLOR[2]) <= tolerance)
    )
    
    print(f"   🎯 Found {np.sum(layered_mask):,} underground+crosswalk pixels")
    return layered_mask

def detect_underground_plus_pedestrian(img_array, tolerance=8):
    """Detect underground+pedestrian layered color RGB(238,183,185)"""
    print(f"🔍 Detecting underground+pedestrian layered areas RGB{tuple(UNDERGROUND_PLUS_PEDESTRIAN_COLOR)}...")
    
    img_array = ensure_rgb_image(img_array)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Detect the specific layered color
    layered_mask = (
        (abs(r - UNDERGROUND_PLUS_PEDESTRIAN_COLOR[0]) <= tolerance) &
        (abs(g - UNDERGROUND_PLUS_PEDESTRIAN_COLOR[1]) <= tolerance) &
        (abs(b - UNDERGROUND_PLUS_PEDESTRIAN_COLOR[2]) <= tolerance)
    )
    
    print(f"   🎯 Found {np.sum(layered_mask):,} underground+pedestrian pixels")
    return layered_mask

def detect_pure_crosswalk(img_array, tolerance=8):
    """Detect pure crosswalk color RGB(223,230,238)"""
    print(f"🦓 Detecting pure crosswalk areas RGB{tuple(CROSSWALK_COLOR)}...")
    
    img_array = ensure_rgb_image(img_array)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Detect exact crosswalk color
    crosswalk_mask = (
        (abs(r - CROSSWALK_COLOR[0]) <= tolerance) &
        (abs(g - CROSSWALK_COLOR[1]) <= tolerance) &
        (abs(b - CROSSWALK_COLOR[2]) <= tolerance)
    )
    
    print(f"   🎯 Found {np.sum(crosswalk_mask):,} pure crosswalk pixels")
    return crosswalk_mask

def detect_similar_crosswalk_colors(img_array):
    """Detect crosswalk-like colors (including thin frames)"""
    print(f"🔍 Detecting similar crosswalk colors (including thin frames)...")
    
    img_array = ensure_rgb_image(img_array)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    brightness = (r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32)) / 3.0
    
    # Look for colors similar to crosswalk color
    # Should be light gray, not too bright (not sidewalk), not too dark (not road)
    similar_crosswalk = (
        (brightness > 200) & (brightness < 250) &  # Light but not super bright
        (r > 200) & (g > 200) & (b > 200) &        # All channels reasonably high
        (abs(r - g) < 25) & (abs(g - b) < 25) & (abs(r - b) < 25) &  # Neutral gray
        # Close to crosswalk color range
        (abs(r - CROSSWALK_COLOR[0]) < 25) &
        (abs(g - CROSSWALK_COLOR[1]) < 25) &
        (abs(b - CROSSWALK_COLOR[2]) < 25)
    )
    
    print(f"   🎯 Found {np.sum(similar_crosswalk):,} similar crosswalk pixels")
    return similar_crosswalk

def detect_standard_pink_areas(img_array):
    """Detect standard pink underground areas (not layered with crosswalks or pedestrians)"""
    img_array = ensure_rgb_image(img_array)
    
    r = img_array[:,:,0].astype(np.float32)
    g = img_array[:,:,1].astype(np.float32)
    b = img_array[:,:,2].astype(np.float32)
    brightness = (r + g + b) / 3.0
    
    # Dark pink = roads (but NOT the layered colors)
    road_pink_mask = (
        (r >= 180) & (r <= 225) &
        (g >= 130) & (g <= 175) &
        (b >= 150) & (b <= 190) &
        (r > g + 15) & (r > b + 10) &
        (brightness < 185) &
        # Exclude the layered colors
        ~((abs(r - UNDERGROUND_PLUS_CROSSWALK_COLOR[0]) <= 8) &
          (abs(g - UNDERGROUND_PLUS_CROSSWALK_COLOR[1]) <= 8) &
          (abs(b - UNDERGROUND_PLUS_CROSSWALK_COLOR[2]) <= 8)) &
        ~((abs(r - UNDERGROUND_PLUS_PEDESTRIAN_COLOR[0]) <= 8) &
          (abs(g - UNDERGROUND_PLUS_PEDESTRIAN_COLOR[1]) <= 8) &
          (abs(b - UNDERGROUND_PLUS_PEDESTRIAN_COLOR[2]) <= 8))
    )
    
    # Light pink = pedestrian
    pedestrian_pink_mask = (
        (r >= 200) & (r <= 250) &
        (g >= 160) & (g <= 220) &
        (b >= 170) & (b <= 230) &
        (r > g + 5) & (r > b + 5) &
        (brightness >= 185)
    )
    
    from scipy.ndimage import binary_closing, binary_opening
    
    road_pink_mask = binary_closing(road_pink_mask, structure=np.ones((2,2)))
    road_pink_mask = binary_opening(road_pink_mask, structure=np.ones((2,2)))
    
    pedestrian_pink_mask = binary_closing(pedestrian_pink_mask, structure=np.ones((2,2)))
    pedestrian_pink_mask = binary_opening(pedestrian_pink_mask, structure=np.ones((2,2)))
    
    return road_pink_mask, pedestrian_pink_mask

def two_step_ultimate_cleaner(image_path, output_path):
    """Two-step ultimate cleaner with proper crosswalk handling"""
    print("🎯 TWO-STEP ULTIMATE CLEANER")
    print("=" * 50)
    print("🔄 Step 1: Underground+Crosswalk → Pure Crosswalk")
    print("🔄 Step 1b: Underground+Pedestrian → Pure Pedestrian")
    print("🔄 Step 2: Pure Crosswalk → Road Color")
    print("=" * 50)
    
    # Load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    print(f"📊 Image shape: {img_array.shape}")
    
    # STEP 1: Convert layered areas to pure colors
    print(f"\n🔄 STEP 1: Handle Underground+Crosswalk Layered Areas")
    layered_crosswalk_mask = detect_underground_plus_crosswalk(img_array, tolerance=8)
    
    print(f"\n🔄 STEP 1b: Handle Underground+Pedestrian Layered Areas")
    layered_pedestrian_mask = detect_underground_plus_pedestrian(img_array, tolerance=8)
    
    # Create intermediate result
    step1_result = img_array.copy()
    
    # Handle underground+crosswalk layered areas
    if np.any(layered_crosswalk_mask):
        layered_coords = np.where(layered_crosswalk_mask)
        print(f"   🔄 Converting {len(layered_coords[0]):,} layered crosswalk pixels...")
        print(f"   🎨 RGB{tuple(UNDERGROUND_PLUS_CROSSWALK_COLOR)} → RGB{tuple(CROSSWALK_COLOR)}")
        
        for i in range(len(layered_coords[0])):
            y, x = layered_coords[0][i], layered_coords[1][i]
            # Convert to pure crosswalk color
            variation = np.random.randint(-2, 3, 3)
            final_color = np.clip(CROSSWALK_COLOR.astype(np.int16) + variation, 0, 255)
            step1_result[y, x] = final_color.astype(np.uint8)
        
        print(f"   ✅ Step 1 complete: Underground+Crosswalk → Pure Crosswalk")
    else:
        print(f"   ℹ️  No layered crosswalk areas found")
    
    # Handle underground+pedestrian layered areas
    if np.any(layered_pedestrian_mask):
        layered_coords = np.where(layered_pedestrian_mask)
        print(f"   🔄 Converting {len(layered_coords[0]):,} layered pedestrian pixels...")
        print(f"   🎨 RGB{tuple(UNDERGROUND_PLUS_PEDESTRIAN_COLOR)} → RGB{tuple(PERFECT_PEDESTRIAN_COLOR)}")
        
        for i in range(len(layered_coords[0])):
            y, x = layered_coords[0][i], layered_coords[1][i]
            # Convert to pure pedestrian color
            variation = np.random.randint(-1, 2, 3)
            final_color = np.clip(PERFECT_PEDESTRIAN_COLOR.astype(np.int16) + variation, 0, 255)
            step1_result[y, x] = final_color.astype(np.uint8)
        
        print(f"   ✅ Step 1b complete: Underground+Pedestrian → Pure Pedestrian")
    else:
        print(f"   ℹ️  No layered pedestrian areas found")
    
    # STEP 2: Now detect and remove all crosswalk colors
    print(f"\n🔄 STEP 2: Remove All Crosswalk Colors")
    
    # Detect standard pink areas (excluding layered)
    road_pink_mask, pedestrian_pink_mask = detect_standard_pink_areas(step1_result)
    
    # Detect all crosswalk colors (now including converted ones)
    pure_crosswalk_mask = detect_pure_crosswalk(step1_result, tolerance=8)
    similar_crosswalk_mask = detect_similar_crosswalk_colors(step1_result)
    
    # Combine all crosswalk detections for thorough removal
    all_crosswalks_mask = pure_crosswalk_mask | similar_crosswalk_mask
    
    # Summary
    total_road_pink = np.sum(road_pink_mask)
    total_ped_pink = np.sum(pedestrian_pink_mask)
    total_crosswalks = np.sum(all_crosswalks_mask)
    total_layered_crosswalk = np.sum(layered_crosswalk_mask)
    total_layered_pedestrian = np.sum(layered_pedestrian_mask)
    total_clean = total_road_pink + total_ped_pink + total_crosswalks
    
    print(f"\n📊 FINAL DETECTION SUMMARY:")
    print(f"   🔄 Underground+Crosswalk layered: {total_layered_crosswalk:,} pixels (converted)")
    print(f"   🔄 Underground+Pedestrian layered: {total_layered_pedestrian:,} pixels (converted)")
    print(f"   🛣️  Underground road areas: {total_road_pink:,} pixels")
    print(f"   🚶 Underground pedestrian areas: {total_ped_pink:,} pixels")
    print(f"   🦓 All crosswalk colors: {total_crosswalks:,} pixels")
    print(f"   📊 TOTAL to clean in Step 2: {total_clean:,} pixels")
    
    # Create final result
    final_result = step1_result.copy()
    
    # Remove underground roads (pink → road color)
    if total_road_pink > 0:
        print(f"\n🛣️  Cleaning underground roads...")
        road_coords = np.where(road_pink_mask)
        
        for i in range(len(road_coords[0])):
            y, x = road_coords[0][i], road_coords[1][i]
            variation = np.random.randint(-2, 3, 3)
            final_color = np.clip(PERFECT_ROAD_COLOR.astype(np.int16) + variation, 0, 255)
            final_result[y, x] = final_color.astype(np.uint8)
        
        print(f"   ✅ {total_road_pink:,} underground road pixels cleaned")
    
    # Remove underground pedestrians (pink → light color)
    if total_ped_pink > 0:
        print(f"🚶 Cleaning underground pedestrians...")
        ped_coords = np.where(pedestrian_pink_mask)
        
        for i in range(len(ped_coords[0])):
            y, x = ped_coords[0][i], ped_coords[1][i]
            variation = np.random.randint(-1, 2, 3)
            final_color = np.clip(PERFECT_PEDESTRIAN_COLOR.astype(np.int16) + variation, 0, 255)
            final_result[y, x] = final_color.astype(np.uint8)
        
        print(f"   ✅ {total_ped_pink:,} underground pedestrian pixels cleaned")
    
    # Remove ALL crosswalks (thorough removal)
    if total_crosswalks > 0:
        print(f"🦓 THOROUGH crosswalk removal...")
        crosswalk_coords = np.where(all_crosswalks_mask)
        
        for i in range(len(crosswalk_coords[0])):
            y, x = crosswalk_coords[0][i], crosswalk_coords[1][i]
            # Replace with road color
            variation = np.random.randint(-3, 4, 3)
            final_color = np.clip(PERFECT_ROAD_COLOR.astype(np.int16) + variation, 0, 255)
            final_result[y, x] = final_color.astype(np.uint8)
        
        print(f"   ✅ {total_crosswalks:,} crosswalk pixels thoroughly removed")
        print(f"   🎯 Including thin frames and similar colors")
    
    # Ultra-thorough final cleanup
    print(f"\n🧹 ULTRA-THOROUGH final cleanup...")
    
    result_r = final_result[:,:,0].astype(np.float32)
    result_g = final_result[:,:,1].astype(np.float32)
    result_b = final_result[:,:,2].astype(np.float32)
    
    # Check for any remaining problematic colors
    remaining_pink = (
        (result_r > result_g + 8) & (result_r > result_b + 6) & 
        (result_r > 175)
    )
    
    remaining_layered_crosswalk = (
        (abs(result_r - UNDERGROUND_PLUS_CROSSWALK_COLOR[0]) <= 5) &
        (abs(result_g - UNDERGROUND_PLUS_CROSSWALK_COLOR[1]) <= 5) &
        (abs(result_b - UNDERGROUND_PLUS_CROSSWALK_COLOR[2]) <= 5)
    )
    
    remaining_layered_pedestrian = (
        (abs(result_r - UNDERGROUND_PLUS_PEDESTRIAN_COLOR[0]) <= 5) &
        (abs(result_g - UNDERGROUND_PLUS_PEDESTRIAN_COLOR[1]) <= 5) &
        (abs(result_b - UNDERGROUND_PLUS_PEDESTRIAN_COLOR[2]) <= 5)
    )
    
    remaining_crosswalks = (
        (abs(result_r - CROSSWALK_COLOR[0]) <= 5) &
        (abs(result_g - CROSSWALK_COLOR[1]) <= 5) &
        (abs(result_b - CROSSWALK_COLOR[2]) <= 5)
    )
    
    cleanup_total = np.sum(remaining_pink) + np.sum(remaining_layered_crosswalk) + np.sum(remaining_layered_pedestrian) + np.sum(remaining_crosswalks)
    
    if cleanup_total > 0:
        print(f"   🧹 Final cleanup: {cleanup_total:,} remaining traces...")
        
        # Clean remaining layered pedestrian traces with pedestrian color
        if np.sum(remaining_layered_pedestrian) > 0:
            cleanup_coords = np.where(remaining_layered_pedestrian)
            for i in range(len(cleanup_coords[0])):
                y, x = cleanup_coords[0][i], cleanup_coords[1][i]
                final_result[y, x] = PERFECT_PEDESTRIAN_COLOR
        
        # Clean all other remaining traces with road color
        cleanup_mask = remaining_pink | remaining_layered_crosswalk | remaining_crosswalks
        cleanup_coords = np.where(cleanup_mask)
        
        for i in range(len(cleanup_coords[0])):
            y, x = cleanup_coords[0][i], cleanup_coords[1][i]
            final_result[y, x] = PERFECT_ROAD_COLOR
        
        print(f"   ✅ All traces eliminated!")
    else:
        print(f"   ✅ Perfect - no remaining traces!")
    
    # Save result
    result_img = Image.fromarray(final_result.astype(np.uint8))
    result_img.save(output_path)
    
    print(f"\n🏆 TWO-STEP ULTIMATE CLEANING COMPLETE!")
    print(f"✨ Step 1: Underground+Crosswalk → Pure Crosswalk ✅")
    print(f"✨ Step 1b: Underground+Pedestrian → Pure Pedestrian ✅")
    print(f"✨ Step 2: All Crosswalks → Road Color ✅")
    print(f"🎯 PERFECT clean intersection achieved!")
    print(f"📁 Result: {output_path}")
    
    return output_path

def main():
    if len(sys.argv) < 2:
        print("❌ connection_id argument is required")
        sys.exit(1)
    
    connection_id = sys.argv[1]
    
    # Input and output paths
    input_path = "roadmap.png"  # Downloaded from S3 by docker-entrypoint.py
    output_path = f"{connection_id}_cleaned_roadmap.png"
    
    print(f"🎯 MAP CLEANER for connection_id: {connection_id}")
    print(f"📍 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    # Apply two-step cleaning
    try:
        two_step_ultimate_cleaner(input_path, output_path)
        print(f"\n🎉 MAP CLEANING SUCCESS!")
        print(f"🏆 Clean map saved as: {output_path}")
    except Exception as e:
        print(f"💥 Error during map cleaning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()