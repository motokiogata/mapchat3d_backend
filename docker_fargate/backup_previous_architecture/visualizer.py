# visualizer.py
import cv2
import numpy as np
import os
from config import CANVAS_SIZE, OUTPUT_IMG

class NetworkVisualizer:
    def __init__(self, roads, intersections, common_language_vocab, edge_analysis_summary):
        self.roads = roads
        self.intersections = intersections
        self.common_language_vocabulary = common_language_vocab
        self.edge_analysis_summary = edge_analysis_summary

    def visualize_integrated_network(self):
        """Create comprehensive visualization of integrated network with user-friendly labels"""
        canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)

        # Colors
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # Draw roads with edge highlighting and user-friendly names
        for i, road in enumerate(self.roads):
            color = colors[i % len(colors)]
            points = road["points"]
            
            # Highlight edge-connected roads
            edge_analysis = road['metadata']['edge_analysis']
            line_thickness = 6 if edge_analysis['has_edge_connection'] else 3
            
            for j in range(len(points) - 1):
                pt1 = (int(points[j][0]), int(points[j][1]))
                pt2 = (int(points[j+1][0]), int(points[j+1][1]))
                cv2.line(canvas, pt1, pt2, color, line_thickness)
            
            # Mark edge points
            if points:
                start_pt = (int(points[0][0]), int(points[0][1]))
                end_pt = (int(points[-1][0]), int(points[-1][1]))
                
                start_edge = edge_analysis.get('start_edge', {})
                end_edge = edge_analysis.get('end_edge', {})
                
                if start_edge.get('is_edge', False):
                    cv2.circle(canvas, start_pt, 12, (0, 255, 0), -1)
                    if start_edge.get('edge_id'):
                        cv2.putText(canvas, start_edge['edge_id'], 
                                   (start_pt[0] + 15, start_pt[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    cv2.circle(canvas, start_pt, 6, (255, 255, 255), -1)
                
                if end_edge.get('is_edge', False):
                    cv2.circle(canvas, end_pt, 12, (0, 0, 255), -1)
                    if end_edge.get('edge_id'):
                        cv2.putText(canvas, end_edge['edge_id'], 
                                   (end_pt[0] + 15, end_pt[1] + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.circle(canvas, end_pt, 6, (0, 0, 0), -1)
                
                # Road label with USER-FRIENDLY NAME
                mid_idx = len(points) // 2
                mid_pt = (int(points[mid_idx][0]), int(points[mid_idx][1]))
                
                # Use user-friendly name instead of generic road name
                road_aliases = self.common_language_vocabulary['roads'].get(road['id'], {})
                user_friendly_name = road_aliases.get('primary_name', f'Road {road["id"]}')
                road_id = road['id']
                
                edge_indicator = ""
                if edge_analysis['has_edge_connection']:
                    edge_sides = '/'.join(edge_analysis['edge_sides'][:2])
                    edge_indicator = f"[{edge_sides}]"
                
                label = f"[{road_id}]{user_friendly_name}{edge_indicator}"
                cv2.putText(canvas, label, (mid_pt[0], mid_pt[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw intersections with user-friendly names
        for intersection in self.intersections:
            center_x, center_y = intersection['center']
            int_id = intersection['id']
            metadata = intersection['metadata']
            
            radius = 8 + metadata['connected_roads_count']
            cv2.circle(canvas, (int(center_x), int(center_y)), radius, (0, 255, 255), -1)
            cv2.circle(canvas, (int(center_x), int(center_y)), radius, (255, 255, 255), 2)
            
            # Intersection label with USER-FRIENDLY NAME
            int_aliases = self.common_language_vocabulary['intersections'].get(int_id, {})
            user_friendly_name = int_aliases.get('primary_name', f'Intersection {int_id}')
            
            label = f"[{int_id}]{user_friendly_name}"
            
            # Add edge indicator
            edge_analysis = metadata.get('edge_analysis', {})
            if edge_analysis.get('is_edge_intersection', False):
                edge_sides = '/'.join(edge_analysis.get('edge_sides', []))
                label += f"[{edge_sides}]"
            
            cv2.putText(canvas, label,
                       (int(center_x) + radius + 5, int(center_y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Title with statistics including user-friendly info
        title = f"Integrated Network with User-Friendly Aliases: {len(self.roads)} roads, {len(self.intersections)} intersections"
        cv2.putText(canvas, title, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        edge_stats = f"Edge connections: {self.edge_analysis_summary['roads_with_edge_connections']} roads, {len(self.edge_analysis_summary['all_edge_ids'])} IDs"
        cv2.putText(canvas, edge_stats, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        alias_info = f"User-friendly aliases: {len(self.common_language_vocabulary['roads'])} roads, {len(self.common_language_vocabulary['intersections'])} intersections"
        cv2.putText(canvas, alias_info, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        consistency_info = "✅ Common Language Ready for AI-Human Communication"
        cv2.putText(canvas, consistency_info, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(OUTPUT_IMG, canvas)
        print(f"✅ Saved integrated network visualization with user-friendly labels to {OUTPUT_IMG}")