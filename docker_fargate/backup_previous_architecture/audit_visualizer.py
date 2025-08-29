#!/usr/bin/env python3
# audit_visualizer.py
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
from config import *

class JSONAuditVisualizer:
    def __init__(self, connection_id=None):
        self.connection_id = connection_id
        self.colors = {
            'road': '#2E86AB',
            'intersection': '#A23B72',
            'lane': '#F18F01',
            'edge': '#C73E1D',
            'landmark': '#6A994E',
            'background': '#F5F5F5',
            'text': '#2D3436',
            'highlight': '#FDCB6E'
        }
        
    def create_road_network_audit(self, integrated_data, output_path="audit_road_network.png"):
        """Create comprehensive road network visualization"""
        print(f"📊 Creating road network audit visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Road Network Audit Report', fontsize=20, fontweight='bold')
        
        # Panel 1: Road Network Overview
        self._plot_road_network_overview(ax1, integrated_data)
        
        # Panel 2: Edge Analysis
        self._plot_edge_analysis(ax2, integrated_data)
        
        # Panel 3: Statistics Dashboard
        self._plot_statistics_dashboard(ax3, integrated_data)
        
        # Panel 4: Common Language Vocabulary
        self._plot_vocabulary_summary(ax4, integrated_data)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ Saved road network audit to {output_path}")
        return output_path
    
    def create_intersections_audit(self, intersections_data, output_path="audit_intersections.png"):
        """Create detailed intersections audit"""
        print(f"📊 Creating intersections audit visualization...")
        
        intersections = intersections_data.get('intersections', [])
        if not intersections:
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Intersections Audit Report', fontsize=20, fontweight='bold')
        
        # Panel 1: Intersection Locations
        self._plot_intersection_locations(ax1, intersections)
        
        # Panel 2: Intersection Types
        self._plot_intersection_types(ax2, intersections)
        
        # Panel 3: Connection Analysis
        self._plot_intersection_connections(ax3, intersections)
        
        # Panel 4: User-Friendly Names
        self._plot_intersection_names(ax4, intersections)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ Saved intersections audit to {output_path}")
        return output_path
    
    def create_lane_trees_audit(self, lane_data, output_path="audit_lane_trees.png"):
        """Create lane trees audit visualization"""
        print(f"📊 Creating lane trees audit visualization...")
        
        lane_trees = lane_data.get('lane_trees', [])
        if not lane_trees:
            return None
            
        # Create multi-page visualization for lane trees
        fig = plt.figure(figsize=(24, 18))
        
        # Main title
        fig.suptitle('Lane Trees Audit Report', fontsize=24, fontweight='bold', y=0.95)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Lane Tree Overview (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_lane_tree_overview(ax1, lane_trees)
        
        # Panel 2: Lane Statistics
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_lane_statistics(ax2, lane_data)
        
        # Panel 3: Edge Connections (spans full width)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_lane_edge_connections(ax3, lane_trees)
        
        # Panel 4: Individual Lane Tree Details (3 columns)
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        ax6 = fig.add_subplot(gs[2, 2])
        
        self._plot_individual_lane_details(ax4, ax5, ax6, lane_trees[:3])
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ Saved lane trees audit to {output_path}")
        return output_path
    
    def create_vocabulary_audit(self, vocab_data, output_path="audit_vocabulary.png"):
        """Create common language vocabulary audit"""
        print(f"📊 Creating vocabulary audit visualization...")
        
        if 'common_language_vocabulary' not in vocab_data:
            return None
            
        vocab = vocab_data['common_language_vocabulary']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Common Language Vocabulary Audit', fontsize=20, fontweight='bold')
        
        # Panel 1: Vocabulary Overview
        self._plot_vocabulary_overview(ax1, vocab)
        
        # Panel 2: Road Names
        self._plot_road_vocabulary(ax2, vocab.get('roads', {}))
        
        # Panel 3: Intersection Names  
        self._plot_intersection_vocabulary(ax3, vocab.get('intersections', {}))
        
        # Panel 4: Landmarks and Spatial Relationships
        self._plot_landmarks_and_spatial(ax4, vocab)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ Saved vocabulary audit to {output_path}")
        return output_path
    
    def create_metadata_summary(self, integrated_data, output_path="audit_metadata_summary.png"):
        """Create comprehensive metadata summary"""
        print(f"📊 Creating metadata summary visualization...")
        
        # Create a large summary image with PIL for better text handling
        img_width, img_height = 1600, 1200
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 32)
            font_header = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_body = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font_title = ImageFont.load_default()
            font_header = ImageFont.load_default()
            font_body = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Title
        draw.text((50, 30), "INTEGRATED ROAD NETWORK - METADATA SUMMARY", 
                 fill='black', font=font_title)
        
        y_pos = 100
        
        # Basic Statistics
        metadata = integrated_data.get('metadata', {})
        draw.text((50, y_pos), "📊 BASIC STATISTICS", fill='#2E86AB', font=font_header)
        y_pos += 40
        
        stats_text = [
            f"Total Roads: {metadata.get('total_roads', 'N/A')}",
            f"Total Intersections: {metadata.get('total_intersections', 'N/A')}",
            f"Canvas Size: {metadata.get('processing_parameters', {}).get('canvas_size', 'N/A')}",
            f"Coordinate System: {metadata.get('coordinate_system', 'N/A')}"
        ]
        
        for stat in stats_text:
            draw.text((70, y_pos), stat, fill='black', font=font_body)
            y_pos += 25
        
        y_pos += 20
        
        # Edge Analysis
        edge_summary = integrated_data.get('edge_analysis_summary', {})
        draw.text((50, y_pos), "🌍 EDGE ANALYSIS", fill='#A23B72', font=font_header)
        y_pos += 40
        
        edge_stats = [
            f"Roads with Edge Connections: {edge_summary.get('roads_with_edge_connections', 'N/A')}",
            f"Intersections at Edges: {edge_summary.get('intersections_at_edges', 'N/A')}",
            f"Total Edge IDs: {len(edge_summary.get('all_edge_ids', []))}"
        ]
        
        for stat in edge_stats:
            draw.text((70, y_pos), stat, fill='black', font=font_body)
            y_pos += 25
        
        # Edge Distribution
        if 'edge_distribution' in edge_summary:
            y_pos += 10
            draw.text((70, y_pos), "Edge Distribution by Direction:", fill='#666666', font=font_small)
            y_pos += 20
            
            for direction, edges in edge_summary['edge_distribution'].items():
                draw.text((90, y_pos), f"{direction.upper()}: {len(edges)} roads", 
                         fill='black', font=font_small)
                y_pos += 18
        
        y_pos += 30
        
        # Common Language Vocabulary
        vocab = integrated_data.get('common_language_vocabulary', {})
        draw.text((50, y_pos), "🗣️ COMMON LANGUAGE VOCABULARY", fill='#F18F01', font=font_header)
        y_pos += 40
        
        vocab_stats = [
            f"Road Aliases: {len(vocab.get('roads', {}))}",
            f"Intersection Aliases: {len(vocab.get('intersections', {}))}",
            f"Lane Aliases: {len(vocab.get('lanes', {}))}",
            f"Landmarks: {len(vocab.get('landmarks', {}))}",
            f"Spatial Relationships: {len(vocab.get('spatial_relationships', {}))}"
        ]
        
        for stat in vocab_stats:
            draw.text((70, y_pos), stat, fill='black', font=font_body)
            y_pos += 25
        
        y_pos += 30
        
        # Navigation Features
        nav_features = metadata.get('navigation_features', {})
        draw.text((50, y_pos), "🧭 NAVIGATION FEATURES", fill='#6A994E', font=font_header)
        y_pos += 40
        
        features = [
            ("Turn-by-turn Navigation", nav_features.get('supports_turn_by_turn', False)),
            ("Landmark Navigation", nav_features.get('supports_landmark_navigation', False)),
            ("Route Planning", nav_features.get('supports_route_planning', False)),
            ("Narrative Parsing", nav_features.get('supports_narrative_parsing', False)),
            ("User-Friendly Aliases", nav_features.get('supports_user_friendly_aliases', False)),
            ("Common Language", nav_features.get('supports_common_language_vocabulary', False))
        ]
        
        for feature, supported in features:
            status = "✅" if supported else "❌"
            draw.text((70, y_pos), f"{status} {feature}", fill='black', font=font_body)
            y_pos += 25
        
        # Save the image
        img.save(output_path)
        print(f"✅ Saved metadata summary to {output_path}")
        return output_path
    
    # Helper methods for specific plot types
    def _plot_road_network_overview(self, ax, integrated_data):
        """Plot road network overview"""
        ax.set_title('Road Network Overview', fontsize=14, fontweight='bold')
        
        roads = integrated_data.get('roads', [])
        intersections = integrated_data.get('intersections', [])
        
        if not roads and not intersections:
            ax.text(0.5, 0.5, 'No road network data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return
        
        # Plot roads as lines
        for road in roads:
            points = road.get('points', [])
            if len(points) >= 2:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                # Color roads based on metadata
                metadata = road.get('metadata', {})
                edge_analysis = metadata.get('edge_analysis', {})
                has_edge = edge_analysis.get('has_edge_connection', False)
                
                color = self.colors['edge'] if has_edge else self.colors['road']
                width = 2 if has_edge else 1
                
                ax.plot(x_coords, y_coords, color=color, linewidth=width, alpha=0.8)
        
        # Plot intersections as circles
        for intersection in intersections:
            center = intersection.get('center', [0, 0])
            metadata = intersection.get('metadata', {})
            int_type = metadata.get('intersection_type', 'unknown')
            
            # Color code by intersection type
            if int_type == 'T_junction':
                color = '#FF6B6B'
            elif int_type == 'four_way':
                color = '#4ECDC4'
            elif int_type == 'complex':
                color = '#45B7D1'
            else:
                color = self.colors['intersection']
            
            circle = Circle(center, 8, color=color, alpha=0.8, zorder=5)
            ax.add_patch(circle)
            
            # Add intersection ID
            ax.text(center[0], center[1], str(intersection.get('id', '?')), 
                ha='center', va='center', fontweight='bold', 
                color='white', fontsize=7)
        
        # Set proper axis limits and styling
        if roads:
            all_x = []
            all_y = []
            for road in roads:
                for point in road.get('points', []):
                    all_x.append(point[0])
                    all_y.append(point[1])
            
            if all_x and all_y:
                margin = 50
                ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        else:
            # Fallback to canvas size
            ax.set_xlim(0, CANVAS_SIZE[0])
            ax.set_ylim(0, CANVAS_SIZE[1])
        
        ax.invert_yaxis()  # Invert Y-axis for image coordinates
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.colors['road'], lw=2, label='Regular Roads'),
            Line2D([0], [0], color=self.colors['edge'], lw=2, label='Edge Roads'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['intersection'], 
                markersize=8, label='Intersections', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Add summary statistics
        summary_text = f"Roads: {len(roads)}\nIntersections: {len(intersections)}"
        edge_roads = sum(1 for road in roads 
                        if road.get('metadata', {}).get('edge_analysis', {}).get('has_edge_connection', False))
        if edge_roads > 0:
            summary_text += f"\nEdge Roads: {edge_roads}"
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_edge_analysis(self, ax, data):
        """Plot edge analysis"""
        ax.set_title('Edge Connections Analysis', fontsize=14, fontweight='bold')
        
        edge_summary = data.get('edge_analysis_summary', {})
        edge_dist = edge_summary.get('edge_distribution', {})
        
        directions = list(edge_dist.keys())
        counts = [len(edge_dist[d]) for d in directions]
        
        if directions and counts:
            bars = ax.bar(directions, counts, color=[self.colors['edge']] * len(directions),
                         alpha=0.7)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of Roads')
        ax.set_xlabel('Edge Direction')
        ax.grid(True, alpha=0.3)
        
        # Add summary text
        total_edge_roads = edge_summary.get('roads_with_edge_connections', 0)
        ax.text(0.02, 0.98, f'Total Edge Roads: {total_edge_roads}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_statistics_dashboard(self, ax, data):
        """Plot statistics dashboard"""
        ax.set_title('Network Statistics Dashboard', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        metadata = data.get('metadata', {})
        edge_summary = data.get('edge_analysis_summary', {})
        vocab = data.get('common_language_vocabulary', {})
        
        stats = [
            ('Roads', metadata.get('total_roads', 0), self.colors['road']),
            ('Intersections', metadata.get('total_intersections', 0), self.colors['intersection']),
            ('Edge Roads', edge_summary.get('roads_with_edge_connections', 0), self.colors['edge']),
            ('Road Aliases', len(vocab.get('roads', {})), self.colors['landmark']),
            ('Landmarks', len(vocab.get('landmarks', {})), self.colors['highlight'])
        ]
        
        # Create pie chart
        values = [stat[1] for stat in stats if stat[1] > 0]
        labels = [stat[0] for stat in stats if stat[1] > 0]
        colors = [stat[2] for stat in stats if stat[1] > 0]
        
        if values:
            wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors,
                                            autopct='%1.0f', startangle=90)
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_color('white')
    
    def _plot_vocabulary_summary(self, ax, data):
        """Plot vocabulary summary"""
        ax.set_title('Common Language Vocabulary', fontsize=14, fontweight='bold')
        
        vocab = data.get('common_language_vocabulary', {})
        quick_ref = data.get('user_friendly_quick_reference', {})
        
        categories = ['Roads', 'Intersections', 'Lanes', 'Landmarks', 'Spatial Relations']
        counts = [
            len(vocab.get('roads', {})),
            len(vocab.get('intersections', {})),
            len(vocab.get('lanes', {})),
            len(vocab.get('landmarks', {})),
            len(vocab.get('spatial_relationships', {}))
        ]
        
        colors_list = [self.colors['road'], self.colors['intersection'], 
                      self.colors['lane'], self.colors['landmark'], self.colors['highlight']]
        
        bars = ax.barh(categories, counts, color=colors_list, alpha=0.7)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   str(count), ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Number of Aliases/Items')
        ax.grid(True, alpha=0.3, axis='x')
        
        total_vocab = sum(counts)
        ax.text(0.98, 0.02, f'Total Vocabulary Items: {total_vocab}',
               transform=ax.transAxes, horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_vocabulary_overview(self, ax, vocab):
        """Plot vocabulary overview"""
        ax.set_title('Vocabulary Categories Overview', fontsize=14, fontweight='bold')
        
        # ✅ Safely get counts, handling both dict and list formats
        spatial_rels = vocab.get('spatial_relationships', [])
        spatial_count = 0
        if isinstance(spatial_rels, dict):
            spatial_count = len(spatial_rels)
        elif isinstance(spatial_rels, list):
            spatial_count = len(spatial_rels)
        
        categories = {
            'Roads': len(vocab.get('roads', {})) if isinstance(vocab.get('roads', {}), dict) else 0,
            'Intersections': len(vocab.get('intersections', {})) if isinstance(vocab.get('intersections', {}), dict) else 0,
            'Lanes': len(vocab.get('lanes', {})) if isinstance(vocab.get('lanes', {}), dict) else 0,
            'Landmarks': len(vocab.get('landmarks', {})) if isinstance(vocab.get('landmarks', {}), dict) else 0,
            'Spatial Relations': spatial_count
        }
        
        labels = list(categories.keys())
        sizes = list(categories.values())
        colors = [self.colors['road'], self.colors['intersection'], 
                self.colors['lane'], self.colors['landmark'], self.colors['highlight']]
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                            autopct='%1.0f', startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, 'No vocabulary data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)


    def _plot_intersection_locations(self, ax, intersections):
        """Plot intersection locations"""
        ax.set_title('Intersection Locations', fontsize=14, fontweight='bold')
        
        for i, intersection in enumerate(intersections):
            center = intersection['center']
            metadata = intersection.get('metadata', {})
            int_type = metadata.get('intersection_type', 'unknown')
            
            # Color code by type
            color = self.colors['intersection']
            if int_type == 'T_junction':
                color = '#FF6B6B'
            elif int_type == 'four_way':
                color = '#4ECDC4'
            elif int_type == 'complex':
                color = '#45B7D1'
            
            circle = Circle(center, 12, color=color, alpha=0.8, zorder=5)
            ax.add_patch(circle)
            
            # Add intersection ID
            ax.text(center[0], center[1], str(intersection['id']), 
                   ha='center', va='center', fontweight='bold', 
                   color='white', fontsize=8)
        
        ax.set_xlim(0, CANVAS_SIZE[0])
        ax.set_ylim(0, CANVAS_SIZE[1])
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_intersection_types(self, ax, intersections):
        """Plot intersection types distribution"""
        ax.set_title('Intersection Types Distribution', fontsize=14, fontweight='bold')
        
        type_counts = {}
        for intersection in intersections:
            metadata = intersection.get('metadata', {})
            int_type = metadata.get('intersection_type', 'unknown')
            type_counts[int_type] = type_counts.get(int_type, 0) + 1
        
        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            
            bars = ax.bar(types, counts, color=self.colors['intersection'], alpha=0.7)
            
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Count')
        ax.set_xlabel('Intersection Type')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_intersection_connections(self, ax, intersections):
        """Plot intersection connections analysis"""
        ax.set_title('Road Connections per Intersection', fontsize=14, fontweight='bold')
        
        connection_counts = []
        for intersection in intersections:
            metadata = intersection.get('metadata', {})
            road_count = metadata.get('connected_roads_count', 0)
            connection_counts.append(road_count)
        
        if connection_counts:
            ax.hist(connection_counts, bins=range(1, max(connection_counts) + 2), 
                   color=self.colors['intersection'], alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Number of Connected Roads')
        ax.set_ylabel('Number of Intersections')
        ax.grid(True, alpha=0.3)
    
    def _plot_intersection_names(self, ax, intersections):
        """Plot intersection user-friendly names"""
        ax.set_title('User-Friendly Intersection Names', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        y_pos = 0.95
        count = 0
        
        for intersection in intersections[:10]:  # Show first 10
            metadata = intersection.get('metadata', {})
            common_lang = metadata.get('common_language', {})
            primary_name = common_lang.get('primary_user_name', f"Intersection {intersection['id']}")
            description = common_lang.get('natural_description', 'Standard intersection')
            
            text = f"ID {intersection['id']}: {primary_name}\n  → {description}"
            ax.text(0.05, y_pos, text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors['background'], alpha=0.7))
            
            y_pos -= 0.18
            count += 1
            
            if y_pos < 0.1:
                break
        
        if len(intersections) > 10:
            ax.text(0.05, 0.05, f"... and {len(intersections) - 10} more intersections",
                   transform=ax.transAxes, fontsize=9, style='italic')
    
    def _plot_lane_tree_overview(self, ax, lane_trees):
        """Plot lane tree overview"""
        ax.set_title('Lane Trees Overview', fontsize=14, fontweight='bold')
        
        for i, tree in enumerate(lane_trees[:20]):  # Show first 20
            road_id = tree.get('road_id', f'Road_{i}')
            branch_count = len(tree.get('branches', []))
            metadata = tree.get('metadata', {})
            
            # Color based on edge connection
            edge_analysis = metadata.get('edge_analysis', {})
            has_edge = edge_analysis.get('has_edge_connection', False)
            
            color = self.colors['edge'] if has_edge else self.colors['lane']
            
            # Plot as horizontal bar
            ax.barh(i, branch_count, color=color, alpha=0.7)
            
            # Add road name
            user_name = metadata.get('common_language', {}).get('primary_user_name', f'Road {road_id}')
            ax.text(-0.5, i, f"{user_name[:15]}...", va='center', ha='right', fontsize=8)
        
        ax.set_xlabel('Number of Branches')
        ax.set_ylabel('Lane Trees')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_lane_statistics(self, ax, lane_data):
        """Plot lane statistics"""
        ax.set_title('Lane Statistics', fontsize=14, fontweight='bold')
        
        stats = lane_data.get('statistics', {})
        
        stat_items = [
            ('Total Trees', stats.get('total_trees', 0)),
            ('Total Branches', stats.get('total_branches', 0)),
            ('Trees with Branches', stats.get('trees_with_branches', 0)),
            ('Edge Trees', stats.get('edge_trees', 0)),
            ('Entry Points', stats.get('entry_point_trees', 0))
        ]
        
        # Create vertical bar chart
        labels = [item[0] for item in stat_items]
        values = [item[1] for item in stat_items]
        
        bars = ax.bar(range(len(labels)), values, color=self.colors['lane'], alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                   str(value), ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    def _plot_lane_edge_connections(self, ax, lane_trees):
        """Plot lane edge connections"""
        ax.set_title('Lane Trees with Edge Connections', fontsize=14, fontweight='bold')
        
        edge_directions = {'west': 0, 'east': 0, 'north': 0, 'south': 0}
        
        for tree in lane_trees:
            metadata = tree.get('metadata', {})
            edge_analysis = metadata.get('edge_analysis', {})
            edge_sides = edge_analysis.get('edge_sides', [])
            
            for side in edge_sides:
                if side in edge_directions:
                    edge_directions[side] += 1
        
        directions = list(edge_directions.keys())
        counts = list(edge_directions.values())
        
        if any(counts):
            bars = ax.bar(directions, counts, color=self.colors['edge'], alpha=0.7)
            
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of Lane Trees')
        ax.set_xlabel('Edge Direction')
        ax.grid(True, alpha=0.3)
    
    def _plot_individual_lane_details(self, ax1, ax2, ax3, lane_trees):
        """Plot individual lane tree details"""
        axes = [ax1, ax2, ax3]
        
        for i, (ax, tree) in enumerate(zip(axes, lane_trees)):
            road_id = tree.get('road_id', f'Road_{i}')
            metadata = tree.get('metadata', {})
            branches = tree.get('branches', [])
            
            user_name = metadata.get('common_language', {}).get('primary_user_name', f'Road {road_id}')
            ax.set_title(f'{user_name[:20]}', fontsize=12, fontweight='bold')
            
            # Show branch information
            if branches:
                branch_types = {}
                for branch in branches:
                    branch_type = branch.get('type', 'unknown')
                    branch_types[branch_type] = branch_types.get(branch_type, 0) + 1
                
                if branch_types:
                    types = list(branch_types.keys())
                    counts = list(branch_types.values())
                    
                    ax.pie(counts, labels=types, autopct='%1.0f',
                          colors=[self.colors['lane']] * len(types))
            else:
                ax.text(0.5, 0.5, 'No Branches', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
       
    def _plot_road_vocabulary(self, ax, roads_vocab):
        """Plot road vocabulary details"""
        ax.set_title('Road User-Friendly Names (Sample)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        y_pos = 0.95
        count = 0
        
        for road_id, details in list(roads_vocab.items())[:8]:  # Show first 8
            primary_name = details.get('primary_name', f'Road {road_id}')
            description = details.get('natural_description', 'Standard road')
            
            text = f"Road {road_id}: {primary_name}\n  → {description}"
            ax.text(0.05, y_pos, text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=self.colors['background'], alpha=0.7))
            
            y_pos -= 0.22
            count += 1
        
        if len(roads_vocab) > 8:
            ax.text(0.05, 0.05, f"... and {len(roads_vocab) - 8} more roads",
                   transform=ax.transAxes, fontsize=9, style='italic')
    
    def _plot_intersection_vocabulary(self, ax, intersections_vocab):
        """Plot intersection vocabulary details"""
        ax.set_title('Intersection User-Friendly Names (Sample)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        y_pos = 0.95
        
        for int_id, details in list(intersections_vocab.items())[:8]:  # Show first 8
            primary_name = details.get('primary_name', f'Intersection {int_id}')
            description = details.get('natural_description', 'Standard intersection')
            
            text = f"Int {int_id}: {primary_name}\n  → {description}"
            ax.text(0.05, y_pos, text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=self.colors['background'], alpha=0.7))
            
            y_pos -= 0.22
        
        if len(intersections_vocab) > 8:
            ax.text(0.05, 0.05, f"... and {len(intersections_vocab) - 8} more intersections",
                   transform=ax.transAxes, fontsize=9, style='italic')

    def _plot_landmarks_and_spatial(self, ax, vocab):
        """Plot landmarks and spatial relationships"""
        ax.set_title('Landmarks & Spatial Relationships', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        landmarks = vocab.get('landmarks', {})
        spatial_rels = vocab.get('spatial_relationships', {})
        
        # Landmarks section
        ax.text(0.05, 0.95, '🏢 LANDMARKS:', transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=self.colors['landmark'])
        
        y_pos = 0.85
        if isinstance(landmarks, dict):
            for landmark_id, details in list(landmarks.items())[:5]:
                name = details.get('user_friendly_name', landmark_id) if isinstance(details, dict) else str(details)
                ax.text(0.1, y_pos, f"• {name}", transform=ax.transAxes, fontsize=10)
                y_pos -= 0.08
            
            if len(landmarks) > 5:
                ax.text(0.1, y_pos, f"... and {len(landmarks) - 5} more landmarks",
                    transform=ax.transAxes, fontsize=9, style='italic')
        else:
            ax.text(0.1, y_pos, "No landmarks data available", transform=ax.transAxes, fontsize=10)
            y_pos -= 0.08
        
        # Spatial relationships section
        ax.text(0.05, 0.45, '📍 SPATIAL RELATIONSHIPS:', transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=self.colors['highlight'])
        
        y_pos = 0.35
        
        # ✅ FIX: Handle both list and dict formats for spatial relationships
        if isinstance(spatial_rels, dict):
            # Dictionary format
            for rel_id, details in list(spatial_rels.items())[:5]:
                description = details.get('description', rel_id) if isinstance(details, dict) else str(details)
                ax.text(0.1, y_pos, f"• {description}", transform=ax.transAxes, fontsize=10)
                y_pos -= 0.08
            
            if len(spatial_rels) > 5:
                ax.text(0.1, y_pos, f"... and {len(spatial_rels) - 5} more relationships",
                    transform=ax.transAxes, fontsize=9, style='italic')
        
        elif isinstance(spatial_rels, list):
            # ✅ List format - handle properly
            for i, rel_item in enumerate(spatial_rels[:5]):
                if isinstance(rel_item, dict):
                    description = rel_item.get('description', f'Relationship {i+1}')
                else:
                    description = str(rel_item)
                ax.text(0.1, y_pos, f"• {description}", transform=ax.transAxes, fontsize=10)
                y_pos -= 0.08
            
            if len(spatial_rels) > 5:
                ax.text(0.1, y_pos, f"... and {len(spatial_rels) - 5} more relationships",
                    transform=ax.transAxes, fontsize=9, style='italic')
        
        else:
            # ✅ Handle other data types or empty data
            ax.text(0.1, y_pos, "No spatial relationships data available", 
                transform=ax.transAxes, fontsize=10)


    def audit_all_jsons(self, json_files_paths=None):
        """Create audit visualizations for all JSON files"""
        print("\n🎨 CREATING AUDIT VISUALIZATIONS FOR ALL JSON FILES")
        print("=" * 60)
        
        if json_files_paths is None:
            json_files_paths = {
                'integrated': OUTPUT_INTEGRATED_JSON,
                'intersections': OUTPUT_INTERSECTIONS_JSON,
                'lane_trees': OUTPUT_LANE_TREES_JSON,
                'centerlines': OUTPUT_CENTERLINES_JSON
            }
        
        audit_files = []
        
        # Audit integrated data
        if os.path.exists(json_files_paths['integrated']):
            with open(json_files_paths['integrated'], 'r') as f:
                integrated_data = json.load(f)
            
            # Main network audit
            network_audit = self.create_road_network_audit(
                integrated_data, f"audit_road_network_{self.connection_id or 'local'}.png"
            )
            if network_audit:
                audit_files.append(network_audit)
            
            # Vocabulary audit
            vocab_audit = self.create_vocabulary_audit(
                integrated_data, f"audit_vocabulary_{self.connection_id or 'local'}.png"
            )
            if vocab_audit:
                audit_files.append(vocab_audit)
            
            # Metadata summary
            metadata_audit = self.create_metadata_summary(
                integrated_data, f"audit_metadata_summary_{self.connection_id or 'local'}.png"
            )
            if metadata_audit:
                audit_files.append(metadata_audit)
        
        # Audit intersections
        if os.path.exists(json_files_paths['intersections']):
            with open(json_files_paths['intersections'], 'r') as f:
                intersections_data = json.load(f)
            
            intersections_audit = self.create_intersections_audit(
                intersections_data, f"audit_intersections_{self.connection_id or 'local'}.png"
            )
            if intersections_audit:
                audit_files.append(intersections_audit)
        
        # Audit lane trees
        if os.path.exists(json_files_paths['lane_trees']):
            with open(json_files_paths['lane_trees'], 'r') as f:
                lane_data = json.load(f)
            
            lane_trees_audit = self.create_lane_trees_audit(
                lane_data, f"audit_lane_trees_{self.connection_id or 'local'}.png"
            )
            if lane_trees_audit:
                audit_files.append(lane_trees_audit)
        
        print(f"\n✅ AUDIT VISUALIZATIONS COMPLETE!")
        print(f"📊 Generated {len(audit_files)} audit files:")
        for audit_file in audit_files:
            print(f"  - {audit_file}")
        
        return audit_files

def main():
    """Main function to run audit visualization"""
    connection_id = sys.argv[1] if len(sys.argv) > 1 else None
    auditor = JSONAuditVisualizer(connection_id)
    audit_files = auditor.audit_all_jsons()
    
    print(f"\n🎯 All audit visualizations saved!")
    print(f"📁 Check the generated PNG files for human-readable network analysis")

if __name__ == '__main__':
    import sys
    main()