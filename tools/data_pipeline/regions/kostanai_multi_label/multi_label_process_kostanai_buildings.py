import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.geometry import Polygon, Point, LineString, MultiPoint, MultiPolygon
from shapely.affinity import scale, translate
from scipy.spatial import ConvexHull
from tqdm import tqdm
from collections import deque
import math
from scipy.spatial.distance import cdist
import os
from datetime import datetime
import shutil
from pathlib import Path
from shapely.validation import make_valid


class ProcessKostanaiBuildings:
    def __init__(self, distance_threshold=0.15, min_intersection_length=2.0, 
                 max_group_size=100, min_area_threshold=5.0, use_scaling_merge=False, target_intersection_ratio=0.02, max_scale=3.0, output_dir=None):
        """
        Initialize the Kostanai Buildings processor.
        
        Args:
            distance_threshold: Maximum distance for grouping buildings (default: 0.15m)
            min_intersection_length: Minimum intersection length for grouping (default: 2.0m)
            max_group_size: Maximum number of buildings in a group (default: 100)
            min_area_threshold: Minimum area to keep buildings (default: 5.0 sq units)
            output_dir: Directory to save processed shapefiles (required)
        """
        self.distance_threshold = distance_threshold
        self.min_intersection_length = min_intersection_length
        self.max_group_size = max_group_size
        self.min_area_threshold = min_area_threshold
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.use_scaling_merge = use_scaling_merge
        self.target_intersection_ratio = target_intersection_ratio
        self.max_scale = max_scale
        
    def clean_geometry(self, gdf):
        """Clean the GeoDataFrame by removing rows with null geometry and fixing invalid geometries."""
        print("Cleaning geometry data...")
        gdf_clean = gdf.dropna(subset=['geometry'])
        print(f"Total rows after cleaning: {len(gdf_clean)}")
        print(f"Rows removed: {len(gdf) - len(gdf_clean)}")
        print(f"Remaining missing geometry: {gdf_clean.geometry.isna().sum()}")

        # Set building_type for multi-class instance segmentation
        gdf_clean['building_type'] = gdf_clean['name_usl'].apply(
            lambda x: 'residential' if x == 'Здания и сооружения жилые' else 'non-residential')

        # Fix invalid geometries using make_valid, fallback to buffer(0)
        def fix_geom(geom):
            if geom is None or geom.is_empty:
                return geom
            if not geom.is_valid:
                try:
                    fixed = make_valid(geom)
                    if fixed.is_valid:
                        return fixed
                except Exception:
                    pass
                try:
                    fixed = geom.buffer(0)
                    if fixed.is_valid:
                        return fixed
                except Exception:
                    pass
                print("Warning: Could not fix invalid geometry, setting to None.")
                return None
            return geom
        gdf_clean['geometry'] = gdf_clean['geometry'].apply(fix_geom)
        # Drop any that are still None or empty
        gdf_clean = gdf_clean.dropna(subset=['geometry'])
        gdf_clean = gdf_clean[~gdf_clean.geometry.is_empty]
        gdf_clean = gdf_clean.reset_index(drop=True)
        return gdf_clean
    
    def compute_area_if_missing(self, gdf):
        """Compute area if shape_Area column is null or missing."""
        print("Checking and computing areas...")
        
        # Create shape_Area column if it doesn't exist
        if 'shape_Area' not in gdf.columns:
            gdf['shape_Area'] = np.nan
        
        # Compute area for null values
        null_mask = gdf['shape_Area'].isna()
        if null_mask.any():
            print(f"Computing area for {null_mask.sum()} rows with missing area values")
            gdf.loc[null_mask, 'shape_Area'] = gdf.loc[null_mask, 'geometry'].area
        
        return gdf
    
    def get_intersection_length(self, geom1, geom2, buffer_distance=None):
        """Calculate the intersection length between two geometries after buffering."""
        if buffer_distance is None:
            buffer_distance = self.distance_threshold
            
        try:
            buffered1 = geom1.buffer(buffer_distance, join_style='mitre', mitre_limit=5.0)
            buffered2 = geom2.buffer(buffer_distance, join_style='mitre', mitre_limit=5.0)
            intersection = buffered1.intersection(buffered2)
            
            if intersection.is_empty:
                return 0
            
            if hasattr(intersection, 'length'):
                return intersection.length
            elif hasattr(intersection, 'geoms'):
                total_length = 0
                for geom in intersection.geoms:
                    if hasattr(geom, 'length'):
                        total_length += geom.length
                    elif hasattr(geom, 'area') and geom.area > 0:
                        total_length += geom.length
                return total_length
            elif hasattr(intersection, 'area') and intersection.area > 0:
                return intersection.length
            else:
                return 0
        except Exception as e:
            print(f"Error calculating intersection length: {e}")
            return 0
    
    def find_all_connected_components(self, gdf):
        """Find all connected components using Union-Find algorithm."""
        if len(gdf) == 0:
            return []
        
        n = len(gdf)
        
        # Create spatial index for efficient spatial queries
        tree = STRtree(gdf.geometry)
        
        # Union-Find data structure with group size tracking
        parent = list(range(n))
        rank = [0] * n
        group_size = [1] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False  # Already in same group
            
            # Check if union would exceed max_group_size
            if group_size[px] + group_size[py] > self.max_group_size:
                return False  # Reject union to maintain size limit
            
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            group_size[px] += group_size[py]
            if rank[px] == rank[py]:
                rank[px] += 1
            return True  # Union successful
        
        # Pre-filter: exclude canopy structures from grouping
        valid_indices = []
        for i in range(n):
            if gdf.iloc[i].get('name_usl', '') != "Навесы и перекрытия между зданиями":
                valid_indices.append(i)
        
        print(f"Processing {len(valid_indices)} non-canopy geometries out of {n} total...")
        print(f"Maximum group size limited to {self.max_group_size} objects")
        
        # Build adjacency relationships in a single pass
        processed_pairs = set()
        successful_unions = 0
        rejected_unions = 0
        
        for i in valid_indices:
            geom_i = gdf.geometry.iloc[i]
            
            # Find all geometries within distance using spatial index
            buffered = geom_i.buffer(self.distance_threshold)
            candidates = tree.query(buffered, predicate='intersects')
            
            for j in candidates:
                if j <= i or j not in valid_indices:  # Avoid duplicate pairs and invalid indices
                    continue
                
                pair = (i, j)
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                geom_j = gdf.geometry.iloc[j]
                
                # Check distance threshold
                if geom_i.distance(geom_j) <= self.distance_threshold:
                    # Check intersection length
                    intersection_length = self.get_intersection_length(geom_i, geom_j)
                    
                    if intersection_length >= self.min_intersection_length:
                        if union(i, j):
                            successful_unions += 1
                        else:
                            rejected_unions += 1
        
        print(f"Successful unions: {successful_unions}")
        print(f"Rejected unions (would exceed max size): {rejected_unions}")
        
        # Group indices by their root parent
        groups = {}
        for i in valid_indices:
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Return only groups with multiple geometries
        connected_components = []
        for group in groups.values():
            if len(group) > 1:
                if len(group) <= self.max_group_size:
                    connected_components.append(group)
                else:
                    print(f"Warning: Found group with {len(group)} objects (should not happen)")
        
        print(f"Found {len(connected_components)} connected components")
        group_sizes = [len(group) for group in connected_components]
        if group_sizes:
            print(f"Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, avg={sum(group_sizes)/len(group_sizes):.1f}")
        
        return connected_components
    
    def create_multipolygons_from_groups(self, gdf, groups):
        """Create multipolygons from groups of geometries, ensuring validity before union."""
        multipolygon_rows = []
        grouped_indices = set()
        for group in groups:
            group_df = gdf.iloc[group]
            geometries = [make_valid(geom) if not geom.is_valid else geom for geom in group_df.geometry]
            # Remove any empty or None
            geometries = [g for g in geometries if g is not None and not g.is_empty]
            if not geometries:
                continue
            try:
                merged = unary_union(geometries)
            except Exception as e:
                print(f"Warning: Failed to union group due to: {e}. Attempting buffer(0) fallback.")
                try:
                    merged = unary_union([g.buffer(0) for g in geometries])
                except Exception as e2:
                    print(f"Error: Could not fix group, skipping. {e2}")
                    continue
            multipolygon_row = group_df.iloc[0].copy()
            multipolygon_row['geometry'] = merged
            multipolygon_rows.append(multipolygon_row)
            grouped_indices.update(group)
        # Combine with ungrouped features
        ungrouped_indices = [i for i in range(len(gdf)) if i not in grouped_indices]
        ungrouped_gdf = gdf.iloc[ungrouped_indices]
        if multipolygon_rows:
            multipolygon_gdf = gpd.GeoDataFrame(multipolygon_rows, crs=gdf.crs)
            result = pd.concat([ungrouped_gdf, multipolygon_gdf], ignore_index=True)
        else:
            result = ungrouped_gdf
        return result
    
    def expand_merge_shrink_to_original(self, geom, linear_expand_dist=0.15):
        """Expand geometry by fixed distance, merge parts, then shrink back to original area."""
        if geom.is_empty or geom.area == 0:
            return geom

        # CRITICAL FIX: Calculate the actual union area (removing overlaps)
        # instead of summing individual polygon areas
        if isinstance(geom, MultiPolygon):
            # For MultiPolygon, first create union to get true area without overlaps
            union_geom = unary_union(geom)
            original_area = union_geom.area
            original_centroid = union_geom.centroid
            # Work with the union geometry to avoid overlap issues
            polygons = [union_geom] if isinstance(union_geom, Polygon) else list(union_geom.geoms)
        elif isinstance(geom, Polygon):
            original_area = geom.area
            original_centroid = geom.centroid
            polygons = [geom]
        else:
            return geom  # Skip non-polygon geometries

        expanded_parts = []
        for poly in polygons:
            area = poly.area
            if area == 0:
                continue

            # Estimate a characteristic radius assuming circular shape
            r = math.sqrt(area / math.pi)
            scale_factor = 1 + (linear_expand_dist / r)

            # Scale polygon outward (preserving shape)
            expanded = scale(poly, xfact=scale_factor, yfact=scale_factor, origin='centroid')
            expanded_parts.append(expanded)

        # Merge expanded parts into a single shape
        merged = unary_union(expanded_parts)

        # Now shrink dynamically to restore original area
        current_area = merged.area
        if current_area == 0:
            return merged  # avoid division by zero
        
        shrink_factor = math.sqrt(original_area / current_area)
        restored = scale(merged, xfact=shrink_factor, yfact=shrink_factor, origin='centroid')
        
        # Fix position shift by translating back to original centroid
        restored_centroid = restored.centroid
        dx = original_centroid.x - restored_centroid.x
        dy = original_centroid.y - restored_centroid.y
        restored = translate(restored, xoff=dx, yoff=dy)

        return restored
    
    def scale_until_intersection(self, geom1, geom2, target_intersection_ratio=0.02, max_scale=3.0, scale_step=0.01):
        """
        Scale two geometries until their intersection area is at least target_intersection_ratio of their total area.
        
        Args:
            geom1, geom2: Input geometries
            target_intersection_ratio: Target intersection ratio (default: 2%)
            max_scale: Maximum scaling factor (default: 3.0)
            scale_step: Scaling increment (default: 0.01)
        
        Returns:
            tuple: (scaled_geom1, scaled_geom2, achieved_ratio, scale_factor)
        """
        if geom1.is_empty or geom2.is_empty:
            return geom1, geom2, 0, 1.0
        
        original_area1 = geom1.area
        original_area2 = geom2.area
        total_original_area = original_area1 + original_area2
        
        current_scale = 1.0
        best_scale = 1.0
        best_ratio = 0
        
        while current_scale <= max_scale:
            # Scale both geometries from their centroids
            scaled_geom1 = scale(geom1, xfact=current_scale, yfact=current_scale, origin='centroid')
            scaled_geom2 = scale(geom2, xfact=current_scale, yfact=current_scale, origin='centroid')
            
            # Calculate intersection
            intersection = scaled_geom1.intersection(scaled_geom2)
            
            if not intersection.is_empty:
                intersection_area = intersection.area
                current_ratio = intersection_area / total_original_area
                
                if current_ratio >= target_intersection_ratio:
                    return scaled_geom1, scaled_geom2, current_ratio, current_scale
                
                # Keep track of best achieved ratio
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_scale = current_scale
            
            current_scale += scale_step
        
        # If target not reached, return best achieved
        if best_scale > 1.0:
            scaled_geom1 = scale(geom1, xfact=best_scale, yfact=best_scale, origin='centroid')
            scaled_geom2 = scale(geom2, xfact=best_scale, yfact=best_scale, origin='centroid')
            return scaled_geom1, scaled_geom2, best_ratio, best_scale
        
        return geom1, geom2, 0, 1.0

    def merge_buildings_by_scaling(self, geom1, geom2, target_intersection_ratio=0.02, max_scale=3.0):
        """
        Merge two buildings by scaling them until intersection, then union and scale back.
        
        Args:
            geom1, geom2: Input geometries to merge
            target_intersection_ratio: Target intersection ratio (default: 2%)
            max_scale: Maximum scaling factor
        
        Returns:
            Merged geometry with original total area preserved
        """
        if geom1.is_empty or geom2.is_empty:
            return unary_union([geom1, geom2])
        
        # Store original areas and centroids
        original_area1 = geom1.area
        original_area2 = geom2.area
        total_original_area = original_area1 + original_area2
        
        original_centroid1 = geom1.centroid
        original_centroid2 = geom2.centroid
        
        # Scale until intersection
        scaled_geom1, scaled_geom2, achieved_ratio, scale_factor = self.scale_until_intersection(
            geom1, geom2, target_intersection_ratio, max_scale
        )
        
        # If no meaningful intersection achieved, return original union
        if achieved_ratio < 0.001:  # Less than 0.1%
            return unary_union([geom1, geom2])
        
        # Union the scaled geometries
        union_geom = unary_union([scaled_geom1, scaled_geom2])
        
        if union_geom.is_empty:
            return unary_union([geom1, geom2])
        
        # Calculate scale-down factor to restore original total area
        current_area = union_geom.area
        if current_area <= 0:
            return unary_union([geom1, geom2])
        
        area_scale_factor = math.sqrt(total_original_area / current_area)
        
        # Scale down to original area
        restored_geom = scale(union_geom, xfact=area_scale_factor, yfact=area_scale_factor, origin='centroid')
        
        # Optionally adjust position to be between original centroids
        restored_centroid = restored_geom.centroid
        target_centroid_x = (original_centroid1.x + original_centroid2.x) / 2
        target_centroid_y = (original_centroid1.y + original_centroid2.y) / 2
        
        dx = target_centroid_x - restored_centroid.x
        dy = target_centroid_y - restored_centroid.y
        
        final_geom = translate(restored_geom, xoff=dx, yoff=dy)
        
        return final_geom

    def check_buildings_for_scaling_merge(self, geom1, geom2, distance_threshold=None, 
                                        target_intersection_ratio=0.02, max_scale=3.0):
        """
        Check if two buildings should be merged using the scaling method.
        
        Args:
            geom1, geom2: Geometries to check
            distance_threshold: Maximum distance to consider for merging
            target_intersection_ratio: Required intersection ratio after scaling
            max_scale: Maximum allowed scaling factor
        
        Returns:
            bool: True if buildings should be merged
        """
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
        
        # Basic distance check
        if geom1.distance(geom2) > distance_threshold:
            return False
        
        # Check if scaling can achieve target intersection
        _, _, achieved_ratio, scale_factor = self.scale_until_intersection(
            geom1, geom2, target_intersection_ratio, max_scale
        )
        
        # Merge if we can achieve target intersection within reasonable scaling
        return achieved_ratio >= target_intersection_ratio and scale_factor <= max_scale

    def find_connected_components_with_scaling(self, gdf, target_intersection_ratio=0.02, max_scale=3.0):
        """
        Find connected components using the new scaling-based intersection method.
        Modified version of find_all_connected_components.
        """
        if len(gdf) == 0:
            return []
        
        n = len(gdf)
        tree = STRtree(gdf.geometry)
        
        # Union-Find data structure
        parent = list(range(n))
        rank = [0] * n
        group_size = [1] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            
            if group_size[px] + group_size[py] > self.max_group_size:
                return False
            
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            group_size[px] += group_size[py]
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Pre-filter valid indices (exclude canopies)
        valid_indices = []
        for i in range(n):
            if gdf.iloc[i].get('name_usl', '') != "Навесы и перекрытия между зданиями":
                valid_indices.append(i)
        
        print(f"Processing {len(valid_indices)} geometries with scaling-based intersection...")
        
        processed_pairs = set()
        successful_unions = 0
        rejected_unions = 0
        
        for i in valid_indices:
            geom_i = gdf.geometry.iloc[i]
            
            # Find candidates using spatial index
            buffered = geom_i.buffer(self.distance_threshold)
            candidates = tree.query(buffered, predicate='intersects')
            
            for j in candidates:
                if j <= i or j not in valid_indices:
                    continue
                
                pair = (i, j)
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                geom_j = gdf.geometry.iloc[j]
                
                # Use new scaling-based check
                if self.check_buildings_for_scaling_merge(
                    geom_i, geom_j, self.distance_threshold, 
                    target_intersection_ratio, max_scale
                ):
                    if union(i, j):
                        successful_unions += 1
                    else:
                        rejected_unions += 1
        
        print(f"Scaling-based unions: {successful_unions}")
        print(f"Rejected unions: {rejected_unions}")
        
        # Group indices by root parent
        groups = {}
        for i in valid_indices:
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Return groups with multiple geometries
        connected_components = []
        for group in groups.values():
            if len(group) > 1:
                connected_components.append(group)
        
        return connected_components

    def create_multipolygons_with_scaling_merge(self, gdf, groups, target_intersection_ratio=0.02, max_scale=3.0):
        """
        Create multipolygons using the scaling-based merge method.
        Modified version of create_multipolygons_from_groups.
        """
        multipolygon_rows = []
        grouped_indices = set()
        
        for group in groups:
            group_df = gdf.iloc[group]
            geometries = group_df.geometry.tolist()
            
            # For pairs, use scaling merge; for larger groups, use traditional approach
            if len(geometries) == 2:
                merged_geom = self.merge_buildings_by_scaling(
                    geometries[0], geometries[1], target_intersection_ratio, max_scale
                )
            else:
                # For larger groups, merge pairwise or use traditional union
                merged_geom = unary_union(geometries)
            
            # Create multipolygon row
            multipolygon_row = group_df.iloc[0].copy()
            multipolygon_row['geometry'] = merged_geom
            multipolygon_row['name_usl'] = 'scaling_merged_group'
            multipolygon_row['shape_Area'] = merged_geom.area
            
            if 'area_build' in multipolygon_row.index:
                multipolygon_row['area_build'] = group_df['area_build'].sum()
            
            multipolygon_rows.append(multipolygon_row)
            grouped_indices.update(group)
        
        # Combine with ungrouped features
        ungrouped_indices = [i for i in range(len(gdf)) if i not in grouped_indices]
        ungrouped_gdf = gdf.iloc[ungrouped_indices]
        
        if multipolygon_rows:
            multipolygon_gdf = gpd.GeoDataFrame(multipolygon_rows, crs=gdf.crs)
            result = pd.concat([ungrouped_gdf, multipolygon_gdf], ignore_index=True)
        else:
            result = ungrouped_gdf
        
        return result
    
    def buffer_expand_merge_shrink(self, geom, buffer_distance=0.02):
        """Buffer-based expand-merge-shrink preserving original area and position."""
        if geom.is_empty or geom.area == 0:
            return geom

        # CRITICAL FIX: Calculate the actual union area (removing overlaps)
        if isinstance(geom, MultiPolygon):
            # For MultiPolygon, first create union to get true area without overlaps
            union_geom = unary_union(geom)
            original_area = union_geom.area
            original_centroid = union_geom.centroid
            # Work with the union geometry
            working_geom = union_geom
        else:
            original_area = geom.area
            original_centroid = geom.centroid
            working_geom = geom

        # Apply buffer expansion with mitre join style
        buffered = working_geom.buffer(buffer_distance, join_style='mitre', mitre_limit=5.0)
        
        # Merge any overlapping parts
        merged = unary_union([buffered]) if isinstance(buffered, MultiPolygon) else buffered

        # Calculate required shrink factor to restore original area
        current_area = merged.area
        if current_area == 0:
            return merged
        shrink_factor = math.sqrt(original_area / current_area)
        
        # Shrink back to original area
        restored = scale(merged, xfact=shrink_factor, yfact=shrink_factor, origin='centroid')
        
        # Fix position shift by translating back to original centroid
        restored_centroid = restored.centroid
        dx = original_centroid.x - restored_centroid.x
        dy = original_centroid.y - restored_centroid.y
        restored = translate(restored, xoff=dx, yoff=dy)

        return restored
    
    def sharpen_building_edges(self, geom, buffer_size=0.05, simplify_tolerance=0.2):
        """Aggressively sharpen building edges by removing rounded corners."""
        if geom.is_empty or geom.area == 0:
            return geom
        
        original_area = geom.area
        
        try:
            # Method 1: Aggressive negative-positive buffering with mitre join style
            sharpened = geom.buffer(-buffer_size, join_style='mitre', mitre_limit=5.0).buffer(buffer_size, join_style='mitre', mitre_limit=5.0)
            
            if sharpened.is_empty or sharpened.area < original_area * 0.7:
                # If too much area lost, try smaller buffer
                sharpened = geom.buffer(-buffer_size/2, join_style='mitre', mitre_limit=5.0).buffer(buffer_size/2, join_style='mitre', mitre_limit=5.0)
            
            if sharpened.is_empty:
                # Fallback to original if buffering failed
                sharpened = geom
            
            # Method 2: Simplification to remove small segments
            if not sharpened.is_empty:
                simplified = sharpened.simplify(simplify_tolerance, preserve_topology=True)
                if not simplified.is_empty and simplified.area > original_area * 0.8:
                    sharpened = simplified
            
            # Method 3: Multiple rounds of small negative buffering
            if not sharpened.is_empty:
                for i in range(3):
                    small_buffer = buffer_size / 10
                    temp = sharpened.buffer(-small_buffer, join_style='mitre', mitre_limit=5.0).buffer(small_buffer, join_style='mitre', mitre_limit=5.0)
                    if not temp.is_empty and temp.area > original_area * 0.75:
                        sharpened = temp
                    else:
                        break
            
            return sharpened if not sharpened.is_empty else geom
            
        except Exception as e:
            print(f"Warning: Sharpening failed, returning original geometry: {e}")
            return geom
    
    def remove_small_holes(self, geom, min_hole_ratio=0.05):
        """Remove holes that are significantly smaller than the main polygon area."""
        if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
            return geom
        
        def process_polygon(poly):
            if not isinstance(poly, Polygon) or len(poly.interiors) == 0:
                return poly
            
            exterior = poly.exterior
            poly_area = poly.area
            
            # Keep only holes that are large enough relative to the polygon
            kept_holes = []
            for interior in poly.interiors:
                hole_poly = Polygon(interior)
                hole_area = hole_poly.area
                
                # Keep hole if it's large enough or if polygon is very small
                if hole_area / poly_area >= min_hole_ratio or poly_area < 100:
                    kept_holes.append(interior)
            
            return Polygon(exterior, holes=kept_holes)
        
        if isinstance(geom, MultiPolygon):
            processed_parts = []
            for poly in geom.geoms:
                processed = process_polygon(poly)
                if not processed.is_empty:
                    processed_parts.append(processed)
            
            if processed_parts:
                return MultiPolygon(processed_parts) if len(processed_parts) > 1 else processed_parts[0]
            else:
                return geom
        else:
            return process_polygon(geom)
    
    def process_building_geometry(self, geom, linear_expand_dist=0.15, buffer_distance=0.02, 
                                sharpen_buffer=0.05, sharpen_tolerance=0.2, min_hole_ratio=0.05):
        """Complete building processing pipeline."""
        # Stage 1: Expand-merge-shrink with scaling
        geom = self.expand_merge_shrink_to_original(geom, linear_expand_dist)
        
        # Stage 2: Buffer-based expand-merge-shrink
        geom = self.buffer_expand_merge_shrink(geom, buffer_distance)
        
        # Stage 3: Sharpen edges
        # geom = self.sharpen_building_edges(geom, buffer_size=sharpen_buffer, 
        #                                  simplify_tolerance=sharpen_tolerance)
        
        # Stage 4: Remove small holes
        geom = self.remove_small_holes(geom, min_hole_ratio)
        
        return geom
    
    def apply_area_filter(self, gdf):
        """Apply minimum area filter to the GeoDataFrame."""
        print(f"Applying {self.min_area_threshold}m² area filter...")
        
        # Ensure area is computed for comparison
        gdf = self.compute_area_if_missing(gdf)
        
        original_count = len(gdf)
        gdf_filtered = gdf[gdf['shape_Area'] >= self.min_area_threshold]
        gdf_filtered = gdf_filtered.reset_index(drop=True)
        
        removed_count = original_count - len(gdf_filtered)
        print(f"Removed {removed_count} buildings below {self.min_area_threshold}m² threshold")
        
        return gdf_filtered
    
    def process_geodataframe_optimized(self, gdf, apply_geometry_processing=True, save_to_shapefile=True):
        """Main processing pipeline for Kostanai buildings."""
        print(f"Original GDF shape: {gdf.shape}")
        
        # Step 1: Clean geometry
        gdf = self.clean_geometry(gdf)
        
        # Step 2: Compute areas if missing
        gdf = self.compute_area_if_missing(gdf)

        # Step 3: Choose merging method        
        if self.use_scaling_merge:
            print("Using scaling-based intersection merging...")
            connected_components = self.find_connected_components_with_scaling(
                gdf, self.target_intersection_ratio, self.max_scale
            )
            gdf = self.create_multipolygons_with_scaling_merge(
                gdf, connected_components, self.target_intersection_ratio, self.max_scale
            )
        else:
            print("Using traditional buffer-based merging...")
            connected_components = self.find_all_connected_components(gdf)
            gdf = self.create_multipolygons_from_groups(gdf, connected_components)
        
        # Step 3: Find connected components and create multipolygons
        print("Finding connected components of close geometries...")
        connected_components = self.find_all_connected_components(gdf)
        
        print("Creating multipolygons from connected components...")
        gdf = self.create_multipolygons_from_groups(gdf, connected_components)
        
        print(f"After creating multipolygons: {gdf.shape}")
        print("Multipolygon groups count:", (gdf['name_usl'] == 'multipolygon_group').sum())
        
        # Step 4: Apply geometry processing if requested
        if apply_geometry_processing:
            print(f"Processing {len(gdf)} building polygons with geometry enhancement...")
            gdf['geometry'] = gdf['geometry'].apply(self.process_building_geometry)
            print("Geometry processing completed")
        
        # Step 5: Apply area filter
        gdf = self.apply_area_filter(gdf)
        
        print("\n=== FINAL RESULTS ===")
        print(f"Final GDF shape: {gdf.shape}")
        print("\nValue counts:")
        if 'name_usl' in gdf.columns:
            print(gdf['name_usl'].value_counts())
            print(f"\nCanopy structures preserved: {(gdf['name_usl'] == 'Навесы и перекрытия между зданиями').sum()}")
        
        if save_to_shapefile:
            self.save_processed_data(gdf)
            
        return gdf
    
    def save_processed_data(self, gdf):
        """Save the processed GeoDataFrame to a shapefile."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime columns to strings
        for col in gdf.columns:
            if pd.api.types.is_datetime64_any_dtype(gdf[col]):
                gdf[col] = gdf[col].astype(str)
        
        # Remove everything in the output directory
        for item in self.output_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Save to shapefile
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f'buildings_processed_{timestamp}.shp'
        gdf.to_file(output_path, driver='ESRI Shapefile')
        
        print(f"✅ Saved processed shapefile to:\n{output_path}")
        return output_path

if __name__ == "__main__":
    """ 
    python tools/data_pipeline/regions/kostanai/process_kostanai_buildings.py

    """
    
    from merge_raw_kostanai_buildings import MergeKostanaiBuildings
    
    # Initialize with custom parameters
    processor = ProcessKostanaiBuildings(
        distance_threshold=0.15,
        min_intersection_length=2.0,  # Not used in scaling method
        max_group_size=100,
        min_area_threshold=5.0,
        use_scaling_merge=True,       # New scaling method
        target_intersection_ratio=0.02,  # 2% intersection requirement
        max_scale=1.5              # Maximum 1.5x scaling
    )

    # processor_traditional = ProcessKostanaiBuildings(
    #     distance_threshold=0.15,
    #     min_intersection_length=2.0,
    #     max_group_size=100,
    #     min_area_threshold=5.0,
    #     use_scaling_merge=False  # Traditional method
    # )
    
    merger = MergeKostanaiBuildings()
    gdf = merger.process()
    processed_gdf = processor.process_geodataframe_optimized(gdf)