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


class ProcessKostanaiBuildings:
    def __init__(self, distance_threshold=0.15, min_intersection_length=2.0, 
                 max_group_size=100, min_area_threshold=5.0):
        """
        Initialize the Kostanai Buildings processor.
        
        Args:
            distance_threshold: Maximum distance for grouping buildings (default: 0.15m)
            min_intersection_length: Minimum intersection length for grouping (default: 2.0m)
            max_group_size: Maximum number of buildings in a group (default: 100)
            min_area_threshold: Minimum area to keep buildings (default: 5.0 sq units)
        """
        self.distance_threshold = distance_threshold
        self.min_intersection_length = min_intersection_length
        self.max_group_size = max_group_size
        self.min_area_threshold = min_area_threshold
        self.output_dir = Path(r"./data/raw/labels/kostanai/buildings_kostanai_processed/")
        
    def clean_geometry(self, gdf):
        """Clean the GeoDataFrame by removing rows with null geometry."""
        print("Cleaning geometry data...")
        gdf_clean = gdf.dropna(subset=['geometry'])
        
        print(f"Total rows after cleaning: {len(gdf_clean)}")
        print(f"Rows removed: {len(gdf) - len(gdf_clean)}")
        print(f"Remaining missing geometry: {gdf_clean.geometry.isna().sum()}")
        
        return gdf_clean.reset_index(drop=True)
    
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
            buffered1 = geom1.buffer(buffer_distance)
            buffered2 = geom2.buffer(buffer_distance)
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
        """Create multipolygons from groups of close geometries."""
        multipolygon_rows = []
        grouped_indices = set()
        
        for group in groups:
            group_df = gdf.iloc[group]
            
            # Get geometries to combine into multipolygon
            geometries = group_df.geometry.tolist()
            
            # Create multipolygon from close geometries
            polygon_list = []
            for geom in geometries:
                if geom.geom_type == 'Polygon':
                    polygon_list.append(geom)
                elif geom.geom_type == 'MultiPolygon':
                    polygon_list.extend(list(geom.geoms))
            
            if len(polygon_list) > 1:
                multipolygon_geom = MultiPolygon(polygon_list)
            elif len(polygon_list) == 1:
                multipolygon_geom = polygon_list[0]
            else:
                continue
            
            # Create multipolygon row
            multipolygon_row = group_df.iloc[0].copy()
            multipolygon_row['geometry'] = multipolygon_geom
            multipolygon_row['name_usl'] = 'multipolygon_group'
            multipolygon_row['shape_Area'] = multipolygon_geom.area
            
            # Sum area_build if available
            if 'area_build' in multipolygon_row.index:
                multipolygon_row['area_build'] = group_df['area_build'].sum()
            
            multipolygon_rows.append(multipolygon_row)
            grouped_indices.update(group)
        
        # Create final GeoDataFrame with ungrouped features
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

        original_area = geom.area
        original_centroid = geom.centroid

        # Split into polygon parts
        if isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        elif isinstance(geom, Polygon):
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
    
    def buffer_expand_merge_shrink(self, geom, buffer_distance=0.02):
        """Buffer-based expand-merge-shrink preserving original area and position."""
        if geom.is_empty or geom.area == 0:
            return geom

        original_area = geom.area
        original_centroid = geom.centroid

        # Apply buffer expansion
        buffered = geom.buffer(buffer_distance)
        
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
            # Method 1: Aggressive negative-positive buffering
            sharpened = geom.buffer(-buffer_size).buffer(buffer_size)
            
            if sharpened.is_empty or sharpened.area < original_area * 0.7:
                # If too much area lost, try smaller buffer
                sharpened = geom.buffer(-buffer_size/2).buffer(buffer_size/2)
            
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
                    temp = sharpened.buffer(-small_buffer).buffer(small_buffer)
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
        geom = self.sharpen_building_edges(geom, buffer_size=sharpen_buffer, 
                                         simplify_tolerance=sharpen_tolerance)
        
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
        output_path = self.output_dir / f'buildings_kostanai_processed_{timestamp}.shp'
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
        distance_threshold=0.15,      # 15cm grouping distance
        min_intersection_length=2.0,  # 2m minimum intersection
        max_group_size=100,           # Max 100 buildings per group
        min_area_threshold=5.0        # 5m² minimum area
    )
    
    merger = MergeKostanaiBuildings()
    gdf = merger.process()
    processed_gdf = processor.process_geodataframe_optimized(
        gdf, 
        apply_geometry_processing=True  # Set False to skip geometry enhancement
    )