import cv2
import numpy as np
import os
import json
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box, Polygon
from shapely.ops import transform, unary_union
from shapely.validation import make_valid
from shapely.strtree import STRtree
import pyproj
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
import shutil
import hashlib
import random

warnings.filterwarnings('ignore')

class OptimizedTIFFChunkerWithShapefiles:
    def __init__(self, 
                 tiff_dir, 
                 region_shapefile_path, 
                 labels_gdf=None,
                 output_dir=None,
                 chunk_size=(640, 640, 3),
                 overlap=0,
                 original_resolution=5.0,
                 target_resolution=30.0,
                 train_split=0.7,
                 val_split=0.15,
                 test_split=0.15,
                 min_valid_pixels=0.3,
                 rewrite_output_dir=False,
                 use_train_val_test=True):
        """
        Initialize the optimized TIFF chunker with shapefile integration.
        
        Args:
            tiff_dir: Path to directory containing TIFF files
            region_shapefile_path: Path to region shapefile
            labels_gdf: GeoDataFrame containing building labels (optional, can be loaded from path)
            output_dir: Output directory for processed data
            chunk_size: Size of image chunks (height, width, channels)
            overlap: Overlap between chunks in pixels
            original_resolution: Original image resolution in meters
            target_resolution: Target resolution in meters
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            min_valid_pixels: Minimum fraction of valid pixels in a chunk
            rewrite_output_dir: If True, DELETE existing output directory contents. WARNING: This will permanently delete all existing data! (default: False)
            use_train_val_test: If True, create train/val/test structure. If False, preserve existing structure for custom splits.
        """
        self.tiff_dir = Path(tiff_dir)
        self.region_shapefile_path = region_shapefile_path
        self.labels_gdf = labels_gdf
        self.output_dir = Path(output_dir) if output_dir else None
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.original_resolution = original_resolution
        self.target_resolution = target_resolution
        self.scale_factor = original_resolution / target_resolution
        self.min_valid_pixels = min_valid_pixels
        self.rewrite_output_dir = rewrite_output_dir
        self.use_train_val_test = use_train_val_test
        
        # Split ratios
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Create output directories with updated structure (only if output_dir is provided)
        if self.output_dir:
            # Only rewrite output directory if explicitly requested AND we're using train/val/test structure
            if self.use_train_val_test and self.rewrite_output_dir and self.output_dir.exists():
                print(f"⚠️  WARNING: DELETING existing output directory: {self.output_dir}")
                print("   This will permanently delete all existing data!")
                print("   Set rewrite_output_dir=False to preserve existing data.")
                shutil.rmtree(self.output_dir)
            elif self.use_train_val_test and self.output_dir.exists() and not self.rewrite_output_dir:
                print(f"📁 Using existing output directory: {self.output_dir}")
                print("   Existing data will be preserved. Set rewrite_output_dir=True to delete and recreate.")
            
            # Only create train/val/test directories if we're using that structure
            if self.use_train_val_test:
                self.train_dir = self.output_dir / 'train'
                self.val_dir = self.output_dir / 'val'
                self.test_dir = self.output_dir / 'test'
                
                # Create image subdirectories
                self.train_img_dir = self.train_dir / 'images'
                self.val_img_dir = self.val_dir / 'images'
                self.test_img_dir = self.test_dir / 'images'
                
                for dir_path in [self.train_dir, self.val_dir, self.test_dir,
                                self.train_img_dir, self.val_img_dir, self.test_img_dir]:
                    dir_path.mkdir(parents=True, exist_ok=True)
            else:
                # For custom splits, just ensure the base output directory exists
                self.output_dir.mkdir(parents=True, exist_ok=True)
                # Initialize these as None for custom splits
                self.train_dir = None
                self.val_dir = None
                self.test_dir = None
                self.train_img_dir = None
                self.val_img_dir = None
                self.test_img_dir = None
        
        # Load shapefiles and create spatial index
        self.region_gdf = None
        self.labels_tree = None
        self._load_shapefiles()
        
        # COCO data structures
        self.coco_data = {
            'train': self._init_coco_structure(),
            'val': self._init_coco_structure(),
            'test': self._init_coco_structure()
        }
        
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        
        # Cache for transformations
        self.transform_cache = {}
    
    def _init_coco_structure(self):
        """Initialize COCO format structure."""
        return {
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "supercategory": "building",
                    "id": 0,
                    "name": "residential"
                },
                {
                    "supercategory": "building",
                    "id": 1,
                    "name": "non-residential"
                }
            ],
            "info": {
                "description": "Residential and non-residential segmentation dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Optimized TIFF Chunker with Shapefiles",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 4,
                    "name": "Attribution License",
                    "url": "http://creativecommons.org/licenses/by/2.0/"
                }
            ]
        }
    
    def _make_valid_geometry(self, geom):
        """Ensure geometry is valid, fixing if necessary."""
        if not geom.is_valid:
            return make_valid(geom)
        return geom
    
    def _load_shapefiles(self):
        """Load and validate shapefiles, create spatial index."""
        print("Loading shapefiles...")
        
        # Load region shapefile
        self.region_gdf = gpd.read_file(self.region_shapefile_path)
        if self.region_gdf.crs.to_epsg() != 32641:
            self.region_gdf = self.region_gdf.to_crs('EPSG:32641')
        
        # Validate and fix geometries
        self.region_gdf['geometry'] = self.region_gdf['geometry'].apply(self._make_valid_geometry)
        
        # Load labels shapefile only if not provided
        if self.labels_gdf is None:
            raise ValueError("labels_gdf must be provided as a parameter")
        
        # Ensure labels_gdf is in the correct CRS
        if self.labels_gdf.crs.to_epsg() != 32641:
            self.labels_gdf = self.labels_gdf.to_crs('EPSG:32641')
        
        # Validate and fix geometries
        self.labels_gdf['geometry'] = self.labels_gdf['geometry'].apply(self._make_valid_geometry)
        
        # Create spatial index for faster queries
        print("Creating spatial index...")
        self.labels_tree = STRtree(self.labels_gdf.geometry.values)
        
        # Get the region polygon (assuming single row)
        if len(self.region_gdf) > 1:
            print(f"Region shapefile has {len(self.region_gdf)} rows. Merging into a single geometry.")
            # Merge all geometries in the region shapefile into one
            geometries = [self._make_valid_geometry(geom) for geom in self.region_gdf.geometry]
            self.region_polygon = unary_union(geometries)
        elif len(self.region_gdf) == 1:
            self.region_polygon = self._make_valid_geometry(self.region_gdf.geometry.iloc[0])
        else:
            raise ValueError("Region shapefile is empty and does not contain any polygons.")
            
        print(f"Loaded {len(self.labels_gdf)} labels with spatial index")
    
    def _get_tiff_transform_info(self, tiff_path):
        """Get and cache TIFF transformation information."""
        if tiff_path not in self.transform_cache:
            with rasterio.open(tiff_path) as src:
                transform = src.transform
                src_crs = src.crs
                
                # Create transformer if needed
                transformer = None
                if src_crs.to_epsg() != 32641:
                    transformer = pyproj.Transformer.from_crs(src_crs, 'EPSG:32641', always_xy=True)
                
                self.transform_cache[tiff_path] = {
                    'transform': transform,
                    'src_crs': src_crs,
                    'transformer': transformer
                }
        
        return self.transform_cache[tiff_path]
    
    def _batch_pixel_to_geo_coords(self, tiff_path, pixel_coords_list):
        """Convert multiple pixel coordinates to geographic coordinates efficiently."""
        transform_info = self._get_tiff_transform_info(tiff_path)
        transform = transform_info['transform']
        
        geo_coords_list = []
        for pixel_coords in pixel_coords_list:
            geo_coords = []
            for px_x, px_y in pixel_coords:
                geo_x, geo_y = rasterio.transform.xy(transform, px_y, px_x)
                geo_coords.append((geo_x, geo_y))
            geo_coords_list.append(geo_coords)
        
        return geo_coords_list
    
    def _batch_geo_to_pixel_coords(self, tiff_path, geo_coords_list):
        """Convert multiple geographic coordinates to pixel coordinates efficiently."""
        transform_info = self._get_tiff_transform_info(tiff_path)
        transform = transform_info['transform']
        
        pixel_coords_list = []
        for geo_coords in geo_coords_list:
            pixel_coords = []
            for geo_x, geo_y in geo_coords:
                px_x, px_y = rasterio.transform.rowcol(transform, geo_x, geo_y)
                pixel_coords.append((px_y, px_x))
            pixel_coords_list.append(pixel_coords)
        
        return pixel_coords_list
    
    def _check_tiff_region_overlap(self, tiff_path):
        """Check if TIFF overlaps with region polygon."""
        with rasterio.open(tiff_path) as src:
            bounds = src.bounds
            src_crs = src.crs
            
            # Transform to EPSG:32641 if needed
            if src_crs.to_epsg() != 32641:
                transformer = pyproj.Transformer.from_crs(src_crs, 'EPSG:32641', always_xy=True)
                min_x, min_y = transformer.transform(bounds.left, bounds.bottom)
                max_x, max_y = transformer.transform(bounds.right, bounds.top)
                bounds = (min_x, min_y, max_x, max_y)
            else:
                bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            tiff_polygon = self._make_valid_geometry(box(*bounds))
            return self.region_polygon.intersects(tiff_polygon)
    
    def _create_mask_for_region(self, tiff_path, img_shape):
        """Create a mask where True=pixels inside region, False=outside"""
        try:
            with rasterio.open(tiff_path) as src:
                # Calculate transform for the downscaled image
                scale_transform = src.transform * src.transform.scale(
                    (src.width / img_shape[1]),
                    (src.height / img_shape[0])
                )
                
                # Create mask
                mask = features.geometry_mask(
                    [self.region_polygon],
                    out_shape=img_shape[:2],
                    transform=scale_transform,
                    invert=True,  # True=inside region
                    all_touched=True
                )
                
                # Convert to 3D if needed
                if len(img_shape) == 3:
                    mask = np.repeat(mask[:, :, np.newaxis], img_shape[2], axis=2)
                
                return mask
        
        except Exception as e:
            print(f"Error creating mask for {tiff_path}: {str(e)}")
            # Return all False mask if there's an error
            if len(img_shape) == 3:
                return np.zeros(img_shape, dtype=bool)
            return np.zeros(img_shape[:2], dtype=bool)
        
    def _batch_process_chunks(self, tiff_path, chunks_info):
        """Process multiple chunks at once for better efficiency."""
        # Get all chunk bounds
        chunk_bounds_list = [chunk['bounds'] for chunk in chunks_info]
        
        # Convert all bounds to geo coordinates at once
        pixel_corners_list = []
        for bounds in chunk_bounds_list:
            min_x, min_y, max_x, max_y = bounds
            pixel_corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            pixel_corners_list.append(pixel_corners)
        
        geo_corners_list = self._batch_pixel_to_geo_coords(tiff_path, pixel_corners_list)
        
        # Process chunks
        valid_chunks = []
        chunk_polygons = []
        
        for i, (chunk_info, geo_corners) in enumerate(zip(chunks_info, geo_corners_list)):
            try:
                chunk_polygon = self._make_valid_geometry(Polygon(geo_corners))
                chunk_polygons.append(chunk_polygon)
                
                # Calculate intersection with region polygon
                intersection = self._make_valid_geometry(self.region_polygon.intersection(chunk_polygon))
                
                if intersection.is_empty:
                    continue
                
                # Calculate area ratio
                valid_pixels = intersection.area / chunk_polygon.area
                
                if valid_pixels >= self.min_valid_pixels:
                    chunk_info['valid_pixels'] = valid_pixels
                    chunk_info['polygon'] = chunk_polygon
                    valid_chunks.append(chunk_info)
                    
            except Exception as e:
                continue
        
        # Get annotations for all valid chunks at once
        if valid_chunks:
            self._batch_get_annotations(tiff_path, valid_chunks)
        
        return valid_chunks
    
    def _batch_get_annotations(self, tiff_path, chunks_info):
        """Get annotations for multiple chunks efficiently using spatial index."""
        # Query spatial index for all chunks at once
        all_chunk_polygons = [chunk['polygon'] for chunk in chunks_info]
        
        for chunk_info in chunks_info:
            chunk_polygon = chunk_info['polygon']
            annotations = []
            
            # Use spatial index to find potential intersections
            possible_matches_idx = list(self.labels_tree.query(chunk_polygon))
            
            if possible_matches_idx:
                # Get actual intersections
                possible_matches = self.labels_gdf.iloc[possible_matches_idx]
                intersecting_labels = possible_matches[possible_matches.geometry.intersects(chunk_polygon)]
                
                for idx, label_row in intersecting_labels.iterrows():
                    try:
                        intersection = self._make_valid_geometry(label_row.geometry.intersection(chunk_polygon))
                        
                        if intersection.is_empty:
                            continue
                        
                        # Determine category_id from building_type
                        building_type = label_row.get('building_type', None)
                        if building_type is not None:
                            if building_type == 'residential':
                                category_id = 0
                            else:
                                category_id = 1
                        else:
                            # fallback: use name_usl if building_type missing
                            name_usl = label_row.get('name_usl', '')
                            if name_usl == 'Здания и сооружения жилые':
                                category_id = 0
                            else:
                                category_id = 1
                        
                        # Handle different geometry types
                        if intersection.geom_type == 'GeometryCollection':
                            for geom in intersection.geoms:
                                if geom.geom_type == 'Polygon':
                                    self._process_single_polygon_optimized(geom, tiff_path, chunk_info, annotations, category_id)
                        elif intersection.geom_type == 'Polygon':
                            self._process_single_polygon_optimized(intersection, tiff_path, chunk_info, annotations, category_id)
                        elif intersection.geom_type == 'MultiPolygon':
                            for polygon in intersection.geoms:
                                self._process_single_polygon_optimized(polygon, tiff_path, chunk_info, annotations, category_id)
                    except Exception as e:
                        continue
            
            chunk_info['annotations'] = annotations
    
    def _process_single_polygon_optimized(self, polygon, tiff_path, chunk_info, annotations, category_id):
        """Optimized polygon processing for annotation creation."""
        try:
            # Convert polygon coordinates
            geo_coords = list(polygon.exterior.coords)
            pixel_coords = self._batch_geo_to_pixel_coords(tiff_path, [geo_coords])[0]
            
            # Adjust coordinates relative to chunk
            min_x, min_y = chunk_info['bounds'][:2]
            relative_coords = []
            for px_x, px_y in pixel_coords:
                rel_x = (px_x - min_x) * self.scale_factor
                rel_y = (px_y - min_y) * self.scale_factor
                relative_coords.extend([rel_x, rel_y])
            
            # Calculate bbox and area
            x_coords = [relative_coords[i] for i in range(0, len(relative_coords), 2)]
            y_coords = [relative_coords[i] for i in range(1, len(relative_coords), 2)]
            
            if len(x_coords) >= 3:  # Valid polygon
                bbox_x = min(x_coords)
                bbox_y = min(y_coords)
                bbox_w = max(x_coords) - bbox_x
                bbox_h = max(y_coords) - bbox_y
                area = bbox_w * bbox_h
                
                annotation = {
                    "id": self.annotation_id_counter,
                    "image_id": None,  # Will be set later
                    "segmentation": [relative_coords],
                    "area": area,
                    "iscrowd": 0,
                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                    "category_id": category_id
                }
                
                annotations.append(annotation)
                self.annotation_id_counter += 1
        except Exception as e:
            pass
    
    def _process_tiff_file(self, tiff_path):
        if not self._check_tiff_region_overlap(tiff_path):
            return []

        try:
            img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error loading image: {tiff_path}")
                return []
                
            # Handle dimension variations
            if len(img.shape) == 2:
                img = img[..., np.newaxis]  # Convert to HxWx1
            elif len(img.shape) != 3:
                print(f"Unexpected image dimensions in {tiff_path}: {img.shape}")
                return []

            # Downscaling
            if self.scale_factor != 1.0:
                h, w = img.shape[:2]
                new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create region mask (2D array)
            region_mask = self._create_mask_for_region(tiff_path, img.shape[:2])
            
            # Apply mask using broadcasting
            img[~region_mask] = 0

            # Handle channels
            if len(img.shape) == 2:
                if self.chunk_size[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3:
                if img.shape[2] != self.chunk_size[2]:
                    img = img[:, :, :self.chunk_size[2]]

            # Generate chunks
            height, width = img.shape[:2]
            chunk_height, chunk_width = self.chunk_size[0], self.chunk_size[1]
            step_height = chunk_height - self.overlap
            step_width = chunk_width - self.overlap

            # Calculate chunk grid
            rows = (height - self.overlap) // step_height + (1 if (height - self.overlap) % step_height > 0 else 0)
            cols = (width - self.overlap) // step_width + (1 if (width - self.overlap) % step_width > 0 else 0)

            # Create a hash of the TIFF path for unique chunk filenames
            tiff_hash = hashlib.md5(str(tiff_path).encode()).hexdigest()[:8]

            chunks_info = []
            for row in range(rows):
                for col in range(cols):
                    # Calculate chunk boundaries
                    start_row = row * step_height
                    end_row = min(start_row + chunk_height, height)
                    start_col = col * step_width
                    end_col = min(start_col + chunk_width, width)

                    # Calculate original coordinates for annotation purposes
                    original_start_col = int(start_col / self.scale_factor)
                    original_start_row = int(start_row / self.scale_factor)
                    original_end_col = int(end_col / self.scale_factor)
                    original_end_row = int(end_row / self.scale_factor)

                    # Extract chunk
                    chunk = img[start_row:end_row, start_col:end_col]

                    # Pad if needed
                    if chunk.shape[0] < chunk_height or chunk.shape[1] < chunk_width:
                        if len(chunk.shape) == 2:
                            padded_chunk = np.zeros((chunk_height, chunk_width), dtype=img.dtype)
                        else:
                            padded_chunk = np.zeros((chunk_height, chunk_width, chunk.shape[2]), dtype=img.dtype)
                        padded_chunk[:chunk.shape[0], :chunk.shape[1]] = chunk
                        chunk = padded_chunk

                    # Prepare chunk info with unique filename
                    chunk_filename = f"{Path(tiff_path).stem}_{tiff_hash}_chunk_{row:03d}_{col:03d}.TIF"
                    chunks_info.append({
                        'filename': chunk_filename,
                        'chunk': chunk,
                        'image_id': self.image_id_counter,
                        'bounds': (original_start_col, original_start_row, original_end_col, original_end_row),
                        'shape': chunk.shape
                    })
                    self.image_id_counter += 1

            # Process annotations for all chunks
            valid_chunks = self._batch_process_chunks(tiff_path, chunks_info)
            return valid_chunks

        except Exception as e:
            print(f"Error processing {tiff_path}: {str(e)}")
            return []
        
    def _split_chunks(self, all_chunks):
        """Split chunks into train/val/test sets."""
        if len(all_chunks) == 0:
            return [], [], []
        
        # First split into train and temp
        train_chunks, temp_chunks = train_test_split(
            all_chunks, 
            test_size=(self.val_split + self.test_split), 
            random_state=42
        )
        
        # Split temp into val and test
        if len(temp_chunks) > 1:
            val_chunks, test_chunks = train_test_split(
                temp_chunks,
                test_size=self.test_split / (self.val_split + self.test_split),
                random_state=42
            )
        else:
            val_chunks = temp_chunks
            test_chunks = []
        
        return train_chunks, val_chunks, test_chunks
    
    def _save_chunks_and_update_coco(self, chunks, split_name, output_dir, img_dir):
        """Save chunks and update COCO data."""
        for chunk_info in chunks:
            # Save chunk image to images subdirectory
            chunk_path = img_dir / chunk_info['filename']
            
            # Convert None values to transparent pixels if saving as PNG
            if str(chunk_path).lower().endswith('.png'):
                # For PNG with transparency
                if len(chunk_info['chunk'].shape) == 3:
                    # For RGB images, add alpha channel
                    alpha_channel = np.ones(chunk_info['chunk'].shape[:2], dtype=np.uint8) * 255
                    alpha_channel[np.all(chunk_info['chunk'] == 0, axis=2)] = 0
                    img_with_alpha = cv2.cvtColor(chunk_info['chunk'], cv2.COLOR_BGR2BGRA)
                    img_with_alpha[:, :, 3] = alpha_channel
                    cv2.imwrite(str(chunk_path), img_with_alpha)
                else:
                    # For grayscale images
                    alpha_channel = np.ones(chunk_info['chunk'].shape, dtype=np.uint8) * 255
                    alpha_channel[chunk_info['chunk'] == 0] = 0
                    img_with_alpha = cv2.merge([chunk_info['chunk'], alpha_channel])
                    cv2.imwrite(str(chunk_path), img_with_alpha)
            else:
                # For other formats (like TIFF), just save as is with None values
                cv2.imwrite(str(chunk_path), chunk_info['chunk'])
            
            # Add image to COCO data
            image_data = {
                "license": 4,
                "file_name": chunk_info['filename'],
                "coco_url": "",
                "height": self.chunk_size[0],
                "width": self.chunk_size[1],
                "date_captured": "",
                "flickr_url": "",
                "id": chunk_info['image_id']
            }
            
            self.coco_data[split_name]['images'].append(image_data)
            
            # Add annotations to COCO data
            for annotation in chunk_info['annotations']:
                annotation['image_id'] = chunk_info['image_id']
                self.coco_data[split_name]['annotations'].append(annotation)
    
    def _save_coco_json(self, split_name, output_dir):
        """Save COCO JSON file."""
        json_path = output_dir / f"{split_name}.json"
        with open(json_path, 'w') as f:
            json.dump(self.coco_data[split_name], f, indent=2)
    
    def process_all_tiffs(self):
        """Process all TIFF files in the directory."""
        # Find all TIFF files with exact extensions only (recursive)
        tiff_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF']:
            tiff_files.extend([f for f in self.tiff_dir.rglob(f'*{ext}') if f.name.endswith(ext)])
        
        print(f"Found {len(tiff_files)} TIFF files")
        
        if len(tiff_files) == 0:
            print("No TIFF files found!")
            return
        
        # Process all TIFF files with single progress bar
        all_chunks = []
        with tqdm(total=len(tiff_files), desc="Processing TIFF files") as pbar:
            for tiff_path in tiff_files:
                pbar.set_postfix_str(f"Processing {tiff_path.name}")
                chunks = self._process_tiff_file(tiff_path)
                all_chunks.extend(chunks)
                pbar.update(1)
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        if len(all_chunks) == 0:
            print("No valid chunks created!")
            return
        
        # Split into train/val/test
        train_chunks, val_chunks, test_chunks = self._split_chunks(all_chunks)
        
        print(f"Split: Train={len(train_chunks)}, Val={len(val_chunks)}, Test={len(test_chunks)}")
        
        # Save chunks and create COCO JSON files only if output_dir is provided
        if self.output_dir:
            if train_chunks:
                self._save_chunks_and_update_coco(train_chunks, 'train', self.train_dir, self.train_img_dir)
                self._save_coco_json('train', self.train_dir)
            
            if val_chunks:
                self._save_chunks_and_update_coco(val_chunks, 'val', self.val_dir, self.val_img_dir)
                self._save_coco_json('val', self.val_dir)
            
            if test_chunks:
                self._save_chunks_and_update_coco(test_chunks, 'test', self.test_dir, self.test_img_dir)
                self._save_coco_json('test', self.test_dir)
            
            print("\n" + "="*60)
            print("PROCESSING COMPLETE!")
            print(f"Output directory: {self.output_dir}")
            print(f"Train: {len(train_chunks)} chunks")
            print(f"Val: {len(val_chunks)} chunks")
            print(f"Test: {len(test_chunks)} chunks")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("PROCESSING COMPLETE (No output directory specified - data not saved)")
            print(f"Train: {len(train_chunks)} chunks")
            print(f"Val: {len(val_chunks)} chunks")
            print(f"Test: {len(test_chunks)} chunks")
            print("="*60)
        
        return {
            'train_chunks': train_chunks,
            'val_chunks': val_chunks,
            'test_chunks': test_chunks,
            'coco_data': self.coco_data
        }

    def process_custom_splits(self, split_sizes):
        """
        Custom split: Instead of train/val/test, create N splits (split0, split1, ...) with user-specified sizes.
        Args:
            split_sizes: List of fractions (e.g., [0.5, 0.3, 0.2]) or counts (e.g., [1000, 500, 500]) summing to 1.0 or total chunk count.
        """
        # Find all TIFF files with exact extensions only (recursive)
        tiff_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF']:
            tiff_files.extend([f for f in self.tiff_dir.rglob(f'*{ext}') if f.name.endswith(ext)])
        print(f"Found {len(tiff_files)} TIFF files")
        if len(tiff_files) == 0:
            print("No TIFF files found!")
            return
        # Process all TIFF files
        all_chunks = []
        with tqdm(total=len(tiff_files), desc="Processing TIFF files") as pbar:
            for tiff_path in tiff_files:
                pbar.set_postfix_str(f"Processing {tiff_path.name}")
                chunks = self._process_tiff_file(tiff_path)
                all_chunks.extend(chunks)
                pbar.update(1)
        print(f"\nTotal chunks created: {len(all_chunks)}")
        if len(all_chunks) == 0:
            print("No valid chunks created!")
            return
        # Determine split indices
        n_chunks = len(all_chunks)
        # If split_sizes are fractions, convert to counts
        if all(isinstance(s, float) and s <= 1.0 for s in split_sizes):
            counts = [int(round(s * n_chunks)) for s in split_sizes]
            # Adjust last count to ensure sum == n_chunks
            counts[-1] += n_chunks - sum(counts)
        else:
            counts = split_sizes
        assert sum(counts) == n_chunks, f"Split sizes {counts} do not sum to total chunks {n_chunks}"
        # Shuffle chunks for randomness
        random.seed(42)
        random.shuffle(all_chunks)
        # Prepare splits
        splits = []
        idx = 0
        for count in counts:
            splits.append(all_chunks[idx:idx+count])
            idx += count
        # Save each split
        for i, split_chunks in enumerate(splits):
            split_dir = self.output_dir / f'split{i}'
            img_dir = split_dir / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            # COCO structure
            coco = self._init_coco_structure()
            for chunk_info in split_chunks:
                # Save image
                chunk_path = img_dir / chunk_info['filename']
                cv2.imwrite(str(chunk_path), chunk_info['chunk'])
                # Add image entry
                image_data = {
                    "license": 4,
                    "file_name": chunk_info['filename'],
                    "coco_url": "",
                    "height": self.chunk_size[0],
                    "width": self.chunk_size[1],
                    "date_captured": "",
                    "flickr_url": "",
                    "id": chunk_info['image_id']
                }
                coco['images'].append(image_data)
                # Add annotations
                for annotation in chunk_info['annotations']:
                    annotation = annotation.copy()
                    annotation['image_id'] = chunk_info['image_id']
                    coco['annotations'].append(annotation)
            # Save COCO JSON
            json_path = split_dir / f'split{i}.json'
            with open(json_path, 'w') as f:
                json.dump(coco, f, indent=2)
            print(f"Saved split {i}: {len(split_chunks)} images, {len(coco['annotations'])} annotations")
        print("\nCustom splitting complete!")
        return splits

    def process_custom_splits_by_count(self, split_size=100):
        """
        Custom split: Instead of train/val/test, create splits (split0, split1, ...) with a fixed number of images per split.
        Args:
            split_size: Number of images per split (default: 100)
        """
        # Find all TIFF files with exact extensions only (recursive)
        tiff_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF']:
            tiff_files.extend([f for f in self.tiff_dir.rglob(f'*{ext}') if f.name.endswith(ext)])
        print(f"Found {len(tiff_files)} TIFF files")
        if len(tiff_files) == 0:
            print("No TIFF files found!")
            return
        # Process all TIFF files
        all_chunks = []
        with tqdm(total=len(tiff_files), desc="Processing TIFF files") as pbar:
            for tiff_path in tiff_files:
                pbar.set_postfix_str(f"Processing {tiff_path.name}")
                chunks = self._process_tiff_file(tiff_path)
                all_chunks.extend(chunks)
                pbar.update(1)
        print(f"\nTotal chunks created: {len(all_chunks)}")
        if len(all_chunks) == 0:
            print("No valid chunks created!")
            return
        # Shuffle chunks for randomness
        random.seed(42)
        random.shuffle(all_chunks)
        # Split into groups of split_size
        n_chunks = len(all_chunks)
        n_splits = (n_chunks + split_size - 1) // split_size
        for i in range(n_splits):
            split_chunks = all_chunks[i*split_size:(i+1)*split_size]
            split_dir = self.output_dir / f'split{i}'
            img_dir = split_dir / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            # COCO structure
            coco = self._init_coco_structure()
            for chunk_info in split_chunks:
                # Save image
                chunk_path = img_dir / chunk_info['filename']
                cv2.imwrite(str(chunk_path), chunk_info['chunk'])
                # Add image entry
                image_data = {
                    "license": 4,
                    "file_name": chunk_info['filename'],
                    "coco_url": "",
                    "height": self.chunk_size[0],
                    "width": self.chunk_size[1],
                    "date_captured": "",
                    "flickr_url": "",
                    "id": chunk_info['image_id']
                }
                coco['images'].append(image_data)
                # Add annotations
                for annotation in chunk_info['annotations']:
                    annotation = annotation.copy()
                    annotation['image_id'] = chunk_info['image_id']
                    coco['annotations'].append(annotation)
            # Save COCO JSON
            json_path = split_dir / f'split{i}.json'
            with open(json_path, 'w') as f:
                json.dump(coco, f, indent=2)
            print(f"Saved split {i}: {len(split_chunks)} images, {len(coco['annotations'])} annotations")
        print("\nCustom splitting complete!")
        return n_splits


def main():
    # Configuration
    config = {
        'tiff_dir': r"D:\Sagi\BuildingSegmentation\data\Kazgisa_layers\Kostanai\Images\ortho kostanay",
        'region_shapefile_path': r"D:\Sagi\BuildingSegmentation\data\Kazgisa_layers\Kostanai\Polygons\large-bbox.shp",
        'output_dir': r"D:\Sagi\GCP\GCP\data\kostanai",
        'chunk_size': (512, 512, 3),
        'overlap': 0,
        'original_resolution': 5.0,
        'target_resolution': 30.0,
        'train_split': 0.9,
        'val_split': 0.05,
        'test_split': 0.05,
        'min_valid_pixels': 0.3,
        'rewrite_output_dir': False,  # WARNING: Set to True only if you want to delete existing data!
        'use_train_val_test': True
    }
    
    # Note: labels_gdf should be provided from the processed buildings data
    # This is just an example - in practice, you would pass the GeoDataFrame from the previous step
    print("Note: This script now requires labels_gdf to be passed as a parameter.")
    print("Please use this class from the main.py script or provide the GeoDataFrame directly.")
    print("\n⚠️  WARNING: rewrite_output_dir=False by default to preserve existing data.")
    print("   Set rewrite_output_dir=True only if you explicitly want to delete existing data!")
    
    # Example of how to use with a GeoDataFrame:
    # labels_gdf = gpd.read_file("path_to_processed_buildings.shp")
    # config['labels_gdf'] = labels_gdf
    # config['rewrite_output_dir'] = False  # Preserve existing data
    # chunker = OptimizedTIFFChunkerWithShapefiles(**config)
    # chunker.process_all_tiffs()

if __name__ == "__main__":
    main()