# Google Satellite Imagery Download and Tiling with MSCOCO Annotations
# Based on the provided Google satellite imagery download code

import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import math
import os
import json
from datetime import datetime
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import logging
from typing import Tuple, List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Install required packages if not available
try:
    import mercantile
except ImportError:
    os.system('pip install mercantile rasterio')
    import mercantile

class GoogleSatelliteTiler:
    def __init__(self, output_dir: str = "satellite_tiles"):
        """
        Initialize the Google Satellite image tiler.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = r"D:\Sagi\GCP\GCP\data\google-satellite-test\images"
        self.tile_size = 512
        os.makedirs(self.output_dir, exist_ok=True)
        
    def deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude/longitude to tile numbers"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    def num2deg(self, xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
        """Convert tile numbers to latitude/longitude"""
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)

    def calculate_ground_resolution(self, latitude: float, zoom_level: int) -> float:
        """
        Calculate ground resolution in meters per pixel
        Based on Web Mercator projection
        """
        # Earth's circumference at equator in meters
        earth_circumference = 40075016.686
        
        # Pixels per tile (standard is 256)
        pixels_per_tile = 256
        
        # Calculate resolution at equator
        equator_resolution = earth_circumference / (pixels_per_tile * (2 ** zoom_level))
        
        # Adjust for latitude (Web Mercator distortion)
        lat_correction = math.cos(math.radians(latitude))
        ground_resolution = equator_resolution / lat_correction
        
        return ground_resolution

    def download_google_satellite_tile(self, x: int, y: int, z: int, max_retries: int = 3) -> Optional[Image.Image]:
        """
        Download a single Google Satellite tile
        Using the URL pattern: 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        """
        # Rotate between different servers
        server = (x + y) % 4
        url = f"https://mt{server}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
        
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Convert to PIL Image
                image = Image.open(BytesIO(response.content))
                return image
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for tile {x},{y},{z}: {e}")
                if attempt == max_retries - 1:
                    return None
        
        return None

    def download_satellite_area(self, lat_center: float, lon_center: float, 
                              zoom_level: int, tile_radius: int = 5) -> Tuple[Dict, Tuple[int, int, int, int]]:
        """
        Download satellite imagery for an area around a center point
        
        Parameters:
        - lat_center, lon_center: Center coordinates
        - zoom_level: Zoom level (higher = more detailed)
        - tile_radius: Number of tiles in each direction from center
        """
        # Get center tile coordinates
        center_x, center_y = self.deg2num(lat_center, lon_center, zoom_level)
        
        # Calculate tile bounds
        min_x = center_x - tile_radius
        max_x = center_x + tile_radius
        min_y = center_y - tile_radius
        max_y = center_y + tile_radius
        
        # Create array to store tiles
        tile_width = (max_x - min_x + 1)
        tile_height = (max_y - min_y + 1)
        
        logger.info(f"Downloading {tile_width * tile_height} tiles...")
        
        # Download tiles
        tiles = {}
        successful_downloads = 0
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile = self.download_google_satellite_tile(x, y, zoom_level)
                if tile:
                    tiles[(x, y)] = tile
                    successful_downloads += 1
                    logger.debug(f"Downloaded tile ({x}, {y})")
                else:
                    logger.warning(f"Failed to download tile ({x}, {y})")
        
        logger.info(f"Successfully downloaded {successful_downloads}/{tile_width * tile_height} tiles")
        return tiles, (min_x, min_y, max_x, max_y)

    def stitch_tiles(self, tiles: Dict, bounds: Tuple[int, int, int, int]) -> Optional[Image.Image]:
        """Stitch downloaded tiles into a single image"""
        min_x, min_y, max_x, max_y = bounds
        
        if not tiles:
            return None
        
        # Get tile size (assuming all tiles are same size)
        sample_tile = next(iter(tiles.values()))
        tile_size = sample_tile.size[0]  # Assuming square tiles
        
        # Create final image
        width = (max_x - min_x + 1) * tile_size
        height = (max_y - min_y + 1) * tile_size
        final_image = Image.new('RGB', (width, height))
        
        # Place tiles
        for (x, y), tile in tiles.items():
            paste_x = (x - min_x) * tile_size
            paste_y = (y - min_y) * tile_size
            final_image.paste(tile, (paste_x, paste_y))
        
        return final_image

    def save_as_geotiff(self, image: Image.Image, bounds: Tuple[int, int, int, int], 
                       lat_center: float, lon_center: float, zoom_level: int, 
                       filename: str) -> str:
        """
        Save the stitched image as a GeoTIFF with proper georeferencing
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Calculate geographic bounds
        top_left_lat, top_left_lon = self.num2deg(min_x, min_y, zoom_level)
        bottom_right_lat, bottom_right_lon = self.num2deg(max_x + 1, max_y + 1, zoom_level)
        
        # Create transform
        width, height = image.size
        transform = from_bounds(top_left_lon, bottom_right_lat, bottom_right_lon, top_left_lat, width, height)
        
        # Convert PIL image to numpy array
        image_array = np.array(image)
        
        # Rearrange dimensions for rasterio (bands, height, width)
        if len(image_array.shape) == 3:
            image_array = np.transpose(image_array, (2, 0, 1))
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Write GeoTIFF
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=image_array.shape[0],
            dtype=image_array.dtype,
            crs=CRS.from_epsg(4326),
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(image_array)
        
        logger.info(f"Saved GeoTIFF: {filepath}")
        return filepath

    def create_512x512_tiles(self, geotiff_path: str, location_name: str) -> List[str]:
        """
        Create 512x512 tiles from the GeoTIFF image
        """
        tile_files = []
        
        with rasterio.open(geotiff_path) as src:
            # Get image dimensions
            width = src.width
            height = src.height
            
            # Calculate number of tiles
            tiles_x = math.ceil(width / self.tile_size)
            tiles_y = math.ceil(height / self.tile_size)
            
            logger.info(f"Creating {tiles_x * tiles_y} tiles of size {self.tile_size}x{self.tile_size}")
            
            # Create tiles
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    # Calculate tile bounds
                    x_start = tx * self.tile_size
                    y_start = ty * self.tile_size
                    x_end = min(x_start + self.tile_size, width)
                    y_end = min(y_start + self.tile_size, height)
                    
                    # Read tile data
                    window = rasterio.windows.Window(x_start, y_start, 
                                                   x_end - x_start, y_end - y_start)
                    tile_data = src.read(window=window)
                    
                    # Create output filename
                    tile_filename = f"{location_name}_tile_{tx:04d}_{ty:04d}.tif"
                    tile_path = os.path.join(self.output_dir, tile_filename)
                    
                    # Calculate geographic bounds for this tile
                    tile_transform = rasterio.windows.transform(window, src.transform)
                    
                    # Pad tile to 512x512 if needed
                    actual_height, actual_width = tile_data.shape[1], tile_data.shape[2]
                    if actual_height < self.tile_size or actual_width < self.tile_size:
                        padded_data = np.zeros((tile_data.shape[0], self.tile_size, self.tile_size), 
                                             dtype=tile_data.dtype)
                        padded_data[:, :actual_height, :actual_width] = tile_data
                        tile_data = padded_data
                        
                        # Adjust transform for padding
                        tile_transform = from_bounds(
                            *rasterio.transform.array_bounds(self.tile_size, self.tile_size, tile_transform),
                            self.tile_size, self.tile_size
                        )
                    
                    # Write tile
                    with rasterio.open(
                        tile_path,
                        'w',
                        driver='GTiff',
                        height=self.tile_size,
                        width=self.tile_size,
                        count=tile_data.shape[0],
                        dtype=tile_data.dtype,
                        crs=src.crs,
                        transform=tile_transform,
                        compress='lzw'
                    ) as dst:
                        dst.write(tile_data)
                    
                    tile_files.append(tile_filename)
                    
        logger.info(f"Created {len(tile_files)} tiles")
        return tile_files

    def create_mscoco_annotation(self, tile_files: List[str], location_info: Dict) -> Dict:
        """
        Create MSCOCO format annotation file with empty annotations
        """
        # MSCOCO format structure
        coco_data = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": f"Google Satellite imagery tiles for {location_info.get('name', 'Unknown Location')}",
                "contributor": "Google Satellite Tiler",
                "url": "https://maps.google.com",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Google Maps/Earth",
                    "url": "https://www.google.com/permissions/geoguidelines/"
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "satellite_tile",
                    "supercategory": "imagery"
                }
            ],
            "images": [],
            "annotations": []
        }
        
        # Add images
        for i, tile_file in enumerate(tile_files):
            # Calculate tile position from filename
            parts = tile_file.replace('.tif', '').split('_')
            tile_x = int(parts[-2])
            tile_y = int(parts[-1])
            
            image_info = {
                "id": i + 1,
                "width": self.tile_size,
                "height": self.tile_size,
                "file_name": tile_file,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().isoformat(),
                "tile_x": tile_x,
                "tile_y": tile_y,
                "zoom_level": location_info.get('zoom_level', 0),
                "ground_resolution_m_per_px": location_info.get('ground_resolution', 0)
            }
            coco_data["images"].append(image_info)
        
        # Annotations are left empty as requested
        # You can add annotations later for object detection tasks
        
        return coco_data

    def process_location(self, lat_center: float, lon_center: float, 
                        zoom_level: int = 18, tile_radius: int = 5,
                        location_name: str = "location") -> Tuple[List[str], str]:
        """
        Main processing function to download, stitch, and tile satellite imagery
        
        Args:
            lat_center: Latitude of center point
            lon_center: Longitude of center point
            zoom_level: Google Maps zoom level (18-20 for high resolution)
            tile_radius: Radius of tiles to download around center
            location_name: Name for output files
            
        Returns:
            Tuple of (tile_files, annotation_path)
        """
        logger.info(f"Processing location: {location_name}")
        logger.info(f"Center: {lat_center:.6f}, {lon_center:.6f}")
        logger.info(f"Zoom level: {zoom_level}, Tile radius: {tile_radius}")
        
        # Calculate ground resolution
        ground_resolution = self.calculate_ground_resolution(lat_center, zoom_level)
        logger.info(f"Ground resolution: {ground_resolution:.2f} meters/pixel")
        
        # Download tiles
        tiles, bounds = self.download_satellite_area(lat_center, lon_center, zoom_level, tile_radius)
        
        if not tiles:
            raise Exception("No tiles downloaded successfully")
        
        # Stitch tiles
        logger.info("Stitching tiles...")
        stitched_image = self.stitch_tiles(tiles, bounds)
        
        if not stitched_image:
            raise Exception("Failed to stitch tiles")
        
        # Save as GeoTIFF
        geotiff_filename = f"{location_name}_satellite.tif"
        geotiff_path = self.save_as_geotiff(
            stitched_image, bounds, lat_center, lon_center, zoom_level, geotiff_filename
        )
        
        # Create 512x512 tiles
        logger.info("Creating 512x512 tiles...")
        tile_files = self.create_512x512_tiles(geotiff_path, location_name)
        
        # Create MSCOCO annotation
        location_info = {
            "name": location_name,
            "center_lat": lat_center,
            "center_lon": lon_center,
            "zoom_level": zoom_level,
            "ground_resolution": ground_resolution,
            "tile_radius": tile_radius
        }
        
        coco_data = self.create_mscoco_annotation(tile_files, location_info)
        
        # Save annotation file
        annotation_filename = f"{location_name}_annotations.json"
        annotation_path = os.path.join(self.output_dir, annotation_filename)
        with open(annotation_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Processing complete!")
        logger.info(f"Created {len(tile_files)} tiles")
        logger.info(f"Tiles saved to: {self.output_dir}")
        logger.info(f"Annotations saved to: {annotation_path}")
        
        return tile_files, annotation_path

    def visualize_coverage(self, lat_center: float, lon_center: float, 
                          zoom_level: int, tile_radius: int):
        """
        Visualize the coverage area that will be downloaded
        """
        # Calculate approximate coverage
        ground_resolution = self.calculate_ground_resolution(lat_center, zoom_level)
        
        # Each Google tile is 256x256 pixels
        tile_size_m = 256 * ground_resolution
        coverage_width_m = (2 * tile_radius + 1) * tile_size_m
        coverage_height_m = (2 * tile_radius + 1) * tile_size_m
        
        total_pixels = (2 * tile_radius + 1) ** 2 * 256 * 256
        tiles_512 = math.ceil(total_pixels / (512 * 512))
        
        print(f"Coverage Preview:")
        print(f"├── Center: {lat_center:.6f}°, {lon_center:.6f}°")
        print(f"├── Zoom level: {zoom_level}")
        print(f"├── Ground resolution: {ground_resolution:.2f} m/pixel")
        print(f"├── Coverage area: {coverage_width_m:.0f} × {coverage_height_m:.0f} meters")
        print(f"├── Coverage area: {coverage_width_m/1000:.2f} × {coverage_height_m/1000:.2f} km")
        print(f"├── Google tiles to download: {(2 * tile_radius + 1) ** 2}")
        print(f"└── Expected 512×512 tiles: ~{tiles_512}")


def main():
    """
    Example usage of the GoogleSatelliteTiler
    """
    # Initialize tiler
    tiler = GoogleSatelliteTiler("satellite_output")
    
    # Example locations
    locations = [
        {
            "name": "central_park_nyc",
            "lat": 40.7829, 
            "lon": -73.9654,
            "zoom": 18,
            "radius": 3
        },
        {
            "name": "Almaty",
            "lat": 43.227,
            "lon": 76.895,
            "zoom": 17,
            "radius": 5
        },
        {
            "name": "kostanai1",
            "lat": 53.20383,
            "lon": 63.60149,
            "zoom": 19,
            "radius": 5
        }
    ]
    
    # Process a location
    location = locations[1]  # Astana example
    
    try:
        # Preview coverage
        tiler.visualize_coverage(
            location["lat"], location["lon"], 
            location["zoom"], location["radius"]
        )
        
        # Confirm processing
        response = input("\nProceed with download? (y/n): ")
        if response.lower() == 'y':
            # Process the location
            tile_files, annotation_path = tiler.process_location(
                lat_center=location["lat"],
                lon_center=location["lon"],
                zoom_level=location["zoom"],
                tile_radius=location["radius"],
                location_name=location["name"]
            )
            
            print(f"\n✅ Successfully created {len(tile_files)} tiles")
            print(f"📁 Output directory: {tiler.output_dir}")
            print(f"📄 Annotation file: {annotation_path}")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()