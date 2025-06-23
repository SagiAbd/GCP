# Google Satellite Imagery Download and Tiling with Flexible Zoom Levels
# Modified to support any target zoom level with intelligent downloading strategy

import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
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
        self.output_dir = output_dir
        self.tile_size = 512
        self.max_zoom = 20  # Maximum zoom level available from Google
        self.min_zoom = 1   # Minimum practical zoom level
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

    def calculate_ground_resolution(self, latitude: float, zoom_level: float) -> float:
        """
        Calculate ground resolution in meters per pixel
        Based on Web Mercator projection
        
        Args:
            latitude: Latitude in degrees
            zoom_level: Zoom level (can be fractional)
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

    def determine_download_strategy(self, target_zoom: float) -> Tuple[int, float]:
        """
        Determine the best download strategy for a given target zoom level.
        
        Args:
            target_zoom: Target zoom level (can be fractional)
            
        Returns:
            Tuple of (download_zoom, scale_factor)
        """
        # Clamp target zoom to reasonable bounds
        target_zoom = max(self.min_zoom, min(self.max_zoom, target_zoom))
        
        # For fractional zoom levels, download from higher zoom and downscale
        if target_zoom != int(target_zoom):
            # Use ceiling to get higher resolution source
            download_zoom = min(math.ceil(target_zoom), self.max_zoom)
            scale_factor = 2 ** (download_zoom - target_zoom)
        else:
            # For integer zoom levels, download directly
            download_zoom = int(target_zoom)
            scale_factor = 1.0
            
        return download_zoom, scale_factor

    def download_google_satellite_tile(self, x: int, y: int, z: int, max_retries: int = 3) -> Optional[Image.Image]:
        """
        Download a single Google Satellite tile
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

    def scale_image(self, image: Image.Image, scale_factor: float) -> Image.Image:
        """
        Scale image by given factor with high quality resampling
        
        Args:
            image: PIL Image to scale
            scale_factor: Scale factor (>1 for upscaling, <1 for downscaling)
        """
        if scale_factor == 1.0:
            return image
            
        original_width, original_height = image.size
        new_width = int(original_width / scale_factor)
        new_height = int(original_height / scale_factor)
        
        # Use appropriate resampling method
        if scale_factor > 1.0:
            # Downscaling - use LANCZOS for best quality
            resampling = Image.Resampling.LANCZOS
        else:
            # Upscaling - use BICUBIC for smooth results
            resampling = Image.Resampling.BICUBIC
        
        scaled = image.resize((new_width, new_height), resampling)
        
        # Optional sharpening for downscaled images
        if scale_factor > 1.0 and scale_factor < 2.0:
            scaled = scaled.filter(ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=2))
        
        logger.debug(f"Scaled from {original_width}x{original_height} to {new_width}x{new_height} (factor: {scale_factor:.3f})")
        return scaled

    def download_and_process_area(self, lat_center: float, lon_center: float, 
                                 target_zoom: float, tile_radius: int = 5) -> Tuple[Dict, Tuple[int, int, int, int], int, float]:
        """
        Download satellite imagery and process to target zoom level
        
        Args:
            lat_center, lon_center: Center coordinates
            target_zoom: Target zoom level (can be fractional)
            tile_radius: Number of tiles in each direction from center
            
        Returns:
            Tuple of (tiles, bounds, download_zoom, scale_factor)
        """
        # Determine download strategy
        download_zoom, scale_factor = self.determine_download_strategy(target_zoom)
        
        logger.info(f"Target zoom: {target_zoom}, Download zoom: {download_zoom}, Scale factor: {scale_factor:.3f}")
        
        # Calculate tile coverage
        if scale_factor > 1.0:
            # Downloading higher resolution - need more tiles
            download_radius = tile_radius * int(math.ceil(scale_factor))
        else:
            # Downloading same or lower resolution
            download_radius = tile_radius
        
        # Get center tile coordinates at download zoom
        center_x, center_y = self.deg2num(lat_center, lon_center, download_zoom)
        
        # Calculate tile bounds at download zoom
        min_x = center_x - download_radius
        max_x = center_x + download_radius
        min_y = center_y - download_radius
        max_y = center_y + download_radius
        
        # Create array to store tiles
        tile_width = (max_x - min_x + 1)
        tile_height = (max_y - min_y + 1)
        
        logger.info(f"Downloading {tile_width * tile_height} tiles at zoom {download_zoom}...")
        
        # Download tiles
        downloaded_tiles = {}
        successful_downloads = 0
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile = self.download_google_satellite_tile(x, y, download_zoom)
                if tile:
                    downloaded_tiles[(x, y)] = tile
                    successful_downloads += 1
                    logger.debug(f"Downloaded tile ({x}, {y}) at zoom {download_zoom}")
                else:
                    logger.warning(f"Failed to download tile ({x}, {y}) at zoom {download_zoom}")
        
        logger.info(f"Successfully downloaded {successful_downloads}/{tile_width * tile_height} tiles")
        
        # Stitch all downloaded tiles into one large image
        if not downloaded_tiles:
            return {}, (0, 0, 0, 0), download_zoom, scale_factor
        
        # Get tile size
        sample_tile = next(iter(downloaded_tiles.values()))
        tile_size = sample_tile.size[0]
        
        # Create large stitched image
        width = tile_width * tile_size
        height = tile_height * tile_size
        large_image = Image.new('RGB', (width, height))
        
        # Place tiles
        for (x, y), tile in downloaded_tiles.items():
            paste_x = (x - min_x) * tile_size
            paste_y = (y - min_y) * tile_size
            large_image.paste(tile, (paste_x, paste_y))
        
        logger.info(f"Stitched image size: {large_image.size}")
        
        # Scale the entire image to target zoom level
        if scale_factor != 1.0:
            scaled_image = self.scale_image(large_image, scale_factor)
            logger.info(f"Scaled to: {scaled_image.size} (target zoom {target_zoom})")
        else:
            scaled_image = large_image
        
        # Calculate equivalent bounds at target zoom level
        target_zoom_int = int(target_zoom) if target_zoom == int(target_zoom) else int(math.floor(target_zoom))
        target_center_x, target_center_y = self.deg2num(lat_center, lon_center, target_zoom_int)
        target_min_x = target_center_x - tile_radius
        target_max_x = target_center_x + tile_radius
        target_min_y = target_center_y - tile_radius
        target_max_y = target_center_y + tile_radius
        
        target_bounds = (target_min_x, target_min_y, target_max_x, target_max_y)
        
        # Store the processed image
        processed_tiles = {('processed_image', 0): scaled_image}
        
        return processed_tiles, target_bounds, download_zoom, scale_factor

    def stitch_tiles(self, tiles: Dict, bounds: Tuple[int, int, int, int]) -> Optional[Image.Image]:
        """Stitch downloaded tiles into a single image"""
        if not tiles:
            return None
        
        # Check if we have a single processed image
        if ('processed_image', 0) in tiles:
            return tiles[('processed_image', 0)]
        
        # Original stitching logic for regular tiles
        min_x, min_y, max_x, max_y = bounds
        
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
                       lat_center: float, lon_center: float, zoom_level: float, 
                       filename: str) -> str:
        """
        Save the stitched image as a GeoTIFF with proper georeferencing
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Use integer zoom for coordinate calculations
        zoom_int = int(zoom_level) if zoom_level == int(zoom_level) else int(math.floor(zoom_level))
        
        # Calculate geographic bounds
        top_left_lat, top_left_lon = self.num2deg(min_x, min_y, zoom_int)
        bottom_right_lat, bottom_right_lon = self.num2deg(max_x + 1, max_y + 1, zoom_int)
        
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
        target_zoom = location_info.get('zoom_level', 18)
        zoom_description = f"zoom {target_zoom}" if target_zoom == int(target_zoom) else f"zoom {target_zoom:.1f}"
        
        # MSCOCO format structure
        coco_data = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": f"Google Satellite imagery tiles for {location_info.get('name', 'Unknown Location')} ({zoom_description})",
                "contributor": "Google Satellite Tiler",
                "url": "https://maps.google.com",
                "date_created": datetime.now().isoformat(),
                "processing_info": f"Downloaded at zoom {location_info.get('download_zoom', 'unknown')} and processed to {zoom_description}"
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
                "target_zoom_level": target_zoom,
                "download_zoom_level": location_info.get('download_zoom', 'unknown'),
                "scale_factor": location_info.get('scale_factor', 1.0),
                "ground_resolution_m_per_px": location_info.get('ground_resolution', 0)
            }
            coco_data["images"].append(image_info)
        
        return coco_data

    def process_location(self, lat_center: float, lon_center: float, 
                        target_zoom: float, tile_radius: int = 5,
                        location_name: str = "location") -> Tuple[List[str], str]:
        """
        Main processing function to download and process satellite imagery
        
        Args:
            lat_center: Latitude of center point
            lon_center: Longitude of center point
            target_zoom: Target zoom level (can be fractional, e.g., 18.5)
            tile_radius: Radius of tiles to download around center
            location_name: Name for output files
            
        Returns:
            Tuple of (tile_files, annotation_path)
        """
        logger.info(f"Processing location: {location_name}")
        logger.info(f"Center: {lat_center:.6f}, {lon_center:.6f}")
        logger.info(f"Target zoom level: {target_zoom}")
        logger.info(f"Tile radius: {tile_radius}")
        
        # Calculate ground resolution
        ground_resolution = self.calculate_ground_resolution(lat_center, target_zoom)
        logger.info(f"Ground resolution: {ground_resolution:.2f} meters/pixel")
        
        # Download and process tiles
        tiles, bounds, download_zoom, scale_factor = self.download_and_process_area(
            lat_center, lon_center, target_zoom, tile_radius
        )
        
        if not tiles:
            raise Exception("No tiles processed successfully")
        
        # Process tiles
        logger.info("Processing image...")
        stitched_image = self.stitch_tiles(tiles, bounds)
        
        if not stitched_image:
            raise Exception("Failed to process tiles")
        
        # Save as GeoTIFF
        zoom_str = f"{target_zoom:.1f}".replace('.', '_') if target_zoom != int(target_zoom) else str(int(target_zoom))
        geotiff_filename = f"{location_name}_satellite_zoom{zoom_str}.tif"
        geotiff_path = self.save_as_geotiff(
            stitched_image, bounds, lat_center, lon_center, target_zoom, geotiff_filename
        )
        
        # Create 512x512 tiles
        logger.info("Creating 512x512 tiles...")
        tile_files = self.create_512x512_tiles(geotiff_path, location_name)
        
        # Create MSCOCO annotation
        location_info = {
            "name": location_name,
            "center_lat": lat_center,
            "center_lon": lon_center,
            "zoom_level": target_zoom,
            "download_zoom": download_zoom,
            "scale_factor": scale_factor,
            "ground_resolution": ground_resolution,
            "tile_radius": tile_radius
        }
        
        coco_data = self.create_mscoco_annotation(tile_files, location_info)
        
        # Save annotation file
        annotation_filename = f"{location_name}_annotations_zoom{zoom_str}.json"
        annotation_path = os.path.join(self.output_dir, annotation_filename)
        with open(annotation_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Processing complete!")
        logger.info(f"Created {len(tile_files)} tiles at zoom {target_zoom}")
        logger.info(f"Download strategy: zoom {download_zoom} -> {target_zoom} (scale: {scale_factor:.3f})")
        logger.info(f"Tiles saved to: {self.output_dir}")
        logger.info(f"Annotations saved to: {annotation_path}")
        
        return tile_files, annotation_path

    def visualize_coverage(self, lat_center: float, lon_center: float, 
                          target_zoom: float, tile_radius: int):
        """
        Visualize the coverage area that will be downloaded
        """
        download_zoom, scale_factor = self.determine_download_strategy(target_zoom)
        ground_resolution = self.calculate_ground_resolution(lat_center, target_zoom)
        
        # Each tile will be 512x512 pixels at target resolution
        tile_size_m = 512 * ground_resolution
        coverage_width_m = (2 * tile_radius + 1) * tile_size_m
        coverage_height_m = (2 * tile_radius + 1) * tile_size_m
        
        # Calculate download requirements
        if scale_factor > 1.0:
            download_radius = tile_radius * int(math.ceil(scale_factor))
        else:
            download_radius = tile_radius
        download_tiles = (2 * download_radius + 1) ** 2
        output_tiles = (2 * tile_radius + 1) ** 2
        
        print(f"Coverage Preview:")
        print(f"├── Center: {lat_center:.6f}°, {lon_center:.6f}°")
        print(f"├── Target zoom level: {target_zoom}")
        print(f"├── Download zoom level: {download_zoom}")
        print(f"├── Scale factor: {scale_factor:.3f}")
        print(f"├── Ground resolution: {ground_resolution:.2f} m/pixel")
        print(f"├── Coverage area: {coverage_width_m:.0f} × {coverage_height_m:.0f} meters")
        print(f"├── Coverage area: {coverage_width_m/1000:.2f} × {coverage_height_m/1000:.2f} km")
        print(f"├── Tiles to download: {download_tiles}")
        print(f"└── Output 512×512 tiles: {output_tiles}")


def main():
    """
    Example usage of the flexible GoogleSatelliteTiler
    """
    # Initialize tiler
    tiler = GoogleSatelliteTiler(r"data/google-satellite-test-christchurch")
    
    # Example locations with different zoom levels
    locations = [
        {
            "name": "christchurch_new_zealand_z18.2",
            "lat": -43.5321, 
            "lon": 172.6362,
            "zoom": 18.2,  # Fractional zoom
            "radius": 3
        },
        {
            "name": "astana_z19",
            "lat": 51.1694,
            "lon": 71.4491,
            "zoom": 18.2,  # Integer zoom
            "radius": 2
        },
        {
            "name": "kostanai_z18_7",
            "lat": 53.20383,
            "lon": 63.60149,
            "zoom": 18.7,  # Fractional zoom
            "radius": 4
        },
        {
            "name": "test_location_z15_5",
            "lat": 40.0,
            "lon": -74.0,
            "zoom": 15.5,  # Lower zoom with fractional
            "radius": 3
        }
    ]
    
    # Process a location
    location = locations[0]  # Central Park example
    
    try:
        # Preview coverage
        tiler.visualize_coverage(
            location["lat"], location["lon"], 
            location["zoom"], location["radius"]
        )
        
        # Confirm processing
        download_zoom, scale_factor = tiler.determine_download_strategy(location["zoom"])
        download_radius = location["radius"] * int(math.ceil(scale_factor)) if scale_factor > 1.0 else location["radius"]
        download_tiles = (2 * download_radius + 1) ** 2
        
        response = input(f"\nProceed with download? This will download {download_tiles} tiles at zoom {download_zoom}. (y/n): ")
        if response.lower() == 'y':
            # Process the location
            tile_files, annotation_path = tiler.process_location(
                lat_center=location["lat"],
                lon_center=location["lon"],
                target_zoom=location["zoom"],
                tile_radius=location["radius"],
                location_name=location["name"]
            )
            
            print(f"\n✅ Successfully created {len(tile_files)} tiles at zoom {location['zoom']}")
            print(f"📁 Output directory: {tiler.output_dir}")
            print(f"📄 Annotation file: {annotation_path}")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()