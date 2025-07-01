"""
python tools/data_pipeline/merge_raw_kostanai_buildings.py
"""

import os
import shutil
import geopandas as gpd
import pandas as pd
import fiona
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


class MergeKostanaiBuildings:
    """
    A utility class to merge multiple .gdb (Geodatabase) files containing 
    building information from the Kostanai region.
    """

    def __init__(self, gdb_dir, output_dir):
        self.gdf = None
        self.gdb_dir = Path(gdb_dir)
        self.output_dir = Path(output_dir)

    def _find_data_files(self):
        # Find .gdb, .gpkg, and .shp files recursively
        gdb_files = list(self.gdb_dir.rglob("*.gdb"))
        gpkg_files = list(self.gdb_dir.rglob("*.gpkg"))
        shp_files = list(self.gdb_dir.rglob("*.shp"))
        return gdb_files + gpkg_files + shp_files

    def _read_and_merge_data(self, layer_name: str = "invsitibuild"):
        data_files = self._find_data_files()
        if not data_files:
            raise FileNotFoundError(f"No .gdb, .gpkg, or .shp files found in {self.gdb_dir}")

        gdfs = []
        target_crs = "EPSG:32641"
        for data_file in tqdm(data_files, desc="Merging GDB/GPKG/SHP files"):
            ext = data_file.suffix.lower()
            if ext == ".shp":
                # .shp files do not have layers, just read geometry
                gdf = gpd.read_file(data_file)
            else:
                # List layers and pick the provided or first one
                layers = fiona.listlayers(data_file)
                chosen_layer = layer_name if layer_name in layers else layers[0]
                gdf = gpd.read_file(data_file, layer=chosen_layer)
            if gdf.crs is None or str(gdf.crs) != target_crs:
                print(f"Reprojecting {data_file} to {target_crs}")
                gdf = gdf.to_crs(target_crs)
            gdfs.append(gdf)

        self.gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=target_crs)
        print(f"Merged GeoDataFrame CRS: {self.gdf.crs}")

    def save_to_shapefile(self):
        if self.gdf is None:
            raise ValueError("No data to save. Run `process()` first.")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Convert datetime columns to strings
        for col in self.gdf.columns:
            if pd.api.types.is_datetime64_any_dtype(self.gdf[col]):
                self.gdf[col] = self.gdf[col].astype(str)

        # Remove everything in the output directory
        for item in self.output_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        # Save shapefile
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"buildings_merged_{timestamp}.shp"
        self.gdf.to_file(output_path)

        print(f"Saved merged shapefile to {output_path}")
    
    def process(self, layer_name: str = "invsitibuild", save_to_shapefile: bool = True):
        """Check for existing shapefile and load it if present. Otherwise, merge GDBs and GPKGs."""
        self._read_and_merge_data(layer_name=layer_name)
        if save_to_shapefile:
            self.save_to_shapefile()
        return self.gdf


if __name__ == "__main__":
    merger = MergeKostanaiBuildings(gdb_dir=r"./data/raw/labels/kostanai/buildings_kostanai_tiles_raw/", output_dir=r"./data/raw/labels/kostanai/buildings_kostanai_tiles_merged/")
    merger.process()
