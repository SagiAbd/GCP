"""
python tools/data_pipeline/regions/kostanai/merge_kostanai_buildings.py
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

    def __init__(self):
        self.gdf = None
        self.gdb_dir = Path(r"./data/raw/labels/kostanai/buildings_kostanai_tiles_raw/")
        self.output_dir = Path(r"./data/raw/labels/kostanai/buildings_kostanai_tiles_merged/")

    def _find_gdb_files(self):
        return list(self.gdb_dir.glob("*.gdb"))

    def _read_and_merge_gdbs(self, layer_name: str = "invsitibuild"):
        gdb_files = self._find_gdb_files()
        if not gdb_files:
            raise FileNotFoundError(f"No .gdb files found in {self.gdb_dir}")

        gdfs = []
        for gdb in tqdm(gdb_files, desc="Merging GDB files"):
            layers = fiona.listlayers(gdb)
            gdf = gpd.read_file(gdb, layer=layer_name)
            gdfs.append(gdf)

        self.gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        print("Finished merging.")

    def save_to_shapefile(self):
        if self.gdf is None:
            raise ValueError("No data to save. Run `process()` first.")

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
        output_path = self.output_dir / f"buildings_kostanai_merged_{timestamp}.shp"
        self.gdf.to_file(output_path)

        print(f"Saved merged shapefile to {output_path}")
    
    def process(self, layer_name: str = "invsitibuild", save_to_shapefile: bool = True):
        """Check for existing shapefile and load it if present. Otherwise, merge GDBs."""
        self._read_and_merge_gdbs(layer_name=layer_name)
        
        if save_to_shapefile:
            self.save_to_shapefile()
        
        return self.gdf


if __name__ == "__main__":
    merger = MergeKostanaiBuildings()
    merger.process()
