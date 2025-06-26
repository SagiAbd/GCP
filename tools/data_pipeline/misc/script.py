import os
import geopandas as gpd
import fiona

def merge_gdbs_to_shapefile(gdb_dir, output_shapefile, layer_name=None):
    all_gdfs = []

    for file in os.listdir(gdb_dir):
        if file.endswith(".gdb"):
            gdb_path = os.path.join(gdb_dir, file)
            print(f"Reading GDB: {gdb_path}")

            # If no layer_name is specified, list available ones
            if layer_name is None:
                layers = fiona.listlayers(gdb_path)
                print(f"Available layers in {file}: {layers}")
                # You can choose first one by default or raise error
                current_layer = layers[0]
            else:
                current_layer = layer_name

            gdf = gpd.read_file(gdb_path, layer=current_layer)
            all_gdfs.append(gdf)

    if not all_gdfs:
        raise ValueError("No .gdb files or valid layers found in the directory.")

    merged_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
    merged_gdf.to_file(output_shapefile)
    print(f"Merged shapefile saved to: {output_shapefile}")

merge_gdbs_to_shapefile(
    gdb_dir=r"D:\Sagi\BuildingSegmentation\data\Kazgisa_layers\Kostanai\Polygons\готовая оцифровка Костанай",
    output_shapefile=r"D:\Sagi\BuildingSegmentation\data\Kazgisa_layers\Kostanai\Polygons\combined_data.shp" # optional; if not provided, uses the first found in each
)
