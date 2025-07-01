#!/usr/bin/env python3
"""
Main script for Kostanai data processing pipeline.

This script orchestrates the execution of three data processing steps:
1. Stage 1: Merge raw Kostanai building data from multiple GDB files
2. Stage 2: Process and clean the merged building geometries
3. Stage 3: Split the processed data into train/val/test sets with image chunking

Usage:
    python tools/data_pipeline/main.py [stage1|stage2|stage3|all]
    
Examples:
    python tools/data_pipeline/main.py stage1    # Run only stage 1
    python tools/data_pipeline/main.py stage2    # Run only stage 2
    python tools/data_pipeline/main.py stage3    # Run only stage 3
    python tools/data_pipeline/main.py all       # Run all stages
    python tools/data_pipeline/main.py           # Run all stages (default)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import geopandas as gpd  # <-- Ensure geopandas is imported

# Add the parent directory to the path to import the modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Global variables to store results between stages
merged_gdf = None
processed_gdf = None

# =====================
# Centralized Pipeline Configurations
# =====================
PIPELINE_CONFIGS = {
    'merge_kostanai_buildings': {
        'gdb_dir': r"D:\Sagi\GCP\GCP\data\raw\labels\kostanai\buildings_kostanai_tiles_raw",
        'output_dir': r"./data/raw/labels/kostanai/buildings_kostanai_tiles_merged/",
        'layer_name': "invsitibuild"
    },
    'process_kostanai_buildings': {
        'distance_threshold': 0.15,      # 15cm grouping distance
        'min_intersection_length': 2.0,  # 2m minimum intersection
        'max_group_size': 50,           # Max 50 buildings per group
        'min_area_threshold': 10,       # 10m² minimum area
        'use_scaling_merge': False,     # Use scaling merge
        'target_intersection_ratio': 0.02, # 2% intersection requirement
        'max_scale': 1.5,                # Maximum 1.5x scaling
        'output_dir': r"./data/raw/labels/shymkent/buildings_shymkent_processed/"
    },
    'data_split': {
        'tiff_dir': r"D:\Sagi\GCP\GCP\data\raw\images\shymkent",
        'region_shapefile_path': None,
        'output_dir': r"D:\Sagi\GCP\GCP\data\shymkent",
        'chunk_size': (512, 512, 3),
        'overlap': 0,
        'original_resolution': 5.0,
        'target_resolution': 30.0,
        'train_split': 0.949,
        'val_split': 0.05,
        'test_split': 0.001,
        'min_valid_pixels': 0.3,
        'use_train_val_test': True
    },
    'custom_split': {
        'tiff_dir': r"D:\Sagi\GCP\GCP\data\raw\images\shymkent",
        'region_shapefile_path': None,
        'output_dir': r"D:\Sagi\GCP\GCP\data\shymkent",
        'chunk_size': (512, 512, 3),
        'overlap': 0,
        'original_resolution': 5.0,
        'target_resolution': 30.0,
        'min_valid_pixels': 0.3,
        'use_train_val_test': False
    },
    'complete_dataset': {
        'tiff_dir': r"D:\Sagi\GCP\GCP\data\raw\images\kostanai\Images\ortho kostanay",
        'region_shapefile_path': r"D:\Sagi\GCP\GCP\data\raw\labels\kostanai\region_bbox\region_bbox.shp",
        'output_dir': r"D:\Sagi\GCP\GCP\data\kostanai",
        'chunk_size': (512, 512, 3),
        'overlap': 0,
        'original_resolution': 5.0,
        'target_resolution': 30.0,
        'min_valid_pixels': 0.3,
        'use_train_val_test': False
    }
}

def run_stage1():
    """Stage 1: Merge raw Kostanai building data from multiple GDB files."""
    global merged_gdf
    
    print(f"\n{'='*60}")
    print(f"STAGE 1: Merge raw Kostanai building data from multiple GDB files")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        from merge_raw_kostanai_buildings import MergeKostanaiBuildings
        merge_cfg = PIPELINE_CONFIGS['merge_kostanai_buildings'].copy()
        merger = MergeKostanaiBuildings(gdb_dir=merge_cfg['gdb_dir'], output_dir=merge_cfg['output_dir'])
        merged_gdf = merger.process(layer_name=merge_cfg['layer_name'])
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ SUCCESS: Stage 1 completed")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Merged {len(merged_gdf)} building records")
        
        return True
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n❌ ERROR: Stage 1 failed")
        print(f"Error: {str(e)}")
        print(f"Duration: {duration:.2f} seconds")
        
        return False

def run_stage2():
    """Stage 2: Process and clean building geometries."""
    global merged_gdf, processed_gdf
    
    print(f"\n{'='*60}")
    print(f"STAGE 2: Process and clean building geometries")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        from process_kostanai_buildings import ProcessKostanaiBuildings
        
        # If merged_gdf is not available, run stage 1 first
        if merged_gdf is None:
            print("Merged data not available, running Stage 1 first...")
            if not run_stage1():
                return False
        
        # Initialize processor with custom parameters
        process_cfg = PIPELINE_CONFIGS['process_kostanai_buildings'].copy()
        processor = ProcessKostanaiBuildings(**process_cfg)
        
        # Process the merged data
        processed_gdf = processor.process_geodataframe_optimized(merged_gdf)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ SUCCESS: Stage 2 completed")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Processed {len(processed_gdf)} building records")
        
        return True
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n❌ ERROR: Stage 2 failed")
        print(f"Error: {str(e)}")
        print(f"Duration: {duration:.2f} seconds")
        
        return False

def run_stage3_with_delete(force_delete=False):
    global processed_gdf
    
    print(f"\n{'='*60}")
    print(f"STAGE 3: Split data into train/val/test sets with image chunking")
    if force_delete:
        print(f"⚠️  WARNING: Force delete mode enabled - existing data will be deleted!")
    else:
        print(f"📁 Preserving existing data (use --force-delete to delete existing data)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        from data_train_test_split import OptimizedTIFFChunkerWithShapefiles
        
        # If processed_gdf is not available, try to load from shapefile if not force_delete
        if processed_gdf is None:
            if not force_delete:
                # Try to load the latest processed shapefile from stage 2 output dir
                stage2_outdir = PIPELINE_CONFIGS['process_kostanai_buildings']['output_dir']
                stage2_outdir = Path(stage2_outdir)
                shp_files = list(stage2_outdir.glob('buildings_processed_*.shp'))
                if shp_files:
                    # Sort by timestamp in filename, descending
                    shp_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    latest_shp = shp_files[0]
                    print(f"Loading processed buildings from: {latest_shp}")
                    processed_gdf = gpd.read_file(str(latest_shp))
                else:
                    print("No processed shapefile found in stage 2 output directory. Running Stage 2...")
                    if not run_stage2():
                        return False
            else:
                print("Processed data not available, running Stage 2 first...")
                if not run_stage2():
                    return False
        
        # Configuration for the data split
        config = PIPELINE_CONFIGS['data_split'].copy()
        config['labels_gdf'] = processed_gdf
        config['rewrite_output_dir'] = force_delete  # Only delete if explicitly requested
        # If region_shapefile_path is not specified, set to None
        if not config.get('region_shapefile_path'):
            config['region_shapefile_path'] = None
        
        # Create and run chunker
        chunker = OptimizedTIFFChunkerWithShapefiles(**config)
        result = chunker.process_all_tiffs()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ SUCCESS: Stage 3 completed")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Created {len(result['train_chunks'])} train chunks")
        print(f"Created {len(result['val_chunks'])} validation chunks")
        print(f"Created {len(result['test_chunks'])} test chunks")
        
        return True
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n❌ ERROR: Stage 3 failed")
        print(f"Error: {str(e)}")
        print(f"Duration: {duration:.2f} seconds")
        
        return False

def run_all_stages(force_delete=False):
    """Run all stages in sequence."""
    print("🚀 KOSTANAI DATA PROCESSING PIPELINE - ALL STAGES")
    if force_delete:
        print("⚠️  WARNING: Force delete mode enabled - existing data will be deleted!")
    else:
        print("📁 Preserving existing data (use --force-delete to delete existing data)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Track overall success
    overall_start_time = time.time()
    successful_stages = 0
    total_stages = 3
    
    # Run each stage
    stages = [
        ("Stage 1", run_stage1),
        ("Stage 2", run_stage2),
        ("Stage 3", lambda: run_stage3_with_delete(force_delete))
    ]
    
    for i, (stage_name, stage_func) in enumerate(stages, 1):
        print(f"\n📋 Running {stage_name} ({i}/{total_stages})")
        
        success = stage_func()
        
        if success:
            successful_stages += 1
        else:
            print(f"\n⚠️  Pipeline failed at {stage_name}")
            print("You may need to fix the error and restart from this stage.")
            break
    
    # Final summary
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    
    print(f"\n{'='*60}")
    print("🏁 PIPELINE COMPLETION SUMMARY")
    print(f"{'='*60}")
    print(f"Total stages: {total_stages}")
    print(f"Successful stages: {successful_stages}")
    print(f"Failed stages: {total_stages - successful_stages}")
    print(f"Overall duration: {overall_duration:.2f} seconds")
    
    if successful_stages == total_stages:
        print("\n🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
        print("The Kostanai dataset is now ready for training.")
    else:
        print(f"\n⚠️  Pipeline completed with {total_stages - successful_stages} failure(s)")
        print("Please check the error messages above and fix any issues.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_customsplit(split_size=100, force_delete=False):
    global processed_gdf

    print(f"\n{'='*60}")
    print(f"CUSTOM SPLIT: Split data into user-defined splits with image chunking")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        from data_train_test_split import OptimizedTIFFChunkerWithShapefiles

        config = PIPELINE_CONFIGS['custom_split'].copy()
        labels_gdf = processed_gdf
        # If labels_gdf is not available, try to load from shapefile if not force_delete
        if labels_gdf is None:
            if not force_delete:
                # Try to load the latest processed shapefile from stage 2 output dir
                stage2_outdir = PIPELINE_CONFIGS['process_kostanai_buildings']['output_dir']
                stage2_outdir = Path(stage2_outdir)
                shp_files = list(stage2_outdir.glob('buildings_processed_*.shp'))
                if shp_files:
                    # Sort by timestamp in filename, descending
                    shp_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    latest_shp = shp_files[0]
                    print(f"Loading processed buildings from: {latest_shp}")
                    labels_gdf = gpd.read_file(str(latest_shp))
                else:
                    print("No processed shapefile found in stage 2 output directory. Running Stage 2...")
                    if not run_stage2():
                        return False
            else:
                print("Processed data not available, running Stage 2 first...")
                if not run_stage2():
                    return False
        config['labels_gdf'] = labels_gdf
        config['rewrite_output_dir'] = False
        if not config.get('region_shapefile_path'):
            config['region_shapefile_path'] = None

        chunker = OptimizedTIFFChunkerWithShapefiles(**config)
        chunker.process_custom_splits_by_count(split_size)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ SUCCESS: Custom split completed")
        print(f"Duration: {duration:.2f} seconds")

        return True

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n❌ ERROR: Custom split failed")
        print(f"Error: {str(e)}")
        print(f"Duration: {duration:.2f} seconds")

        return False

def run_complete_dataset():
    global processed_gdf

    print(f"\n{'='*60}")
    print(f"COMPLETE DATASET: Create all chunks in a single folder for annotation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        from data_train_test_split import OptimizedTIFFChunkerWithShapefiles

        if processed_gdf is None:
            print("Processed data not available, running Stage 2 first...")
            if not run_stage2():
                return False

        config = PIPELINE_CONFIGS['complete_dataset'].copy()
        config['labels_gdf'] = processed_gdf
        config['rewrite_output_dir'] = False
        if not config.get('region_shapefile_path'):
            config['region_shapefile_path'] = None

        chunker = OptimizedTIFFChunkerWithShapefiles(**config)
        
        # Create complete_dataset folder
        complete_dataset_dir = Path(config['output_dir']) / 'complete_dataset'
        complete_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all TIFF files with exact extensions only (recursive)
        tiff_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF']:
            tiff_files.extend([f for f in Path(config['tiff_dir']).rglob(f'*{ext}') if f.name.endswith(ext)])
        
        print(f"Found {len(tiff_files)} TIFF files")
        if len(tiff_files) == 0:
            print("No TIFF files found!")
            return False
            
        # Process all TIFF files
        all_chunks = []
        from tqdm import tqdm
        with tqdm(total=len(tiff_files), desc="Processing TIFF files") as pbar:
            for tiff_path in tiff_files:
                pbar.set_postfix_str(f"Processing {tiff_path.name}")
                chunks = chunker._process_tiff_file(tiff_path)
                all_chunks.extend(chunks)
                pbar.update(1)
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        if len(all_chunks) == 0:
            print("No valid chunks created!")
            return False
        
        # Save all chunks to complete_dataset folder
        img_dir = complete_dataset_dir / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO structure for complete dataset
        coco = chunker._init_coco_structure()
        
        for chunk_info in all_chunks:
            # Save image
            chunk_path = img_dir / chunk_info['filename']
            import cv2
            cv2.imwrite(str(chunk_path), chunk_info['chunk'])
            
            # Add image entry
            image_data = {
                "license": 4,
                "file_name": chunk_info['filename'],
                "coco_url": "",
                "height": config['chunk_size'][0],
                "width": config['chunk_size'][1],
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
        import json
        json_path = complete_dataset_dir / 'complete_dataset.json'
        with open(json_path, 'w') as f:
            json.dump(coco, f, indent=2)
        
        print(f"Saved complete dataset: {len(all_chunks)} images, {len(coco['annotations'])} annotations")
        print(f"Location: {complete_dataset_dir}")

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ SUCCESS: Complete dataset created")
        print(f"Duration: {duration:.2f} seconds")

        return True

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n❌ ERROR: Complete dataset creation failed")
        print(f"Error: {str(e)}")
        print(f"Duration: {duration:.2f} seconds")

        return False

def main():
    """Main function to handle command line arguments and run appropriate stages."""
    parser = argparse.ArgumentParser(
        description="Kostanai data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/data_pipeline/main.py stage1    # Run only stage 1
  python tools/data_pipeline/main.py stage2    # Run only stage 2
  python tools/data_pipeline/main.py stage3    # Run only stage 3 (preserves existing data)
  python tools/data_pipeline/main.py stage3 --force-delete  # Run stage 3 and delete existing data
  python tools/data_pipeline/main.py customsplit  # Create numbered splits (split0, split1, etc.)
  python tools/data_pipeline/main.py complete_dataset  # Save all chunks into 'complete_dataset' for annotation
  python tools/data_pipeline/main.py all       # Run all stages
  python tools/data_pipeline/main.py           # Run all stages (default)
        """
    )
    
    parser.add_argument(
        'stage',
        nargs='?',
        default='all',
        choices=['stage1', 'stage2', 'stage3', 'customsplit', 'complete_dataset', 'all'],
        help='Which stage(s) to run (default: all)'
    )
    parser.add_argument(
        '--split-size',
        type=int,
        default=200,
        help='Number of images per split (default: 100)'
    )
    parser.add_argument(
        '--force-delete',
        action='store_true',
        help='WARNING: Force deletion of existing output data. Use with extreme caution!'
    )
    args = parser.parse_args()
    
    # Run the appropriate stage(s)
    if args.stage == 'stage1':
        run_stage1()
    elif args.stage == 'stage2':
        run_stage2()
    elif args.stage == 'stage3':
        run_stage3_with_delete(args.force_delete)
    elif args.stage == 'customsplit':
        run_customsplit(args.split_size)
    elif args.stage == 'complete_dataset':
        run_complete_dataset()
    elif args.stage == 'all':
        run_all_stages(args.force_delete)

if __name__ == "__main__":
    """
    python tools/data_pipeline/main.py [stage1|stage2|stage3|customsplit|complete_dataset|all]
    """
    main()
