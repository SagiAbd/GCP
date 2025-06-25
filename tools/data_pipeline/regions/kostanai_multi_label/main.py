#!/usr/bin/env python3
"""
Main script for Kostanai data processing pipeline (multi-label: residential/non-residential).

This script orchestrates the execution of three data processing steps:
1. Stage 1: Merge raw Kostanai building data from multiple GDB files
2. Stage 2: Process and clean the merged building geometries, assigning 'residential' or 'non-residential' labels
3. Stage 3: Split the processed data into train/val/test sets with image chunking and export COCO with two categories

Usage:
    python tools/data_pipeline/regions/kostanai_multi_label/main.py [stage1|stage2|stage3|all]
    
Examples:
    python tools/data_pipeline/regions/kostanai_multi_label/main.py stage1    # Run only stage 1
    python tools/data_pipeline/regions/kostanai_multi_label/main.py stage2    # Run only stage 2
    python tools/data_pipeline/regions/kostanai_multi_label/main.py stage3    # Run only stage 3
    python tools/data_pipeline/regions/kostanai_multi_label/main.py all       # Run all stages
    python tools/data_pipeline/regions/kostanai_multi_label/main.py           # Run all stages (default)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import the modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Global variables to store results between stages
merged_gdf = None
processed_gdf = None

def run_stage1():
    """Stage 1: Merge raw Kostanai building data from multiple GDB files."""
    global merged_gdf
    
    print(f"\n{'='*60}")
    print(f"STAGE 1: Merge raw Kostanai building data from multiple GDB files")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        from multi_label_merge_raw_kostanai_buildings import MergeKostanaiBuildings
        merger = MergeKostanaiBuildings()
        merged_gdf = merger.process()
        
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
        from multi_label_process_kostanai_buildings import ProcessKostanaiBuildings
        
        # If merged_gdf is not available, run stage 1 first
        if merged_gdf is None:
            print("Merged data not available, running Stage 1 first...")
            if not run_stage1():
                return False
        
        # Initialize processor with custom parameters
        processor = ProcessKostanaiBuildings(
            distance_threshold=0.15,      # 15cm grouping distance
            min_intersection_length=2.0,  # 2m minimum intersection
            max_group_size=10,           # Max 10 buildings per group
            min_area_threshold=5.0        # 5m² minimum area
        )
        
        # Process the merged data
        processed_gdf = processor.process_geodataframe_optimized(
            merged_gdf, 
            apply_geometry_processing=True
        )
        
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

def run_stage3():
    """Stage 3: Split data into train/val/test sets with image chunking."""
    global processed_gdf
    
    print(f"\n{'='*60}")
    print(f"STAGE 3: Split data into train/val/test sets with image chunking")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        from multi_label_data_train_test_split import OptimizedTIFFChunkerWithShapefiles
        
        # If processed_gdf is not available, run stage 2 first
        if processed_gdf is None:
            print("Processed data not available, running Stage 2 first...")
            if not run_stage2():
                return False
        
        # Configuration for the data split
        config = {
            'tiff_dir': r"D:\Sagi\GCP\GCP\data\raw\images\kostanai\Images\ortho kostanay",
            'region_shapefile_path': r"D:\Sagi\GCP\GCP\data\raw\labels\kostanai\region_bbox\region_bbox.shp",
            'labels_gdf': processed_gdf,  # Pass the processed GeoDataFrame directly (with building_type column)
            'output_dir': r"D:\Sagi\GCP\GCP\data\processed\multi-label-kostanai",
            'chunk_size': (512, 512, 3),
            'overlap': 0,
            'original_resolution': 5.0,
            'target_resolution': 30.0,
            'train_split': 0.5,
            'val_split': 0.05,
            'test_split': 0.45,
            'min_valid_pixels': 0.7,
            'rewrite_output_dir': True
        }
        
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

def run_all_stages():
    """Run all stages in sequence."""
    print("🚀 KOSTANAI DATA PROCESSING PIPELINE - ALL STAGES")
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
        ("Stage 3", run_stage3)
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

def main():
    """Main function to handle command line arguments and run appropriate stages."""
    parser = argparse.ArgumentParser(
        description="Kostanai data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/data_pipeline/regions/kostanai_multi_label/main.py stage1    # Run only stage 1
  python tools/data_pipeline/regions/kostanai_multi_label/main.py stage2    # Run only stage 2
  python tools/data_pipeline/regions/kostanai_multi_label/main.py stage3    # Run only stage 3
  python tools/data_pipeline/regions/kostanai_multi_label/main.py all       # Run all stages
  python tools/data_pipeline/regions/kostanai_multi_label/main.py           # Run all stages (default)
        """
    )
    
    parser.add_argument(
        'stage',
        nargs='?',
        default='all',
        choices=['stage1', 'stage2', 'stage3', 'all'],
        help='Which stage(s) to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Run the appropriate stage(s)
    if args.stage == 'stage1':
        run_stage1()
    elif args.stage == 'stage2':
        run_stage2()
    elif args.stage == 'stage3':
        run_stage3()
    elif args.stage == 'all':
        run_all_stages()

if __name__ == "__main__":
    """
    python tools/data_pipeline/regions/kostanai_multi_label/main.py [stage1|stage2|stage3|all]
    """
    main()
