import os
import sys
import argparse
from pathlib import Path
import mmcv
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
import json
import numpy as np
from pycocotools import mask as maskUtils

def get_image_files(input_dir):
    input_dir = Path(input_dir)
    return sorted([str(f) for f in input_dir.glob('*.tif')])

def calculate_polygon_area(polygons, height, width):
    """Calculate exact area of polygons using pycocotools"""
    try:
        rle = maskUtils.frPyObjects(polygons, height, width)
        if isinstance(rle, list):
            rle = maskUtils.merge(rle)
        return float(maskUtils.area(rle))
    except:
        # Fallback to simple polygon area calculation
        total_area = 0
        for polygon in polygons:
            if len(polygon) >= 6:  # At least 3 points
                coords = np.array(polygon).reshape(-1, 2)
                # Shoelace formula
                area = 0.5 * abs(sum(coords[i][0] * (coords[i+1][1] - coords[i-1][1]) 
                                   for i in range(len(coords))))
                total_area += area
        return float(total_area)

def main():
    """
    python tools/inference.py --config D:\Sagi\GCP\GCP\configs\gcp\gcp_r50_kazgisa-kostanai.py --checkpoint D:\Sagi\GCP\GCP\checkpoints\gcp_e5_lre-4_kostanai-afs_v1.pth --input_dir D:\Sagi\GCP\GCP\data\shymkent\images
    """
    parser = argparse.ArgumentParser(description="MMDetection instance segmentation inference for building detection")
    parser.add_argument('--config', type=str, required=True, help='Path to mmdet config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with .tif images')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference')
    parser.add_argument('--score_thr', type=float, default=0.7, help='Score threshold for predictions')
    args = parser.parse_args()

    # Prepare output path
    input_dir = Path(args.input_dir)
    output_json = input_dir.parent / 'predictions.json'

    # Load model
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Prepare image list
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"No .tif images found in {input_dir}")
        sys.exit(1)

    # Prepare COCO-style results
    coco_results = []
    image_info = []

    for idx, img_path in enumerate(tqdm(image_files, desc='Processing images')):
        img = mmcv.imread(img_path)
        height, width = img.shape[:2]
        result = inference_detector(model, img)
        
        # Prepare image info
        file_name = os.path.basename(img_path)
        image_id = idx + 1
        
        image_info.append({
            'id': image_id,
            'file_name': file_name,
            'width': width,
            'height': height
        })

        # Handle results - expecting (bbox_results, segm_results)
        if isinstance(result, tuple) and len(result) == 2:
            bbox_results, segm_results = result
        else:
            print(f"Warning: Unexpected result format for {file_name}")
            continue

        # Process detections (assuming single class - building)
        class_id = 0  # Single class
        if class_id < len(bbox_results) and class_id < len(segm_results):
            bboxes = bbox_results[class_id]
            polygons_list = segm_results[class_id]
            
            # Process each detection
            for bbox, polygons in zip(bboxes, polygons_list):
                score = float(bbox[4])
                if score < args.score_thr:
                    continue
                
                # Convert bbox to COCO format [x, y, width, height]
                x1, y1, x2, y2 = bbox[:4]
                bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                # Calculate exact polygon area
                area = calculate_polygon_area(polygons, height, width)
                
                # Create COCO annotation
                coco_det = {
                    'id': len(coco_results) + 1,  # Unique annotation ID
                    'image_id': image_id,
                    'category_id': 1,  # Building class (1-based)
                    'bbox': bbox_coco,
                    'segmentation': polygons,
                    'area': area,
                    'score': score,
                    'iscrowd': 0
                }
                
                coco_results.append(coco_det)

    # Create complete COCO-style output
    output_data = {
        'images': image_info,
        'annotations': coco_results,
        'categories': [
            {
                'id': 1,
                'name': 'building',
                'supercategory': 'structure'
            }
        ]
    }

    # Save predictions.json
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(coco_results)} building detections to {output_json}")
    print(f"Processed {len(image_info)} images")

if __name__ == '__main__':
    main()