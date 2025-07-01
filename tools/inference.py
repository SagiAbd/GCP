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
import cv2

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
        file_name = os.path.basename(img_path)
        image_id = idx + 1
        image_info.append({
            'id': image_id,
            'file_name': file_name,
            'width': width,
            'height': height
        })

        # Handle results - support both old and new MMDet formats
        if hasattr(result, 'pred_instances'):
            pred_instances = result.pred_instances
            if len(pred_instances) == 0:
                continue
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            # Prefer polygons if present
            has_polygons = hasattr(pred_instances, 'segmentations') and pred_instances.segmentations is not None
            has_masks = hasattr(pred_instances, 'masks') and pred_instances.masks is not None
            for i in range(len(bboxes)):
                score = float(scores[i])
                if score < args.score_thr:
                    continue
                x1, y1, x2, y2 = bboxes[i]
                bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                polygons = []
                # 1. Use polygons from model if available
                if has_polygons and pred_instances.segmentations[i] is not None and len(pred_instances.segmentations[i]) > 0:
                    polygons = pred_instances.segmentations[i]
                # 2. Fallback: convert mask to polygons
                elif has_masks:
                    mask_array = pred_instances.masks[i].cpu().numpy()
                    mask_uint8 = (mask_array > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) >= 3:
                            polygon = contour.flatten().tolist()
                            if len(polygon) >= 6:
                                polygons.append(polygon)
                if not polygons:
                    print(f"Warning: No valid polygons for detection {i} in {file_name}")
                    continue
                area = calculate_polygon_area(polygons, height, width)
                coco_det = {
                    'id': len(coco_results) + 1,
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': bbox_coco,
                    'segmentation': polygons,
                    'area': area,
                    'score': score,
                    'iscrowd': 0
                }
                coco_results.append(coco_det)
        elif isinstance(result, tuple) and len(result) == 2:
            # Old MMDet format (fallback)
            class_id = 0
            bbox_results, segm_results = result
            if (segm_results is not None and class_id < len(bbox_results) and class_id < len(segm_results)):
                bboxes_old = bbox_results[class_id]
                polygons_list = segm_results[class_id]
                for i, (bbox, polygons) in enumerate(zip(bboxes_old, polygons_list)):
                    score = float(bbox[4])
                    if score < args.score_thr:
                        continue
                    x1, y1, x2, y2 = bbox[:4]
                    bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    # polygons_list is already polygons (from mask2poly)
                    if not polygons or not any(len(poly) >= 6 for poly in polygons):
                        # fallback: try to convert mask to polygon if possible (not implemented here)
                        continue
                    area = calculate_polygon_area(polygons, height, width)
                    coco_det = {
                        'id': len(coco_results) + 1,
                        'image_id': image_id,
                        'category_id': 1,
                        'bbox': bbox_coco,
                        'segmentation': polygons,
                        'area': area,
                        'score': score,
                        'iscrowd': 0
                    }
                    coco_results.append(coco_det)
        else:
            print(f"Warning: Unsupported result format for {file_name}: {type(result)}")
            continue

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