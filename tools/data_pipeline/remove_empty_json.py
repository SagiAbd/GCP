import json
import os
import argparse
# python tools/data_pipeline/remove_empty_json.py D:\Sagi\GCP\GCP\data\kostanai\train\train.json^C
def is_empty_annotation(ann):
    # Consider empty if segmentation is empty, bbox is empty, or area == 0
    return (not ann.get("segmentation")) or (not ann.get("bbox")) or (ann.get("area", 1) == 0)

def remove_empty_annotations(input_path):
    # Load original JSON
    with open(input_path, 'r') as f:
        coco = json.load(f)

    # Filter annotations
    original_count = len(coco.get('annotations', []))
    filtered_annotations = [ann for ann in coco.get('annotations', []) if not is_empty_annotation(ann)]
    filtered_count = len(filtered_annotations)

    # Update annotations
    coco['annotations'] = filtered_annotations

    # Write output JSON
    base, ext = os.path.splitext(input_path)
    output_path = base + '_cleaned' + ext
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Original annotations: {original_count}")
    print(f"Filtered annotations: {filtered_count}")
    print(f"Saved cleaned file to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove empty annotations from COCO JSON')
    parser.add_argument('input_json', help='Path to input COCO JSON file')
    args = parser.parse_args()

    remove_empty_annotations(args.input_json)
