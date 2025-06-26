import os
import shutil
import json
from pathlib import Path

def combine_coco_splits(data_dir, output_dir='complete_dataset'):
    data_dir = Path(data_dir)
    output_dir = data_dir / output_dir
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'val', 'test']
    all_images = []
    all_annotations = []
    image_id_map = {}
    next_image_id = 1
    next_ann_id = 1
    categories = None
    info = None
    licenses = None

    for split in splits:
        split_dir = data_dir / split
        json_path = split_dir / f'{split}.json'
        if not json_path.exists():
            print(f"Warning: {json_path} does not exist, skipping.")
            continue
        with open(json_path, 'r') as f:
            coco = json.load(f)
        if categories is None:
            categories = coco['categories']
        if info is None:
            info = coco.get('info', {})
        if licenses is None:
            licenses = coco.get('licenses', [])

        for img in coco['images']:
            old_id = img['id']
            img['id'] = next_image_id
            image_id_map[(split, old_id)] = next_image_id
            # Copy image file
            src_img = split_dir / 'images' / img['file_name']
            dst_img = images_dir / img['file_name']
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            all_images.append(img)
            next_image_id += 1

        for ann in coco['annotations']:
            ann['id'] = next_ann_id
            ann['image_id'] = image_id_map[(split, ann['image_id'])]
            all_annotations.append(ann)
            next_ann_id += 1

    merged_coco = {
        'images': all_images,
        'annotations': all_annotations,
        'categories': categories,
        'info': info,
        'licenses': licenses
    }
    out_json = output_dir / 'complete_dataset.json'
    with open(out_json, 'w') as f:
        json.dump(merged_coco, f, indent=2)
    print(f"Combined dataset saved to {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Combine COCO train/val/test splits into a single dataset.")
    parser.add_argument('data_dir', help="Directory containing train/val/test splits")
    parser.add_argument('--output_dir', default='complete_dataset', help="Output directory name (default: complete_dataset)")
    args = parser.parse_args()
    combine_coco_splits(args.data_dir, args.output_dir)