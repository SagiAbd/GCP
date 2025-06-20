import cv2
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

def chop_tiff_to_chunks(input_path, output_dir, chunk_size=(1024, 1024, 3), overlap=0, create_coco_json=True, 
                       original_resolution=5.0, target_resolution=10.0):
    """
    Chop a TIFF image into chunks of specified size and save to output directory.
    
    Args:
        input_path (str): Path to input TIFF image
        output_dir (str): Directory to save chunks
        chunk_size (tuple): Size of chunks (height, width, channels)
        overlap (int): Overlap between chunks in pixels (default: 0)
        create_coco_json (bool): Whether to create COCO-style JSON file
        original_resolution (float): Original resolution in cm/px (default: 5.0)
        target_resolution (float): Target resolution in cm/px (default: 10.0)
    
    Returns:
        tuple: (list of saved chunk file paths, path to JSON file if created)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read the TIFF image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image from {input_path}")
    
    print(f"Original image shape: {img.shape}")
    print(f"Original resolution: {original_resolution} cm/px")
    print(f"Target resolution: {target_resolution} cm/px")
    
    # Calculate downscaling factor
    scale_factor = original_resolution / target_resolution
    print(f"Scale factor: {scale_factor}")
    
    # Downscale the image if needed
    if scale_factor != 1.0:
        original_height, original_width = img.shape[:2]
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)
        
        print(f"Downscaling from {original_height}x{original_width} to {new_height}x{new_width}")
        
        # Use INTER_AREA for downscaling (best quality for shrinking)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Downscaled image shape: {img.shape}")
    
    # Get image dimensions after potential downscaling
    if len(img.shape) == 2:  # Grayscale
        height, width = img.shape
        channels = 1
        # Convert to 3-channel if needed
        if chunk_size[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            channels = 3
    else:
        height, width, channels = img.shape
    
    print(f"Final image shape for chunking: {img.shape}")
    print(f"Target chunk size: {chunk_size}")
    print(f"Final resolution: {target_resolution} cm/px")
    
    # Ensure image has the required number of channels
    if channels < chunk_size[2]:
        if channels == 1 and chunk_size[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 3 and chunk_size[2] == 4:
            # Add alpha channel
            alpha = np.ones((height, width, 1), dtype=img.dtype) * 255
            img = np.concatenate([img, alpha], axis=2)
        channels = img.shape[2]
    elif channels > chunk_size[2]:
        # Take only the required channels
        img = img[:, :, :chunk_size[2]]
        channels = chunk_size[2]
    
    chunk_height, chunk_width = chunk_size[0], chunk_size[1]
    step_height = chunk_height - overlap
    step_width = chunk_width - overlap
    
    saved_chunks = []
    chunk_count = 0
    coco_images = []  # For COCO JSON format
    
    # Calculate number of chunks
    rows = (height - overlap) // step_height + (1 if (height - overlap) % step_height > 0 else 0)
    cols = (width - overlap) // step_width + (1 if (width - overlap) % step_width > 0 else 0)
    
    print(f"Creating {rows} x {cols} = {rows * cols} chunks")
    
    # Extract chunks
    for row in range(rows):
        for col in range(cols):
            # Calculate chunk boundaries
            start_row = row * step_height
            end_row = min(start_row + chunk_height, height)
            start_col = col * step_width
            end_col = min(start_col + chunk_width, width)
            
            # Extract chunk
            chunk = img[start_row:end_row, start_col:end_col]
            
            # Pad chunk if it's smaller than target size
            if chunk.shape[0] < chunk_height or chunk.shape[1] < chunk_width:
                padded_chunk = np.zeros((chunk_height, chunk_width, channels), dtype=img.dtype)
                padded_chunk[:chunk.shape[0], :chunk.shape[1]] = chunk
                chunk = padded_chunk
            
            # Generate filename
            base_name = Path(input_path).stem
            chunk_filename = f"{base_name}_chunk_{row:03d}_{col:03d}.TIF"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            # Save chunk
            cv2.imwrite(chunk_path, chunk)
            saved_chunks.append(chunk_path)
            
            # Add to COCO format data
            if create_coco_json:
                image_id = int(f"{row}{col:03d}")  # Create unique ID from row/col
                coco_images.append({
                    "license": 4,
                    "file_name": chunk_filename,
                    "coco_url": "",
                    "height": chunk_height,
                    "width": chunk_width,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": image_id
                })
            
            chunk_count += 1
            
            print(f"Saved chunk {chunk_count}: {chunk_filename} - Shape: {chunk.shape}")
    
    print(f"Successfully created {chunk_count} chunks in {output_dir}")
    
    # Create COCO JSON file
    json_path = None
    if create_coco_json:
        json_path = create_coco_json_file(output_dir, coco_images)
        print(f"Created COCO JSON file: {json_path}")
    
    return saved_chunks, json_path

def create_coco_json_file(output_dir, images_list, filename="test.json"):
    """
    Create a COCO-style JSON file with image information.
    
    Args:
        output_dir (str): Directory to save the JSON file
        images_list (list): List of image dictionaries in COCO format
        filename (str): Name of the JSON file
    
    Returns:
        str: Path to created JSON file
    """
    coco_data = {
        "images": images_list,
        "annotations": [],  # Empty for now, can be populated later
        "categories": [],   # Empty for now, can be populated later
        "info": {
            "description": "Generated chunks dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "TIFF Chunker",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 4,
                "name": "Attribution License",
                "url": "http://creativecommons.org/licenses/by/2.0/"
            }
        ]
    }
    
    json_path = os.path.join(output_dir, filename)
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    return json_path

def chop_tiff_advanced(input_path, output_dir, chunk_size=(1024, 1024, 3), 
                      overlap=0, prefix="chunk", save_format="TIF", create_coco_json=True,
                      original_resolution=5.0, target_resolution=10.0):
    """
    Advanced version with more options.
    
    Args:
        input_path (str): Path to input TIFF image
        output_dir (str): Directory to save chunks
        chunk_size (tuple): Size of chunks (height, width, channels)
        overlap (int): Overlap between chunks in pixels
        prefix (str): Prefix for chunk filenames
        save_format (str): Output format ('TIF', 'png', 'jpg')
        create_coco_json (bool): Whether to create COCO-style JSON file
        original_resolution (float): Original resolution in cm/px (default: 5.0)
        target_resolution (float): Target resolution in cm/px (default: 10.0)
    
    Returns:
        dict: Information about the chunking process
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image from {input_path}")
    
    original_shape = img.shape
    
    # Calculate and apply downscaling
    scale_factor = original_resolution / target_resolution
    if scale_factor != 1.0:
        original_height, original_width = img.shape[:2]
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)
        
        # Use INTER_AREA for downscaling (best quality for shrinking)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Handle channel conversion
    if len(img.shape) == 2:
        if chunk_size[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        if img.shape[2] != chunk_size[2]:
            if img.shape[2] == 1 and chunk_size[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 3 and chunk_size[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=2)
            else:
                img = img[:, :, :chunk_size[2]]
    
    height, width = img.shape[:2]
    chunk_height, chunk_width = chunk_size[0], chunk_size[1]
    step_height = chunk_height - overlap
    step_width = chunk_width - overlap
    
    chunks_info = []
    coco_images = []  # For COCO JSON format
    
    # Process chunks
    row_idx = 0
    while row_idx * step_height < height:
        col_idx = 0
        while col_idx * step_width < width:
            # Calculate boundaries
            start_row = row_idx * step_height
            end_row = min(start_row + chunk_height, height)
            start_col = col_idx * step_width
            end_col = min(start_col + chunk_width, width)
            
            # Extract and pad chunk
            chunk = img[start_row:end_row, start_col:end_col]
            
            if chunk.shape[0] < chunk_height or chunk.shape[1] < chunk_width:
                if len(chunk.shape) == 2:
                    padded_chunk = np.zeros((chunk_height, chunk_width), dtype=img.dtype)
                else:
                    padded_chunk = np.zeros((chunk_height, chunk_width, chunk.shape[2]), dtype=img.dtype)
                padded_chunk[:chunk.shape[0], :chunk.shape[1]] = chunk
                chunk = padded_chunk
            
            # Save chunk
            chunk_filename = f"{prefix}_{row_idx:03d}_{col_idx:03d}.{save_format}"
            chunk_path = os.path.join(output_dir, chunk_filename)
            cv2.imwrite(chunk_path, chunk)
            
            # Create unique ID based on row and column
            image_id = int(f"{row_idx}{col_idx:03d}")
            
            chunks_info.append({
                'filename': chunk_filename,
                'path': chunk_path,
                'row': row_idx,
                'col': col_idx,
                'bbox': (start_row, start_col, end_row, end_col),
                'shape': chunk.shape,
                'id': image_id
            })
            
            # Add to COCO format data
            if create_coco_json:
                coco_images.append({
                    "license": 4,
                    "file_name": chunk_filename,
                    "coco_url": "",
                    "height": chunk_height,
                    "width": chunk_width,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": image_id
                })
            
            col_idx += 1
        row_idx += 1
    
    # Create COCO JSON file
    json_path = None
    if create_coco_json:
        json_path = create_coco_json_file(output_dir, coco_images)
    
    return {
        'original_shape': original_shape,
        'final_shape': img.shape,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'total_chunks': len(chunks_info),
        'chunks': chunks_info,
        'output_dir': output_dir,
        'json_path': json_path,
        'original_resolution': original_resolution,
        'target_resolution': target_resolution,
        'scale_factor': original_resolution / target_resolution
    }

# Example usage
if __name__ == "__main__":
    # Basic usage
    input_tiff = r"D:\Sagi\GCP\GCP\data\raw\Pavlodar_AFS_2019_22_11.tif"
    output_directory = r"D:\Sagi\GCP\GCP\data\kazgisa-pavlodar-test"
    
    try:
        # Simple chunking with resolution downscaling
        saved_files, json_file = chop_tiff_to_chunks(
            input_path=input_tiff,
            output_dir=output_directory,
            chunk_size=(512, 512, 3),
            overlap=0,
            create_coco_json=True,
            original_resolution=5.0,  # 5 cm/px
            target_resolution=30    # 30 cm/px
        )
        
        print(f"\nChunking complete! Created {len(saved_files)} chunks.")
        if json_file:
            print(f"Created COCO JSON file: {json_file}")
        
        # Advanced chunking with overlap and resolution change
        # result = chop_tiff_advanced(
        #     input_path=input_tiff,
        #     output_dir="chunks_with_overlap",
        #     chunk_size=(1024, 1024, 3),
        #     overlap=128,
        #     prefix="tile",
        #     save_format="TIF",
        #     create_coco_json=True,
        #     original_resolution=5.0,
        #     target_resolution=10.0
        # )
        # print(f"Advanced chunking: {result['total_chunks']} chunks created")
        # print(f"Scale factor applied: {result['scale_factor']}")
        # if result['json_path']:
        #     print(f"JSON file created: {result['json_path']}")
        
    except Exception as e:
        print(f"Error: {e}")