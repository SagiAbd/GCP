vastai algorithm:
1) git clone GCP
2) install datasets
3) change config for dataset paths (for dataset and model separately). Batch size too
4) change train/launch_train scripts as needed

GPU Configuration	Total Batch Size	Estimated Training Time
2x RTX 4090	            6	                    ~1.8 days (44 hours)
2x RTX 5090	            10	                    ~1 day (25 hours)



# Building Instance Segmentation Project

## Project Overview
Automatic detection and generation of building outlines on images.

## Author
KazGisa - Sagi Abdashim

## 🛠️ Technical Specification

### Task
Develop a model for automatic recognition and drawing of building and structure outlines based on Google Maps overlays (or Google satellite images), or on aerial photography (AFS).

### Requirements

#### Functionality
- Automatic detection and generation of building outlines on images.
- If a new building outline appears on the overlay/AFS, the system should automatically add it to the database and render it.
- Rescan the area and update data upon detecting changes (optional).

#### Accuracy
- The geometric accuracy and shape of the outlines must match the example provided in the reference image.
- Deviation in coordinates should not exceed X meters/pixels (to be specified based on the example).

#### Data Source
- Satellite imagery / Google Maps overlays (or Google Satellite)
- Available aerial photographs with 5cm accuracy.

#### Output Data
- Vector outlines of buildings (formats: GeoJSON, Shapefile).
- Accompanying attribute information:
  - Date of appearance
  - Building status (residential, non-residential, etc.) # Optional
  - Area
  - Coordinates

## 📌 Features
- [x] Instance segmentation support via MMDetection
- [x] Command-line interface for training, evaluation, and visualization
- [x] Integration with Weights & Biases for experiment tracking
- [ ] TorchScript and ONNX export (planned)
- [ ] Post-processing (planned)

## 🚀 Getting Started
*Instructions for setup and usage will be provided in a future update.*

## 🧠 Model Architecture and Approach
Initially, we used P2PFormer ([paper](https://arxiv.org/pdf/2406.02930v1)), but later adopted the GCP model ([paper](https://arxiv.org/pdf/2505.01385)) with a ResNet50 backbone for improved performance. Training GCP involves two stages ([GitHub repo](https://github.com/zhu-xlab/GCP)):
1. Train Mask2Former for mask extraction
2. Train GCP for polygionization of masks by adding a polygonizer head, freezing all other layers.

The multi-class instance segmentation approach was abandoned due to the complexity of distinguishing residential from non-residential buildings.

### Alternative Experiments
- Multi-class instance segmentation (abandoned)
- Training only on AFS (aerial) images (in progress)
- Training only on Google Satellite images (planned)
- Combining both datasets (planned)

