# SAM3 Tracking to KWCoco Pipeline

This repository contains a script to perform open-vocabulary object tracking using Meta's SAM3 and serialize the output masks, bounding boxes, and track IDs into the KWCoco (Kitware COCO) format.

This output is designed to be ingested by bespoke visualization tools or downstream machine learning pipelines.

## Prerequisites

1. **Google Cloud Workstation**: A fresh instance running Debian/Ubuntu.

2. **Hugging Face Account**: Meta's SAM3 weights are gated.

   * Go to the [SAM 3 Hugging Face repository](https://huggingface.co/facebook/sam3) and agree to the terms.

   * Generate a "Read" Access Token in your Hugging Face account settings.

## Step 1: System Dependencies

Install required system libraries for git, video processing (ffmpeg/OpenCV), and Python environments:

```bash
sudo apt update
sudo apt install -y git wget ffmpeg libsm6 libxext6 python3-venv
```

## Step 2: Python Environment Setup

Create a virtual environment and install the required dependencies:

```bash
# 1. Create and activate a virtual environment
python3 -m venv sam3_env
source sam3_env/bin/activate

# 2. Install PyTorch
pip install --upgrade pip
pip install torch torchvision torchaudio

# 3. Clone and install SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..

# 4. Install serialization and helper libraries
pip install opencv-python kwimage kwcoco huggingface_hub
```

## Step 3: Authenticate with Hugging Face

You need to log in to the Hugging Face CLI so the script can download the SAM3 weights automatically:

```bash
huggingface-cli login
```

*(Paste your Access Token when prompted. It will be hidden as you type).*

## Step 4: Running the Script

The main script is `sam3_to_kwcoco.py`. It accepts arguments for the input video, the text prompt, the output JSON name, and the folder to dump extracted frames into.

### Example Usage:

```bash
# Optional: Download a sample video if you don't have one
wget https://github.com/opengeos/datasets/releases/download/videos/cars.mp4 -O sample_video.mp4

# Run the serialization script
python sam3_to_kwcoco.py \
    --video sample_video.mp4 \
    --prompt "car" \
    --output car_tracking.kwcoco.json \
    --frames-dir sample_frames
```

### Outputs:

1. **`car_tracking.kwcoco.json`**: The standard KWCoco manifest containing image paths, categories, annotations (with bounding boxes and RLE segmentations), and track IDs.

2. **`sample_frames/`**: A directory containing the extracted JPEG frames from the video. The JSON manifest references these relative paths.



# Contributing

We would love to have your contributions that improve current functionality, fix bugs, or add new features.  See [the contributing guidelines](CONTRIBUTING.md) for more info.

# Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and
Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project
code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the
Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub
project will be governed by all applicable Federal law. Any reference to specific commercial products,
processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or
imply their endorsement, recommendation or favoring by the Department of Commerce. The Department
of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to
imply endorsement of any commercial product or activity by DOC or the United States Government.
