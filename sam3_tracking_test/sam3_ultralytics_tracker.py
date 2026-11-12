import os
import argparse
import numpy as np
import cv2
import kwimage
import kwcoco
from ultralytics.models.sam import SAM3VideoSemanticPredictor

def get_video_metadata(video_path: str) -> tuple[int, int, int]:
    """
    Reads the video header to extract metadata without decoding the pixels.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    cap.release()
    print(f"Video metadata loaded: {total_frames} frames, {w}x{h} resolution.")
    return total_frames, w, h

def generate_kwcoco_manifest(
    video_path: str, 
    text_prompt: str, 
    output_json: str, 
    conf_threshold: float = 0.50, 
    mask_threshold: float = 0.0,
    max_mem_frames: int = 30
) -> None:
    """
    Runs Ultralytics SAM on a video stream and serializes tracking data to KWCoco.
    """
    # 1. Fetch metadata
    total_frames, w, h = get_video_metadata(video_path)
    
    # 2. Initialize KWCoco Dataset
    dset = kwcoco.CocoDataset()
    cat_id = dset.add_category(name=text_prompt)
    vid_id = dset.add_video(name=os.path.basename(video_path))
    
    # Pre-register all virtual frames
    img_ids = {}
    for i in range(total_frames):
        img_id = dset.add_image(
            name=f"frame_{i:05d}",
            video_id=vid_id, 
            frame_index=i, 
            width=w, 
            height=h
        )
        img_ids[i] = img_id

    # 3. Load Ultralytics SAM Engine
    print("Loading Ultralytics SAM3VideoSemanticPredictor engine...")
    
    overrides = {
        "conf": conf_threshold,
        "task": "segment",
        "mode": "predict",
        "model": "/home/user/sam3/checkpoints/sam3.pt",
        "imgsz": max(w, h), # Dynamically forces native resolution,
        "half": True 
    }
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    
    print(f"Propagating masks natively through the video stream (stream=True)...")
    
    # 4. Stream and Track (This loop replaces the Meta predictor)
    # The stream=True argument guarantees frames are processed and discarded one-by-one
    results = predictor(
        source=video_path,
        text=[text_prompt],
        stream=True
    )
    
    for frame_idx, result in enumerate(results):
        
        # Ensure we have a registered virtual image
        if frame_idx not in img_ids:
            img_ids[frame_idx] = dset.add_image(
                name=f"frame_{frame_idx:05d}", video_id=vid_id, frame_index=frame_idx, width=w, height=h
            )
            
        current_img_id = img_ids[frame_idx]
        
        # Check if the model found any objects in this frame
        if result.masks is not None and result.boxes is not None:
            # Extract PyTorch tensors to CPU NumPy arrays
            masks_np = result.masks.data.cpu().numpy().astype(np.uint8)
            boxes_np = result.boxes.xyxy.cpu().numpy()
            
            # SAM3VideoSemanticPredictor tracks instances natively. 
            # Fallback to sequential index if the Ultralytics YOLO ID field isn't populated
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                track_ids = list(range(len(masks_np)))
            
            for mask_arr, bbox, track_id in zip(masks_np, boxes_np, track_ids):
                # Wrap in KWImage Mask object
                kw_mask = kwimage.Mask(mask_arr, 'c_mask')
                
                # Convert to compressed COCO RLE format
                coco_rle = kw_mask.to_coco(style='rle')
                
                # Use the bounding box directly from the Ultralytics tracker
                # Format is [x_min, y_min, width, height] for COCO. Ensure they are native floats for JSON.
                x1, y1, x2, y2 = map(float, bbox)
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                
                # Add annotation to manifest
                dset.add_annotation(
                    image_id=current_img_id,
                    category_id=cat_id,
                    bbox=coco_bbox,
                    segmentation=coco_rle,
                    track_id=track_id
                )

    # 5. Save the dataset
    print(f"Dumping completed KWCoco dataset to {output_json}...")
    dset.dump(output_json, newlines=True)
    print("Serialization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ultralytics SAM natively on a video and export tracking data to KWCoco format.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input MP4 video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the object to track.")
    parser.add_argument("--output", type=str, default="tracking_output.kwcoco.json", help="Path for the output KWCoco JSON file.")
    
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold for presence (0.0 to 1.0).")
    parser.add_argument("--mask-thresh", type=float, default=0.0, help="Mask binarization threshold (adjusts mask tightness/coarseness).")
    parser.add_argument("--max-mem-frames", type=int, default=30, help="(Legacy arg) Unused by Ultralytics, handled internally.")
    
    args = parser.parse_args()
    
    generate_kwcoco_manifest(
        video_path=args.video,
        text_prompt=args.prompt,
        output_json=args.output,
        conf_threshold=args.conf,
        mask_threshold=args.mask_thresh,
        max_mem_frames=args.max_mem_frames
    )
