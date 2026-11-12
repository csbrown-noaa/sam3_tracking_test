import os
import argparse
import torch
import cv2
import numpy as np
import kwimage
import kwcoco
from tqdm import tqdm

# Import the Hugging Face SAM 3 implementation
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

def get_video_metadata(video_path: str) -> tuple[int, int, int]:
    """Reads the video header to extract metadata."""
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
    # 1. Fetch metadata
    total_frames, w, h = get_video_metadata(video_path)
    
    # 2. Initialize KWCoco Dataset
    dset = kwcoco.CocoDataset()
    cat_id = dset.add_category(name=text_prompt)
    vid_id = dset.add_video(name=os.path.basename(video_path))
    
    # Pre-register virtual frames
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

    # 3. Load Hugging Face SAM 3 Model & Processor
    print("Loading Hugging Face SAM 3 Tracker...")
    model_id = "meta-facebook/sam3-base"  # Base model identifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)
    model = Sam3TrackerVideoModel.from_pretrained(model_id).to(device)
    
    # 4. Initialize bounded streaming session
    print(f"Initializing streaming session (VRAM Memory Bank Limit: {max_mem_frames} frames)...")
    session_state = model.init_video_session(max_vision_features_cache_size=max_mem_frames)
    
    # 5. Stream and process video frame-by-frame to prevent System RAM OOM
    print("Propagating masks natively through the video stream...")
    cap = cv2.VideoCapture(video_path)
    
    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert OpenCV BGR to standard RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare inputs for the processor
            if frame_idx == 0:
                print(f"\nAdding text prompt: '{text_prompt}' to frame 0...")
                inputs = processor(images=rgb_frame, text=text_prompt, return_tensors="pt").to(device)
                outputs = model.add_prompt(
                    session_state=session_state,
                    frame_index=frame_idx,
                    **inputs
                )
            else:
                inputs = processor(images=rgb_frame, return_tensors="pt").to(device)
                outputs = model.propagate_in_video(
                    session_state=session_state,
                    frame_index=frame_idx,
                    **inputs
                )
                
            # 6. Extract masks and serialize
            # HF outputs raw logits, we apply the mask_threshold to binarize
            pred_masks = outputs.pred_masks.squeeze(0).cpu().numpy()  # Shape: (num_objects, H, W)
            pred_scores = outputs.iou_scores.squeeze(0).cpu().numpy() # Shape: (num_objects,)
            
            for obj_id, (mask_logit, score) in enumerate(zip(pred_masks, pred_scores)):
                # Skip if the model is not confident
                if score < conf_threshold:
                    continue
                    
                # Binarize the mask
                binary_mask = (mask_logit > mask_threshold).astype(np.uint8)
                
                # Skip empty masks
                if binary_mask.sum() == 0:
                    continue
                
                # Wrap in KWImage Mask object
                kw_mask = kwimage.Mask(binary_mask, 'c_mask')
                
                # Convert to compressed COCO RLE format and bbox
                coco_rle = kw_mask.to_coco(style='rle')
                coco_bbox = kw_mask.to_box().to_coco()
                
                # Add annotation to manifest
                dset.add_annotation(
                    image_id=img_ids[frame_idx],
                    category_id=cat_id,
                    bbox=coco_bbox,
                    segmentation=coco_rle,
                    track_id=int(obj_id),
                    score=float(score)
                )
                
            frame_idx += 1
            pbar.update(1)
            
    cap.release()
            
    # 7. Save the dataset
    print(f"\nDumping completed KWCoco dataset to {output_json}...")
    dset.dump(output_json, newlines=True)
    print("Serialization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM3 via Hugging Face on a video and export tracking data to KWCoco format.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input MP4 video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the object to track.")
    parser.add_argument("--output", type=str, default="tracking_output.kwcoco.json", help="Path for the output KWCoco JSON file.")
    
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold for presence (0.0 to 1.0).")
    parser.add_argument("--mask-thresh", type=float, default=0.0, help="Mask binarization threshold (adjusts mask tightness/coarseness).")
    parser.add_argument("--max-mem-frames", type=int, default=30, help="Maximum number of past frames to keep in the memory bank (prevents OOM on long videos).")
    
    args = parser.parse_args()
    
    # Run the pipeline
    generate_kwcoco_manifest(
        video_path=args.video,
        text_prompt=args.prompt,
        output_json=args.output,
        conf_threshold=args.conf,
        mask_threshold=args.mask_thresh,
        max_mem_frames=args.max_mem_frames
    )
