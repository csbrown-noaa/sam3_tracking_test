import os
import argparse
import torch
import cv2
import numpy as np
import kwimage
import kwcoco
from sam3.model_builder import build_sam3_video_predictor

def extract_frames(video_path: str, out_dir: str) -> tuple[list[str], int, int]:
    """
    Extracts frames from an MP4 video so KWCoco has tangible images to reference.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    out_dir : str
        Directory path where the extracted frames will be saved.

    Returns
    -------
    tuple[list[str], int, int]
        A tuple containing:
        - list of str: Paths to the extracted frame images.
        - int: Width of the video frames.
        - int: Height of the video frames.

    Raises
    ------
    ValueError
        If the video file cannot be opened.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    idx = 0
    print(f"Extracting frames from {video_path} to {out_dir}...")
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame_name = f"frame_{idx:05d}.jpg"
        frame_path = os.path.join(out_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        idx += 1
        
    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames, w, h

def generate_kwcoco_manifest(
    video_path: str, 
    text_prompt: str, 
    output_json: str, 
    frames_dir: str, 
    conf_threshold: float = 0.50, 
    mask_threshold: float = 0.0
) -> None:
    """
    Runs SAM3 on a video with a text prompt and serializes the tracking data to KWCoco.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    text_prompt : str
        The object category to search for and track (e.g., 'car').
    output_json : str
        Path to the output KWCoco JSON manifest file.
    frames_dir : str
        Directory to store the extracted frames.
    conf_threshold : float, optional
        Confidence threshold for object presence. Defaults to 0.50.
    mask_threshold : float, optional
        Threshold for mask binarization (controls mask tightness). Defaults to 0.0.

    Returns
    -------
    None
    """
    # 1. Prepare video frames
    frame_paths, w, h = extract_frames(video_path, frames_dir)
    
    # 2. Initialize KWCoco Dataset
    dset = kwcoco.CocoDataset()
    cat_id = dset.add_category(name=text_prompt)
    vid_id = dset.add_video(name=os.path.basename(video_path))
    
    # Pre-register all frames
    img_ids = {}
    for i, fpath in enumerate(frame_paths):
        img_id = dset.add_image(
            file_name=fpath, 
            video_id=vid_id, 
            frame_index=i, 
            width=w, 
            height=h
        )
        img_ids[i] = img_id

    # 3. Load SAM3 with exposed hyperparameters
    print(f"Loading SAM3 video predictor (conf={conf_threshold}, mask_thresh={mask_threshold})...")
    video_predictor = build_sam3_video_predictor(
        conf=conf_threshold,
        mask_threshold=mask_threshold
    )
    
    # Start Session
    print("Initializing SAM3 session...")
    response = video_predictor.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = response["session_id"]
    
    # Add prompt to the first frame
    print(f"Adding text prompt: '{text_prompt}' to frame 0...")
    video_predictor.handle_request(
        request=dict(
            type="add_prompt", 
            session_id=session_id, 
            frame_index=0, 
            text=text_prompt
        )
    )
    
    # 4. Propagate masks and serialize
    print("Propagating masks through the video...")
    for response in video_predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        frame_idx = response["frame_index"]
        current_img_id = img_ids[frame_idx]
        
        outputs = response["outputs"]
        masks = outputs.get("masks", [])
        obj_ids = outputs.get("object_ids", range(len(masks)))
        
        for mask_tensor, obj_id in zip(masks, obj_ids):
            # Convert PyTorch tensor to numpy
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
            
            # Wrap in KWImage Mask object
            kw_mask = kwimage.Mask(mask_np, 'c_mask')
            
            # Convert to compressed COCO RLE format
            coco_rle = kw_mask.to_coco(style='rle')
            
            # Naively generate bounding box from the segmentation mask
            coco_bbox = kw_mask.to_box().to_coco()
            
            # Add annotation to manifest
            dset.add_annotation(
                image_id=current_img_id,
                category_id=cat_id,
                bbox=coco_bbox,
                segmentation=coco_rle,
                track_id=int(obj_id) 
            )
            
    # 5. Save the dataset
    print(f"Dumping completed KWCoco dataset to {output_json}...")
    dset.dump(output_json, newlines=True)
    print("Serialization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM3 and export tracking data to KWCoco format.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input MP4 video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the object to track.")
    parser.add_argument("--output", type=str, default="tracking_output.kwcoco.json", help="Path for the output KWCoco JSON file.")
    parser.add_argument("--frames-dir", type=str, default="extracted_frames", help="Directory to store extracted frames.")
    
    # Exposing tuning hyperparameters
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold for presence (0.0 to 1.0).")
    parser.add_argument("--mask-thresh", type=float, default=0.0, help="Mask binarization threshold (adjusts mask tightness/coarseness).")
    
    args = parser.parse_args()
    
    generate_kwcoco_manifest(
        video_path=args.video,
        text_prompt=args.prompt,
        output_json=args.output,
        frames_dir=args.frames_dir,
        conf_threshold=args.conf,
        mask_threshold=args.mask_thresh
    )
