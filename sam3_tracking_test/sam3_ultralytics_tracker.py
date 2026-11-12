import os
import argparse
import numpy as np
import kwimage
import kwcoco
from ultralytics.models.sam import SAM3VideoSemanticPredictor

def ultralytics_tracker_to_kwcoco(results, source: str, category_name: str, output_path: str) -> None:
    """
    Consumes a lazy stream of Ultralytics tracking results and serializes them to KWCoco.
    Supports both segmentation-based trackers (masks) and bounding-box-only trackers.
    """
    dset = kwcoco.CocoDataset()
    cat_id = dset.add_category(name=category_name)
    vid_id = dset.add_video(name=os.path.basename(source))
    
    # Process the lazy generator frame-by-frame
    for frame_idx, result in enumerate(results):
        # Dynamically fetch resolution from the first result (result.orig_shape is (h, w))
        h, w = result.orig_shape
        
        # Register the frame on the fly
        img_id = dset.add_image(
            name=f"frame_{frame_idx:05d}",
            video_id=vid_id, 
            frame_index=frame_idx, 
            width=w, 
            height=h
        )
        
        # Check if the model found any objects in this frame
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_np = result.boxes.xyxy.cpu().numpy()
            
            # Fetch tracking IDs (fallback to sequential if missing)
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                track_ids = list(range(len(boxes_np)))
                
            # Check if masks exist (SAM vs YOLO bounding-box trackers)
            has_masks = result.masks is not None
            if has_masks:
                masks_np = result.masks.data.cpu().numpy().astype(np.uint8)
            
            for i in range(len(boxes_np)):
                bbox = boxes_np[i]
                track_id = track_ids[i]
                
                # Format is [x_min, y_min, width, height] for COCO. Ensure they are native floats.
                x1, y1, x2, y2 = map(float, bbox)
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                
                coco_rle = None
                if has_masks:
                    mask_arr = masks_np[i]
                    kw_mask = kwimage.Mask(mask_arr, 'c_mask')
                    coco_rle = kw_mask.to_coco(style='rle')
                
                # Add annotation to manifest
                dset.add_annotation(
                    image_id=img_id,
                    category_id=cat_id,
                    bbox=coco_bbox,
                    segmentation=coco_rle,
                    track_id=track_id
                )

    print(f"Dumping completed KWCoco dataset to {output_path}...")
    dset.dump(output_path, newlines=True)
    print("Serialization complete.")

def generate_kwcoco_manifest(
    source: str, 
    text: str, 
    output: str, 
    conf: float = 0.50, 
    max_obs_in_memory: int = 30,
    imgsz: int = 1024,
    vid_stride: int = 1,
    half: bool = True,
    retina_masks: bool = False,
    iou: float = 0.70
) -> None:
    """
    Runs Ultralytics SAM on a video stream and serializes tracking data to KWCoco.
    """
    print("Loading Ultralytics SAM3VideoSemanticPredictor engine...")
    
    overrides = {
        "conf": conf,
        "iou": iou,
        "task": "segment",
        "mode": "predict",
        "model": "/home/user/sam3/checkpoints/sam3.pt",
        "imgsz": imgsz,
        "vid_stride": vid_stride,
        "half": half,
        "retina_masks": retina_masks,
        "max_obs_in_memory": max_obs_in_memory 
    }
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    
    print(f"Propagating masks natively through the video stream (stream=True)...")
    
    # 1. Stream and Track (This loop replaces the Meta predictor)
    # The stream=True argument guarantees frames are processed and discarded one-by-one
    results = predictor(
        source=source,
        text=[text],
        stream=True
    )
    
    # 2. Pass the lazy generator to the serializer
    ultralytics_tracker_to_kwcoco(
        results=results,
        source=source,
        category_name=text,
        output_path=output
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ultralytics SAM natively on a video and export tracking data to KWCoco format.")
    parser.add_argument("--source", type=str, required=True, help="Path to the input MP4 video (Ultralytics 'source').")
    parser.add_argument("--text", type=str, required=True, help="Text prompt for the object to track (Ultralytics 'text').")
    parser.add_argument("--output", type=str, default="tracking_output.kwcoco.json", help="Path for the output KWCoco JSON file.")
    
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold for presence (0.0 to 1.0).")
    parser.add_argument("--max-obs-in-memory", type=int, default=30, help="Maximum number of past frames to keep in the memory bank (prevents OOM).")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution. Higher means better accuracy but more VRAM usage.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Frame stride. 2 means skip every other frame.")
    parser.add_argument("--no-half", action="store_false", dest="half", help="Disable FP16 half-precision and force FP32.")
    parser.add_argument("--retina-masks", action="store_true", help="Evaluate final mask boundaries at native image resolution.")
    parser.add_argument("--iou", type=float, default=0.70, help="Intersection Over Union (IoU) threshold for object tracking/overlap.")
    
    args = parser.parse_args()
    
    generate_kwcoco_manifest(
        source=args.source,
        text=args.text,
        output=args.output,
        conf=args.conf,
        max_obs_in_memory=args.max_obs_in_memory,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        half=args.half,
        retina_masks=args.retina_masks,
        iou=args.iou
    )
