import cv2
import numpy as np

def letterbox(img, size=640) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (nw, nh))

    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left

    boxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return boxed

def unletterbox(mask_640, orig_shape, size=640) -> np.ndarray:
    """
    mask_640: model output (H=640, W=640) single-channel
    orig_shape: (orig_h, orig_w)
    """
    orig_h, orig_w = orig_shape

    # Recompute the same geometry
    scale = size / max(orig_h, orig_w)
    nh, nw = int(orig_h * scale), int(orig_w * scale)

    top = (size - nh) // 2
    left = (size - nw) // 2

    # 1. Crop padding away
    unpadded = mask_640[top:top+nh, left:left+nw]

    # 2. Resize back to original resolution
    restored = cv2.resize(unpadded, (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST)

    return restored

def scale_boxes(
    boxes: np.ndarray, 
    orig_shape: tuple[int, int], 
    size: int = 640
) -> np.ndarray:
    """
    Rescale YOLO boxes (x1, y1, x2, y2, conf, cls) from 640x640 letterbox 
    back to original image resolution.
    
    Args:
        boxes: NumPy array of shape (N, 6)
        orig_shape: (height, width) of the original high-res frame
        size: The letterbox size (usually 640)
    """
    if boxes.size == 0:
        return boxes

    oh, ow = orig_shape
    
    # 1. Replicate the exact scale used in your letterbox()
    scale = size / max(oh, ow)
    nh, nw = int(oh * scale), int(ow * scale)

    # 2. Replicate the exact padding used in your letterbox()
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2

    # 3. Create copy to avoid side-effects
    scaled_boxes = boxes.copy()
    
    # 4. Apply transformation: (coord - pad) / scale
    # Scale x coordinates (indices 0 and 2)
    scaled_boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    # Scale y coordinates (indices 1 and 3)
    scaled_boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    # 5. Optional: Clip boxes to image boundaries to prevent coordinates like -1 or 1921
    scaled_boxes[:, [0, 2]] = np.clip(scaled_boxes[:, [0, 2]], 0, ow)
    scaled_boxes[:, [1, 3]] = np.clip(scaled_boxes[:, [1, 3]], 0, oh)

    return scaled_boxes