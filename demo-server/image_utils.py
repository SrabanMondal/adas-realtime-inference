import cv2
import numpy as np


def decode_jpeg_bytes(data: bytes) -> np.ndarray:
    np_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid JPEG data")
    return img
