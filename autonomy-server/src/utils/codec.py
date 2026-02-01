"""
Msgpack Codec Utilities for WebSocket Communication
"""
from typing import Type, TypeVar
from pydantic import BaseModel
import msgpack
import numpy as np
import cv2

T = TypeVar("T", bound=BaseModel)


def decode_msgpack(data: bytes, model: Type[T]) -> T:
    """Decode msgpack bytes into a Pydantic model."""
    raw = msgpack.unpackb(data, raw=False)
    return model.model_validate(raw)


def encode_msgpack(model: BaseModel) -> bytes:
    """Encode a Pydantic model into msgpack bytes."""
    data = msgpack.packb(model.model_dump(), use_bin_type=True)
    assert isinstance(data, (bytes, bytearray))
    return bytes(data)


def decode_jpeg_bytes(data: bytes) -> np.ndarray:
    """Decode JPEG bytes into a BGR numpy array."""
    np_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid JPEG data")
    return img
