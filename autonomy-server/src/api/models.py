from typing import List, Literal, Optional, Type, TypeVar, Tuple
from pydantic import BaseModel
import msgpack
import numpy as np
import cv2

# ---------- Models ----------
Point = List[int]  # [x, y]

class GpsData(BaseModel):
    lat: float
    lon: float
    accuracy: Optional[float] = None

class SensorPacket(BaseModel):
    timestamp: float
    image: bytes          # RAW JPEG bytes
    gps: GpsData

class SensorMessage(BaseModel):
    type: Literal["sensor"]
    payload: SensorPacket

ClientToServerMessage = SensorMessage

class Control(BaseModel):
    steeringAngle: float
    confidence: float

class AutonomyState(BaseModel):
    laneLines: List[List[Point]]
    trajectory: List[Tuple[int,int]]
    control: Control
    status: Literal["NORMAL", "WARNING", "ERROR","FINISHED"]

class AutonomyMessage(BaseModel):
    type: Literal["autonomy"]
    payload: AutonomyState

ServerToClientMessage = AutonomyMessage

# ---------- Codec Utilities ----------
T = TypeVar("T", bound=BaseModel)

def decode_msgpack(data: bytes, model: Type[T]) -> T:
    raw = msgpack.unpackb(data, raw=False)
    return model.model_validate(raw)

def encode_msgpack(model: BaseModel) -> bytes:
    data = msgpack.packb(model.model_dump(), use_bin_type=True)
    assert isinstance(data, (bytes, bytearray))
    return bytes(data)

def decode_jpeg_bytes(data: bytes) -> np.ndarray:
    np_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid JPEG data")
    return img