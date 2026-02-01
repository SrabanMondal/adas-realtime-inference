"""
Pydantic Schemas for WebSocket API Communication

Client -> Server: SensorMessage (image + GPS)
Server -> Client: AutonomyMessage (steering + trajectory)
"""
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel

# Re-export codec utilities for backward compatibility
from src.utils.codec import decode_msgpack, encode_msgpack, decode_jpeg_bytes

# =============================================================================
# Type Aliases
# =============================================================================
Point = List[int]  # [x, y]


# =============================================================================
# Client -> Server Messages
# =============================================================================
class GpsData(BaseModel):
    """GPS coordinates from device."""
    lat: float
    lon: float
    accuracy: Optional[float] = None


class SensorPacket(BaseModel):
    """Sensor data packet containing image and GPS."""
    timestamp: float
    image: bytes  # RAW JPEG bytes
    gps: GpsData


class SensorMessage(BaseModel):
    """Message wrapper for sensor data."""
    type: Literal["sensor"]
    payload: SensorPacket


ClientToServerMessage = SensorMessage


# =============================================================================
# Server -> Client Messages
# =============================================================================
class Control(BaseModel):
    """Steering control output."""
    steeringAngle: float
    confidence: float


class AutonomyState(BaseModel):
    """Complete autonomy state sent to client."""
    laneLines: List[List[Point]]
    trajectory: List[Tuple[int, int]]
    control: Control
    status: Literal["NORMAL", "WARNING", "ERROR", "FINISHED"]


class AutonomyMessage(BaseModel):
    """Message wrapper for autonomy state."""
    type: Literal["autonomy"]
    payload: AutonomyState


ServerToClientMessage = AutonomyMessage