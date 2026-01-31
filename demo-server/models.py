from typing import List, Literal, Optional
from pydantic import BaseModel


# ---------- Basic types ----------

Point = List[int]  # [x, y]


class GpsData(BaseModel):
    lat: float
    lon: float
    accuracy: Optional[float] = None


# ---------- Client → Server ----------

class SensorPacket(BaseModel):
    timestamp: float
    image: bytes          # RAW JPEG bytes
    gps: GpsData


class SensorMessage(BaseModel):
    type: Literal["sensor"]
    payload: SensorPacket


ClientToServerMessage = SensorMessage


# ---------- Server → Client ----------

class Control(BaseModel):
    steeringAngle: float
    confidence: float


class AutonomyState(BaseModel):
    laneLines: List[List[Point]]
    trajectory: List[Point]
    control: Control
    status: Literal["NORMAL", "WARNING", "ERROR"]


class AutonomyMessage(BaseModel):
    type: Literal["autonomy"]
    payload: AutonomyState


ServerToClientMessage = AutonomyMessage
