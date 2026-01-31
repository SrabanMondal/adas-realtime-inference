// app/types/protocol.ts

import { SensorPacket } from "./sensor"
import { AutonomyState } from "./autonomy"

export type ClientToServerMessage = {
  type: "sensor"
  payload: SensorPacket
}

export type ServerToClientMessage = {
  type: "autonomy"
  payload: AutonomyState
}
