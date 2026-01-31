// app/types/autonomy.ts

export type Point2D = [number, number]

export interface ControlCommand {
  steeringAngle: number       // degrees (left -, right +)
  yawRate?: number            // rad/s (optional)
  confidence: number          // 0–1
}

export interface AutonomyState {
  laneLines: Point2D[][]      // multiple polylines
  trajectory: Point2D[]       // chosen local path
  control: ControlCommand
  status: "NORMAL" | "SLOW" | "STOP"
}
