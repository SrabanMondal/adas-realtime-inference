"use client"

import { useEffect, useRef } from "react"
import { AutonomyState} from "@/types"
import { drawArrow, drawLaneLines, drawStatus, drawTrajectory } from "@/utils/draw"

interface OverlayCanvasProps {
  data: AutonomyState | null
  width: number
  height: number
}

export function OverlayCanvas({
  data,
  width,
  height,
}: OverlayCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")!
    ctx.clearRect(0, 0, width, height)

    // Black background
    ctx.fillStyle = "black"
    ctx.fillRect(0, 0, width, height)

    if (!data) return

    drawLaneLines(ctx, data.laneLines)
    drawTrajectory(ctx, data.trajectory)
    drawArrow(ctx, data.control.steeringAngle, width, height)
    //drawStatus(ctx, data.status, data.control.confidence)

  }, [data, width, height])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        width: "100%",
        height: "100%",
        background: "black",
      }}
    />
  )
}
