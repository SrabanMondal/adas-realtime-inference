import { Point2D } from "@/types"

export function drawLaneLines(
  ctx: CanvasRenderingContext2D,
  lanes: Point2D[][]
) {
  ctx.strokeStyle = "#ff3333" // red
  ctx.lineWidth = 3

  lanes.forEach(line => {
    ctx.beginPath()
    line.forEach(([x, y], i) => {
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.stroke()
  })
}

export function drawTrajectory(
  ctx: CanvasRenderingContext2D,
  traj: Point2D[]
) {
  if (!traj.length) return

  ctx.strokeStyle = "#ffd000" // yellow
  ctx.lineWidth = 2
  ctx.setLineDash([6, 6])

  ctx.beginPath()
  traj.forEach(([x, y], i) => {
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  })
  ctx.stroke()

  ctx.setLineDash([])
}

export function drawArrow(
  ctx: CanvasRenderingContext2D,
  angleDeg: number,
  w: number,
  h: number
) {
  const cx = w / 2
  const cy = h * 0.85   // near bottom
  const len = h * 0.15 // scale with screen

  const rad = (angleDeg * Math.PI) / 180
  const tx = cx + len * Math.sin(rad)
  const ty = cy - len * Math.cos(rad)

  ctx.strokeStyle = "#00e5ff"
  ctx.lineWidth = 5

  ctx.beginPath()
  ctx.moveTo(cx, cy)
  ctx.lineTo(tx, ty)
  ctx.stroke()

  ctx.beginPath()
  ctx.arc(tx, ty, 8, 0, Math.PI * 2)
  ctx.fillStyle = "#00e5ff"
  ctx.fill()
}


export function drawStatus(
  ctx: CanvasRenderingContext2D,
  status: string,
  confidence: number
) {
  ctx.fillStyle = "rgba(0,0,0,0.6)"
  ctx.fillRect(0, 0, 180, 60)

  ctx.fillStyle =
    status === "STOP" ? "red" :
    status === "SLOW" ? "yellow" :
    "lime"

  ctx.font = "16px monospace"
  ctx.fillText(`STATUS: ${status}`, 10, 25)

  ctx.fillStyle = "white"
  ctx.fillText(
    `CONF: ${(confidence * 100).toFixed(0)}%`,
    10,
    45
  )
}

