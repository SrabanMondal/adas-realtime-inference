"use client"

import { useEffect, useRef } from "react"
import { SensorPacket, ClientToServerMessage, GpsData } from "@/types"

export function useSensorStream(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  gps: GpsData,
  send: (msg: ClientToServerMessage) => void
) {
  const lastSent = useRef(0)

  // ❌ document usage removed from top-level
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    // ✅ SSR guard
    if (typeof window === "undefined") return

    if (!videoRef.current) return

    // ✅ Create canvas lazily (client-only)
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas")
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")

    if (!ctx) return

    let running = true

    const loop = (time: number) => {
      if (!running) return
      requestAnimationFrame(loop)

      // ~8 FPS throttle
      if (time - lastSent.current < 120) return
      lastSent.current = time

      const w = video.videoWidth
      const h = video.videoHeight
      if (!w || !h) return

      // Resize once video metadata is ready
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      
      canvas.toBlob(
        async blob => {
          if (!blob) return

          const buffer = await blob.arrayBuffer()
          const imageBytes = new Uint8Array(buffer)

          const packet: SensorPacket = {
            timestamp: performance.now(),
            image: imageBytes,
            gps,
          }

          send({
            type: "sensor",
            payload: packet,
          })
        },
        "image/jpeg",
        0.6
      )
    }

    requestAnimationFrame(loop)

    return () => {
      running = false
    }
  }, [videoRef, gps, send])
}
