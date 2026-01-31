"use client"

import { useEffect, useRef, useState } from "react"
import { OverlayCanvas } from "@/components/OverlayCanvas"
import { useCamera } from "@/hooks/useCamera"
import { useGPS } from "@/hooks/useGPS"
import { useWebSocket } from "@/hooks/useWebSocket"
import { useSensorStream } from "@/hooks/useSensorStream"
import { AutonomyState } from "@/types"

export default function Page() {
  const videoRef = useRef<HTMLVideoElement>(null)

  useCamera(videoRef)
  const gps = useGPS()

  // Network
  const { send, latestMessage } =
    useWebSocket("wss://192.168.1.7:8000/ws")

  // Stream sensor data
  useSensorStream(videoRef, gps, send)

  // Extract autonomy payload
  const autonomy: AutonomyState | null =
    latestMessage?.type === "autonomy"
      ? latestMessage.payload
      : null
      
  const [width, setwidth] = useState<number|null>(null)
  const [height, setheight] = useState<number|null>(null)

 useEffect(() => {
  const video = videoRef.current
  if (!video) return

  const updateSize = () => {
    if (video.videoWidth && video.videoHeight) {
      setwidth(video.videoWidth)
      setheight(video.videoHeight)
    }
  }
  video.addEventListener("loadedmetadata", updateSize)
  return () => {
    video.removeEventListener("loadedmetadata", updateSize)
  }
}, [])

  
  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        background: "black",
        overflow: "hidden",
      }}
    >
      {/* Hidden camera feed */}
      <video
        ref={videoRef}
        playsInline
        muted
        style={{ display: "none" }}
      />

      {/* Autonomy visualization */}
      {width && height &&
      <OverlayCanvas data={autonomy} width={width} height={height} />
      }
    </div>
  )
}
