// app/hooks/useCamera.ts
import { useEffect } from "react"

export function useCamera(
  videoRef: React.RefObject<HTMLVideoElement|null>
) {
  useEffect(() => {
    let stream: MediaStream | null = null

    async function init() {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      })

      if (videoRef && videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
    }

    init()

    return () => {
      const track = stream?.getVideoTracks()[0]
      console.log(track?.getSettings())
      stream?.getTracks().forEach(t => t.stop())
    }
  }, [videoRef])
}
