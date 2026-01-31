// app/hooks/useGPS.ts
import { useEffect, useState } from "react"
import { GpsData } from "@/types"

export function useGPS() {
  const [gps, setGps] = useState<GpsData>({
    lat: 0,
    lon: 0,
  })

  useEffect(() => {
    const id = navigator.geolocation.watchPosition(
      pos => {
        setGps({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
        })
      },
      err => console.warn("GPS error", err),
      { enableHighAccuracy: true, maximumAge: 0, timeout: 10000, }
    )

    return () => navigator.geolocation.clearWatch(id)
  }, [])

  return gps
}
