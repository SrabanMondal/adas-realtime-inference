// app/types/sensor.ts

export interface GpsData {
  lat: number
  lon: number
  accuracy?: number
}

export interface SensorPacket {
  timestamp: number           // performance.now()
  image: Uint8Array               
  gps: GpsData
}
