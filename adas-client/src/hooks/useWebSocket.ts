import { useEffect, useRef, useState } from "react"
import {
  encodeClientMessage,
  decodeServerMessage,
} from "@/lib/wsCodec"
import {
  ClientToServerMessage,
  ServerToClientMessage,
} from "@/types"

export function useWebSocket(url: string) {
  const wsRef = useRef<WebSocket | null>(null)
  const [latestMessage, setLatestMessage] =
    useState<ServerToClientMessage | null>(null)

  useEffect(() => {
    const ws = new WebSocket(url)
    ws.binaryType = "arraybuffer"
    wsRef.current = ws

    ws.onmessage = ev => {
      if (ev.data instanceof ArrayBuffer) {
        setLatestMessage(decodeServerMessage(ev.data))
      } else {
        // optional backward compatibility
        setLatestMessage(JSON.parse(ev.data))
      }
    }

    return () => ws.close()
  }, [url])

  function send(msg: ClientToServerMessage) {
    const encoded = encodeClientMessage(msg)
    wsRef.current?.send(encoded)
  }

  return { send, latestMessage }
}
