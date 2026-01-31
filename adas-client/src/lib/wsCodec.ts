import { encode, decode } from "@msgpack/msgpack"
import {
  ClientToServerMessage,
  ServerToClientMessage,
} from "@/types"

export function encodeClientMessage(
  msg: ClientToServerMessage
): Uint8Array {
  return encode(msg)
}

export function decodeServerMessage(
  data: ArrayBuffer
): ServerToClientMessage {
  return decode(new Uint8Array(data)) as ServerToClientMessage
}
