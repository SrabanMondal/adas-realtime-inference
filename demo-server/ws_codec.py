import msgpack
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def decode_msgpack(data: bytes, model: Type[T]) -> T:
    raw = msgpack.unpackb(data, raw=False)
    return model.model_validate(raw)

def encode_msgpack(model: BaseModel) -> bytes:
    data = msgpack.packb(
        model.model_dump(),
        use_bin_type=True,
    )

    assert isinstance(data, (bytes, bytearray))
    return bytes(data)
