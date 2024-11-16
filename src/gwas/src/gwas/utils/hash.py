import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from hashlib import sha1
from typing import Any, Mapping

from upath import UPath


class JSONEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)

        if isinstance(value, Mapping):
            return dict(value)

        if isinstance(value, UPath):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        return super().default(value)


def hex_digest(value: Any) -> str:
    hash = sha1()
    str_representation = json.dumps(value, cls=JSONEncoder, sort_keys=True)
    hash.update(str_representation.encode())
    digest = hash.hexdigest()
    # logger.debug(f'Hashed "{str_representation}" to "{digest}"')
    return digest
