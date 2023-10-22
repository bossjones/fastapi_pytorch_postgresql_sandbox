"""fastapi_pytorch_postgresql_sandbox.factories.cmd_factory."""
# pylint: disable=no-value-for-parameter
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from pydantic import validate_arguments

from fastapi_pytorch_postgresql_sandbox.factories import SerializerFactory

# fastapi_pytorch_postgresql_sandbox.web.factories


# SOURCE: https://stackoverflow.com/questions/54863458/force-type-conversion-in-python-dataclass-init-method
@validate_arguments
@dataclass
class CmdSerializer(SerializerFactory):
    name: str
    cmd: Optional[str]
    uri: Optional[str]

    @staticmethod
    def create(d: Dict) -> CmdSerializer:
        return CmdSerializer(name=d["name"])
