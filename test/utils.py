from datetime import datetime
from pathlib import Path

import pytest


def get_output_dir() -> Path:
    result = Path("output")
    result.mkdir(exist_ok=True, parents=True)
    return result


def make_test_dir(request: pytest.FixtureRequest) -> Path:
    result = (
        get_output_dir()
        / str(request.node.originalname)
        / datetime.now().isoformat(sep="_")
    )
    result.mkdir(parents=True)
    return result
