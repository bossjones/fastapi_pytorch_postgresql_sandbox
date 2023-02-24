"""fastapi_pytorch_postgresql_sandbox models."""
from pathlib import Path
import pkgutil


def load_all_models() -> None:
    """Load all models from this folder."""
    package_dir = Path(__file__).resolve().parent
    modules = pkgutil.walk_packages(
        path=[str(package_dir)],
        prefix="fastapi_pytorch_postgresql_sandbox.db.models.",
    )
    for module in modules:
        __import__(module.name)  # noqa: WPS421
