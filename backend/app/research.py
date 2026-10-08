"""Load the exact HiTSKT source without benchmark-only package side effects.

ktbench.models.__init__ imports every benchmark family, including RKT's Linux
resource module. Serving HiTSKT on Windows does not need those imports. This
loader executes the original Performer and HiTSKT files; it copies/modifies no
research source and preserves checkpoint keys and model arithmetic.
"""
import importlib.util
import sys
from functools import lru_cache

from app.config import ROOT


def _source_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model source at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@lru_cache(maxsize=1)
def hitskt_types():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _source_module("ktbench.models.performer", ROOT / "ktbench" / "models" / "performer.py")
    module = _source_module("_lumen_original_hitskt", ROOT / "ktbench" / "models" / "hitskt.py")
    return module.HiTSKT, module.HiTSKTConfig
