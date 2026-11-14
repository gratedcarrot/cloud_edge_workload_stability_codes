"""Repository-relative input and output locations shared by reproduction commands."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REFERENCE_RESULTS = ROOT / "results"
RESULTS = Path(os.environ.get("CLOUD_EDGE_OUTPUT", str(ROOT / "outputs"))).resolve()
FIGURES = RESULTS / "figures"
for path in (ROOT / "engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
def result_file(relative: str) -> Path:
    """Prefer a regenerated result; otherwise use the bundled reference result."""
    generated = RESULTS / relative
    reference = REFERENCE_RESULTS / relative
    if generated.is_file():
        return generated
    if reference.is_file():
        return reference
    raise FileNotFoundError(f"Required result is missing: {relative}. Run its experiment first.")
