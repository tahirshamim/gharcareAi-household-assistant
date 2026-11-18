"""Imported first by every example script: makes `import gharcare` load THIS folder's src/, not an older copy
that an earlier `pip install -e .` in another unzipped folder still points to (that happened in Step 6.4:
new example scripts silently ran old library code)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gharcare  # noqa: E402

print(f"gharcare code: {Path(gharcare.__file__).resolve().parent}", file=sys.stderr)
