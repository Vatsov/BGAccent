"""Generate the runnable synthetic-augmentation workflow from the committed template.

The Workflow runtime has no filesystem/args access, so the worklist must be inlined
into the script text. The worklist carries copyrighted corpus excerpts, so the runnable
file is NOT committed: this builder injects the gitignored data/homographs/augment_worklist.json
into tools/homographs/augment_minority.workflow.js (template) and writes the gitignored
tools/homographs/augment_minority.local.js, which you then run via:

    Workflow({scriptPath: ".../tools/homographs/augment_minority.local.js"})
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = HERE / "augment_minority.workflow.js"
WORKLIST = ROOT / "data" / "homographs" / "augment_worklist.json"
OUT = HERE / "augment_minority.local.js"
TOKEN = "__WORKLIST__"


def main() -> int:
    if not WORKLIST.exists():
        print(f"missing {WORKLIST} — build it from the corpus pipeline first", file=sys.stderr)
        return 1
    template = TEMPLATE.read_text()
    if TOKEN not in template:
        print(f"template {TEMPLATE.name} has no {TOKEN} placeholder", file=sys.stderr)
        return 1
    worklist = json.loads(WORKLIST.read_text())
    literal = json.dumps(worklist, ensure_ascii=False)  # JS array literal
    OUT.write_text(template.replace(TOKEN, literal))
    print(f"wrote {OUT.relative_to(ROOT)} with {len(worklist)} variants embedded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
