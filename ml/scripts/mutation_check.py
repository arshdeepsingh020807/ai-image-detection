"""Manual mutation check: break one line at a time and confirm the test suite notices.

Usage: python ml/scripts/mutation_check.py <source.py> <tests path> <mutants.txt>
mutants.txt holds one `old ==> new` replacement per line. Bytecode caching is disabled because
same-size mutations within one second would otherwise reuse a stale .pyc and give wrong verdicts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    target, tests, mutants_file = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    mutants = [line.split(" ==> ") for line in mutants_file.read_text().splitlines() if " ==> " in line]
    original = target.read_text()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    survived = 0
    try:
        for old, new in mutants:
            if old not in original:
                print("MISSING pattern:", old)
                continue
            target.write_text(original.replace(old, new, 1))
            command = [sys.executable, "-m", "pytest", tests, "-x", "-q", "-p", "no:cacheprovider"]
            killed = subprocess.run(command, capture_output=True, text=True, env=env).returncode != 0
            survived += not killed
            print("KILLED  " if killed else "SURVIVED", old.strip()[:60], "->", new.strip()[:40])
    finally:
        target.write_text(original)
    print(f"\n{len(mutants) - survived}/{len(mutants)} killed")


if __name__ == "__main__":
    main()
