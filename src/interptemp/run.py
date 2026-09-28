"""CLI: `uv run interp-run experiments/foo/config.yaml [key.sub=value ...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from interptemp.config import load_config
from interptemp.experiment import Experiment
from interptemp.registry import import_object


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument(
        "overrides", nargs="*", help="dotted overrides, e.g. model.name=Qwen/Qwen3-0.6B"
    )
    args = ap.parse_args(argv)

    load_dotenv()
    # Make `experiments.*` importable when running from the repo root.
    sys.path.insert(0, str(Path.cwd()))
    cfg = load_config(args.config, args.overrides)
    cls = import_object(cfg.target)
    if not (isinstance(cls, type) and issubclass(cls, Experiment)):
        raise TypeError(f"{cfg.target} is not an Experiment subclass")
    cls(cfg).execute()


if __name__ == "__main__":
    main()
