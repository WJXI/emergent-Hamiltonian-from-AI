"""Generate the eight independent explicit-water Phenol holdout families."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
for source in (EXPERIMENT_ROOT / "src",):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
import fresh_physical_families as fresh  # noqa: E402


DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "fresh_physical_phenol_dsm_final_v1.json"
fresh.SOURCE_PATH = Path(__file__).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "pack", "minimize", "sample", "qualify", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = fresh.load_config(args.config)
    if args.stage == "freeze":
        result: Any = fresh.freeze(config)
    elif args.stage == "pack":
        result = fresh.build_packings(config)
    elif args.stage == "minimize":
        result = fresh.minimize_packings(config)
    elif args.stage == "sample":
        result = fresh.run_sampling(config)
    elif args.stage == "qualify":
        result = fresh.qualify(config)
    else:
        protocol = fresh.freeze(config)
        packings = fresh.build_packings(config)
        minimized = fresh.minimize_packings(config)
        sampling = fresh.run_sampling(config)
        qualification = fresh.qualify(config) if sampling.name == "sampling_manifest.json" else "sampling incomplete"
        result = {
            "protocol": str(protocol),
            "packings": str(packings),
            "minimized": str(minimized),
            "sampling": str(sampling),
            "qualification": str(qualification),
        }
    print(json.dumps(result if isinstance(result, dict) else {"result": str(result)}, indent=2))


if __name__ == "__main__":
    main()
