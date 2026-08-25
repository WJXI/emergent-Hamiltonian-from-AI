"""Train only the configuration-independent structured Network 2 arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PHENOL_ROOT = Path(__file__).resolve().parent
CONTROLLED_SRC = PHENOL_ROOT / "phenol_dsm_controlled" / "src"
if str(CONTROLLED_SRC) not in sys.path:
    sys.path.insert(0, str(CONTROLLED_SRC))

import independent_sigma_dsm_experiment as experiment  # noqa: E402
import unified_dsm_experiment as base  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PHENOL_ROOT / "phenol_dsm_controlled" / "configs" / "clean_limit_protocol_v2.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PHENOL_ROOT / "reproduced" / "network_2")
    parser.add_argument("--seed-limit", type=int, default=None)
    args = parser.parse_args()

    config = experiment.load_protocol(args.config.resolve())
    train, _train_family, validation, _validation_family, data_audit = base.load_coordinate_splits(config)
    output = args.output_dir.resolve()
    result = experiment.run_arm("true_topology", config, train, validation, output, args.seed_limit)
    report = {
        "schema_version": 1,
        "network": "Network 2",
        "definition": "240 retained structured basis functions with configuration-independent coefficients",
        "data": data_audit,
        "result": result,
    }
    base.atomic_json(output / "network_2_summary.json", report)
    print(json.dumps({"output": str(output / "network_2_summary.json"), "clean_readout": result["clean_readout"]}, indent=2))


if __name__ == "__main__":
    main()
