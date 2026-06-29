#!/usr/bin/env python3
"""dh2mujoco – main entry point.

Usage
-----
    python -m dh2mujoco.main              # mode 4 (default), all-zeros FK
    python -m dh2mujoco.main --mode 1     # Modified DH, theta from table
    python -m dh2mujoco.main --mode 3     # Standard DH
    python -m dh2mujoco.main --random     # also run random joint config
    python -m dh2mujoco.main --no-verify  # skip FK verification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the repo root
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dh2mujoco.config import Config
from dh2mujoco.dh_parser import compute_dh_fk
from dh2mujoco.mjcf_writer import MJCFWriter
from dh2mujoco.robots.sixdof import make_config, make_table
from dh2mujoco.utils import print_fk_chain, print_transform
from dh2mujoco.verification import verify


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DH → MuJoCo MJCF generator")
    p.add_argument("--mode", type=int, default=4, choices=[1, 2, 3, 4],
                   help="DH convention mode (default: 4 = Modified DH, offsets explicit)")
    p.add_argument("--random", action="store_true",
                   help="Also evaluate FK at a random joint configuration")
    p.add_argument("--no-verify", dest="verify", action="store_false",
                   help="Skip the FK verification step")
    p.add_argument("--no-xml", dest="export_xml", action="store_false",
                   help="Skip XML file generation")
    p.add_argument("--show-matrix", action="store_true",
                   help="Print full 4×4 transforms at each frame")
    p.add_argument("--quiet", action="store_true",
                   help="Only print errors / summary, suppress per-frame output")
    p.set_defaults(verify=True, export_xml=True)
    return p.parse_args()


def run(config: Config, random_q: bool, do_verify: bool, show_matrix: bool) -> None:
    np.set_printoptions(precision=6, suppress=True)

    table = make_table(config.column_order)

    # ------------------------------------------------------------------ FK
    configs_to_test = [
        ("home (q=0)", np.zeros(table.n_joints)),
    ]
    if random_q:
        rng = np.random.default_rng(42)
        q_rand = rng.uniform(-np.pi, np.pi, table.n_joints)
        configs_to_test.append(("random", q_rand))

    for label, q in configs_to_test:
        chain = compute_dh_fk(table, q, config, return_chain=True)

        print(f"\n{'#'*62}")
        print(f"  Forward Kinematics  [{label}]  Mode {config.MODE}")
        print(f"{'#'*62}")
        print(f"  q = {np.round(q, 4).tolist()}")

        if config.SHOW_FRAME:
            print_fk_chain(chain, table.n_joints, show_matrix=show_matrix)
        else:
            # Always print at least the EE frame
            print_transform(
                f"End-Effector (T_{len(chain)-1})",
                chain[-1],
                show_matrix=show_matrix,
            )

    # ------------------------------------------------------------------ XML
    writer = MJCFWriter(table, config)
    xml_str, body_chain = writer.build()

    if config.EXPORT_XML:
        out_path = Path(config.OUTPUT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(xml_str, encoding="utf-8")
        print(f"\n  XML written → {out_path.resolve()}")

    # ------------------------------------------------------------------ Verify
    if do_verify:
        for label, q in configs_to_test:
            verify(table, config, body_chain, q=q, label=label)


def main() -> None:
    args = _parse_args()

    config = make_config(mode=args.mode)
    config.SHOW_FRAME = not args.quiet
    config.SHOW_MATRIX = args.show_matrix
    config.EXPORT_XML = args.export_xml

    run(config, random_q=args.random, do_verify=args.verify, show_matrix=args.show_matrix)


if __name__ == "__main__":
    main()
