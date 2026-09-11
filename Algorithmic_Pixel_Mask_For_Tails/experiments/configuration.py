"""Translate a fully specified experiment configuration to the unchanged CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import algorithmic_tail_mask as atm


def baseline_arguments(config: dict, input_root: Path, output_root: Path) -> tuple[list[str], argparse.Namespace]:
    """Reject incomplete/stale configs instead of silently adopting changed defaults."""
    original_argv = sys.argv
    try:
        sys.argv = ["algorithmic_tail_mask.py"]
        defaults = vars(atm.parse_args())
    finally:
        sys.argv = original_argv
    supplied = dict(config["arguments"])
    expected = set(defaults) - {"input_root", "output_root"}
    if set(supplied) != expected:
        raise ValueError(f"Configuration keys differ: missing={sorted(expected-set(supplied))}, extra={sorted(set(supplied)-expected)}")
    if supplied["overwrite"] or supplied["limit"] is not None:
        raise ValueError("R0 freezes all inputs and never overwrites an output root")
    argv = ["--input-root", str(input_root), "--output-root", str(output_root)]
    boolean_optional = {"path_v2", "path_fast_isolated", "save_intermediates"}
    for name, value in supplied.items():
        flag = name.replace("_", "-")
        if name in boolean_optional:
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
            argv.append(f"--{'' if value else 'no-'}{flag}")
        elif isinstance(defaults[name], bool):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
            if value:
                argv.append(f"--{flag}")
        elif value is not None:
            argv.extend([f"--{flag}", str(value)])
    try:
        sys.argv = ["algorithmic_tail_mask.py", *argv]
        args = atm.parse_args()
    finally:
        sys.argv = original_argv
    if args.path_mode == "legacy":
        args.path_v2 = False
    for name, value in supplied.items():
        if getattr(args, name) != value:
            raise ValueError(f"Effective CLI value for {name} differs from frozen config")
    return argv, args
