#!/usr/bin/env python3
import sys, os
import yaml

IN_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]
RUN_DIR = sys.argv[3]

with open(IN_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Try common keys; if none exist, add run_dir.
candidates = ["run_dir", "out_dir", "results_dir", "exp_dir", "log_dir"]
set_ok = False
for k in candidates:
    if isinstance(cfg, dict) and k in cfg:
        cfg[k] = RUN_DIR
        set_ok = True
        break

if not set_ok:
    if not isinstance(cfg, dict):
        raise RuntimeError("Config root is not a dict, cannot set run_dir.")
    cfg["run_dir"] = RUN_DIR

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[OK] wrote {OUT_PATH} with run_dir={RUN_DIR}")