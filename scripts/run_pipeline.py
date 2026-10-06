"""Run the full pipeline with the settings of the final submission, after checking the machine.

    python scripts/run_pipeline.py --data-dir data
    python scripts/run_pipeline.py --data-dir data --dev 100000        # quick check, test skipped
    python scripts/run_pipeline.py --data-dir data -- --seeds 42,7,2026  # extra flags for run.py
All model settings come from entity_resolution/config.py.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        gpus = [[x.strip() for x in line.split(",")] for line in out.strip().splitlines()]
        best = max(gpus, key=lambda g: float(g[2]))
        return best[0], (f"{best[1]} (nvidia-smi #{best[0]}), {float(best[2]) / 1024:.0f} GB free "
                         f"of {float(best[3]) / 1024:.0f} GB")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return "0", "nvidia-smi not available"


def check_environment(env):
    check = subprocess.run([sys.executable, "-c",
                            "import torch, xgboost, sentence_transformers; "
                            "print(torch.__version__, torch.cuda.is_available())"],
                           env=env, capture_output=True, text=True)
    print("torch :", check.stdout.strip() or check.stderr.strip()[-400:])
    if "True" not in check.stdout:
        sys.exit("PyTorch cannot use the GPU (or a library is missing) - see requirements.txt")
    # XGBoost 3.x failed at scale on large GPUs (CUDA_ERROR_INVALID_VALUE in cuMemCreate) while
    # tiny tests passed, so test at a realistic size; 2.1.x is the tested version
    xgb_test = ("import numpy as np, xgboost as xgb; X = np.random.rand(2_000_000, 50).astype(np.float32); "
                "d = xgb.QuantileDMatrix(X, (X[:, 0] > .5).astype(np.float32)); "
                "xgb.train({'device': 'cuda', 'tree_method': 'hist', 'max_depth': 8}, d, 5); "
                "print('xgboost', xgb.__version__, 'GPU training OK')")
    chk = subprocess.run([sys.executable, "-c", xgb_test], env=env, capture_output=True, text=True)
    if chk.returncode != 0:
        print((chk.stdout + chk.stderr).strip()[-600:])
        sys.exit("XGBoost cannot train on this GPU - run: pip install xgboost==2.1.4")
    print("xgb   :", chk.stdout.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"), help="folder with train/ and test/")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--work-dir", default=os.path.join(ROOT, "work"), help="caches and checkpoints")
    ap.add_argument("--log", default=os.path.join(ROOT, "logs", "run.txt"))
    ap.add_argument("--no-cross-encoder", action="store_true", help="skip the cross-encoder")
    ap.add_argument("--dev", type=int, default=0, help="quick check on N train S1 entities, test skipped")
    ap.add_argument("extra", nargs="*", help="further flags for entity_resolution.run (after --)")
    args = ap.parse_args()

    env = os.environ.copy()
    if env.pop("PYTHONPATH", None):
        print("note  : ignoring PYTHONPATH (an IDE can point it at another environment)")
    env["PYTHONNOUSERSITE"] = "1"
    # string hashing is randomised per process by default; it changes set order and therefore how
    # ties in the top-k blocking are broken, so fix it for run-to-run reproducible candidates
    env.setdefault("PYTHONHASHSEED", "0")
    gpu, desc = pick_gpu()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"python: {sys.executable}\ngpu   : {desc}")
    check_environment(env)
    for split in ("train", "test"):
        if not os.path.exists(os.path.join(args.data_dir, split, f"{split}_source1.tsv")):
            sys.exit(f"dataset not found in {args.data_dir}/{split} - pass --data-dir")

    cmd = [sys.executable, "-u", "-m", "entity_resolution.run", "--data-dir", args.data_dir,
           "--out-dir", args.out_dir, "--work-dir", args.work_dir]
    if not args.no_cross_encoder:
        cmd.append("--cross-encoder")
    if args.dev:
        cmd += ["--dev", str(args.dev)]
    cmd += args.extra
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    print("run   :", " ".join(cmd[2:]), f"\nlog   : {args.log}\n", flush=True)
    t0 = time.time()
    env.setdefault("COLUMNS", "130")              # progress-bar width when output goes to a pipe
    with open(args.log, "wb") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pending = b""
        while True:
            chunk = proc.stdout.read1(65536)      # forward at once: progress bars redraw with \r
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for ln in lines:                      # the log keeps only the final state of each bar
                fh.write(ln.rsplit(b"\r", 1)[-1] + b"\n")
            fh.flush()
        if pending:
            fh.write(pending.rsplit(b"\r", 1)[-1] + b"\n")
        proc.wait()
    mins = (time.time() - t0) / 60
    if proc.returncode == 0:
        print(f"\n===== finished after {mins:.1f} min: {args.out_dir}/matching_results.tsv")
    else:
        print(f"\n===== FAILED (exit {proc.returncode}) after {mins:.1f} min - see {args.log}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
