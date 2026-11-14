"""TRACE-RAG ticket 05: WICE hidden-state extraction + head training on Kaggle.

Pipeline (2xT4, HF backend, Qwen3.5-2B):
  1. clone repo, link dataset input -> repo/data/
  2. extract feats for uni:wice_p1p2 (P1-train/P1-val/P2) -> runs/wice_p1p2
  3. extract feats for uni:wice validation+test (pooled dev+test) -> runs/wice_pool
  4. train VerdictModel head on wice_p1p2 (val fold = P1-val, test = P2)
  5. scores_from_cache -> ClaimScore npz for all four pools -> results/

Outputs land in /kaggle/working/results/ (auto-downloaded by notebook-run).
"""

import os, shutil, subprocess, sys, json
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "repo"
RUNS = WORK / "runs"
RESULTS = WORK / "results"
for d in (RUNS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# locate the dataset anywhere under /kaggle/input (mount path varies;
# may still be processing -> wait for content, not just the dir)
import time
def dataset_root():
    for _ in range(60):
        inp = Path("/kaggle/input")
        if inp.exists():
            # prefer a dir literally containing our files
            for marker in ("p1_parents.txt", "universal"):
                hits = [p for p in inp.rglob(marker)]
                if hits:
                    p = hits[0]
                    # walk up to the ancestor containing 'universal/'
                    for anc in [p] + list(p.parents):
                        if (anc / "universal").is_dir():
                            return anc
                    return p.parent
            zips = list(inp.rglob("*.zip"))
            if zips:
                import zipfile
                zp = zips[0]
                with zipfile.ZipFile(zp) as z:
                    z.extractall(zp.parent)
                # re-scan for content after extraction
                continue
        time.sleep(20)
    return None

DS = dataset_root()
if DS is None:
    raise SystemExit("dataset content not found under /kaggle/input")
print("dataset root:", DS, "| contents:", sorted(p.name for p in DS.iterdir()))

# dataset layout on mount: universal/{wice,wice_p1p2}/ + wice/*_parents.txt
def find(rel):
    # strip the leading 'data/' the repo layout uses; the upload dropped it
    tail = rel[len("data/"):] if rel.startswith("data/") else rel
    cand = DS / tail
    if not cand.exists():
        hits = [p for p in DS.rglob(Path(tail).name) if p.is_dir()
                and (p / ("train.jsonl" if "universal" in tail
                          else "p1_parents.txt")).exists()]
        cand = hits[0] if hits else cand
    if not cand.exists():
        raise SystemExit(f"missing {rel} (looked for {cand}) under {DS}")
    return cand

print("== clone repo ==")
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/ramanbansal1/akarshak.git", str(REPO)],
                   check=True)

# link dataset files into the repo layout the loaders expect
for rel in ("data/universal/wice", "data/universal/wice_p1p2", "data/wice"):
    dst = REPO / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(find(rel), dst)
print("== data linked ==")

os.chdir(REPO)
sys.path.insert(0, str(REPO))

MODEL = "Qwen/Qwen3.5-2B"

# ---- optional W&B: needs a Kaggle secret (or env) named WANDB_API_KEY ----
_wandb_run = None
try:
    _key = os.environ.get("WANDB_API_KEY")
    if not _key:
        try:
            from kaggle_secrets import UserSecretsClient
            _key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            _key = None
    if _key:
        os.environ["WANDB_API_KEY"] = _key
        os.environ.setdefault("WANDB_PROJECT", "trace-rag-wice")
        import wandb
        _wandb_run = wandb.init(
            project="trace-rag-wice", name="kaggle-p1p2",
            config=dict(model=MODEL, backend="hf", dtype="float16",
                        pools=["wice_p1p2:train", "wice_p1p2:validation",
                               "wice_p1p2:test", "wice:validation",
                               "wice:test"],
                        protocol="P1-train head / P1-val early stop / "
                                 "P2 calib / dev+test pooled"))
        print("wandb: logging to", _wandb_run.url, flush=True)
    else:
        print("wandb: no WANDB_API_KEY secret — file logs only", flush=True)
except Exception as e:
    print("wandb init failed:", e, flush=True)
    _wandb_run = None


def wb_mark(stage):
    if _wandb_run is not None:
        try:
            _wandb_run.log({f"stage/{stage}": 1})
        except Exception:
            pass


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
EXTRACT = ["python", "-m", "src.pre_computing.extract",
           "--model", MODEL, "--backend", "hf", "--dtype", "float16",
           "--hf_batch_tokens", "8192", "--tp", "1"]

def run(cmd, log):
    """Stream child stdout/stderr live to the kernel log AND the log file —
    tqdm progress bars and tracebacks are visible in real time, not just the
    last 3000 chars after exit."""
    print(f"$ {' '.join(cmd)}", flush=True)
    log_path = WORK / log
    with open(log_path, "wb") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        while True:
            chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") \
                else p.stdout.read(4096)
            if not chunk:
                break
            f.write(chunk)
            f.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        rc = p.wait()
    if rc != 0:
        raise SystemExit(f"FAILED ({rc}): {' '.join(cmd)} (see {log})")

# 1) P1/P2 pools (all three cache slots: train=P1-train, val=P1-val, test=P2)
run(EXTRACT + ["--dataset", "uni:wice_p1p2", "--out_dir", str(RUNS / "wice_p1p2")],
    "extract_p1p2.log")
wb_mark("extract_p1p2")

# 2) pooled dev+test parents (validation->dev.jsonl, test->test.jsonl)
run(EXTRACT + ["--dataset", "uni:wice", "--splits", "validation", "test",
               "--out_dir", str(RUNS / "wice_pool")],
    "extract_pool.log")
wb_mark("extract_pool")

# 3) head training: P1-train -> P1-val early stop -> P2 test metrics
run(["python", "-m", "src.entailment_network.train",
     "--out_dir", str(RUNS / "wice_p1p2")], "train.log")
wb_mark("train")

# 3b) training-curve figures from head_history.csv -> results/plots/
run(["python", "experiments/plot_training.py",
     "--glob", f"{RUNS}/*/head_history.csv",
     "--out", str(RESULTS / "plots")], "plots.log")

# 4) ClaimScore npz per pool (tau_s/tau_r calibrated on P2 = wice_p1p2:test)
run(["python", "experiments/scores_from_cache.py",
     "--ckpt", str(RUNS / "wice_p1p2/head.pt"),
     "--calib", f"uni:wice_p1p2:test={RUNS}/wice_p1p2/feats_test.pt",
     "--pairs",
     f"uni:wice_p1p2:validation={RUNS}/wice_p1p2/feats_validation.pt",
     f"uni:wice_p1p2:test={RUNS}/wice_p1p2/feats_test.pt",
     f"uni:wice:validation={RUNS}/wice_pool/feats_validation.pt",
     f"uni:wice:test={RUNS}/wice_pool/feats_test.pt",
     "--out", str(RESULTS / "scores"), "--alpha", "0.10"],
    "scores.log")
wb_mark("scores")

# 5) copy checkpoints + metrics manifest for download
shutil.copytree(RUNS, RESULTS / "runs", dirs_exist_ok=True)
(RESULTS / "manifest.json").write_text(json.dumps(dict(
    model=MODEL, backend="hf", dtype="float16",
    pools=["wice_p1p2:train", "wice_p1p2:validation", "wice_p1p2:test",
           "wice:validation", "wice:test"],
    protocol="P1-train head / P1-val early stop / P2 calib / dev+test pooled")))

# 6) push generated figures to W&B, then close the run
if _wandb_run is not None:
    try:
        import wandb
        for png in sorted((RESULTS / "plots").glob("*.png")):
            _wandb_run.log({png.stem: wandb.Image(str(png))})
        _wandb_run.finish()
    except Exception as e:
        print("wandb figure upload failed:", e, flush=True)
print("== DONE ==", flush=True)
