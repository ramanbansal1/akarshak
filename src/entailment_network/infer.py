import argparse, json
import torch
from src.pre_computing.hs_extract import HiddenStateExtractor
from .verifier_common import *
from .data import DataCfg, RowEncoder, load_universal


class Verifier:
    def __init__(self, ckpt_path, max_model_len=8192, gpu_mem=0.85, shm_tokens=150_000, device="cuda",
                 dtype="bfloat16", tp=1, shm_root="/dev/shm/hs_extract", backend="auto"):
        ck = torch.load(ckpt_path, map_location=" cpu")
        self.dev, self.thr = device, ck["thresholds"]
        self.model = VerdictModel(**ck["model_cfg"]); self.model.load_state_dict(ck["state_dict"])
        self.model.to(device).eval()
        from transformers import AutoTokenizer
        self.enc = RowEncoder(AutoTokenizer.from_pretrained(ck["backbone"]), DataCfg(**ck["data_cfg"]))
        self.ext = HiddenStateExtractor(ck["backbone"], ck["layers"], max_model_len,
                                        gpu_mem=gpu_mem, max_chunk_tokens=shm_tokens,
                                        dtype=dtype, tp=tp, shm_root=shm_root, backend=backend)

    @torch.inference_mode()
    def verify(self, rows):
        """rows: [{'claim': str, 'evidence': [sentence, ...]}] ->
        [{'verdict', 'scores', 'chunks': [{'span':[s0,s1), 'probs', 'verdict'}]}]"""
        meta = []

        def gen():
            for i, r in enumerate(rows):
                for s in self.enc.encode_row(i, r["claim"], r["evidence"]):
                    meta.append((s.row, s.spans))
                    yield s.ids, s.marks

        chunks, flat_p, flat_row = [[] for _ in rows], [], []
        for k, st in enumerate(self.ext.extract(gen())):
            row, spans = meta[k]
            st = st.to(self.dev)
            p = self.model(st[:1], st[1:].unsqueeze(0))[0].softmax(-1).cpu()          # [N,3]
            for sp, pr in zip(spans, p):
                chunks[row].append(dict(span=list(sp), probs=pr.tolist(), verdict=LABELS[int(pr.argmax())]))
            flat_p.append(p); flat_row += [row] * len(spans)
        probs = torch.cat(flat_p) if flat_p else torch.zeros(0, 3)
        s = row_scores(probs, torch.tensor(flat_row, dtype=torch.long), len(rows))
        pred = decide(s, *self.thr)
        return [dict(verdict=LABELS[int(pred[i])], scores=dict(zip(LABELS, s[i].tolist())), chunks=chunks[i])
                for i in range(len(rows))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="jon-tow/wice"); p.add_argument("--config", default="subclaim")
    p.add_argument("--split", default="test")
    p.add_argument("--jsonl", default=None, help="file with {'claim':..., 'evidence':[...]} per line")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="preds.jsonl")
    p.add_argument("--dtype", default="bfloat16"); p.add_argument("--tp", type=int, default=1)
    p.add_argument("--shm_tokens", type=int, default=150_000); p.add_argument("--shm_root", default="/dev/shm/hs_extract")
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    a = p.parse_args()

    if a.jsonl:
        rows = [json.loads(l) for l in open(a.jsonl)]
    elif a.dataset.startswith("uni:"):
        rows = load_universal(a.dataset[4:], a.split)
        if a.limit:
            rows = rows.select(range(min(a.limit, len(rows))))
    else:
        from datasets import load_dataset
        rows = load_dataset(a.dataset, a.config, split=a.split)
        if a.limit:
            rows = rows.select(range(min(a.limit, len(rows))))
    rows = list(rows)
    v = Verifier(a.ckpt, dtype=a.dtype, tp=a.tp, shm_tokens=a.shm_tokens, shm_root=a.shm_root,
                 backend=a.backend)
    res = v.verify(rows)
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    if "label" in rows[0]:
        y = torch.tensor([parse_label(r["label"]) for r in rows])
        pred = torch.tensor([LABELS.index(r["verdict"]) for r in res])
        f1, per = macro_f1(pred, y)
        print(f"row acc {(pred == y).float().mean():.4f} macro_f1 {f1:.4f} per-class {per}")


if __name__ == "__main__":
    main()
