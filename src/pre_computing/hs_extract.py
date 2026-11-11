"""HiddenStateExtractor: (ids, marks) stream -> per-sequence [M, L, d] tensors.

For each input sequence, `marks` gives token positions whose hidden states are
collected: marks[0] is the claim-end position, marks[1:] the evidence-chunk-end
positions. Output row i is the hidden state at marks[i], for each cached layer:
shape [len(marks), L, d]. Row 0 becomes the claim state (hq), rows 1: the chunk
states (he) in save_cache.

Backends:
  "hf"   - transformers AutoModel forward passes (always works, any layers)
  "vllm" - vLLM pooling runner returning all-token last hidden states (fast,
           last layer only); falls back to "hf" automatically on failure
  "auto" - vllm if installed + CUDA, else hf

Layer ids are 0-based hidden-layer indices (see default_layers). With the HF
backend, hidden_states[layer_id + 1] is used (index 0 is the embedding output);
the final layer uses the model's normed last_hidden_state. vLLM returns only
last-layer states, so non-final cached layers are duplicated with a warning.
"""

from __future__ import annotations

import warnings

import torch

_DTYPE_MAP = {"float16": torch.float16, "fp16": torch.float16,
              "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
_WINDOW = 128  # sequences buffered per backend attempt


class HiddenStateExtractor:
    def __init__(self, model: str, layers: list[int], max_model_len: int = 8192,
                 gpu_mem: float = 0.85, max_chunk_tokens: int = 150_000,
                 dtype: str = "bfloat16", tp: int = 1,
                 shm_root: str = "/dev/shm/hs_extract", backend: str = "auto",
                 hf_batch_tokens: int = 65_536):
        self.model_id, self.layers = model, list(layers)
        self.max_model_len = max_model_len
        self.dtype = _DTYPE_MAP.get(str(dtype).lower(), torch.bfloat16)
        self.backend = self._pick_backend(backend)
        self.hf_batch_tokens = hf_batch_tokens
        self._hf = self._vllm = self._tok = None
        if self.backend == "vllm":
            try:
                self._init_vllm(tp, gpu_mem)
            except Exception as e:                       # noqa: BLE001 - any vLLM failure -> HF
                warnings.warn(f"vLLM backend failed ({e}); falling back to HF")
                self.backend = "hf"
        if self.backend == "hf":
            self._init_hf()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _pick_backend(backend: str) -> str:
        if backend in ("hf", "vllm"):
            return backend
        try:
            import vllm  # noqa: F401
            return "vllm" if torch.cuda.is_available() else "hf"
        except Exception:                                # noqa: BLE001
            return "hf"

    def _init_vllm(self, tp: int, gpu_mem: float):
        from vllm import LLM
        self._vllm = LLM(
            model=self.model_id,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,
            max_model_len=self.max_model_len,
            dtype=str(self.dtype).replace("torch.", ""),
            enforce_eager=True,
            seed=0,
        )
        if len(self.layers) != 1:
            warnings.warn("vLLM backend returns last-layer states only; "
                          "non-final cached layers will be duplicated")

    def _init_hf(self):
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._hf = AutoModel.from_pretrained(self.model_id, torch_dtype=self.dtype,
                                             device_map="auto" if torch.cuda.is_available() else None)
        self._hf.eval()

    def close(self):
        self._hf = self._vllm = self._tok = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ main
    def extract(self, gen):
        """gen yields (ids: list[int], marks: list[int]). Yields [M, L, d] cpu tensors."""
        window, window_tokens = [], 0
        for item in gen:
            ids, marks = item
            ids, marks = self._fit(ids, marks)
            window.append((ids, marks))
            window_tokens += len(ids)
            full = len(window) >= _WINDOW or window_tokens >= self.hf_batch_tokens
            if full:
                yield from self._extract_window(window)
                window, window_tokens = [], 0
        if window:
            yield from self._extract_window(window)

    def _extract_window(self, window):
        if self.backend == "vllm":
            try:
                results = list(self._extract_vllm(window))   # materialize: no partial yields
                yield from results
                return
            except Exception as e:                       # noqa: BLE001 - API mismatch -> HF
                warnings.warn(f"vLLM extraction failed ({e}); falling back to HF")
                self.backend = "hf"
                self._vllm = None
                if self._hf is None:
                    self._init_hf()
        yield from self._extract_hf(window)

    def _fit(self, ids, marks):
        """Truncate over-long sequences and drop marks past the cutoff."""
        if len(ids) <= self.max_model_len:
            return ids, marks
        return ids[:self.max_model_len], [m for m in marks if m < self.max_model_len]

    # ------------------------------------------------------------------ HF path
    def _extract_hf(self, window):
        model, tok, layers = self._hf, self._tok, self.layers
        n_layers = getattr(model.config, "num_hidden_layers", max(layers) + 1)
        need_all = any(l != n_layers - 1 for l in layers)
        dev = next(model.parameters()).device
        maxlen = max(len(s) for s, _ in window)
        ids = torch.full((len(window), maxlen),
                         tok.pad_token_id if tok.pad_token_id is not None else 0,
                         dtype=torch.long)
        att = torch.zeros((len(window), maxlen), dtype=torch.long)
        for i, (s, _) in enumerate(window):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(dev), att.to(dev)
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=need_all)
        states, last = out.hidden_states, out.last_hidden_state
        for b, (_, marks) in enumerate(window):
            cols = []
            for li in layers:
                h = last[b] if li == n_layers - 1 else states[li + 1][b]
                cols.append(h[marks].float().cpu())        # [M, d]
            yield torch.stack(cols, dim=1)                 # [M, L, d]

    # ------------------------------------------------------------------ vLLM path
    def _extract_vllm(self, window):
        from vllm import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _ in window]
        try:
            outs = self._vllm.encode(prompts, pooling_task="all")
        except TypeError:
            outs = self._vllm.encode(prompts)
        l1 = len(self.layers)
        for (_, marks), o in zip(window, outs):
            data = o.outputs.data
            if data.dim() == 3:
                data = data[0]
            data = data.to("cpu", torch.float32)
            idx = [m for m in marks if m < data.shape[0]]
            st = data[idx]                                 # [M, d]
            yield st.unsqueeze(1).expand(-1, l1, -1).contiguous()  # [M, L, d]
