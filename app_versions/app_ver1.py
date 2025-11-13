"""
ECG Drag‑and‑Drop Web App (Streamlit)
-------------------------------------
Run locally with:
  pip install -r requirements.txt
  streamlit run app.py

Requirements (install them yourself if you don't use requirements.txt):
  streamlit==1.37.0
  numpy
  opencv-python
  torch
  torchvision
  matplotlib
  pandas

Notes:
- This file wraps your testing notebook pipeline into a single‑file app.
- Drop a PNG (and optional matching JSON) to see overlay visualisations + metrics.
- Set your checkpoint paths in the left sidebar.
"""

import os, io, json, math, tempfile
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# UI — Sidebar configuration
# ------------------------------
st.set_page_config(page_title="ECG Digitization — Drag & Drop", layout="wide")
st.title("ECG Digitization (PNG → trace) — Drag & Drop")
st.caption("Drop a PNG (and optional JSON) → we detect lead boxes, reconstruct traces, overlay results, and compute metrics.")

with st.sidebar:
    st.header("Configuration")

    # --- BBox checkpoint (simple text path) ---
    BBOX_CKPT = st.text_input("BBox checkpoint path", value="models/bbox/best_bbox_transformer_rgb_spikecover_13leads.pth")

    # --- Signal checkpoint: discover & choose ---
    def _discover_ckpts(root_dirs: list[str]) -> list[str]:
        found = []
        exts = (".pth", ".pt")
        for root in root_dirs:
            if not root or not os.path.isdir(root):
                continue
            for r, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(exts):
                        found.append(os.path.join(r, f))
        return sorted(found)

    default_dirs = ["./models/signal", "./models", "."]
    discovered = _discover_ckpts(default_dirs)
    use_discovered = st.toggle("Choose signal checkpoint from list", value=bool(discovered), help="Turn off to type a custom path.")

    if use_discovered and discovered:
        # Show friendly names (relative paths)
        display_opts = [os.path.relpath(p) if not os.path.isabs(p) else p for p in discovered]
        sig_choice = st.selectbox("Signal checkpoint (discovered)", options=display_opts, index=0)
        SIG_CKPT = discovered[display_opts.index(sig_choice)]
    else:
        SIG_CKPT  = st.text_input("Signal checkpoint path", value="./ckpts_signal_bbox13_noexpand/best_signal_transformer_13leads.pth")

    # --- Architecture selector ---
    MODEL_TYPE= st.selectbox("Signal architecture", options=["vanilla", "dyt"], index=1, help="Matches your ColumnSignalTransformer (vanilla) or DyT variant.")

    # --- Hyperparameters ---
    H_REF     = st.number_input("H_REF (column height)", min_value=64, max_value=256, value=128, step=8)
    W_MAX     = st.number_input("W_MAX (max width)", min_value=256, max_value=4096, value=1600, step=64)
    MAX_LEN   = st.number_input("MAX_LEN (pos enc length)", min_value=512, max_value=8192, value=4096, step=256)
    IMG_SIZE_BBOX = st.number_input("BBox img size", min_value=128, max_value=512, value=256, step=16)
    st.markdown("---")
    st.caption("Tip: We load models once and reuse them for faster runs. Switching checkpoints reloads automatically.")

# Constants / colors
LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6","II_bottom"]
GT_COLOR   = (0, 255, 255)  # cyan (OpenCV BGR)
PRED_COLOR = (255, 255, 0)  # yellow
BOX_COLOR  = (0, 255, 255)  # cyan

# ------------------------------
# Helpers (ported from your notebook, condensed)
# ------------------------------

def _letterbox_256(rgb: np.ndarray, dst: int=256):
    h, w = rgb.shape[:2]
    scale = min(dst / w, dst / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((dst, dst, 3), dtype=np.uint8)
    px, py = (dst - nw) // 2, (dst - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, scale, px, py

def _map_back(norm_box, scale, px, py, W0, H0, dst=256):
    x1n, y1n, x2n, y2n = norm_box
    x1 = int((x1n * dst - px) / scale); x2 = int((x2n * dst - px) / scale)
    y1 = int((y1n * dst - py) / scale); y2 = int((y2n * dst - py) / scale)
    x1 = max(0, min(W0 - 2, x1)); x2 = max(x1 + 1, min(W0 - 1, x2))
    y1 = max(0, min(H0 - 2, y1)); y2 = max(y1 + 1, min(H0 - 1, y2))
    return x1, y1, x2, y2

def expand_box_to_spikes_enlarge_only(rgb, box_xyxy, pad_px=8, max_expand=32, grad_pct=90, hpad_px=4):
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1 = max(0, min(W - 2, x1)); x2 = max(x1 + 1, min(W - 1, x2))
    y1 = max(0, min(H - 2, y1)); y2 = max(y1 + 1, min(H - 1, y2))
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return x1, y1, x2, y2
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gy   = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=3)
    prof = np.mean(np.abs(gy), axis=1)
    if prof.size < 3 or float(prof.max()) <= 1e-6:
        new_y1 = max(0, y1 - pad_px); new_y2 = min(H - 1, y2 + pad_px)
    else:
        thr  = np.percentile(prof, grad_pct)
        rows = np.where(prof >= thr)[0]
        if rows.size == 0:
            new_y1 = max(0, y1 - pad_px); new_y2 = min(H - 1, y2 + pad_px)
        else:
            rmin, rmax = int(rows.min()), int(rows.max())
            band_y1, band_y2 = y1 + rmin, y1 + rmax
            prop_y1 = max(0, band_y1 - (pad_px + max_expand))
            prop_y2 = min(H - 1, band_y2 + (pad_px + max_expand))
            new_y1 = min(y1, prop_y1); new_y2 = max(y2, prop_y2)
    new_x1 = max(0, x1 - hpad_px); new_x2 = min(W - 1, x2 + hpad_px)
    if new_x2 <= new_x1: new_x2 = min(W - 1, new_x1 + 1)
    new_y1 = min(new_y1, y1); new_y2 = max(new_y2, y2)
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)


def mask_white_on_black(crop_rgb, bold=True):
    if crop_rgb is None or crop_rgb.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    _, _, V = cv2.split(hsv)
    trace_mask = (V < 120).astype(np.uint8) * 255
    k = np.ones((2, 2), np.uint8)
    trace_mask = cv2.morphologyEx(trace_mask, cv2.MORPH_OPEN, k)
    trace_mask = cv2.morphologyEx(trace_mask, cv2.MORPH_CLOSE, k)
    if bold:
        kb = np.ones((3, 3), np.uint8)
        trace_mask = cv2.dilate(trace_mask, kb, iterations=2)
        trace_mask = cv2.morphologyEx(trace_mask, cv2.MORPH_CLOSE, kb)
    return trace_mask


def _prep_input_from_bw(mask_bw, H_ref=128):
    if mask_bw.ndim == 3: mask_bw = mask_bw[...,0]
    Hc, Wc = mask_bw.shape[:2]
    if Hc <= 0 or Wc <= 0:
        return np.zeros((H_ref, 1), np.float32)
    inter = cv2.INTER_AREA if Hc > H_ref else cv2.INTER_LINEAR
    col_scaled = cv2.resize(mask_bw, (Wc, H_ref), interpolation=inter).astype(np.float32) / 255.0
    return col_scaled


def _decimate_width(mask_bw: np.ndarray, w_max: int | None):
    Hc, Wc = mask_bw.shape[:2]
    if (w_max is None) or (Wc <= w_max) or (Wc <= 1):
        return mask_bw, 1
    stride = int(math.ceil(Wc / float(w_max)))
    mask_sub = mask_bw[:, ::stride]
    if mask_sub.shape[1] < 2:
        mask_sub = np.concatenate([mask_sub, mask_bw[:, -1:]], axis=1)
    return mask_sub, stride


def _rasterize_trace_native_width(pts_local, Hc, Wc):
    if Wc <= 1:
        return np.zeros((max(1, Wc),), np.float32)
    col_bins = [[] for _ in range(Wc)]
    for (yy, xx) in pts_local:
        xi = int(round(xx))
        if 0 <= xi < Wc:
            col_bins[xi].append(float(yy))
    y = np.full((Wc,), np.nan, np.float32)
    for i in range(Wc):
        if col_bins[i]: y[i] = np.mean(col_bins[i])
    valid = np.where(~np.isnan(y))[0]
    if valid.size == 0:
        y[:] = (Hc - 1.0) * 0.5
    else:
        first = valid[0]; y[:first] = y[first]
        last  = valid[-1]; y[last+1:] = y[last]
        for a, b in zip(valid[:-1], valid[1:]):
            if b > a + 1:
                ya, yb = y[a], y[b]
                n = b - a
                for k in range(1, n):
                    y[a+k] = ya + (yb - ya) * (k / n)
    return np.clip(y, 0.0, max(0.0, Hc-1.0)).astype(np.float32)


def _clip_points_to_box(pts, box):
    x1,y1,x2,y2 = box
    out = []
    for (yy, xx) in pts:
        if y1 <= yy < y2 and x1 <= xx < x2:
            out.append([float(yy - y1), float(xx - x1)])
    return out

# ------------------------------
# Models (BBox + Signal) — minimal faithful ports
# ------------------------------

class BBoxHeadTransformerRGB(nn.Module):
    def __init__(self, in_ch=3, img_size=256, patch=16, dim=256, depth=4, heads=8, num_leads=13):
        super().__init__()
        self.img_size = img_size
        self.num_leads = num_leads
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n_patches = (img_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_patches, dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_leads * 4))

    def forward(self, rgb):
        tok = self.proj(rgb).flatten(2).transpose(1, 2)
        tok = tok + self.pos
        tok = self.encoder(tok)
        feat = tok.mean(1)
        out  = self.head(feat).view(-1, self.num_leads, 4)
        return torch.sigmoid(out)


def _infer_from_state_training_style(state: Dict[str, torch.Tensor]):
    if "proj.weight" not in state:
        raise RuntimeError("Checkpoint missing 'proj.weight' — not your training-style bbox.")
    w = state["proj.weight"]
    dim   = int(w.shape[0])
    patch = int(w.shape[2])
    out_dim = None
    for k, v in state.items():
        if k.endswith("head.1.weight"):
            out_dim = int(v.shape[0]); break
    if out_dim is None:
        raise RuntimeError("Could not infer num_leads from 'head.1.weight'.")
    num_leads = out_dim // 4
    layer_ids = set()
    import re
    pat = re.compile(r"encoder\\.layers\\.(\\d+)\\.")
    for k in state.keys():
        m = pat.search(k)
        if m: layer_ids.add(int(m.group(1)))
    depth = (max(layer_ids) + 1) if layer_ids else 4
    return dict(patch=patch, dim=dim, depth=depth, num_leads=num_leads)


def load_bbox_head_training_style(ckpt_path: str, img_size=256, in_ch=3, heads=8, device="cpu"):
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("state_dict", raw)
    hp = _infer_from_state_training_style(state)
    model = BBoxHeadTransformerRGB(
        in_ch=in_ch, img_size=img_size,
        patch=hp["patch"], dim=hp["dim"], depth=hp["depth"], heads=heads,
        num_leads=hp["num_leads"]
    ).to(device)
    if "pos" in state and tuple(state["pos"].shape) != tuple(model.pos.shape):
        with torch.no_grad():
            model.pos = nn.Parameter(torch.zeros_like(state["pos"]))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError("BBox checkpoint did not load fully — architecture mismatch.")
    model.eval()
    return model, hp["num_leads"] * 4


class ColumnSignalTransformer(nn.Module):
    def __init__(self, H_ref=128, d_model=256, depth=6, heads=8, mlp=512, max_len=4096):
        super().__init__()
        self.embed_col = nn.Linear(H_ref, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=mlp, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1), nn.Tanh())
    def forward(self, X, attn_mask=None):
        B, Href, W = X.shape
        t = self.embed_col(X.permute(0,2,1))
        pe = self.pos[:, :W, :] if W <= self.pos.shape[1] else torch.cat([self.pos, self.pos[:, :W-self.pos.shape[1], :].clone()], dim=1)
        padmask = (attn_mask == 0.0) if attn_mask is not None else None
        t = self.encoder(t + pe, src_key_padding_mask=padmask)
        return self.head(t).squeeze(-1)


class DyTAct(nn.Module):
    def __init__(self, dim: int, init_alpha: float = 1.0, per_channel: bool = True):
        super().__init__()
        if per_channel:
            self.log_alpha = nn.Parameter(torch.log(torch.ones(dim) * init_alpha))
            self._shape = (1, 1, dim)
        else:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
            self._shape = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp()
        if self._shape is not None: alpha = alpha.view(*self._shape)
        return torch.tanh(alpha * x)


class DyTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0, depth: int = 6):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, bias=False, dropout=dropout)
        self.ff1 = nn.Linear(d_model, dim_ff, bias=False)
        self.act = DyTAct(dim_ff, init_alpha=1.0, per_channel=True)
        self.ff2 = nn.Linear(dim_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        init_gamma = 1.0 / math.sqrt(depth)
        self.gamma_attn = nn.Parameter(torch.tensor(init_gamma))
        self.gamma_ffn  = nn.Parameter(torch.tensor(init_gamma))
    def forward(self, x: torch.Tensor, padmask: torch.Tensor | None = None) -> torch.Tensor:
        aout, _ = self.attn(x, x, x, key_padding_mask=padmask, need_weights=False)
        x = x + self.drop(self.gamma_attn * aout)
        h = self.ff2(self.act(self.ff1(x)))
        x = x + self.drop(self.gamma_ffn * h)
        return x


class ColumnSignalTransformerDyT(nn.Module):
    def __init__(self, H_ref=128, d_model=256, depth=6, heads=8, mlp=512, max_len=4096, dropout=0.0):
        super().__init__()
        self.embed_col = nn.Linear(H_ref, d_model, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([DyTEncoderLayer(d_model=d_model, nhead=heads, dim_ff=mlp, dropout=dropout, depth=depth) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(d_model, 1, bias=False), nn.Tanh())
        nn.init.normal_(self.pos, mean=0.0, std=0.01)
    def forward(self, X: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, Href, W = X.shape
        t = self.embed_col(X.permute(0, 2, 1))
        pe = self.pos[:, :W, :] if W <= self.pos.shape[1] else torch.cat([self.pos, self.pos[:, :W-self.pos.shape[1], :]], dim=1)
        padmask = (attn_mask == 0.0) if attn_mask is not None else None
        t = t + pe
        for lyr in self.layers:
            t = lyr(t, padmask=padmask)
        return self.head(t).squeeze(-1)


# ------------------------------
# Device + cached loaders
# ------------------------------
@st.cache_resource(show_spinner=False)
def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=True)
def load_models_cached(bbox_ckpt: str, sig_ckpt: str, model_type: str, H_ref: int, max_len: int, img_size_bbox: int):
    device = _get_device()
    bbox_head, out_dim = load_bbox_head_training_style(bbox_ckpt, img_size=img_size_bbox, in_ch=3, heads=8, device=device)

    # ---- Smart signal-ckpt inspection to avoid size-mismatch errors ----
    detected_arch = None
    sd = None
    if os.path.isfile(sig_ckpt):
        sd = torch.load(sig_ckpt, map_location=device)
        # Heuristic: look at head.0.weight shape
        w0 = sd.get("head.0.weight", None)
        if isinstance(w0, torch.Tensor):
            if w0.ndim == 2 and w0.shape[0] == 1:          # [1, 256] → Linear first → DyT-style head
                detected_arch = "dyt"
            elif w0.ndim == 1:                              # [256] → LayerNorm first → Vanilla head
                detected_arch = "vanilla"
    # If user-chosen arch mismatches detected, prefer detected but warn in UI later
    effective_arch = detected_arch or model_type

    if effective_arch == "vanilla":
        signal_model = ColumnSignalTransformer(H_ref=H_ref, d_model=256, depth=6, heads=8, mlp=512, max_len=max_len).to(device)
    else:
        signal_model = ColumnSignalTransformerDyT(H_ref=H_ref, d_model=256, depth=6, heads=8, mlp=512, max_len=max_len).to(device)

    # Safely load: drop incompatible keys (e.g., LayerNorm vs Linear) to avoid shape errors
    if sd is not None:
        pruned = sd.copy()
        missing = []
        # Remove keys whose shapes don't match model's current state dict
        cur = signal_model.state_dict()
        for k, v in list(pruned.items()):
            if k in cur and tuple(cur[k].shape) != tuple(v.shape):
                pruned.pop(k)
                missing.append(k)
        signal_model.load_state_dict(pruned, strict=False)
        if missing:
            st.warning(f"Some signal weights could not be loaded due to shape mismatch and were skipped: {', '.join(missing[:6])}{' …' if len(missing)>6 else ''}")
        if detected_arch and detected_arch != model_type:
            st.info(f"Auto-detected architecture from checkpoint: **{detected_arch}** (overrode sidebar selection '{model_type}').")

    bbox_head.eval(); signal_model.eval()
    return device, bbox_head, signal_model, out_dim


# ------------------------------
# Inference core
# ------------------------------

def _unnorm_to_pixels(y_norm: torch.Tensor, Hc: torch.Tensor) -> torch.Tensor:
    Hc_exp = Hc.expand_as(y_norm).to(y_norm.dtype)
    y = (1.0 - y_norm) * 0.5 * (Hc_exp - 1.0)
    zero = torch.zeros_like(y); maxv = (Hc_exp - 1.0)
    y = torch.maximum(y, zero); y = torch.minimum(y, maxv)
    return y


def run_inference(rgb: np.ndarray, json_ann: Dict[str, Any] | None, cfg: Dict[str, Any]):
    device, bbox_head, signal_model, out_dim = load_models_cached(
        cfg["bbox_ckpt"], cfg["sig_ckpt"], cfg["model_type"], cfg["H_REF"], cfg["MAX_LEN"], cfg["IMG_SIZE_BBOX"]
    )
    canvas_256, scale, px, py = _letterbox_256(rgb, dst=cfg["IMG_SIZE_BBOX"])
    x = torch.tensor(canvas_256/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = bbox_head(x)[0].detach().cpu().numpy()

    num_leads = len(LEAD_NAMES)
    if pred_norm.shape[0] >= num_leads:
        pred_norm = pred_norm[:num_leads]
    else:
        pad = np.tile(pred_norm[-1:], (num_leads - pred_norm.shape[0], 1))
        pred_norm = np.concatenate([pred_norm, pad], axis=0)

    H0, W0 = rgb.shape[:2]
    raw_boxes, exp_boxes = [], []
    for i in range(num_leads):
        b = _map_back(pred_norm[i], scale, px, py, W0, H0, dst=cfg["IMG_SIZE_BBOX"])
        raw_boxes.append(b)
        exp_boxes.append(expand_box_to_spikes_enlarge_only(rgb, b))

    # Build inputs
    X_cols, metas = [], []
    for li in range(num_leads):
        x1,y1,x2,y2 = map(int, exp_boxes[li])
        crop = rgb[y1:y2, x1:x2] if (x2>x1 and y2>y1) else np.zeros((32,32,3), np.uint8)
        bw   = mask_white_on_black(crop, bold=True)
        bw_small, x_stride = _decimate_width(bw, cfg["W_MAX"])  # cap width
        x_arr = _prep_input_from_bw(bw_small, H_ref=cfg["H_REF"])
        X_cols.append(torch.tensor(x_arr, dtype=torch.float32))
        metas.append({
            "lead": LEAD_NAMES[li],
            "exp_box": (x1,y1,x2,y2),
            "Hc": bw.shape[0],
            "Wc": bw_small.shape[1],
            "Wc_orig": bw.shape[1],
            "x_stride": int(x_stride),
        })

    # Pack batch
    max_W = max(x.shape[1] for x in X_cols) if len(X_cols)>0 else 1
    N = len(X_cols)
    X = torch.zeros(N, cfg["H_REF"], max_W, dtype=torch.float32)
    mask = torch.zeros(N, max_W, dtype=torch.float32)
    for i, Xi in enumerate(X_cols):
        w = Xi.shape[1]
        X[i, :, :w] = Xi
        mask[i, :w] = 1.0
    X = X.to(device); m_t = mask.to(device)
    Hc_vec = torch.tensor([m["Hc"] for m in metas], dtype=torch.float32, device=device).view(-1,1)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda")):
        y_norm = signal_model(X, attn_mask=m_t)
        y_hat  = _unnorm_to_pixels(y_norm, Hc_vec.expand_as(y_norm))

    y_hat_np = y_hat.detach().cpu().numpy()
    m_np     = mask.detach().cpu().numpy()

    # Full‑image overlay (prediction)
    canvas = rgb.copy()
    for i, meta in enumerate(metas):
        x1,y1,x2,y2 = meta["exp_box"]
        w = int(m_np[i].sum())
        if w < 2: continue
        stride = int(meta["x_stride"])
        yh = y_hat_np[i, :w]
        xs_png = (np.arange(w, dtype=np.int32) * stride) + int(x1)
        ys_pd  = (yh + y1).astype(np.int32)
        pts_pd = np.stack([xs_png, np.clip(ys_pd, 0, canvas.shape[0]-1)], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts_pd], False, PRED_COLOR, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1,y1), (x2,y2), BOX_COLOR, 1)

    # Metrics (if JSON available)
    metrics_df = None
    overlay_both = None
    if json_ann is not None:
        # Build GT per box
        H0, W0 = rgb.shape[:2]
        by_name = {}
        for ld in json_ann.get("leads", []):
            nm = str(ld.get("lead_name", ld.get("name",""))).strip()
            by_name.setdefault(nm, []).append(ld)

        def _entry_box_xyxy(entry, H0, W0):
            if entry is None: return [0,0,1,1]
            if entry.get("lead_bounding_box") is not None:
                bb = entry["lead_bounding_box"]
                xs, ys = [], []
                for k in ("0","1","2","3"):
                    y, x = bb[k]; xs.append(float(x)); ys.append(float(y))
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                pts = entry.get("plotted_pixels", [])
                if not pts: return [0,0,1,1]
                xs = [float(p[1]) for p in pts]; ys = [float(p[0]) for p in pts]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            x1 = max(0, min(W0-2, int(round(x1)))); x2 = max(x1+1, min(W0-1, int(round(x2))))
            y1 = max(0, min(H0-2, int(round(y1)))); y2 = max(y1+1, min(H0-1, int(round(y2))))
            return [x1, y1, x2, y2]

        def _split_II(ii_entries):
            if not ii_entries:
                return {"II": None, "II_bottom": None}
            boxes = [_entry_box_xyxy(e, H0, W0) for e in ii_entries]
            widths = [max(1, b[2]-b[0]) for b in boxes]
            idx_bottom = int(np.argmax(widths)); entry_bottom = ii_entries[idx_bottom]
            idx_top = None
            if len(ii_entries) >= 2:
                cand = [(w, i) for i, w in enumerate(widths) if i != idx_bottom]
                idx_top = max(cand)[1] if cand else None
            entry_top = ii_entries[idx_top] if idx_top is not None else entry_bottom
            return {"II": entry_top, "II_bottom": entry_bottom}

        ii_split = _split_II(by_name.get("II", []))

        entry_map = {}
        for nm in ["I","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]:
            ls = by_name.get(nm, [])
            best = None; best_area = -1
            for e in ls:
                x1,y1,x2,y2 = _entry_box_xyxy(e, H0, W0)
                area = (x2-x1)*(y2-y1)
                if area > best_area:
                    best_area, best = area, e
            entry_map[nm] = best
        entry_map["II"] = ii_split["II"]
        entry_map["II_bottom"] = ii_split["II_bottom"]

        def _clip_points(pts, box):
            x1,y1,x2,y2 = box
            out = []
            for (yy, xx) in pts:
                if y1 <= yy < y2 and x1 <= xx < x2:
                    out.append([float(yy - y1), float(xx - x1)])
            return out

        def _metric_1d(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            L = min(len(y_true), len(y_pred))
            if L < 2: return np.nan, np.nan, np.nan
            yt = y_true[:L]; yp = y_pred[:L]
            std_t = np.std(yt); std_p = np.std(yp)
            pcc = float(np.corrcoef(yt, yp)[0,1]) if (std_t >= 1e-8 and std_p >= 1e-8) else np.nan
            rmse = float(np.sqrt(np.mean((yt - yp)**2)))
            var  = float(np.var(yt, ddof=1)) if L > 1 else 0.0
            mse  = float(np.mean((yt - yp)**2))
            snr  = float(10.0*np.log10((var + 1e-12)/(mse + 1e-12))) if var > 0 else np.nan
            return pcc, rmse, snr

        # Iterate leads for metrics and overlay GT
        rows = []
        canvas_pg = rgb.copy()
        for i, meta in enumerate(metas):
            lead = meta["lead"]
            x1,y1,x2,y2 = meta["exp_box"]
            w = int(m_np[i].sum())
            if w < 2:
                rows.append({"lead": lead, "width": 0, "PCC": np.nan, "RMSE": np.nan, "SNR_dB": np.nan})
                continue
            stride = int(meta["x_stride"])
            yhat = y_hat_np[i, :w]

            e = entry_map.get(lead, None)
            ygt = np.zeros_like(yhat)
            if e is not None:
                pts_all = e.get("plotted_pixels", [])
                if pts_all:
                    local = _clip_points(pts_all, (x1,y1,x2,y2))
                    y_full = _rasterize_trace_native_width(local, y2-y1, x2-x1)
                    ygt = y_full[::stride]
                    if len(ygt) < w:
                        ygt = np.pad(ygt, (0, w - len(ygt)), mode='edge')
                    elif len(ygt) > w:
                        ygt = ygt[:w]
                else:
                    ygt = np.full_like(yhat, (y2 - y1) * 0.5, dtype=np.float32)
            else:
                ygt = np.full_like(yhat, (y2 - y1) * 0.5, dtype=np.float32)

            pcc, rmse, snr = _metric_1d(ygt, yhat)
            rows.append({"lead": lead, "width": w, "PCC": pcc, "RMSE": rmse, "SNR_dB": snr})

            xs_png = (np.arange(w, dtype=np.int32) * stride) + int(x1)
            ys_pd  = (yhat + y1).astype(np.int32)
            ys_gt  = (ygt  + y1).astype(np.int32)
            pts_pd = np.stack([xs_png, np.clip(ys_pd, 0, canvas_pg.shape[0]-1)], axis=1).astype(np.int32)
            pts_gt = np.stack([xs_png, np.clip(ys_gt, 0, canvas_pg.shape[0]-1)], axis=1).astype(np.int32)
            cv2.polylines(canvas_pg, [pts_pd], False, PRED_COLOR, 1, cv2.LINE_AA)
            cv2.polylines(canvas_pg, [pts_gt], False, GT_COLOR,   1, cv2.LINE_AA)
            cv2.rectangle(canvas_pg, (x1,y1), (x2,y2), BOX_COLOR, 1)

        metrics_df = pd.DataFrame(rows).set_index("lead").reindex(LEAD_NAMES)
        wv = metrics_df["width"].fillna(0).values.astype(float)
        def _wmean(vals, w):
            vals = np.asarray(vals, dtype=float)
            mask = ~np.isnan(vals)
            if mask.sum() == 0 or np.sum(w[mask]) == 0: return np.nan
            return float(np.nansum(vals[mask] * w[mask]) / np.sum(w[mask]))
        mean_pcc  = float(np.nanmean(metrics_df["PCC"].values))
        mean_rmse = float(np.nanmean(metrics_df["RMSE"].values))
        mean_snr  = float(np.nanmean(metrics_df["SNR_D"].values)) if "SNR_D" in metrics_df else float(np.nanmean(metrics_df["SNR_dB"].values))
        wmean_pcc  = _wmean(metrics_df["PCC"].values, wv)
        wmean_rmse = _wmean(metrics_df["RMSE"].values, wv)
        wmean_snr  = _wmean(metrics_df["SNR_dB"].values, wv)
        avg_rows = pd.DataFrame([
            {"lead": "__MEAN__",   "width": np.nan,     "PCC": mean_pcc,  "RMSE": mean_rmse,  "SNR_dB": mean_snr},
            {"lead": "__W_MEAN__", "width": np.nansum(wv), "PCC": wmean_pcc, "RMSE": wmean_rmse, "SNR_dB": wmean_snr},
        ]).set_index("lead")
        metrics_df = pd.concat([metrics_df, avg_rows], axis=0)
        overlay_both = canvas_pg

    return {
        "pred_overlay": canvas,
        "metrics": metrics_df,
        "overlay_both": overlay_both,
    }


# ------------------------------
# UI — File upload & results
# ------------------------------
left, right = st.columns([1,1])
with left:
    up_png = st.file_uploader("Drop ECG PNG here", type=["png"], accept_multiple_files=False)
    up_json= st.file_uploader("(Optional) Matching JSON", type=["json"], accept_multiple_files=False)
    run_btn = st.button("Run Inference", type="primary", use_container_width=True)

with right:
    st.info("You can tweak model/type/params in the sidebar, then click **Run Inference**.")

if run_btn and up_png is not None:
    png_bytes = up_png.read()
    file_bytes = np.asarray(bytearray(png_bytes), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Failed to read PNG. Make sure it's a valid image.")
        st.stop()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    ann = None
    if up_json is not None:
        try:
            ann = json.loads(up_json.read().decode("utf-8"))
        except Exception as e:
            st.warning(f"JSON could not be parsed: {e}")

    cfg = dict(
        bbox_ckpt=BBOX_CKPT, sig_ckpt=SIG_CKPT, model_type=MODEL_TYPE,
        H_REF=H_REF, W_MAX=W_MAX, MAX_LEN=MAX_LEN, IMG_SIZE_BBOX=IMG_SIZE_BBOX,
    )

    with st.spinner("Running model…"):
        try:
            out = run_inference(rgb, ann, cfg)
        except Exception as e:
            st.exception(e)
            st.stop()

    # Visualisation tabs
    tab1, tab2, tab3 = st.tabs(["Prediction overlay", "Pred vs GT overlay", "Metrics table & CSV"]) 

    with tab1:
        st.image(out["pred_overlay"], channels="RGB", caption="Prediction overlay (yellow) + boxes (cyan)")
        # Download
        _, buf = cv2.imencode('.png', cv2.cvtColor(out["pred_overlay"], cv2.COLOR_RGB2BGR))
        st.download_button("Download overlay (PNG)", data=buf.tobytes(), file_name="pred_overlay.png", mime="image/png")

    with tab2:
        if out["overlay_both"] is None:
            st.warning("No JSON provided → metrics skipped. Upload a JSON to see GT overlay.")
        else:
            st.image(out["overlay_both"], channels="RGB", caption="Prediction (yellow) vs GT (cyan)")
            _, buf2 = cv2.imencode('.png', cv2.cvtColor(out["overlay_both"], cv2.COLOR_RGB2BGR))
            st.download_button("Download pred_vs_gt (PNG)", data=buf2.tobytes(), file_name="pred_vs_gt_overlay.png", mime="image/png")

    with tab3:
        if out["metrics"] is None:
            st.info("Metrics available only if JSON was provided.")
        else:
            st.dataframe(out["metrics"], use_container_width=True)
            csv_bytes = out["metrics"].reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("Download metrics (CSV)", data=csv_bytes, file_name="metrics.csv", mime="text/csv")

    st.success("Done.")

else:
    st.info("Drop a PNG to begin.")
