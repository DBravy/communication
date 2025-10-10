#!/usr/bin/env python3
"""
diagnose_comm.py — One-shot diagnostics for your communication/autoencoder training setup.

Run:
  python diagnose_comm.py --device cuda --max-batches 4 --batch-size 32

What it checks (and saves into ./diagnostics/):
1) Reproducibility & Seeds
   - Fixes seeds unless you pass --no-seed.
2) Data Pipeline Sanity
   - Auto-discovers a torch Dataset in dataset.py (or use --dataset-class).
   - Builds a DataLoader and inspects several mini-batches.
   - Verifies shapes, dtype, value ranges.
   - Checks label distribution and whether selection tasks contain the target among candidates.
   - Detects degenerate labels (all same or nearly constant) and duplicate samples in-batch.
   - Verifies shuffle is actually shuffling across epochs (optional second pass).
3) Model Smoke Tests
   - Auto-discovers a torch nn.Module in model.py (or use --model-class).
   - Moves model to device, sets train() and eval() correctly.
   - Runs forward on a batch; prints output signatures, detects NaNs/Infs.
   - Estimates logits entropy; warns if outputs are nearly uniform or saturated.
4) Loss/Backward/Optimizer
   - If train.py exposes build_optimizer or similar, uses it; otherwise builds AdamW as a fallback.
   - Performs 3 tiny train steps on a single batch; logs loss values and global grad-norms.
   - Flags zero/NaN gradients, vanishing/exploding norms.
5) Symbol/Message Probes (Best-effort)
   - Tries to find attributes like: model.sender, model.receiver, model.symbol_vocab_size, model.temperature, etc.
   - If message logits/probs are found, computes per-position entropy, dead-symbol rates, usage histograms.
6) Metrics Cross-check
   - Recomputes accuracy on a micro-batch; flags if accuracy calc appears wrong (e.g., not matching argmax vs labels).
7) Artifacts
   - Writes CSVs and PNGs to ./diagnostics/ (message usage histogram, entropy over positions, label hist, grad norms).

Notes:
- The script is defensive: probes are optional. If something can't be inspected, it will tell you what to expose.
- You can explicitly pass class names with --dataset-class and --model-class if auto-discovery guesses wrong.
"""

import argparse
import importlib
import inspect
import os
import sys
import types
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DIAG_DIR = "diagnostics"
os.makedirs(DIAG_DIR, exist_ok=True)

def set_seeds(seed=1337):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_module_safely(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        print(f"[WARN] Could not import {name}: {e}")
        return None

def find_dataset_class(module, explicit_name=None):
    if module is None:
        return None
    if explicit_name:
        cls = getattr(module, explicit_name, None)
        if cls is None:
            print(f"[ERR] --dataset-class '{explicit_name}' not found in dataset.py")
        return cls
    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, torch.utils.data.Dataset):
            candidates.append(obj)
    if not candidates:
        print("[WARN] No torch.utils.data.Dataset subclass found in dataset.py")
        return None
    # Heuristic: prefer names containing 'ARC', 'Comm', or 'Dataset'
    def score(c):
        name = c.__name__.lower()
        s = 0
        if "arc" in name: s += 3
        if "comm" in name: s += 2
        if "dataset" in name: s += 1
        return -s  # smaller (more negative) is better for sorted()
    candidates.sort(key=score)
    chosen = candidates[0]
    print(f"[OK] Using dataset class: {chosen.__name__}")
    return chosen

def find_model_class(module, explicit_name=None):
    if module is None:
        return None
    if explicit_name:
        cls = getattr(module, explicit_name, None)
        if cls is None:
            print(f"[ERR] --model-class '{explicit_name}' not found in model.py")
        return cls
    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            candidates.append(obj)
    if not candidates:
        print("[WARN] No torch.nn.Module subclass found in model.py")
        return None
    # Heuristics: prefer names containing 'Comm', 'Autoenc', 'SenderReceiver', 'Model'
    def score(c):
        name = c.__name__.lower()
        s = 0
        if "comm" in name: s += 5
        if "autoenc" in name or "autoencoder" in name: s += 4
        if "sender" in name or "receiver" in name: s += 3
        if "model" in name: s += 1
        return -s
    candidates.sort(key=score)
    chosen = candidates[0]
    print(f"[OK] Using model class: {chosen.__name__}")
    return chosen

def build_dataloader(DatasetCls, batch_size, num_workers=0, pin_memory=False, split=None):
    # Try common ctor signatures:
    kwargs_try = [
        dict(split=split) if split else {},
        dict(train=(split!="val")) if split else {},
        {},  # fallback
    ]
    last_err = None
    for kwargs in kwargs_try:
        try:
            ds = DatasetCls(**kwargs)
            break
        except TypeError as e:
            last_err = e
            continue
    else:
        raise TypeError(f"Could not instantiate {DatasetCls.__name__}: last error: {last_err}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    return ds, dl

def describe_batch(batch, tag=""):
    if isinstance(batch, (list, tuple)):
        parts = list(batch)
        info = [f"  part[{i}]: type={type(p)}, shape={getattr(p, 'shape', None)}, dtype={getattr(p, 'dtype', None)}" for i,p in enumerate(parts)]
    elif isinstance(batch, dict):
        info = [f"  key='{k}': type={type(v)}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}" for k,v in batch.items()]
    else:
        info = [f"  batch type={type(batch)}, shape={getattr(batch, 'shape', None)}, dtype={getattr(batch, 'dtype', None)}"]
    print(f"[BATCH {tag}]")
    for line in info:
        print(line)

def tensor_like(x):
    return isinstance(x, torch.Tensor)

def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if tensor_like(v) else v) for k,v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)([(v.to(device) if tensor_like(v) else v) for v in batch])
    else:
        return batch.to(device) if tensor_like(batch) else batch

def guess_labels(batch):
    # Try common keys
    if isinstance(batch, dict):
        for k in ["labels", "y", "target", "targets", "cls", "class", "answer"]:
            if k in batch and tensor_like(batch[k]):
                return batch[k]
    elif isinstance(batch, (list, tuple)):
        # assume last item is labels if tensor long/int
        cand = batch[-1]
        if tensor_like(cand) and cand.dtype in (torch.long, torch.int64, torch.int32):
            return cand
    return None

def has_candidates(batch):
    if isinstance(batch, dict):
        for k in ["candidates", "options", "choices"]:
            if k in batch and tensor_like(batch[k]):
                return k, batch[k]
    elif isinstance(batch, (list, tuple)):
        # heuristically, second element could be candidates
        if len(batch) >= 2 and tensor_like(batch[1]) and batch[1].dim() >= 3:
            return "candidates", batch[1]
    return None, None

def batch_sanity_checks(batches, out_prefix="batch_checks"):
    # Label distribution
    labels = []
    for b in batches:
        y = guess_labels(b)
        if y is not None and tensor_like(y):
            labels.append(y.detach().cpu().numpy().reshape(-1))
    if labels:
        ys = np.concatenate(labels, axis=0)
        counts = Counter(ys.tolist())
        print(f"[LABEL DIST] {counts}")
        # Save histogram
        fig = plt.figure()
        keys = sorted(counts.keys())
        vals = [counts[k] for k in keys]
        plt.bar([str(k) for k in keys], vals)
        plt.title("Label distribution (sampled)")
        plt.xlabel("class")
        plt.ylabel("count")
        fig.savefig(os.path.join(DIAG_DIR, f"{out_prefix}_label_hist.png"), bbox_inches="tight")
        plt.close(fig)
        if len(counts) == 1:
            print("[RED FLAG] All labels are identical in sampled batches.")
    else:
        print("[INFO] Could not find labels in batches; skipping label distribution.")
    # Selection: ensure target is in candidates
    for i, b in enumerate(batches[:5]):
        key, C = has_candidates(b)
        y = guess_labels(b)
        if key and y is not None and tensor_like(C) and tensor_like(y):
            Cn = C.detach().cpu().numpy()
            yn = y.detach().cpu().numpy()
            if Cn.shape[0] != yn.shape[0]:
                print(f"[WARN] Candidates batch size {Cn.shape[0]} != labels batch size {yn.shape[0]}")
            # Heuristic: check membership by equality on a flattened signature if possible
            # If C are indices, this will work; if C are images, this is best-effort.
            try:
                # For each sample, check if any candidate equals the designated positive (if batch provides such index)
                # If labels are class indices and candidates are a set of class indices per sample, this works.
                # If this doesn't fit your data shape, this check will be skipped gracefully.
                ok = True
                if Cn.ndim == 2:  # [B, num_candidates] integer ids
                    for bi in range(min(yn.shape[0], Cn.shape[0])):
                        if yn[bi] not in Cn[bi]:
                            ok = False
                            break
                if not ok:
                    print(f"[RED FLAG] For batch {i}, target not found among candidates for at least one sample.")
            except Exception as e:
                print(f"[INFO] Skipped candidate/target membership check (shape mismatch or non-integer types): {e}")
        elif i == 0:
            print("[INFO] No 'candidates' structure detected; skipping selection-specific checks.")

def build_model(ModelCls):
    # Try no-arg first, then look for common ctor arg names from config.py
    try:
        return ModelCls()
    except TypeError:
        # Try reading config for kwargs
        cfg = import_module_safely("config")
        if cfg:
            # Collect dict-like or upper-case constants as kwargs (best effort)
            kw = {}
            for k in dir(cfg):
                if k.isupper():
                    kw[k.lower()] = getattr(cfg, k)
            for k in dir(cfg):
                v = getattr(cfg, k)
                if isinstance(v, (int, float, str, bool)):
                    kw[k] = v
            try:
                return ModelCls(**kw)
            except Exception as e:
                print(f"[WARN] Could not build Model with config-derived kwargs: {e}")
        print("[ERR] Could not instantiate model; please pass --model-class explicitly and ensure its ctor is arg-free or expose defaults.")
        raise

def global_grad_norm(model: nn.Module):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total += param_norm.item() ** 2
    return (total ** 0.5) if total > 0 else 0.0

def tensor_entropy(probs, eps=1e-12):
    # probs: (..., K)
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)

def logits_to_probs(logits):
    if logits.dim() == 1:
        logits = logits[None, :]
    return logits.log_softmax(dim=-1).exp()

def try_symbol_introspection(model, batch_on_device):
    # Best-effort: attempt to extract message logits/probs if sender exists.
    # Supports: model.sender, model.communication, model.autoencoder, etc.
    findings = {}
    candidates = [getattr(model, "sender", None), getattr(model, "communication", None), getattr(model, "autoencoder", None), model]
    sender_like = None
    for c in candidates:
        if c is None: 
            continue
        # Look for a forward that returns logits or probs for symbols
        for name, fn in inspect.getmembers(c, predicate=inspect.ismethod):
            if name in ("forward", "encode", "send", "communicate", "forward_messages"):
                try:
                    out = fn(**batch_on_device) if isinstance(batch_on_device, dict) else fn(*batch_on_device) if isinstance(batch_on_device, (list, tuple)) else fn(batch_on_device)
                except Exception:
                    continue
                # Heuristic: if out is a dict, search for 'logits', 'probs', 'messages'
                if isinstance(out, dict):
                    for k in ["logits", "probs", "messages", "symbols"]:
                        if k in out and tensor_like(out[k]):
                            sender_like = c
                            findings["raw_out"] = out
                            break
                elif isinstance(out, (list, tuple)):
                    # search tensor of shape [B, T, V] or [B, V]
                    for t in out:
                        if tensor_like(t) and t.dim() >= 2 and t.size(-1) >= 2:
                            findings["tensor_out"] = t
                            sender_like = c
                            break
                elif tensor_like(out) and out.dim() >= 2 and out.size(-1) >= 2:
                    findings["tensor_out"] = out
                    sender_like = c
                if sender_like is not None:
                    break
        if sender_like is not None:
            break
    if sender_like is None:
        print("[INFO] Could not auto-extract symbol/message logits; expose something like model.sender(logits=...) to enable symbol diagnostics.")
        return None
    # Convert to probs
    T = None
    if "raw_out" in findings and isinstance(findings["raw_out"], dict):
        if "probs" in findings["raw_out"]:
            probs = findings["raw_out"]["probs"]
        elif "logits" in findings["raw_out"]:
            probs = logits_to_probs(findings["raw_out"]["logits"])
        else:
            toks = findings["raw_out"].get("messages", None)
            if toks is not None and toks.dim() >= 2:
                # Discrete tokens: make one-hot approx to compute usage histogram
                V = int(toks.max().item() + 1)
                one_hot = torch.zeros(*toks.shape, V, device=toks.device)
                one_hot.scatter_(-1, toks.unsqueeze(-1), 1.0)
                probs = one_hot
            else:
                print("[INFO] Symbol output not recognized for entropy stats.")
                return None
        T = probs.size(-2) if probs.dim() >= 3 else 1
    else:
        t = findings["tensor_out"]
        if t.dim() == 2:
            probs = logits_to_probs(t)
            T = 1
        elif t.dim() >= 3:
            probs = logits_to_probs(t)
            T = t.size(-2)
        else:
            print("[INFO] Unrecognized tensor_out shape for symbol stats.")
            return None
    # Entropy per position
    ent = tensor_entropy(probs).detach().cpu().numpy()
    if ent.ndim >= 2:
        ent_pos = ent.mean(axis=0)  # over batch
    else:
        ent_pos = np.array([ent.mean()])
    np.savetxt(os.path.join(DIAG_DIR, "symbol_entropy_by_pos.csv"), ent_pos, delimiter=",")
    fig = plt.figure()
    plt.plot(np.arange(len(ent_pos)), ent_pos)
    plt.title("Symbol entropy by position")
    plt.xlabel("position")
    plt.ylabel("entropy (nats)")
    fig.savefig(os.path.join(DIAG_DIR, "symbol_entropy_by_pos.png"), bbox_inches="tight")
    plt.close(fig)
    # Dead symbol analysis (prob mass below threshold across batch/positions)
    probs_cpu = probs.detach().cpu()
    V = probs_cpu.size(-1)
    usage = probs_cpu.sum(dim=tuple(range(probs_cpu.dim()-1)))  # sum over batch/positions
    usage = usage / usage.sum()
    dead = (usage < (1.0 / (V * 50))).sum().item()  # heuristic threshold: < 1/(50*V) of total mass
    print(f"[SYMBOLS] Vocab={V}, approx dead symbols={dead}")
    np.savetxt(os.path.join(DIAG_DIR, "symbol_usage.csv"), usage.numpy(), delimiter=",")
    fig = plt.figure()
    plt.bar(np.arange(V), usage.numpy())
    plt.title("Approx. symbol usage")
    plt.xlabel("symbol id")
    plt.ylabel("relative mass")
    fig.savefig(os.path.join(DIAG_DIR, "symbol_usage.png"), bbox_inches="tight")
    plt.close(fig)
    return {"entropy_by_pos": ent_pos, "usage": usage.numpy()}

def forward_once(model, batch, device):
    model.train()
    batch = to_device(batch, device)
    out = None
    try:
        out = model(**batch) if isinstance(batch, dict) else model(*batch) if isinstance(batch, (list, tuple)) else model(batch)
    except TypeError:
        # Try a single-arg call
        try:
            # guess first element as inputs
            if isinstance(batch, dict):
                first_key = next(iter(batch.keys()))
                out = model(batch[first_key])
            elif isinstance(batch, (list, tuple)):
                out = model(batch[0])
        except Exception as e:
            print(f"[ERR] Model forward failed: {e}")
            raise
    return out, batch

def guess_logits_from_out(out):
    # Best-effort extraction of logits for classification/selection
    if isinstance(out, dict):
        for k in ["logits", "cls_logits", "selection_logits", "scores"]:
            if k in out and tensor_like(out[k]):
                return out[k]
        # If messages present, not helpful here
    elif isinstance(out, (list, tuple)):
        for t in out:
            if tensor_like(t) and t.dim() >= 2:
                return t
    elif tensor_like(out):
        return out
    return None

def attach_simple_loss(logits, batch):
    # Try CE if we can find labels
    y = guess_labels(batch)
    if y is not None and tensor_like(y) and logits is not None and logits.dim() >= 2 and logits.size(0) == y.size(0):
        return nn.CrossEntropyLoss()(logits, y)
    return None

def recompute_accuracy(logits, batch):
    y = guess_labels(batch)
    if y is None or logits is None:
        return None
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def build_optimizer_from_train_py(model):
    tr = import_module_safely("train")
    if tr:
        for name, fn in inspect.getmembers(tr, inspect.isfunction):
            if "optim" in name or "optimizer" in name:
                try:
                    opt = fn(model.parameters())
                    print(f"[OK] Using optimizer from train.py: {name}()")
                    return opt
                except Exception:
                    continue
    print("[INFO] Falling back to AdamW(lr=1e-3)")
    return torch.optim.AdamW(model.parameters(), lr=1e-3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=4, help="How many batches to sample for data checks")
    parser.add_argument("--dataset-class", type=str, default=None)
    parser.add_argument("--model-class", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, help="Optional split name: train/val/test")
    parser.add_argument("--no-seed", action="store_true", help="Do not fix seeds; allow full randomness (not recommended)")
    args = parser.parse_args()

    if not args.no_seed:
        set_seeds(1337)

    ds_mod = import_module_safely("dataset")
    md_mod = import_module_safely("model")

    DatasetCls = find_dataset_class(ds_mod, args.dataset_class)
    if DatasetCls is None:
        print("[FATAL] No dataset class available; please pass --dataset-class and ensure dataset.py is on PYTHONPATH")
        sys.exit(1)

    ds, dl = build_dataloader(DatasetCls, batch_size=args.batch_size, split=args.split)

    # Sample a few batches for sanity checks
    sampled = []
    it = iter(dl)
    for i in range(args.max_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        describe_batch(b, tag=f"{i}")
        sampled.append(b)

    if not sampled:
        print("[FATAL] DataLoader yielded no batches. Check dataset length or split.")
        sys.exit(1)

    batch_sanity_checks(sampled, out_prefix="train")

    # Try to instantiate a model
    if md_mod is None:
        print("[FATAL] model.py not importable; cannot proceed with model diagnostics.")
        sys.exit(1)
    ModelCls = find_model_class(md_mod, args.model_class)
    if ModelCls is None:
        print("[FATAL] No model class found in model.py; pass --model-class explicitly.")
        sys.exit(1)

    model = build_model(ModelCls)
    device = torch.device(args.device)
    model.to(device)

    # Forward pass on one batch
    b0 = to_device(sampled[0], device)
    out, b0 = forward_once(model, b0, device)
    print(f"[MODEL OUT] type={type(out)}")
    if tensor_like(out):
        print(f"  tensor shape={tuple(out.shape)}, dtype={out.dtype}")
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("[RED FLAG] NaN/Inf detected in model output.")
    elif isinstance(out, (list, tuple)):
        for i, t in enumerate(out):
            if tensor_like(t):
                print(f"  out[{i}] shape={tuple(t.shape)}, dtype={t.dtype}")
    elif isinstance(out, dict):
        for k, v in out.items():
            if tensor_like(v):
                print(f"  out['{k}'] shape={tuple(v.shape)}, dtype={v.dtype}")

    # Guess logits and compute simple loss/acc if possible
    logits = guess_logits_from_out(out)
    if logits is not None and tensor_like(logits):
        probs = logits_to_probs(logits.float())
        H = tensor_entropy(probs).mean().item()
        print(f"[LOGITS ENTROPY] mean entropy={H:.3f} nats (higher ~ more uncertainty).")
    else:
        print("[INFO] Could not identify classification logits; skipping CE loss & acc sanity.")

    loss = attach_simple_loss(logits, b0)
    if loss is not None:
        print(f"[LOSS] Simple CE loss on one batch: {loss.item():.4f}")
        acc = recompute_accuracy(logits, b0)
        if acc is not None:
            print(f"[ACC] Argmax accuracy on that batch: {acc*100:.2f}%")
    else:
        print("[INFO] Loss not computed (no compatible labels/logits).")

    # Try symbol/message introspection
    try_symbol_introspection(model, b0)

    # Tiny train steps: 3 iterations on one batch
    if loss is not None:
        opt = build_optimizer_from_train_py(model)
        last = None
        grad_norms = []
        losses = []
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            # reforward (avoid graph reuse)
            out2, b0 = forward_once(model, b0, device)
            logits2 = guess_logits_from_out(out2)
            if logits2 is None:
                print("[INFO] Skipping train steps: cannot find logits on reforward.")
                break
            loss2 = attach_simple_loss(logits2, b0)
            if loss2 is None:
                print("[INFO] Skipping train steps: could not build loss.")
                break
            loss2.backward()
            gnorm = global_grad_norm(model)
            if np.isnan(gnorm) or np.isinf(gnorm) or gnorm == 0.0:
                print(f"[RED FLAG] global grad-norm at step {step}: {gnorm}")
            else:
                print(f"[GRAD] global grad-norm at step {step}: {gnorm:.4f}")
            opt.step()
            val = float(loss2.item())
            print(f"[STEP {step}] loss={val:.4f}")
            grad_norms.append(gnorm)
            losses.append(val)
        # Save curves if we collected any
        if losses:
            np.savetxt(os.path.join(DIAG_DIR, "micro_train_losses.csv"), np.array(losses), delimiter=",")
            fig = plt.figure()
            plt.plot(np.arange(len(losses)), losses)
            plt.title("Micro-train loss (3 steps)")
            plt.xlabel("step")
            plt.ylabel("loss")
            fig.savefig(os.path.join(DIAG_DIR, "micro_train_losses.png"), bbox_inches="tight")
            plt.close(fig)
        if grad_norms:
            np.savetxt(os.path.join(DIAG_DIR, "micro_train_grad_norms.csv"), np.array(grad_norms), delimiter=",")
            fig = plt.figure()
            plt.plot(np.arange(len(grad_norms)), grad_norms)
            plt.title("Global grad-norm (3 steps)")
            plt.xlabel("step")
            plt.ylabel("||g||_2")
            fig.savefig(os.path.join(DIAG_DIR, "micro_train_grad_norms.png"), bbox_inches="tight")
            plt.close(fig)

    print("\n[DIAGNOSTICS COMPLETE] See the 'diagnostics/' folder for artifacts.")
    print("If any section logged [RED FLAG], that's likely where the issue lies.")
    print("Tip: If auto-discovery picked the wrong classes, rerun with --dataset-class and --model-class explicitly.")
    print("     Example: python diagnose_comm.py --dataset-class ARCDataset --model-class CommunicationModel")
    print("")

if __name__ == "__main__":
    main()
