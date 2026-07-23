# -*- coding: utf-8 -*-
"""
Teacher-low-entropy masked GKD loss for ms-swift.

Mechanism:
- Monkey-patch swift.rlhf_trainers.gkd_trainer.GKDTrainer.generalized_jsd_loss
- Compute per-position TEACHER entropy H = -sum(p_t * log p_t)
  * If teacher_topk_logprobs is provided, entropy is computed over the topk
    subset (renormalized), matching the JSD-on-topk approximation that the
    original loss already uses.
- Keep only the `LOW_ENTROPY_TOP_RATIO` lowest-entropy positions
  (i.e. positions where the teacher is most confident).
- Backprop loss only on kept positions.

Env vars:
  LOW_ENTROPY_TOP_RATIO  (float, default 0.5)   fraction of LOW-entropy teacher tokens to keep
  LOW_ENTROPY_MIN_KEEP   (int,   default 8)     if kept < this, fall back to keep all
  LOW_ENTROPY_ABS_THRESH (float, default 0)     if > 0, keep H_teacher < this (nat); ignores TOP_RATIO
  LOW_ENTROPY_WARMUP     (int,   default 0)     first N steps keep all positions
  DEBUG_LOW_ENTROPY      (0/1,   default 0)     enable debug logging
  DEBUG_LOW_ENTROPY_EVERY(int,   default 50)    log every N calls
  DEBUG_LOW_ENTROPY_FILE (str,   default /tmp/low_entropy_mask_debug.log)

Usage:
  --external_plugins /path/to/custom_loss_scale2_teacher_low.py
  (Do NOT pass --loss_scale; do NOT mount the high-entropy plugin at the
   same time — they patch the same function.)
"""

import os
import sys
import torch
import torch.nn.functional as F


# ------------------------- Config (env) -------------------------
TOP_RATIO   = float(os.environ.get("LOW_ENTROPY_TOP_RATIO", "0.5"))
MIN_KEEP    = int(os.environ.get("LOW_ENTROPY_MIN_KEEP", "8"))
ABS_THRESH  = float(os.environ.get("LOW_ENTROPY_ABS_THRESH", "0"))
WARMUP      = int(os.environ.get("LOW_ENTROPY_WARMUP", "0"))
DEBUG       = os.environ.get("DEBUG_LOW_ENTROPY", "0") == "1"
DEBUG_EVERY = int(os.environ.get("DEBUG_LOW_ENTROPY_EVERY", "50"))
DEBUG_FILE  = os.environ.get("DEBUG_LOW_ENTROPY_FILE", "/tmp/low_entropy_mask_debug.log")


# ------------------------- Logging helpers -------------------------
def _banner(msg: str):
    line = f"[LOW_ENTROPY_MASK] {msg} (pid={os.getpid()})"
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        with open(DEBUG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


_call_count = 0


def _debug(msg: str):
    if not DEBUG:
        return
    global _call_count
    if _call_count % DEBUG_EVERY != 0:
        return
    line = f"[LOW_ENTROPY_MASK#{_call_count}] {msg}"
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        with open(DEBUG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ------------------------- Patched JSD loss -------------------------
def _patched_generalized_jsd_loss(
    self,
    student_logits,
    teacher_logits=None,
    labels=None,
    beta=0.5,
    temperature=1.0,
    chunk_size=512,
    topk=None,
    teacher_topk_logprobs=None,
    teacher_topk_indices=None,
):
    """Drop-in replacement that masks loss to LOW-entropy TEACHER positions."""
    global _call_count
    _call_count += 1

    # ---- Vocab alignment (same as original) ----
    if teacher_logits is not None:
        student_logits, teacher_logits = self._align_vocab_size(student_logits, teacher_logits)

    # ---- Top-k reduction (same as original) ----
    # After this block, teacher_logits is either full logits or topk-logprobs,
    # both shaped [..., V_eff]. We need it to compute teacher entropy below.
    if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
        student_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)
        student_logits = student_logits / temperature
        teacher_logits = teacher_topk_logprobs / temperature
        teacher_is_logprobs = True
        temperature = 1.0
    elif topk is not None and teacher_logits is not None:
        teacher_logits, topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)
        teacher_logits = teacher_logits / temperature
        student_logits = torch.gather(student_logits, dim=-1, index=topk_idx)
        student_logits = student_logits / temperature
        teacher_is_logprobs = False
        temperature = 1.0
    else:
        teacher_is_logprobs = False

    # ---- Flatten with label mask ----
    if labels is not None:
        mask = labels != -100
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
        num_valid = int(mask.sum().item())
    else:
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        num_valid = student_logits.size(0)

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    if num_valid == 0:
        return student_logits.new_zeros(())

    # ---- Pass 1: per-position TEACHER entropy ----
    # Two paths:
    #   teacher_is_logprobs == True  -> teacher_logits is already log p_t over a
    #     topk subset; renormalize within the subset for a proper distribution,
    #     then H = -sum(p * log p).
    #   teacher_is_logprobs == False -> standard log_softmax over the available
    #     last-dim (full vocab or already-topk'd logits).
    with torch.no_grad():
        H_chunks = []
        for start in range(0, num_valid, chunk_size):
            end = min(start + chunk_size, num_valid)
            t_slice = teacher_logits[start:end].float()
            if teacher_is_logprobs:
                # Renormalize log-probs over the topk subset (subset may not sum to 1).
                t_lp = t_slice - torch.logsumexp(t_slice, dim=-1, keepdim=True)
            else:
                t_lp = F.log_softmax(t_slice, dim=-1)
            H = -(t_lp.exp() * t_lp).sum(dim=-1)  # [chunk]
            H_chunks.append(H)
            del t_slice, t_lp, H
        H_all = torch.cat(H_chunks, dim=0)        # [num_valid]
        del H_chunks

    # ---- Decide keep mask: LOW entropy = teacher is confident ----
    global_step = int(getattr(self.state, "global_step", 0))
    keep_all_reason = None

    if global_step < WARMUP:
        keep_mask = torch.ones(num_valid, dtype=torch.bool, device=student_logits.device)
        keep_all_reason = f"warmup({global_step}<{WARMUP})"
    elif ABS_THRESH > 0:
        # Absolute mode: keep positions where teacher entropy is BELOW threshold.
        keep_mask = H_all < ABS_THRESH
        if keep_mask.sum().item() < MIN_KEEP:
            keep_mask = torch.ones_like(keep_mask)
            keep_all_reason = f"kept<{MIN_KEEP}@abs"
    else:
        # Quantile-based: keep BOTTOM TOP_RATIO fraction (lowest entropy).
        try:
            q = max(0.0, min(1.0, TOP_RATIO))   # cut at the TOP_RATIO-th quantile from below
            thr = torch.quantile(H_all, q).item()
        except RuntimeError:
            # quantile size cap: fall back to sort ASCENDING
            sorted_H, _ = torch.sort(H_all, descending=False)
            k = max(1, int(num_valid * TOP_RATIO))
            thr = sorted_H[k - 1].item()
        keep_mask = H_all <= thr
        if keep_mask.sum().item() < MIN_KEEP:
            keep_mask = torch.ones_like(keep_mask)
            keep_all_reason = f"kept<{MIN_KEEP}@quantile"

    n_kept = int(keep_mask.sum().item())
    keep_ratio = n_kept / max(1, num_valid)

    _debug(
        f"num_valid={num_valid} n_kept={n_kept} ratio={keep_ratio:.3f} "
        f"H_teacher[mean={H_all.mean().item():.3f} min={H_all.min().item():.3f} "
        f"max={H_all.max().item():.3f}] "
        f"thr={'-' if keep_all_reason else f'{H_all[keep_mask].max().item():.3f}'} "
        f"fallback={keep_all_reason or '-'} step={global_step} "
        f"src={'topk_logprobs' if teacher_is_logprobs else 'logits'}"
    )

    # ---- Index kept positions only ----
    if n_kept < num_valid:
        idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        student_logits = student_logits.index_select(0, idx).contiguous()
        teacher_logits = teacher_logits.index_select(0, idx).contiguous()
        del idx
    num_valid_int = n_kept
    del H_all, keep_mask

    # ---- Pass 2: JSD loss on kept positions (same math as original) ----
    if beta != 0 and beta != 1:
        beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
        log_beta = torch.log(beta_t)
        log_1_minus_beta = torch.log1p(-beta_t)
    else:
        beta_t = log_beta = log_1_minus_beta = None

    total_loss = student_logits.new_zeros(())
    for start_idx in range(0, num_valid_int, chunk_size):
        end_idx = min(start_idx + chunk_size, num_valid_int)
        s_chunk = student_logits[start_idx:end_idx]
        t_chunk = teacher_logits[start_idx:end_idx]

        s_log_probs = F.log_softmax(s_chunk, dim=-1)
        t_log_probs = F.log_softmax(t_chunk, dim=-1)
        del s_chunk, t_chunk

        if beta == 0:
            jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
        elif beta == 1:
            jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
        else:
            mixture_log_probs = torch.logsumexp(
                torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
            kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
            del mixture_log_probs
            jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
            del kl_teacher, kl_student

        total_loss = total_loss + jsd_chunk.sum()
        del jsd_chunk, s_log_probs, t_log_probs

    return total_loss / num_valid_int


# ------------------------- Apply patch -------------------------
def _apply_patch():
    try:
        from swift.rlhf_trainers.gkd_trainer import GKDTrainer
    except Exception as e:
        _banner(f"FAILED to import GKDTrainer: {e}")
        raise
    GKDTrainer.generalized_jsd_loss = _patched_generalized_jsd_loss
    _banner(
        f"PATCHED GKDTrainer.generalized_jsd_loss (TEACHER-LOW-entropy) | "
        f"TOP_RATIO={TOP_RATIO} MIN_KEEP={MIN_KEEP} ABS_THRESH={ABS_THRESH} "
        f"WARMUP={WARMUP} DEBUG={DEBUG} FILE={DEBUG_FILE}"
    )


_apply_patch()