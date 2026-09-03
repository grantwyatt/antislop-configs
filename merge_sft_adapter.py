#!/usr/bin/env python3
"""Merge copublish's SFT LoRA adapter (trained locally on the P40, see
local-ai-gateway/sft.train.ps1 in the main copublish repo) onto the FULL
PRECISION base model, producing a standalone bf16 model that auto-antislop's
FTPO step can point at directly (as a local path) instead of a stock
third-party base -- this is the 2026-08-29 reordering decision in effect:
SFT establishes voice first, FTPO strips remaining tics from THAT result
second, so FTPO's base has to be copublish's own SFT'd model, not
sam-paech/gemma-3-27b-it-antislop.

Deliberately loads unsloth/gemma-3-27b-it (the full bf16 checkpoint, ~55GB)
rather than the unsloth/gemma-3-27b-it-bnb-4bit checkpoint the adapter was
actually TRAINED against -- this is the standard, correct way to merge a
QLoRA adapter: the LoRA delta is dtype/quantization-agnostic math applied to
the same underlying weights, so merging onto the full-precision original
gives a clean bf16 result without any of bitsandbytes' own (lossy, not
universally supported) 4-bit merge-and-unload path.

Deliberately run HERE, on the RunPod pod (A100 80GB), not on the P40 box:
the P40 has only 24GB VRAM (can't hold a 55GB model at all) and this
project already has a real, documented incident of a full-bf16 Gemma-3-27B
load freezing that exact Windows machine via system-RAM/pagefile exhaustion
(see project memory, project_local_deslop_plan -- the "on-the-fly bf16
quantization crash" saga from the first sft.train.ps1 debugging session).
Merging where there's 80GB of real VRAM headroom sidesteps that risk
entirely, and this pod is already being paid for the FTPO run regardless.

Usage:
  python merge_sft_adapter.py --adapter /workspace/copublish_sft_adapter \
      --output /workspace/copublish_sft_merged
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", default="unsloth/gemma-3-27b-it",
        help="Full-precision base model -- must be the SAME base the "
             "adapter was trained against (the bnb-4bit variant of this), "
             "just not pre-quantized here.",
    )
    parser.add_argument(
        "--adapter", type=Path, required=True,
        help="Path to the LoRA adapter directory (copied from the P40 "
             "box's sft-training/output/final-adapter, or pulled from an "
             "HF repo it was pushed to -- see the runbook).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Where to write the merged bf16 model -- point "
             "configs_copublish.yaml's model_id at this exact path "
             "afterward.",
    )
    args = parser.parse_args()

    if not args.adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {args.adapter}")

    print(f"==> Loading base model (full precision): {args.base_model}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"==> Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, str(args.adapter))

    print("==> Merging adapter into base weights")
    model = model.merge_and_unload()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"==> Saving merged model to {args.output}")
    model.save_pretrained(str(args.output), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output))

    # Gemma-3 is a vision-capable architecture -- vLLM auto-detects this
    # from config.json and always tries to build an image processor for
    # it at startup, even for a text-only run, and hard-fails if
    # preprocessor_config.json isn't present (real failure 2026-09-03:
    # "Can't load image processor for ...", OSError, at vLLM server
    # startup -- this is a DIFFERENT spot than the similar Gemma-3
    # vision-capability issue already hit during SFT training, which was
    # worked around inside trl's SFTTrainer, not fixed at the model-
    # directory level -- that fix doesn't cover this). AutoTokenizer/
    # AutoModelForCausalLM above never touch this file, so it has to be
    # pulled in separately. Lightweight -- config only, no model weights.
    print("==> Saving processor config (image processor -- needed for vLLM's Gemma-3 multimodal auto-detection, even though this is text-only)")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(str(args.output))

    print("==> Done. Sanity check before running FTPO:")
    print(f"    ls -la {args.output}")
    print("    (expect config.json, tokenizer files, and .safetensors shards)")
    print(f"    du -sh {args.output}  (expect roughly the size of the bf16 base, ~55GB)")


if __name__ == "__main__":
    main()
