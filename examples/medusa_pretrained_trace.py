#!/usr/bin/env python3
"""Trace real Medusa-1 greedy decoding, one verification iteration at a time.

This is inference-only. It downloads the official pretrained checkpoint
``FasterDecoding/medusa-vicuna-7b-v1.3`` and uses the official Medusa tree
decoder. The target model and Medusa heads are already trained.

Run on one A100 (after installing the requirements shown in
``requirements-medusa-pretrained.txt``)::

    python examples/medusa_pretrained_trace.py \
        --prompt "Explain why speculative decoding is lossless." \
        --max-new-tokens 96 \
        --output-json medusa_trace.json

The important distinction in every printed iteration is:

* ``accepted_medusa_tokens``: speculative tokens after the root token whose
  values exactly match the target model under greedy decoding.
* ``committed_tokens``: ``1 + accepted_medusa_tokens``. The extra token is the
  root/ordinary target token, which is always committed.

Consequently, zero accepted Medusa tokens does not mean zero progress. Greedy
Medusa still commits one target-model token in that iteration.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from fastchat.model.model_adapter import get_conversation_template
    from medusa.model.kv_cache import initialize_past_key_values
    from medusa.model.medusa_model import MedusaModel
    from medusa.model.utils import (
        evaluate_posterior,
        generate_candidates,
        generate_medusa_buffers,
        initialize_medusa,
        reset_medusa_mode,
        tree_decoding,
        update_inference_inputs,
    )
except ImportError as exc:  # pragma: no cover - depends on the A100 environment
    raise SystemExit(
        "Missing Medusa runtime dependencies. Follow the installation commands "
        "at the top of examples/requirements-medusa-pretrained.txt.\n"
        f"Original import error: {exc}"
    ) from exc


DEFAULT_MODEL = "FasterDecoding/medusa-vicuna-7b-v1.3"


@dataclass
class IterationTrace:
    """The measurements for one target-model tree verification pass."""

    iteration: int
    candidate_paths: int
    candidate_path_width: int
    tree_nodes_verified: int
    best_candidate: int
    best_path_prefix_matches: list[bool]
    accepted_medusa_tokens: int
    committed_tokens: int
    committed_token_ids: list[int]
    committed_text: str
    rejected_candidate_token_id: int | None
    rejected_candidate_token: str | None
    target_token_id_at_rejection: int | None
    target_token_at_rejection: str | None
    target_probability_at_rejection: float | None
    iteration_ms: float
    running_mean_accepted_medusa_tokens: float
    running_mean_committed_tokens: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and trace an official pretrained Medusa-1 model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Official Medusa checkpoint on Hugging Face.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain speculative decoding to a machine learning engineer.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Stop after at least this many committed tokens (a tree step may overshoot).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=128,
        help="Safety limit on target-model verification iterations.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="A100 supports both. float16 matches the official example.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the machine-readable iteration trace.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be at least 1")
    if not args.prompt.strip():
        raise ValueError("--prompt cannot be empty")


def visible_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token while keeping whitespace visible in terminal output."""

    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return repr(text)


def make_vicuna_prompt(user_prompt: str) -> str:
    """Apply the chat template used by the Vicuna-v1.3 base checkpoint."""

    conversation = get_conversation_template("vicuna")
    conversation.append_message(conversation.roles[0], user_prompt)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def greedy_prefix_matches(
    candidates: torch.Tensor,
    verifier_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the official greedy posterior calculation explicitly.

    ``candidates[:, 0]`` is the root token. For each later candidate token, the
    target token is ``argmax(verifier_logits[:, previous_position])``. Once a
    mismatch occurs, later matches on that path do not count; ``cumprod`` turns
    the raw equality mask into a prefix-validity mask.
    """

    target_tokens = torch.argmax(verifier_logits[:, :-1], dim=-1)
    raw_matches = candidates[:, 1:] == target_tokens
    prefix_matches = torch.cumprod(raw_matches.to(torch.int32), dim=1)
    accepted_per_path = prefix_matches.sum(dim=1)
    return target_tokens, raw_matches, accepted_per_path


def print_iteration(trace: IterationTrace) -> None:
    print(f"\niteration {trace.iteration}")
    print(
        "  verified: "
        f"{trace.tree_nodes_verified} tree nodes -> "
        f"{trace.candidate_paths} padded root-to-leaf paths "
        f"(width={trace.candidate_path_width})"
    )
    print(f"  selected candidate path: {trace.best_candidate}")
    print(
        "  target-match mask after root: "
        f"{trace.best_path_prefix_matches}"
    )
    print(
        "  accepted_medusa_tokens="
        f"{trace.accepted_medusa_tokens}; "
        f"committed_tokens={trace.committed_tokens} "
        "(root + accepted Medusa tokens)"
    )
    print(
        f"  committed ids={trace.committed_token_ids}; "
        f"text={trace.committed_text!r}"
    )

    if trace.rejected_candidate_token_id is not None:
        print(
            "  first rejection: Medusa proposed "
            f"id={trace.rejected_candidate_token_id} "
            f"token={trace.rejected_candidate_token}; target wanted "
            f"id={trace.target_token_id_at_rejection} "
            f"token={trace.target_token_at_rejection} "
            f"(p={trace.target_probability_at_rejection:.4f})"
        )
    else:
        print("  first rejection: none; the entire represented path matched")

    print(
        f"  iteration_ms={trace.iteration_ms:.2f}; "
        "running means: "
        f"accepted_medusa={trace.running_mean_accepted_medusa_tokens:.2f}, "
        f"committed={trace.running_mean_committed_tokens:.2f} tokens/iteration"
    )


@torch.inference_mode()
def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("This pretrained 7B trace requires a CUDA GPU.")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    print(f"loading {args.model} ({args.dtype})")
    model = MedusaModel.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    prompt = make_vicuna_prompt(args.prompt)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.base_model.device)
    prompt_length = input_ids.shape[1]

    medusa_choices = model.get_medusa_choice(model.base_model_name_or_path)
    buffers = generate_medusa_buffers(
        medusa_choices,
        device=model.base_model.device,
    )
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.base_model)

    reset_medusa_mode(model)
    medusa_logits, logits = initialize_medusa(
        input_ids,
        model,
        buffers["medusa_attn_mask"],
        past_key_values,
    )

    traces: list[IterationTrace] = []
    new_token = 0

    for iteration in range(1, args.max_iterations + 1):
        synchronize()
        started = time.perf_counter()

        candidates, tree_candidates = generate_candidates(
            medusa_logits,
            logits,
            buffers["tree_indices"],
            buffers["retrieve_indices"],
            temperature=0,
        )
        next_medusa_logits, verifier_logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            buffers["medusa_position_ids"],
            input_ids,
            buffers["retrieve_indices"],
        )

        target_tokens, raw_matches, accepted_per_path = greedy_prefix_matches(
            candidates,
            verifier_logits,
        )
        best_candidate, accept_length = evaluate_posterior(
            verifier_logits,
            candidates,
            temperature=0,
        )
        best_index = int(best_candidate.item())
        accepted = int(accept_length.item())

        reconstructed_accept = int(accepted_per_path.max().item())
        reconstructed_best = (
            0
            if reconstructed_accept == 0
            else int(accepted_per_path.argmax().item())
        )
        assert accepted == reconstructed_accept
        assert best_index == reconstructed_best

        path_indices = buffers["retrieve_indices"][best_index]
        real_path_width = int((path_indices >= 0).sum().item())
        real_future_tokens = real_path_width - 1
        prefix_mask = torch.cumprod(
            raw_matches[best_index].to(torch.int32), dim=0
        ).bool()
        displayed_mask = prefix_mask[:real_future_tokens].tolist()

        committed_ids = candidates[
            best_index, : accepted + 1
        ].tolist()
        committed_text = tokenizer.decode(
            committed_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        rejected_id: int | None = None
        rejected_token: str | None = None
        target_id: int | None = None
        target_token: str | None = None
        target_probability: float | None = None
        if accepted < real_future_tokens:
            rejected_id = int(candidates[best_index, accepted + 1].item())
            target_id = int(target_tokens[best_index, accepted].item())
            rejected_token = visible_token(tokenizer, rejected_id)
            target_token = visible_token(tokenizer, target_id)
            target_probability = float(
                torch.softmax(verifier_logits[best_index, accepted].float(), dim=-1)[
                    target_id
                ].item()
            )

        input_ids, logits, medusa_logits, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            buffers["retrieve_indices"],
            outputs,
            verifier_logits,
            next_medusa_logits,
            new_token,
            past_key_values_data,
            current_length_data,
        )

        synchronize()
        iteration_ms = (time.perf_counter() - started) * 1000
        total_accepted = sum(t.accepted_medusa_tokens for t in traces) + accepted
        total_committed = sum(t.committed_tokens for t in traces) + accepted + 1
        trace = IterationTrace(
            iteration=iteration,
            candidate_paths=candidates.shape[0],
            candidate_path_width=candidates.shape[1],
            tree_nodes_verified=tree_candidates.shape[1],
            best_candidate=best_index,
            best_path_prefix_matches=displayed_mask,
            accepted_medusa_tokens=accepted,
            committed_tokens=accepted + 1,
            committed_token_ids=committed_ids,
            committed_text=committed_text,
            rejected_candidate_token_id=rejected_id,
            rejected_candidate_token=rejected_token,
            target_token_id_at_rejection=target_id,
            target_token_at_rejection=target_token,
            target_probability_at_rejection=target_probability,
            iteration_ms=iteration_ms,
            running_mean_accepted_medusa_tokens=total_accepted / iteration,
            running_mean_committed_tokens=total_committed / iteration,
        )
        traces.append(trace)
        print_iteration(trace)

        generated = input_ids[0, prompt_length:]
        if tokenizer.eos_token_id in generated.tolist():
            print("\nstop: generated EOS")
            break
        if new_token >= args.max_new_tokens:
            print(f"\nstop: committed {new_token} tokens")
            break
    else:
        print(f"\nstop: reached --max-iterations={args.max_iterations}")

    generated_ids = input_ids[0, prompt_length:].tolist()
    acceptance_histogram: dict[str, int] = {}
    for trace in traces:
        key = str(trace.accepted_medusa_tokens)
        acceptance_histogram[key] = acceptance_histogram.get(key, 0) + 1

    summary: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "formatted_prompt_tokens": prompt_length,
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ),
        "iterations": len(traces),
        "accepted_medusa_tokens": sum(
            trace.accepted_medusa_tokens for trace in traces
        ),
        "committed_tokens": sum(trace.committed_tokens for trace in traces),
        "acceptance_length_histogram": acceptance_histogram,
        "mean_accepted_medusa_tokens": (
            sum(trace.accepted_medusa_tokens for trace in traces) / len(traces)
        ),
        "mean_committed_tokens": (
            sum(trace.committed_tokens for trace in traces) / len(traces)
        ),
        "trace": [asdict(trace) for trace in traces],
    }

    print("\nsummary")
    print(json.dumps({key: value for key, value in summary.items() if key != "trace"}, indent=2))

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.output_json}")

    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_trace(args)


if __name__ == "__main__":
    main()
