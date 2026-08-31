#!/usr/bin/env python3
"""Train two Medusa-1 heads and trace greedy verification step by step.

This is an educational implementation, not a speed benchmark.  It uses one
Hugging Face causal LM containing:

    frozen Transformer backbone + original LM head + K Medusa heads

The verifier intentionally expands complete candidate paths into a batch.  That
makes every target probability, equality test, rejection, and accepted prefix
easy to inspect.  The selected verification node supplies the draft logits for
the next iteration, so only the initial prompt needs a separate prefill.
Production Medusa replaces the duplicated path batch with a flattened candidate
tree, custom position IDs, a tree-attention mask, and KV-cache reuse.  The
acceptance calculation itself is the same.

Example:

    pip install -r examples/requirements-medusa-demo.txt

    # Train only the heads on the included tiny demonstration corpus, then run.
    python examples/medusa_smol_lm_demo.py \
        --train-steps 300 \
        --prompt "The small cat"

    # Reuse the saved heads without training again.
    python examples/medusa_smol_lm_demo.py \
        --train-steps 0 \
        --prompt "Once upon a time"

The included corpus is deliberately tiny so the mechanics are runnable and
inspectable.  It is not sufficient for a useful general-purpose checkpoint.
Train the same loss on a real corpus before evaluating latency or acceptance.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"

# Repetition and predictable phrasing make a short head-training demonstration
# possible.  This is intentionally not presented as a real training dataset.
DEMO_CORPUS = [
    "The small cat sat on the warm mat and watched the quiet room.",
    "The small cat sat on the blue rug and watched the morning light.",
    "The young dog ran across the green field and chased the red ball.",
    "The young dog ran across the garden and followed the little bird.",
    "Once upon a time, a curious fox lived beside a quiet forest.",
    "Once upon a time, a kind robot helped people in a small village.",
    "The sun rose over the hills and filled the valley with golden light.",
    "The moon rose over the lake and covered the water with silver light.",
    "Machine learning models predict tokens from the context they receive.",
    "A language model reads a sequence and predicts the next token.",
    "Speculative decoding proposes tokens and verifies them with the target model.",
    "Medusa adds lightweight prediction heads to a frozen language model.",
    "The first Medusa head predicts two positions into the future.",
    "The second Medusa head predicts three positions into the future.",
    "Candidate tokens are accepted only when the target model verifies them.",
    "When a candidate is rejected, all later tokens on that path are discarded.",
]


class ResidualBlock(nn.Module):
    """The residual block used by a Medusa prediction head."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.SiLU()

        # Start as an identity transformation.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.activation(self.linear(hidden_states))


class MedusaHead(nn.Module):
    """One independent future-token head: residual block -> vocabulary logits."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        initial_lm_head_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.residual = ResidualBlock(hidden_size)
        self.vocab_projection = nn.Linear(hidden_size, vocab_size, bias=False)

        # This is only an initialization.  Every Medusa head owns and trains an
        # independent vocabulary projection after the copy.
        with torch.no_grad():
            self.vocab_projection.weight.copy_(initial_lm_head_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.vocab_projection(self.residual(hidden_states))


class SingleModelMedusa(nn.Module):
    """A single target LLM augmented with independent Medusa-1 heads."""

    def __init__(self, base_model: nn.Module, number_of_heads: int) -> None:
        super().__init__()
        self.base_model = base_model

        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

        hidden_size = int(base_model.config.hidden_size)
        vocab_size = int(base_model.config.vocab_size)
        original_lm_head = base_model.get_output_embeddings()
        if original_lm_head is None:
            raise ValueError("The base model does not expose an output embedding layer.")

        initial_weight = original_lm_head.weight.detach().float()
        self.medusa_heads = nn.ModuleList(
            MedusaHead(hidden_size, vocab_size, initial_weight)
            for _ in range(number_of_heads)
        )

    @property
    def number_of_heads(self) -> int:
        return len(self.medusa_heads)

    def base_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the frozen backbone and return the final hidden state."""
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return outputs.hidden_states[-1].float()

    def all_logits(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return original next-token logits and every Medusa-head logit tensor."""
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1].float()
            medusa_logits = [head(hidden_states) for head in self.medusa_heads]
        return outputs.logits.float(), medusa_logits


@dataclass(frozen=True)
class CandidatePath:
    token_ids: tuple[int, ...]
    draft_log_probability: float


@dataclass(frozen=True)
class VerificationCheck:
    medusa_head_index: int
    candidate_token_id: int
    target_token_id: int
    candidate_probability: float
    target_probability: float
    matched: bool


@dataclass(frozen=True)
class VerificationResult:
    candidate: CandidatePath
    committed_length: int
    checks: tuple[VerificationCheck, ...]

    @property
    def accepted_medusa_tokens(self) -> int:
        """Number of verified draft tokens after the always-committed root."""

        return self.committed_length - 1


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shifted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """At position t, compare a head's logits with the label at t + shift."""
    if shift <= 0 or logits.size(1) <= shift:
        raise ValueError(f"Invalid shift={shift} for sequence length={logits.size(1)}")

    shifted_logits = logits[:, :-shift, :].contiguous()
    shifted_labels = labels[:, shift:].contiguous()
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
    )


def build_token_blocks(
    tokenizer,
    texts: Sequence[str],
    sequence_length: int,
    repetitions: int,
) -> torch.Tensor:
    """Create fixed-length token blocks from the small demonstration corpus."""
    separator_id = tokenizer.eos_token_id
    if separator_id is None:
        separator_id = tokenizer.bos_token_id
    if separator_id is None:
        raise ValueError("The tokenizer must define either eos_token_id or bos_token_id.")

    token_stream: list[int] = []
    for _ in range(repetitions):
        shuffled_texts = list(texts)
        random.shuffle(shuffled_texts)
        for text in shuffled_texts:
            token_stream.extend(tokenizer.encode(text, add_special_tokens=False))
            token_stream.append(separator_id)

    number_of_blocks = len(token_stream) // sequence_length
    if number_of_blocks == 0:
        raise ValueError("The training corpus is too short for the sequence length.")

    usable_tokens = token_stream[: number_of_blocks * sequence_length]
    return torch.tensor(usable_tokens, dtype=torch.long).view(
        number_of_blocks, sequence_length
    )


def cache_frozen_hidden_states(
    medusa_model: SingleModelMedusa,
    token_blocks: torch.Tensor,
    device: torch.device,
    cache_batch_size: int,
) -> torch.Tensor:
    """Evaluate the frozen backbone once so head-only training is inexpensive."""
    cached_batches: list[torch.Tensor] = []
    medusa_model.base_model.eval()

    for start in range(0, token_blocks.size(0), cache_batch_size):
        block_batch = token_blocks[start : start + cache_batch_size].to(device)
        hidden_states = medusa_model.base_hidden_states(block_batch)
        cached_batches.append(hidden_states.cpu())

    return torch.cat(cached_batches, dim=0)


def train_medusa_heads(
    medusa_model: SingleModelMedusa,
    cached_hidden_states: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    head_decay: float,
) -> None:
    """Train only auxiliary heads with shifts 2, 3, ..., K+1."""
    medusa_model.medusa_heads.train()
    optimizer = torch.optim.AdamW(
        medusa_model.medusa_heads.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    number_of_examples = labels.size(0)
    report_every = max(1, steps // 10)

    for step in range(1, steps + 1):
        indices = torch.randint(0, number_of_examples, (batch_size,))
        hidden_batch = cached_hidden_states[indices].to(device)
        label_batch = labels[indices].to(device)

        per_head_losses = []
        for head_index, head in enumerate(medusa_model.medusa_heads):
            logits = head(hidden_batch)
            shift = head_index + 2
            per_head_losses.append(shifted_cross_entropy(logits, label_batch, shift))

        weighted_loss = sum(
            head_decay ** (head_index + 1) * loss
            for head_index, loss in enumerate(per_head_losses)
        )

        optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(medusa_model.medusa_heads.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % report_every == 0 or step == steps:
            raw_losses = ", ".join(
                f"head_{index}={loss.item():.4f}"
                for index, loss in enumerate(per_head_losses)
            )
            print(
                f"training step {step:>4}/{steps}: "
                f"weighted={weighted_loss.item():.4f}, {raw_losses}"
            )

    medusa_model.medusa_heads.eval()


def save_medusa_heads(
    checkpoint_path: Path,
    medusa_model: SingleModelMedusa,
    base_model_name: str,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_model_name": base_model_name,
            "number_of_heads": medusa_model.number_of_heads,
            "medusa_heads": medusa_model.medusa_heads.state_dict(),
        },
        checkpoint_path,
    )
    print(f"saved Medusa heads to {checkpoint_path}")


def load_medusa_heads(
    checkpoint_path: Path,
    medusa_model: SingleModelMedusa,
    expected_base_model_name: str,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["base_model_name"] != expected_base_model_name:
        raise ValueError(
            "Checkpoint base model mismatch: "
            f"{checkpoint['base_model_name']!r} != {expected_base_model_name!r}"
        )
    if checkpoint["number_of_heads"] != medusa_model.number_of_heads:
        raise ValueError(
            "Checkpoint Medusa-head count mismatch: "
            f"{checkpoint['number_of_heads']} != {medusa_model.number_of_heads}"
        )
    medusa_model.medusa_heads.load_state_dict(checkpoint["medusa_heads"])
    print(f"loaded Medusa heads from {checkpoint_path}")


def token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return repr(text)


def sequence_text(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def top_k_tokens(
    logits: torch.Tensor,
    top_k: int,
) -> list[tuple[int, float, float]]:
    """Return (token_id, probability, log_probability) for the top-k tokens."""
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(log_probabilities, k=top_k)
    return [
        (int(token_id), math.exp(float(log_probability)), float(log_probability))
        for token_id, log_probability in zip(indices.tolist(), values.tolist())
    ]


def build_candidate_paths(
    base_last_logits: torch.Tensor,
    medusa_last_logits: Sequence[torch.Tensor],
    top_k: int,
) -> tuple[list[CandidatePath], tuple[int, float], list[list[tuple[int, float]]]]:
    """Take the Cartesian product of independent future-token predictions."""
    base_choice = top_k_tokens(base_last_logits, top_k=1)[0]
    root_token_id, root_probability, root_log_probability = base_choice

    choices_for_product: list[list[tuple[int, float]]] = []
    printable_head_choices: list[list[tuple[int, float]]] = []
    for logits in medusa_last_logits:
        choices = top_k_tokens(logits, top_k=top_k)
        choices_for_product.append(
            [(token_id, log_probability) for token_id, _, log_probability in choices]
        )
        printable_head_choices.append(
            [(token_id, probability) for token_id, probability, _ in choices]
        )

    candidate_paths = []
    for future_choices in itertools.product(*choices_for_product):
        future_token_ids = tuple(token_id for token_id, _ in future_choices)
        future_log_probability = sum(log_probability for _, log_probability in future_choices)
        candidate_paths.append(
            CandidatePath(
                token_ids=(root_token_id, *future_token_ids),
                draft_log_probability=root_log_probability + future_log_probability,
            )
        )

    return (
        candidate_paths,
        (root_token_id, root_probability),
        printable_head_choices,
    )


def verify_candidate_paths_batched(
    medusa_model: SingleModelMedusa,
    prefix_ids: torch.Tensor,
    candidate_paths: Sequence[CandidatePath],
) -> tuple[
    list[VerificationResult],
    int,
    tuple[int, ...],
    torch.Tensor,
    list[torch.Tensor],
]:
    """Verify full paths in a batch and return the longest greedy match.

    If the prefix length is P and a candidate is [root, h0, h1], then:

      * root was already selected from target_logits[P - 1]
      * verifier_logits[P] must predict h0
      * verifier_logits[P + 1] must predict h1

    Rejection occurs at the first candidate != argmax(target distribution).
    """
    if prefix_ids.shape[0] != 1:
        raise ValueError("This transparent demo supports one request at a time.")
    if not candidate_paths:
        raise ValueError("At least one candidate path is required.")

    path_length = len(candidate_paths[0].token_ids)
    if any(len(path.token_ids) != path_length for path in candidate_paths):
        raise ValueError("All candidate paths must have the same length.")

    device = prefix_ids.device
    path_tensor = torch.tensor(
        [path.token_ids for path in candidate_paths],
        dtype=torch.long,
        device=device,
    )
    repeated_prefix = prefix_ids.expand(len(candidate_paths), -1)
    verifier_input_ids = torch.cat((repeated_prefix, path_tensor), dim=1)

    with torch.no_grad():
        verifier_outputs = medusa_model.base_model(
            input_ids=verifier_input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        verifier_logits = verifier_outputs.logits.float()
        verifier_hidden_states = verifier_outputs.hidden_states[-1].float()

    prefix_length = prefix_ids.size(1)
    results: list[VerificationResult] = []

    for path_index, candidate in enumerate(candidate_paths):
        # The root is the original target LM's greedy token, so it is accepted.
        committed_length = 1
        checks: list[VerificationCheck] = []

        for path_position in range(1, path_length):
            predictor_position = prefix_length + path_position - 1
            target_distribution = F.log_softmax(
                verifier_logits[path_index, predictor_position], dim=-1
            )
            target_token_id = int(target_distribution.argmax())
            candidate_token_id = candidate.token_ids[path_position]
            matched = candidate_token_id == target_token_id

            checks.append(
                VerificationCheck(
                    medusa_head_index=path_position - 1,
                    candidate_token_id=candidate_token_id,
                    target_token_id=target_token_id,
                    candidate_probability=float(
                        target_distribution[candidate_token_id].exp()
                    ),
                    target_probability=float(target_distribution[target_token_id].exp()),
                    matched=matched,
                )
            )

            if not matched:
                break
            committed_length += 1

        results.append(
            VerificationResult(
                candidate=candidate,
                committed_length=committed_length,
                checks=tuple(checks),
            )
        )

    # Longer accepted prefix wins.  Draft probability only breaks ties between
    # paths that commit the same greedy target prefix.
    best_result_index = max(
        range(len(results)),
        key=lambda index: (
            results[index].committed_length,
            results[index].candidate.draft_log_probability,
        ),
    )
    best_result = results[best_result_index]
    accepted_token_ids = best_result.candidate.token_ids[: best_result.committed_length]

    # A causal hidden state at this position sees exactly the newly accepted
    # prefix.  Its original-LM logits provide the next root, while the Medusa
    # heads provide the future-token candidates for the next iteration.
    final_accepted_position = prefix_length + best_result.committed_length - 1
    next_base_logits = verifier_logits[best_result_index, final_accepted_position]
    final_accepted_hidden_state = verifier_hidden_states[
        best_result_index, final_accepted_position
    ].unsqueeze(0)
    with torch.no_grad():
        next_medusa_logits = [
            head(final_accepted_hidden_state).squeeze(0)
            for head in medusa_model.medusa_heads
        ]

    return (
        results,
        best_result_index,
        accepted_token_ids,
        next_base_logits,
        next_medusa_logits,
    )


def print_draft_trace(
    tokenizer,
    root_choice: tuple[int, float],
    per_head_choices: Sequence[Sequence[tuple[int, float]]],
    candidate_paths: Sequence[CandidatePath],
) -> None:
    root_token_id, root_probability = root_choice
    print("\nDRAFT")
    print(
        "  original LM head, x[t+1]: "
        f"{token_text(tokenizer, root_token_id)}  p={root_probability:.6f}"
    )
    for head_index, choices in enumerate(per_head_choices):
        formatted_choices = ", ".join(
            f"{token_text(tokenizer, token_id)} (p={probability:.6f})"
            for token_id, probability in choices
        )
        print(
            f"  Medusa head {head_index}, x[t+{head_index + 2}] top-k: "
            f"{formatted_choices}"
        )

    print("  Cartesian-product paths:")
    for path_index, path in enumerate(candidate_paths):
        print(
            f"    path {path_index}: "
            f"{sequence_text(tokenizer, path.token_ids)!r}"
        )


def print_verification_trace(
    tokenizer,
    prefix_ids: torch.Tensor,
    candidate_paths: Sequence[CandidatePath],
    results: Sequence[VerificationResult],
    best_result_index: int,
) -> None:
    path_length = len(candidate_paths[0].token_ids)
    verifier_shape = (
        len(candidate_paths),
        prefix_ids.size(1) + path_length,
    )
    print("\nVERIFY WITH THE SAME BASE LLM")
    print(
        "  educational verifier input shape: "
        f"[{verifier_shape[0]}, {verifier_shape[1]}]"
    )
    print(
        "  This batch duplicates complete paths for clarity. Optimized Medusa "
        "packs shared nodes with tree attention."
    )

    prefix_list = prefix_ids[0].tolist()
    for path_index, result in enumerate(results):
        selected_marker = "  <-- longest accepted path" if path_index == best_result_index else ""
        print(
            f"\n  path {path_index}: "
            f"{sequence_text(tokenizer, result.candidate.token_ids)!r}"
            f"{selected_marker}"
        )
        root_id = result.candidate.token_ids[0]
        print(
            "    root: original LM selected "
            f"{token_text(tokenizer, root_id)} -> ACCEPT"
        )

        for check_index, check in enumerate(result.checks):
            # At check i, the target conditions on prefix + root + all earlier
            # Medusa tokens, but not on the candidate currently being checked.
            context_ids = prefix_list + list(result.candidate.token_ids[: check_index + 1])
            context_tail = sequence_text(tokenizer, context_ids[-12:])
            outcome = "ACCEPT" if check.matched else "REJECT"
            equality = "==" if check.matched else "!="
            print(f"    head {check.medusa_head_index} under context {context_tail!r}")
            print(
                "      candidate "
                f"{token_text(tokenizer, check.candidate_token_id)} "
                f"p_target(candidate)={check.candidate_probability:.6f}"
            )
            print(
                "      target argmax "
                f"{token_text(tokenizer, check.target_token_id)} "
                f"p_target(argmax)={check.target_probability:.6f}"
            )
            print(f"      token ids: {check.candidate_token_id} {equality} {check.target_token_id} -> {outcome}")

        rejected = result.committed_length < len(result.candidate.token_ids)
        if rejected:
            print("    stop checking this path after its first rejection")
        accepted_text = sequence_text(
            tokenizer,
            result.candidate.token_ids[: result.committed_length],
        )
        print(
            "    accepted_medusa_tokens="
            f"{result.accepted_medusa_tokens}; "
            f"committed_tokens={result.committed_length}: {accepted_text!r}"
        )


def medusa_greedy_generate(
    medusa_model: SingleModelMedusa,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Generate greedily while printing every Medusa acceptance calculation."""
    current_ids = prompt_ids
    generated_token_ids: list[int] = []
    committed_lengths: list[int] = []
    iteration = 0

    # Initial prompt prefill.  Later draft logits come directly from the final
    # accepted node of the preceding verification pass.
    prefill_base_logits, prefill_medusa_logits = medusa_model.all_logits(current_ids)
    base_last_logits = prefill_base_logits[0, -1]
    medusa_last_logits = [logits[0, -1] for logits in prefill_medusa_logits]

    while len(generated_token_ids) < max_new_tokens:
        iteration += 1
        print("\n" + "=" * 88)
        print(f"ITERATION {iteration}")
        print(f"accepted prefix: {sequence_text(tokenizer, current_ids[0].tolist())!r}")

        candidate_paths, root_choice, per_head_choices = build_candidate_paths(
            base_last_logits=base_last_logits,
            medusa_last_logits=medusa_last_logits,
            top_k=top_k,
        )
        print_draft_trace(
            tokenizer,
            root_choice,
            per_head_choices,
            candidate_paths,
        )

        (
            results,
            best_result_index,
            accepted_token_ids,
            next_base_logits,
            next_medusa_logits,
        ) = verify_candidate_paths_batched(
            medusa_model=medusa_model,
            prefix_ids=current_ids,
            candidate_paths=candidate_paths,
        )
        print_verification_trace(
            tokenizer,
            current_ids,
            candidate_paths,
            results,
            best_result_index,
        )

        remaining_budget = max_new_tokens - len(generated_token_ids)
        tokens_to_commit = list(accepted_token_ids[:remaining_budget])
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None and eos_token_id in tokens_to_commit:
            eos_position = tokens_to_commit.index(eos_token_id)
            tokens_to_commit = tokens_to_commit[: eos_position + 1]

        if not tokens_to_commit:
            raise RuntimeError("Verifier made no progress; the target root must be accepted.")

        committed_lengths.append(len(tokens_to_commit))
        generated_token_ids.extend(tokens_to_commit)
        commit_tensor = torch.tensor(
            [tokens_to_commit],
            dtype=torch.long,
            device=current_ids.device,
        )
        current_ids = torch.cat((current_ids, commit_tensor), dim=1)

        # These were already calculated at the final accepted verification node.
        # No additional backbone call is required to begin the next iteration.
        base_last_logits = next_base_logits
        medusa_last_logits = next_medusa_logits

        print("\nCOMMIT")
        print(f"  token ids: {tokens_to_commit}")
        print(f"  text: {sequence_text(tokenizer, tokens_to_commit)!r}")
        print(f"  new accepted prefix: {sequence_text(tokenizer, current_ids[0].tolist())!r}")

        if eos_token_id is not None and eos_token_id in tokens_to_commit:
            break

    return current_ids, generated_token_ids, committed_lengths


def baseline_greedy_generate(
    base_model: nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> list[int]:
    """Reference greedy decoding used to prove the verifier preserved output."""
    current_ids = prompt_ids.clone()
    generated_token_ids: list[int] = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = base_model(
                input_ids=current_ids,
                use_cache=False,
                return_dict=True,
            ).logits
        next_token_id = int(logits[0, -1].argmax())
        generated_token_ids.append(next_token_id)
        next_token = torch.tensor(
            [[next_token_id]],
            dtype=torch.long,
            device=current_ids.device,
        )
        current_ids = torch.cat((current_ids, next_token), dim=1)
        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break

    return generated_token_ids


def parameter_count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Medusa-1 heads and print greedy verification/rejection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="The small cat")
    parser.add_argument("--number-of-heads", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--corpus-repetitions", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--head-decay", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/smollm2-135m-medusa-heads.pt"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.number_of_heads < 1:
        raise ValueError("--number-of-heads must be at least 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.train_steps < 0:
        raise ValueError("--train-steps cannot be negative")
    if args.sequence_length <= args.number_of_heads + 1:
        raise ValueError("--sequence-length is too short for the requested heads")

    number_of_paths = args.top_k ** args.number_of_heads
    if number_of_paths > 256:
        raise ValueError(
            f"This educational verifier would create {number_of_paths} paths; "
            "reduce --top-k or --number-of-heads."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)

    print(f"loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    )
    base_model.eval()

    medusa_model = SingleModelMedusa(
        base_model=base_model,
        number_of_heads=args.number_of_heads,
    ).to(device)
    medusa_model.base_model.eval()

    base_parameters = parameter_count(medusa_model.base_model.parameters())
    trainable_parameters = parameter_count(medusa_model.medusa_heads.parameters())
    print(f"frozen base parameters: {base_parameters:,}")
    print(f"trainable Medusa-head parameters: {trainable_parameters:,}")

    if args.train_steps > 0:
        print("building a cached hidden-state training set from the tiny demo corpus")
        token_blocks = build_token_blocks(
            tokenizer=tokenizer,
            texts=DEMO_CORPUS,
            sequence_length=args.sequence_length,
            repetitions=args.corpus_repetitions,
        )
        cached_hidden_states = cache_frozen_hidden_states(
            medusa_model=medusa_model,
            token_blocks=token_blocks,
            device=device,
            cache_batch_size=args.cache_batch_size,
        )
        print(
            "training cache shapes: "
            f"hidden={tuple(cached_hidden_states.shape)}, "
            f"labels={tuple(token_blocks.shape)}"
        )
        train_medusa_heads(
            medusa_model=medusa_model,
            cached_hidden_states=cached_hidden_states,
            labels=token_blocks,
            device=device,
            steps=args.train_steps,
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            head_decay=args.head_decay,
        )
        save_medusa_heads(
            checkpoint_path=args.checkpoint,
            medusa_model=medusa_model,
            base_model_name=args.model,
        )
    else:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(
                f"No checkpoint at {args.checkpoint}. Run once with --train-steps > 0."
            )
        load_medusa_heads(
            checkpoint_path=args.checkpoint,
            medusa_model=medusa_model,
            expected_base_model_name=args.model,
        )

    medusa_model.eval()
    prompt_ids = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    print("\n" + "#" * 88)
    print("MEDUSA GREEDY GENERATION WITH EXPLICIT VERIFICATION")
    print("#" * 88)
    final_ids, medusa_generated_ids, committed_lengths = medusa_greedy_generate(
        medusa_model=medusa_model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
    )

    print("\n" + "#" * 88)
    print("COMPARE AGAINST ORDINARY GREEDY DECODING")
    print("#" * 88)
    baseline_generated_ids = baseline_greedy_generate(
        base_model=medusa_model.base_model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
    )

    medusa_text = sequence_text(tokenizer, medusa_generated_ids)
    baseline_text = sequence_text(tokenizer, baseline_generated_ids)
    print(f"Medusa generated:   {medusa_text!r}")
    print(f"Baseline generated: {baseline_text!r}")
    print(f"committed lengths by iteration: {committed_lengths}")
    print(
        "mean committed tokens per iteration: "
        f"{sum(committed_lengths) / len(committed_lengths):.3f}"
    )

    if medusa_generated_ids != baseline_generated_ids:
        raise RuntimeError(
            "Medusa output differed from ordinary greedy output. "
            "Inspect the printed verification trace."
        )

    print("exact token match: PASS")
    print(f"final text: {sequence_text(tokenizer, final_ids[0].tolist())!r}")


if __name__ == "__main__":
    main()
