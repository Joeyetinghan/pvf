from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_PRIORITY_WORDS = [
    # --- 1. Control Flow & Structure (The Skeleton) ---
    "def",
    "class",
    "return",
    "import",
    "from",
    "as",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "break",
    "continue",
    "pass",
    "try",
    "except",
    "raise",
    "finally",
    "with",
    "assert",
    "lambda",
    "yield",
    "global",
    "nonlocal",
    "del",
    # --- 2. Logical & Comparison Operators (The Decision Makers) ---
    "and",
    "or",
    "not",
    "is",
    "in",
    "==",
    "!=",
    ">=",
    "<=",
    ">",
    "<",  # Python specific operators
    # --- 3. Critical Punctuation (The Syntax Glue) ---
    ":",  # Block start
    "(",
    ")",  # Function calls / grouping
    "[",
    "]",  # Lists / indexing
    "{",
    "}",  # Dicts / Sets
    ",",  # Separator
    ".",  # Attribute access (VERY important for methods)
    "->",  # Type hinting (common in HumanEval prompts)
    # --- 4. Constants & Booleans ---
    "self",
    "None",
    "True",
    "False",
    # --- 5. Built-in Functions (Action Anchors) ---
    # Decoding these early constrains the expected arguments immediately.
    "len",
    "range",
    "enumerate",
    "zip",
    "sorted",
    "reversed",
    "int",
    "float",
    "str",
    "list",
    "dict",
    "set",
    "tuple",
    "bool",
    "sum",
    "max",
    "min",
    "abs",
    "round",
    "pow",
    "divmod",
    "print",
    "input",
    "open",
    "map",
    "filter",
    "all",
    "any",
    "isinstance",
    "issubclass",
    "type",
    # --- 6. Common Methods (Data Structure Anchors) ---
    # These often appear right after a variable and dot (e.g., my_list.append)
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "clear",
    "index",
    "count",
    "sort",
    "reverse",  # List
    "get",
    "keys",
    "values",
    "items",
    "update",  # Dict
    "add",
    "union",
    "intersection",
    "difference",  # Set
    "split",
    "join",
    "strip",
    "replace",
    "format",
    "startswith",
    "endswith",
    "lower",
    "upper",  # String
    # --- 7. Reasoning / Math / Comment Anchors ---
    "#",  # Start of comment (Essential for your "Plan First" strategy)
    "=",
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "+=",
    "-=",
    "*=",
    "/=",
]


def _shift_logits(logits: torch.Tensor) -> torch.Tensor:
    """Right-shift logits by one position (required by Dream architecture)."""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def add_gumbel_noise(logits, t):
    if t == 0:
        return logits
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** t)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    gen_length=128,
    block_length=32,
    temp=0.0,
    cfg=0.0,
    mask_id=None,
    conf_thresh=0.9,
    filter_thresh=0.2,
    hub_strategy="heuristic",
    priority_token_ids=None,
    seed=42,
    priority_confidence_threshold=None,
    cumulative_fallback=False,
    cumulative_fallback_num=4,
    unlock_next_block_threshold=None,
    priority_batch_inference=False,
    priority_batch_num=4,
    priority_batch_mode="cumulative",
    priority_selection_criterion="count",
    priority_confidence_upper_bound=0.65,
    cumulative_fallback_order="position",
    high_conf_topk=None,
    track_token_labels=False,
    skip_priority_verification=False,
    model_type="llada",
    track_history=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = prompt.device
    B = prompt.shape[0]
    shift = model_type == "dream"

    if mask_id is None:
        mask_id = tokenizer.mask_token_id

    x = torch.full(
        (B, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.shape[1]] = prompt
    prompt_len = prompt.shape[1]
    history = [x[0].detach().cpu().clone().tolist()] if track_history else None

    num_blocks = gen_length // block_length

    total_calls = 0
    total_strategy_selected = 0  # Only priority tokens
    total_fallback_selected = 0  # Look-ahead fallback tokens
    priority_confidences = []  # Track confidences of selected priority tokens
    # Track priority token stats: {token_str: {"count": N, "high_conf_increases": [...]}}
    priority_token_stats = {}
    priority_set = set(priority_token_ids) if priority_token_ids else set()
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    # Track priority consistency stats
    priority_consistency_passed = 0  # Number of priority tokens that pass consistency check
    priority_consistency_total = 0  # Total priority tokens tested in batch inference

    # Track cumulative fallback path length distribution
    cumulative_fallback_length_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }  # 3+ grouped together

    # Track how often we enter the priority batch inference fallback path
    # (priority batch inference done but no priority token passed, so use baseline for cumulative fallback)
    priority_batch_fallback_count = 0

    # Track how each token was committed (for labeling output)
    # Labels: "H" = high-conf, "P" = priority, "F" = fallback, "X" = force, None = prompt
    token_labels = [None] * (prompt.shape[1] + gen_length) if track_token_labels else None

    # Cache for batch inference results to skip initial forward pass in subsequent iterations
    # When we commit a token at position P, the batch result at that index gives predictions for x_new
    cached_x0 = None
    cached_x0_p = None

    # Outer loop over blocks
    for block_iter in range(num_blocks):
        b_start = prompt_len + block_iter * block_length
        b_end = b_start + block_length
        # Inner loop for current block
        inner_it = 0
        while inner_it < 256:
            mask_index = x == mask_id

            cur_mask = torch.zeros_like(mask_index)
            cur_mask[:, b_start:b_end] = True
            eligible = mask_index & cur_mask

            if not eligible.any():
                break

            # Unlock next block if masked positions in current block fall below threshold
            # Only initialize tracking variables if the option is enabled
            original_b_end = None
            next_block_unlocked = False
            if unlock_next_block_threshold is not None:
                original_b_end = b_end  # Track original block boundary
                num_masked_in_block = eligible.sum().item()
                if num_masked_in_block <= unlock_next_block_threshold:
                    # Extend b_end to include next block
                    new_b_end = min(b_end + block_length, prompt_len + gen_length)
                    next_block_unlocked = True
                    cur_mask[:, b_start:new_b_end] = True
                    eligible = mask_index & cur_mask

            # 1. Forward Pass
            use_cache = cached_x0 is not None and cached_x0_p is not None

            if use_cache:
                # Use cached predictions from previous iteration's batch inference
                x0 = cached_x0
                x0_p = cached_x0_p
                # Clear cache after use
                cached_x0 = None
                cached_x0_p = None
            else:
                if cfg > 0:
                    x_in = torch.cat(
                        [x, x.clone().masked_fill_(x != mask_id, mask_id)],
                        dim=0,
                    )
                    logits_all = model(x_in).logits
                    if shift:
                        logits_all = _shift_logits(logits_all)
                    logits, un_logits = logits_all[:B], logits_all[B:]
                    logits = un_logits + (cfg + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
                    if shift:
                        logits = _shift_logits(logits)
                total_calls += 1

                # 2. Predictions
                x0 = torch.argmax(add_gumbel_noise(logits, temp), dim=-1)
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

            # 3. Commit High Confidence
            if high_conf_topk is not None and high_conf_topk > 0:
                # Top-k selection: get top-k highest confidence positions among eligible
                commit = torch.zeros_like(mask_index)
                for b in range(B):
                    eligible_for_commit = mask_index[b] & cur_mask[b]
                    eligible_indices = torch.where(eligible_for_commit)[0]
                    if len(eligible_indices) > 0:
                        eligible_confs = x0_p[b, eligible_indices]
                        k = min(high_conf_topk, len(eligible_indices))
                        _, topk_local_indices = torch.topk(eligible_confs, k)
                        topk_global_indices = eligible_indices[topk_local_indices]
                        commit[b, topk_global_indices] = True
            else:
                # Threshold-based selection (default)
                commit = mask_index & (x0_p >= conf_thresh) & cur_mask

            # Track high-confidence commits for labeling
            if track_token_labels:
                for b in range(B):
                    for pos in torch.where(commit[b])[0]:
                        token_labels[pos.item()] = "H"

            # 4. Hub / Heuristic Strategy
            hub_commit = torch.zeros_like(commit)

            # Candidates for fallback (uses confidence_filter_threshold)
            candidates_mask = eligible & (x0_p >= filter_thresh)
            candidates_mask = candidates_mask & (~commit)

            # Priority candidates (only uses priority_confidence_threshold, not filter_thresh)
            priority_candidates_mask = eligible & (x0_p < priority_confidence_upper_bound) & (~commit)
            if priority_confidence_threshold is not None:
                priority_candidates_mask = priority_candidates_mask & (x0_p >= priority_confidence_threshold)

            if priority_candidates_mask.any() or candidates_mask.any():
                for b in range(B):
                    selected_idx = None
                    selection_type = None  # "priority" or "fallback"

                    # --- HEURISTIC LOGIC ---
                    if hub_strategy == "heuristic":
                        # First, get priority candidates for this batch element
                        priority_cand_indices = torch.where(priority_candidates_mask[b])[0]

                        priority_tokens = []  # List of (idx, conf) tuples - initialize before conditional
                        if len(priority_cand_indices) > 0:
                            # Collect all priority tokens with their confidences

                            for idx in priority_cand_indices:
                                tok_id = x0[b, idx].item()
                                conf = x0_p[b, idx].item()

                                is_priority = False

                                # Check 1: Explicit Priority Set
                                if priority_set and tok_id in priority_set:
                                    is_priority = True

                                # Check 2: Capital Letter (Start of Sentence)
                                if not is_priority:
                                    tok_str = tokenizer.decode([tok_id])
                                    s = tok_str.strip()
                                    if s and s[0].isupper():
                                        is_priority = True

                                # # Check 3: EOS Token
                                # if not is_priority and eos_token_id is not None and tok_id == eos_token_id:
                                #     is_priority = True

                                if is_priority:
                                    priority_tokens.append((idx, conf))

                            if len(priority_tokens) > 0:
                                # Sort by confidence (highest first)
                                priority_tokens.sort(key=lambda x: x[1], reverse=True)

                                if priority_batch_inference and len(priority_tokens) >= 1:
                                    num_priority_test = min(priority_batch_num, len(priority_tokens))
                                    priority_indices = [pt[0] for pt in priority_tokens[:num_priority_test]]

                                    if priority_batch_mode == "individual":
                                        # Each trajectory with one priority token, pick the one that boosts most confidence
                                        # Include baseline (index 0) for consistency check
                                        batch_size_prio = 1 + num_priority_test  # baseline + individual tests
                                        x_batch_prio = x[b : b + 1].repeat(batch_size_prio, 1)

                                        # Fill commit positions for all sequences
                                        commit_positions = torch.where(commit[b])[0]
                                        for pos in commit_positions:
                                            x_batch_prio[:, pos] = x0[b, pos]

                                        # For each batch sequence (except baseline at index 0), fill ONE priority token
                                        for i in range(num_priority_test):
                                            x_batch_prio[i + 1, priority_indices[i]] = x0[b, priority_indices[i]]

                                        with torch.no_grad():
                                            logits_batch_prio = model(x_batch_prio).logits
                                        if shift:
                                            logits_batch_prio = _shift_logits(logits_batch_prio)
                                        total_calls += 1
                                        # Get predictions and confidences
                                        x0_batch_prio = torch.argmax(logits_batch_prio, dim=-1)
                                        p_batch_prio = F.softmax(logits_batch_prio, dim=-1)
                                        x0_p_batch_prio = torch.gather(
                                            p_batch_prio,
                                            -1,
                                            x0_batch_prio.unsqueeze(-1),
                                        ).squeeze(-1)

                                        # Get baseline high confidence positions for consistency check
                                        # IMPORTANT: must also check mask_index to avoid overwriting already-committed tokens
                                        baseline_high_conf_mask_prio = (x0_p_batch_prio[0] >= conf_thresh) & cur_mask[b] & mask_index[b]
                                        baseline_high_conf_positions_prio = torch.where(baseline_high_conf_mask_prio)[0]

                                        # Find best priority token: one that boosts most confidence overall AND is consistent
                                        best_i_prio = -1
                                        best_high_conf_count_prio = -1

                                        for i in range(num_priority_test):
                                            # Track total priority tokens tested
                                            priority_consistency_total += 1

                                            # Check: priority token position must still be over priority threshold
                                            prio_pos = priority_indices[i]
                                            prio_conf = x0_p_batch_prio[i + 1, prio_pos].item()

                                            if priority_confidence_threshold is None or prio_conf >= priority_confidence_threshold:
                                                # Consistency check: predictions at baseline high-conf positions must match
                                                # Skip this check if skip_priority_verification is True
                                                tokens_match = True
                                                if not skip_priority_verification:
                                                    for pos in baseline_high_conf_positions_prio:
                                                        if pos == prio_pos:
                                                            pass
                                                        if x0_batch_prio[0, pos] != x0_batch_prio[i + 1, pos]:
                                                            tokens_match = False
                                                            break

                                                if tokens_match:
                                                    # Track passed consistency (only when verification is enabled)
                                                    if not skip_priority_verification:
                                                        priority_consistency_passed += 1
                                                    # Score this trajectory based on criterion
                                                    if priority_selection_criterion == "confidence_sum":
                                                        # Sum of confidence values for all masked positions in block
                                                        masked_in_block = cur_mask[b] & mask_index[b]
                                                        score = (
                                                            x0_p_batch_prio[
                                                                i + 1,
                                                                masked_in_block,
                                                            ]
                                                            .sum()
                                                            .item()
                                                        )
                                                    else:  # "count" (default)
                                                        # Count of high-confidence positions
                                                        prio_high_conf_mask = (x0_p_batch_prio[i + 1] >= conf_thresh) & cur_mask[b]
                                                        score = prio_high_conf_mask.sum().item()

                                                    if score > best_high_conf_count_prio:
                                                        best_high_conf_count_prio = score
                                                        best_i_prio = i

                                        # Commit the best priority token
                                        if best_i_prio >= 0:
                                            selected_idx = priority_indices[best_i_prio]
                                            selection_type = "priority"
                                            # Cache batch result for next iteration (skip initial forward pass)
                                            # The batch index is best_i_prio + 1 (index 0 is baseline)
                                            cached_x0 = x0_batch_prio[best_i_prio + 1 : best_i_prio + 2].expand(B, -1).clone()
                                            cached_x0_p = x0_p_batch_prio[best_i_prio + 1 : best_i_prio + 2].expand(B, -1).clone()
                                        else:
                                            # Fall back: no consistent priority token found
                                            pass
                                else:
                                    # Standard: select highest confidence priority token
                                    selected_idx = priority_tokens[0][0]
                                    selection_type = "priority"

                    # --- Fallback Logic ---
                    if selected_idx is None:
                        total_calls += 1
                        cand_indices = torch.where(candidates_mask[b])[0]
                        if len(cand_indices) == 0:
                            continue

                        if hub_strategy == "heuristic":
                            # Special case: we tried priority batch inference but no priority token passed
                            # Use the baseline from batch inference (with high-conf revealed) for cumulative fallback
                            if (
                                len(priority_tokens) > 0
                                and priority_batch_inference
                                and cumulative_fallback
                                and "x0_batch_prio" in dir()
                                and "x0_p_batch_prio" in dir()
                            ):
                                # Track how often we enter this fallback path
                                priority_batch_fallback_count += 1
                                # Get remaining masked positions from baseline trajectory
                                # These are positions that are still masked AND not high confidence in baseline
                                baseline_remaining_mask = (
                                    eligible[b]
                                    & (~baseline_high_conf_mask_prio)  # Not high conf in baseline
                                    & (x0_p_batch_prio[0] >= filter_thresh)  # Above filter threshold
                                )
                                remaining_indices = torch.where(baseline_remaining_mask)[0]

                                if len(remaining_indices) > 0:
                                    num_cumulative_prio = min(
                                        cumulative_fallback_num,
                                        len(remaining_indices),
                                    )
                                    # Sort by position (leftmost first) or confidence (highest first)
                                    if cumulative_fallback_order == "confidence":
                                        # Sort by confidence (highest first)
                                        remaining_with_conf = [
                                            (
                                                idx.item(),
                                                x0_p_batch_prio[0, idx].item(),
                                            )
                                            for idx in remaining_indices
                                        ]
                                        remaining_with_conf.sort(key=lambda x: x[1], reverse=True)
                                        cumulative_remaining = [idx for idx, _ in remaining_with_conf[:num_cumulative_prio]]
                                    else:
                                        # Sort by position (leftmost first) - default
                                        sorted_remaining = sorted(remaining_indices.tolist())
                                        cumulative_remaining = sorted_remaining[:num_cumulative_prio]

                                    # Create batch: baseline (with high-conf from batch inference) + cumulative sequences
                                    batch_size_cum_prio = 1 + num_cumulative_prio
                                    x_batch_cum_prio = x[b : b + 1].repeat(batch_size_cum_prio, 1)

                                    # Fill high-conf positions from the baseline trajectory (using batch inference results)
                                    baseline_high_conf_positions_list = torch.where(baseline_high_conf_mask_prio)[0]
                                    for pos in baseline_high_conf_positions_list:
                                        x_batch_cum_prio[:, pos] = x0_batch_prio[0, pos]

                                    # Also fill original commit positions
                                    commit_positions_prio = torch.where(commit[b])[0]
                                    for pos in commit_positions_prio:
                                        x_batch_cum_prio[:, pos] = x0[b, pos]

                                    # For each cumulative sequence, fill positions 0 to i from baseline predictions
                                    for i in range(num_cumulative_prio):
                                        for j in range(i + 1):
                                            x_batch_cum_prio[i + 1, cumulative_remaining[j]] = x0_batch_prio[0, cumulative_remaining[j]]

                                    # Forward pass on batch
                                    with torch.no_grad():
                                        logits_batch_cum_prio = model(x_batch_cum_prio).logits
                                    if shift:
                                        logits_batch_cum_prio = _shift_logits(logits_batch_cum_prio)

                                    # Get predictions
                                    x0_batch_cum_prio = torch.argmax(logits_batch_cum_prio, dim=-1)

                                    # Step-by-step verification using baseline predictions
                                    longest_valid_path_prio = 0
                                    for i in range(num_cumulative_prio):
                                        # Check if trajectory[i] predicts same token at position i as baseline
                                        if x0_batch_cum_prio[i, cumulative_remaining[i]] == x0_batch_prio[0, cumulative_remaining[i]]:
                                            longest_valid_path_prio = i + 1
                                        else:
                                            break

                                    # Commit positions from baseline + valid cumulative path
                                    # First: update x0 with baseline predictions so normal update flow works correctly
                                    # Then set hub_commit for those positions

                                    # Update x0 with baseline high-conf predictions within current block

                                    for pos in baseline_high_conf_positions_list:
                                        if pos.item() >= b_start and pos.item() < b_end:
                                            x0[b, pos] = x0_batch_prio[0, pos]
                                            hub_commit[b, pos] = True
                                            # Track baseline high-conf for labeling
                                            if track_token_labels:
                                                token_labels[pos.item()] = "H"

                                    # Update x0 and commit cumulative positions
                                    # Track cumulative fallback path length distribution
                                    path_key = min(longest_valid_path_prio, 3)  # Group 3+ together
                                    cumulative_fallback_length_counts[path_key] = cumulative_fallback_length_counts.get(path_key, 0) + 1

                                    if longest_valid_path_prio > 0:
                                        for j in range(longest_valid_path_prio):
                                            x0[b, cumulative_remaining[j]] = x0_batch_prio[0, cumulative_remaining[j]]
                                            hub_commit[b, cumulative_remaining[j]] = True
                                            # Track cumulative fallback for labeling
                                            if track_token_labels:
                                                token_labels[cumulative_remaining[j]] = "F"
                                        total_fallback_selected += longest_valid_path_prio
                                        # Cache batch result for next iteration
                                        # Use the cumulative prio batch result at longest_valid_path_prio
                                        cached_x0 = x0_batch_cum_prio[longest_valid_path_prio : longest_valid_path_prio + 1].expand(B, -1).clone()
                                        p_batch_cum_prio = F.softmax(logits_batch_cum_prio, dim=-1)
                                        x0_p_batch_cum_prio = torch.gather(
                                            p_batch_cum_prio,
                                            -1,
                                            x0_batch_cum_prio.unsqueeze(-1),
                                        ).squeeze(-1)
                                        cached_x0_p = x0_p_batch_cum_prio[longest_valid_path_prio : longest_valid_path_prio + 1].expand(B, -1).clone()
                            elif cumulative_fallback and len(cand_indices) > 0:
                                # Cumulative fallback with step-by-step verification
                                # Each step verifies that the previous trajectory's prediction at the next position matches
                                num_cumulative = min(cumulative_fallback_num, len(cand_indices))
                                # Sort by position (leftmost first) or confidence (highest first)
                                if cumulative_fallback_order == "confidence":
                                    # Sort by confidence (highest first)
                                    cands_with_conf = [(idx.item(), x0_p[b, idx].item()) for idx in cand_indices]
                                    cands_with_conf.sort(key=lambda x: x[1], reverse=True)
                                    cumulative_cands = [idx for idx, _ in cands_with_conf[:num_cumulative]]
                                else:
                                    # Sort by position (leftmost first) - default
                                    sorted_cands = sorted(cand_indices.tolist())
                                    cumulative_cands = sorted_cands[:num_cumulative]

                                # Create batch: baseline + cumulative sequences
                                batch_size_cum = 1 + num_cumulative
                                x_batch_cum = x[b : b + 1].repeat(batch_size_cum, 1)

                                # Fill commit positions for all sequences
                                commit_positions = torch.where(commit[b])[0]
                                for pos in commit_positions:
                                    x_batch_cum[:, pos] = x0[b, pos]

                                # For each cumulative sequence, fill positions 0 to i
                                for i in range(num_cumulative):
                                    for j in range(i + 1):
                                        x_batch_cum[i + 1, cumulative_cands[j]] = x0[b, cumulative_cands[j]]

                                # Forward pass on batch
                                with torch.no_grad():
                                    logits_batch_cum = model(x_batch_cum).logits
                                if shift:
                                    logits_batch_cum = _shift_logits(logits_batch_cum)

                                # Get predictions
                                x0_batch_cum = torch.argmax(logits_batch_cum, dim=-1)

                                # Step-by-step verification:
                                # Check if seq[i]'s prediction at position cumulative_cands[i] matches x0[b, cumulative_cands[i]]
                                # Longest path that satisfies this condition is chosen
                                longest_valid_path = 0

                                for i in range(num_cumulative):
                                    # Check if trajectory[i] (baseline + positions 0..i-1) predicts the correct token at position i
                                    # For i=0, trajectory[0] is baseline, check baseline's prediction at cumulative_cands[0]
                                    # For i=1, trajectory[1] is baseline+first, check its prediction at cumulative_cands[1]
                                    if x0_batch_cum[i, cumulative_cands[i]] == x0[b, cumulative_cands[i]]:
                                        longest_valid_path = i + 1
                                    else:
                                        break  # Stop at first mismatch

                                # Track cumulative fallback path length distribution
                                path_key = min(longest_valid_path, 3)  # Group 3+ together
                                cumulative_fallback_length_counts[path_key] = cumulative_fallback_length_counts.get(path_key, 0) + 1

                                # Commit all positions in the longest valid path
                                if longest_valid_path > 0:
                                    for j in range(longest_valid_path):
                                        hub_commit[b, cumulative_cands[j]] = True
                                        # Track cumulative fallback for labeling
                                        if track_token_labels:
                                            token_labels[cumulative_cands[j]] = "F"
                                    total_fallback_selected += longest_valid_path
                                    # Cache batch result for next iteration
                                    # Batch index longest_valid_path has all committed positions filled
                                    cached_x0 = x0_batch_cum[longest_valid_path : longest_valid_path + 1].expand(B, -1).clone()
                                    # Need to get x0_p from logits for cumulative batch
                                    p_batch_cum = F.softmax(logits_batch_cum, dim=-1)
                                    x0_p_batch_cum = torch.gather(
                                        p_batch_cum,
                                        -1,
                                        x0_batch_cum.unsqueeze(-1),
                                    ).squeeze(-1)
                                    cached_x0_p = x0_p_batch_cum[longest_valid_path : longest_valid_path + 1].expand(B, -1).clone()

                    if selected_idx is not None:
                        hub_commit[b, selected_idx] = True
                        if selection_type == "priority":
                            total_strategy_selected += 1
                            # Track priority token for labeling
                            if track_token_labels:
                                token_labels[selected_idx.item() if hasattr(selected_idx, "item") else selected_idx] = "P"
                            # Track confidence of committed priority token
                            priority_confidences.append(x0_p[b, selected_idx].item())
                            # Track token stats
                            tok_id = x0[b, selected_idx].item()
                            tok_str = tokenizer.decode([tok_id])
                            # Normalize: strip leading space to merge " token" with "token"
                            tok_str_normalized = tok_str.lstrip(" ") if tok_str.lstrip(" ") else tok_str
                            if tok_str_normalized not in priority_token_stats:
                                priority_token_stats[tok_str_normalized] = {
                                    "count": 0,
                                    "high_conf_increases": [],
                                }
                            priority_token_stats[tok_str_normalized]["count"] += 1
                            # Compute confidence increase as AVERAGE of confidence values for masked positions in block
                            # EXCLUDING commit positions and the selected priority token position
                            masked_in_block = cur_mask[b] & mask_index[b]
                            # Exclude commit positions (already high-conf, filled in both)
                            masked_in_block = masked_in_block & (~commit[b])
                            # Exclude the selected priority token position
                            masked_in_block[selected_idx] = False
                            num_masked_tokens = masked_in_block.sum().item()
                            if (
                                priority_batch_inference
                                and priority_batch_mode == "individual"
                                and "x0_p_batch_prio" in dir()
                                and "best_i_prio" in dir()
                                and best_i_prio >= 0
                            ):
                                # Use batch inference results: sum of confidence in trajectory vs baseline
                                conf_sum_trajectory = x0_p_batch_prio[best_i_prio + 1, masked_in_block].sum().item()
                                conf_sum_baseline = x0_p_batch_prio[0, masked_in_block].sum().item()
                                high_conf_increase = conf_sum_trajectory - conf_sum_baseline
                            else:
                                # For standard mode without batch inference, use sum of current confidences as proxy
                                high_conf_increase = x0_p[b, masked_in_block].sum().item()
                            # Normalize by number of masked tokens
                            if num_masked_tokens > 0:
                                high_conf_increase = high_conf_increase / num_masked_tokens
                            priority_token_stats[tok_str_normalized]["high_conf_increases"].append(high_conf_increase)

                        elif selection_type == "fallback":
                            total_fallback_selected += 1

            # 5. Apply Updates
            final_commit = commit | hub_commit
            # Fallback if stuck (No Priority found AND No High Confidence found AND No Fallback selected)
            if not final_commit.any():
                block_scores = torch.where(eligible, x0_p, -1e9)
                force_idx = torch.argmax(block_scores, dim=1)
                x.scatter_(
                    1,
                    force_idx.unsqueeze(1),
                    x0.gather(1, force_idx.unsqueeze(1)),
                )
                # Track force commit for labeling
                if track_token_labels:
                    for b in range(B):
                        token_labels[force_idx[b].item()] = "X"
            else:
                x[final_commit] = x0[final_commit]

            if track_history:
                history.append(x[0].detach().cpu().clone().tolist())

            inner_it += 1

    avg_priority_conf = np.mean(priority_confidences) if priority_confidences else 0.0

    result = (
        x[0].tolist(),
        total_calls,
        total_strategy_selected,
        total_fallback_selected,
        avg_priority_conf,
        priority_token_stats,
        priority_consistency_passed,
        priority_consistency_total,
        cumulative_fallback_length_counts,
        token_labels,
        priority_batch_fallback_count,
    )
    if track_history:
        return (*result, history)
    return result


@torch.no_grad()
def generate_ablation(
    model,
    tokenizer,
    prompt,
    gen_length=128,
    block_length=32,
    temp=0.0,
    cfg=0.0,
    mask_id=None,
    conf_thresh=0.9,
    filter_thresh=0.2,
    hub_strategy="heuristic",
    priority_token_ids=None,
    seed=42,
    priority_confidence_threshold=None,
    unlock_next_block_threshold=None,
    priority_batch_inference=False,
    priority_batch_num=4,
    priority_batch_mode="cumulative",
    priority_selection_criterion="count",
    priority_confidence_upper_bound=0.65,
    high_conf_topk=None,
    ablation_comparison=False,
    priority_pick_strategy="confidence",
    model_type="llada",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = prompt.device
    B = prompt.shape[0]
    shift = model_type == "dream"

    if mask_id is None:
        mask_id = tokenizer.mask_token_id

    x = torch.full(
        (B, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.shape[1]] = prompt
    prompt_len = prompt.shape[1]

    num_blocks = gen_length // block_length

    total_calls = 0
    total_strategy_selected = 0  # Only priority tokens
    priority_confidences = []  # Track confidences of selected priority tokens
    # Track priority token stats: {token_str: {"count": N, "high_conf_increases": [...]}}
    priority_token_stats = {}
    priority_set = set(priority_token_ids) if priority_token_ids else set()
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    # these are for the summary csv
    total_inner_iterations = 0
    extra_commit_iters = 0  # any selected extra token (planning or ablation) committed
    extra_tokens_committed = 0  # total number of extra tokens committed (should match extra_commit_iters if 1 each time)
    tokens_committed_total = 0  # total tokens filled (high-conf + extra + forced)
    planning_tokens_count = 0  # number of candidates that satisfy (1|2|3) when ablation is OFF

    # Cache for batch inference results to skip initial forward pass in subsequent iterations
    # When we commit a token at position P, the batch result at that index gives predictions for x_new
    cached_x0 = None
    cached_x0_p = None

    # Outer loop over blocks
    for block_iter in range(num_blocks):
        b_start = prompt_len + block_iter * block_length
        b_end = b_start + block_length
        # Inner loop for current block
        inner_it = 0
        while inner_it < 256:
            total_inner_iterations += 1

            mask_index = x == mask_id

            cur_mask = torch.zeros_like(mask_index)
            cur_mask[:, b_start:b_end] = True
            eligible = mask_index & cur_mask

            if not eligible.any():
                break

            # 1. Forward Pass
            use_cache = cached_x0 is not None and cached_x0_p is not None

            if use_cache:
                # Use cached predictions from previous iteration's batch inference
                x0 = cached_x0
                x0_p = cached_x0_p
                # Clear cache after use
                cached_x0 = None
                cached_x0_p = None
            else:
                if cfg > 0:
                    x_in = torch.cat(
                        [x, x.clone().masked_fill_(x != mask_id, mask_id)],
                        dim=0,
                    )
                    logits_all = model(x_in).logits
                    if shift:
                        logits_all = _shift_logits(logits_all)
                    logits, un_logits = logits_all[:B], logits_all[B:]
                    logits = un_logits + (cfg + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
                    if shift:
                        logits = _shift_logits(logits)
                total_calls += 1

                # 2. Predictions
                x0 = torch.argmax(add_gumbel_noise(logits, temp), dim=-1)
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

            # 3. Commit High Confidence
            if high_conf_topk is not None and high_conf_topk > 0:
                # Top-k selection: get top-k highest confidence positions among eligible
                commit = torch.zeros_like(mask_index)
                for b in range(B):
                    eligible_for_commit = mask_index[b] & cur_mask[b]
                    eligible_indices = torch.where(eligible_for_commit)[0]
                    if len(eligible_indices) > 0:
                        eligible_confs = x0_p[b, eligible_indices]
                        k = min(high_conf_topk, len(eligible_indices))
                        _, topk_local_indices = torch.topk(eligible_confs, k)
                        topk_global_indices = eligible_indices[topk_local_indices]
                        commit[b, topk_global_indices] = True
            else:
                # Threshold-based selection (default)
                commit = mask_index & (x0_p >= conf_thresh) & cur_mask

            # 4. Hub / Heuristic Strategy
            hub_commit = torch.zeros_like(commit)

            # Filter priority candidates by confidence bounds
            priority_candidates_mask = eligible & (x0_p < priority_confidence_upper_bound) & (~commit)
            if priority_confidence_threshold is not None:
                priority_candidates_mask = priority_candidates_mask & (x0_p >= priority_confidence_threshold)

            if priority_candidates_mask.any():
                for b in range(B):
                    selected_idx = None
                    selection_type = None  # "priority" or "fallback"

                    # --- HEURISTIC LOGIC ---
                    if hub_strategy == "heuristic":
                        # First, get priority candidates for this batch element
                        priority_cand_indices = torch.where(priority_candidates_mask[b])[0]

                        priority_tokens = []  # List of (idx, conf) tuples - initialize before conditional
                        if len(priority_cand_indices) > 0:
                            # Collect all priority tokens with their confidences
                            for idx in priority_cand_indices:
                                tok_id = x0[b, idx].item()
                                conf = x0_p[b, idx].item()

                                if ablation_comparison == True:
                                    is_priority = True
                                else:
                                    is_priority = False
                                    # Check 1: Explicit Priority Set
                                    if priority_set and tok_id in priority_set:
                                        is_priority = True
                                    # Check 2: Capital Letter (Start of Sentence)
                                    if not is_priority:
                                        tok_str = tokenizer.decode([tok_id])
                                        s = tok_str.strip()
                                        if s and s[0].isupper():
                                            is_priority = True
                                    # Check 3: EOS Token
                                    if not is_priority and eos_token_id is not None and tok_id == eos_token_id:
                                        is_priority = True
                                if is_priority:
                                    priority_tokens.append((idx, conf))

                            if (not ablation_comparison) and (len(priority_tokens) > 0):
                                planning_tokens_count += 1
                            # Fallback: if *no* heuristic priority tokens exist, treat all candidates as priority (ablation-like)
                            if (not ablation_comparison) and (len(priority_tokens) == 0):
                                priority_tokens = [(idx, x0_p[b, idx].item()) for idx in priority_cand_indices]

                            if len(priority_tokens) > 0:
                                # priority_tokens is [(idx, conf), ...] where conf = x0_p[b, idx].item()

                                if priority_pick_strategy == "random":
                                    selected_idx = random.choice([idx for (idx, _) in priority_tokens])
                                else:
                                    # highest confidence
                                    selected_idx = max(priority_tokens, key=lambda t: t[1])[0]

                                selection_type = "priority"

                                # commit exactly this one token
                                hub_commit[b, selected_idx] = True
                                extra_commit_iters += 1
                                extra_tokens_committed += 1
                                sel_conf = float(x0_p[b, selected_idx].item())
                                priority_confidences.append(sel_conf)
                                total_strategy_selected += 1

            # 5. Apply Updates
            final_commit = commit | hub_commit
            # Fallback if stuck (No Priority found AND No High Confidence found AND No Fallback selected)
            if not final_commit.any():
                block_scores = torch.where(eligible, x0_p, -1e9)
                force_idx = torch.argmax(block_scores, dim=1)
                x.scatter_(
                    1,
                    force_idx.unsqueeze(1),
                    x0.gather(1, force_idx.unsqueeze(1)),
                )
                tokens_committed_total += int(B)  # one per sample in batch
            else:
                x[final_commit] = x0[final_commit]

            inner_it += 1

    avg_priority_conf = np.mean(priority_confidences) if priority_confidences else 0.0

    # --- build decode_stats for CSV ---
    decode_stats = {
        "total_inner_iterations": int(total_inner_iterations),
        "extra_commit_iters": int(extra_commit_iters),
        "extra_tokens_committed": int(extra_tokens_committed),
        "planning_tokens_count": int(planning_tokens_count),
        "tokens_committed_total": int(tokens_committed_total),
        "avg_tokens_committed_per_inner_iter": float(tokens_committed_total / max(1, total_inner_iterations)),
        "avg_priority_conf": float(avg_priority_conf),
    }

    return (
        x[0].tolist(),
        total_calls,
        total_strategy_selected,
        avg_priority_conf,
        priority_token_stats,
        decode_stats,
    )
