# Documentation review notes

[Draft article](../content/posts/small-vllm-kv-cache-manager.md)

These notes are for the author and reviewers. The images supplied so far are included unchanged. Captions explain the intended behavior, but image corrections should be completed before publication.

## Diagram coverage

Diagrams 1–6 and 8–10 are supplied. Diagram 7, covering prefix misses and the logits limit, still needs an image.

## Diagram 1 revision notes for the author

- Extend each Shared Prefix bracket across both blue blocks, T0–T7.
- Replace the Hash 2 = Hash 1 + tokens label with H2 = hash((H1, (T4, T5, T6, T7))).
- Make both T0–T3 blocks point clearly to Physical KV Block 1, and both T4–T7 blocks point to Physical KV Block 2. Remove arrows that could suggest different token blocks share one physical block.
The supplied image is included unchanged; the caption explains the intended mapping.

## Diagram 3 revision notes for the author

- Route "last reference released" to the right/back (MRU) end of the queue.
- Show the allocation arrow leaving the left/front (LRU) entry, even though the Reassigned box is drawn on the right.
- Remove the duplicate MRU label beneath the queue; retain one LRU label over the left end and one MRU label over the right end.
- A matching-prefix arrow may leave any queue entry because a hit removes its specific block by ID.
The supplied image is included unchanged; the caption specifies the queue operations.

## Diagram 4 revision notes for the author

- Release every reference held by C, not only references to its unshared blocks. Only zero-reference blocks become free.
- Replace "shared/critical" labels with explicit reference states: "shared: still referenced by another request" or "unshared: reference count reaches zero." There is no separate critical-block protection in the supplied code.
- Show another request pointing to any shared surviving physical blocks; sharing means the same physical storage, not separate copies in each row.
- After release, show C.block_ids = []. Draw surviving cached contents separately from C's current ownership, and show C reclaiming them when it resumes.
- A uses the same physical block that was released; draw the transfer or recolor that block instead of implying a new physical block was created.
- The preemption stage appears twice in the supplied image. A single sequence of Before, Release and reassign, and Resume would make the progression easier to follow.
The supplied image is included unchanged; the caption explains the reference-count and cache-survival rules.

## Diagram 6 revision note for the author

In the extending panel, remove the arrow from [I J K L] to H1. Route [I J K L] and the already stored H1 as separate inputs to H2. The new tokens do not update H1.
The supplied image is included unchanged.

## Diagram 8 revision notes for the author

- Keep cached = [17, 42] in all three panels. The method does not remove entries from this input or stop treating 17 as cached when it becomes active.
- In the Before and middle panels, associate the old prefix P with free block 23, not block 42 or 17. Remove P only when the second loop reassigns 23.
- Retain H0 for block 17 and H1 for block 42; their cached KV contents are reused.
- Remove arrows between the final page-table blocks if they suggest that 23 or 31 points to 17 or 42. Each entry independently maps a logical token block to physical storage.
The supplied image is included unchanged; the caption follows the actual code and worked example.

## Diagram 7 illustration plan

*Draw this in Excalidraw:* use two panels. In the first, draw the request's hashes in order: H0, H1, H2, H3. Show arrows from H0 to physical block 17 and H1 to physical block 42. Mark H2 as “miss: stop” and H3 as “not checked,” even if its entry exists. Label the returned list `[17, 42]`. In the second panel, draw an eight-token prompt as two four-token blocks. Mark the first “reuse cached KV” and the second “process again to obtain final-token logits.” Show `(8 - 1) // 4 = 1` above that panel.

## Code behavior to check while completing the series

- Duplicate prefix registrations: `_register_full_blocks` uses `setdefault`, while eviction removes the old hash without verifying which block it points to. Follow this case through the caller and decide whether the implementation needs an ownership check.
- Registration and readiness: allocation can register blocks for an upcoming forward pass. The scheduler and attention walkthroughs must explain when KV writes happen relative to another request reading shared blocks. Registration alone does not prove readiness.
- Preemption progress: the supplied `Scheduler.schedule` does reset `num_computed_tokens` after releasing a victim. Include that caller behavior when writing the scheduler chapter.
- Installation: document actual dependency versions and verify a clean-environment setup. The current repository snapshot has no dependency manifest.
- Execution examples: CLI spellings were checked against `main`; model and GPU examples were not executed during this documentation preparation.

## Scope of this starter

The architecture overview was grounded in the repository source. Foundations and KV explanations were adapted from the existing draft. The inventory was generated from Python syntax trees; remaining methods have not received individual walkthroughs or a runtime correctness audit.

## Website integration

- The article uses the existing frontmatter and `content/posts` loader. Adding the file makes it appear on `/writing/` when deployed.
- Proposed route: `/blog/small-vllm-kv-cache-manager/`.
- Image paths use `/assets/img/small-vllm/`, backed by files under `public/assets/img/small-vllm/`.
- Internal reading links resolve to existing GPTQ, AWQ, and speculative-decoding posts. The GPTQ link points readers to Parts 7 and 8 for packing and unpacking.
- Explicit section anchors are included because the current Markdown renderer does not assign heading IDs.
- The series concerns `small-vLLM`, the CUDA-oriented repository provided for this walkthrough. The existing Tiny-vLLM project page links to a separate CPU/MPS repository and is left unchanged.
- Keep the pull request in draft until image revisions, the missing Diagram 7, and final content review are complete. The deployment workflow publishes pushes to `main`.

## Preservation and validation

- The website body retains the attached blog's wording, examples, comments, and draft placeholders. Only the title placement, section anchors, image paths, and full-size image links are adapted for the site.
- The small-vLLM README links to the website blog. All nine diagram files match both supplied archives byte for byte.
- `npm ci` and `npm run build` passed. The static export includes the article, writing-list entry, and sitemap entry.
- Export checks passed for all nine images, all 19 table-of-contents anchors, and the existing GPTQ, AWQ, and speculative-decoding routes. Chrome preview confirmed the article layout, inline diagrams, and full-size image navigation.
