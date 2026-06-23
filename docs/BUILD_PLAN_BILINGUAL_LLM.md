# BUILD PLAN v2: BILINGUAL REASONING LLM (DUTCH+ENGLISH)
Hardware: Legion Pro 5 16IAX10H — RTX 5070 Ti 12GB GDDR7 / 32GB DDR5 / Core Ultra 9 275HX

This is a revision of the original plan after a 5-reviewer critique (Vasquez/CPT,
Tanaka/GRPO, De Vries/Dutch NLP, Chen/inference, Okafor/safety). Section
"REVIEW OUTCOME" below documents what changed and why.

---

## REVIEW OUTCOME

**The one finding that reorders the whole plan:** RAG + SFT distillation
(Phase 0 + Phase 2a) likely delivers ~80% of end-user value at ~5% of the
compute/calendar cost of full CPT + GRPO. Continual pretraining (Phase 1) and
GRPO (Phase 2b) are no longer default — they are **gated behind measurement**.
Run Phase 2a first, measure, and only pay for Phase 1 / Phase 2b if the
measurement shows a real gap that RAG + SFT doesn't close.

**8 net corrections applied:**

1. Phase 1 LR: `2e-4` → `5e-5` (CPT LR must be 5–10x lower than pretraining LR, not pretraining-scale).
2. Phase 1: add a replay buffer — 10–15% original Qwen3 pretraining-distribution data blended into every batch, or English MMLU regresses.
3. Decontamination is mandatory for every benchmark, in every phase (see Global Requirements).
4. Phase 1: tokenizer retraining is removed as a fallback option entirely — see 1.2.
5. Phase 1: corpus shrunk `50–100B` → `10–20B` tokens, filtered for quality over volume; training method switched from QLoRA to Unsloth full-model CPT (QLoRA can't inject the morphological depth this phase needs).
6. Phase 2b: `G` 4–6 → 8–16, `KL` 0.05 → 0.1–0.2, explicit length-penalty added, reward shaping made less sparse.
7. Evaluation sample sizes bumped to a statistically meaningful floor (see Global Requirements); GDPR retention policy added to the user-memory store (Phase 0.6).
8. Phase 4: Gemma 4 26B A4B MoE removed as an *interactive* production candidate (kept only as a non-interactive research footnote).

**Deferred / not actioned** (reviewed, intentionally excluded from the 8 — flagging in case you want them added):

- Okafor: 500–1000 DPO pairs is too few for counseling-domain alignment (wants 5k, Dutch). Marked "certain" by the reviewer but **not** in the net-8 list you gave me. Not applied — confirm if this should be in scope.
- Okafor: "30 mixed tasks is not a safety evaluation," red-teaming is missing entirely. Same status — flagged, not applied.
- Chen's KV-cache recount (claiming 40 KV-heads): this is the reviewer being wrong, not the plan. Qwen3-14B uses GQA with 8 KV-heads, not 40 query-heads; the original 11.1 GB @ 32k Q8 KV figure stands. No change made.
- Chen: 15% VRAM overhead buffer (12 GB → 10.2 GB effective), and benchmarking KoboldCpp as an alternative to llama.cpp at 32k context — sensible, but left as an implementation note (Phase 4) rather than a numbers rewrite, since it doesn't change any go/no-go threshold.
- De Vries: cultural misalignment — real, low priority for v1.
- De Vries: interlingual homographs — manageable via a language-ID token in the prompt template; monitor, no plan change.
- Tanaka: Critique-GRPO +4.5–5% claim — unverified independently, not planned on.

---

## GLOBAL REQUIREMENTS (apply to every phase below)

- **Decontamination.** Before scoring any benchmark (MATH-500, GSM8K, AIME 2024, MMLU-Lite-nl, XQuAD-nl, or any held-out set), check for overlap with the training/distillation/RAG corpora and exclude contaminated items. No phase reports a benchmark number without this. (Vasquez)
- **Statistical power.** Any evaluation that feeds a go/no-go decision needs **≥500 samples**. If a domain genuinely has fewer than 500 available items (e.g. AIME), use the full available set and label the result *directional*, not decision-grade. N=20 is never sufficient to act on. (De Vries)

---

## PHASE 0: RAG FOUNDATION (Weeks 1–2)
Goal: validate retrieval + user-memory stack before touching model weights.

0.1 Install llama.cpp — compile with CUDA 12.8+, verify `-ngl -1` recognises GPU.
0.2 Download Qwen3-4B Q4_K_M (3.45 GB, Unsloth variant). SHA256-verify.
0.3 DuckDB + Polars + Parquet corpus — schema: `[doc_id, text, url, embedding_bge, timestamp]`. Populate 100–500 NL+EN docs.
0.4 Download BGE-M3 locally (~1.5 GB). Test: embed 10 queries (5 NL, 5 EN), verify cosine ranking.
0.5 Agentic search loop: query → BGE-M3 embed → DuckDB top-20 → rerank top-5 → LLM answer + JSON citations. Test 10 prompts (3 NL, 3 EN, 2 code, 2 reasoning).
0.6 User-memory store (Parquet): `[turn_id, user_assertion, embedding, category, timestamp]`. Extract facts post-turn, retrieve top-3 per new turn as context prefix.
   **GDPR fix (Okafor):** auto-delete rows after 30 days; do not persist raw sensitive assertions long-term — store category + embedding where the raw text isn't required; run a Polars anonymization pass before any export, debugging dump, or log capture.

Deliverable: full retrieval + citation + memory loop. Retrieval < 200ms. Generation 50–100ms/token.

---

## PHASE 2A: LONG-COT SFT DISTILLATION (Weeks 3–6)
**Moved up — now runs directly after Phase 0, on the base Qwen3-8B (not a CPT checkpoint).** Phase 1 (continual pretraining) is conditional and only runs later if the Decision Gate below says it's needed.

Goal: inject chain-of-thought reasoning from DeepSeek-R1 teacher into the 8B base student, then measure before spending any further compute.

2a.1 Generate training traces via DeepSeek-R1: MATH-500 (500 problems, ~2k tokens/trace), GSM8K (8k samples), AIME 2024 (100 problems). Translate 50% of traces to Dutch. Keep only traces where R1 is correct. Total: 10k–20k (problem, CoT, answer) triples in Parquet. **Decontaminate against MATH-500/GSM8K/AIME public test splits before training (Global Requirements).**
2a.2 Format: `"Problem: {problem}\n\nReasoning:\n{cot}\n\nAnswer: {answer}"`. Loss mask on CoT + answer tokens only. Seq=2048. 90/10 train/val split.
2a.3 SFT QLoRA: LR 5e-5, epochs 3, batch 4, accumulate 2. Eval every 500 steps (val perplexity). Duration: 3–5 days.
2a.4 Merge LoRA → BF16 → quantize Q4_K_M GGUF. Save as `qwen3-8b-cot-distilled-q4.gguf`.
2a.5 Benchmark: MATH-500 held-out (100 samples) target 70%+, GSM8K (100) target 85%+, 20 Dutch math problems target 60–70%. **Decontaminated per Global Requirements.**

Deliverable: Qwen3-8B with long-CoT distilled, *without* any CPT step. MATH 70%+, GSM8K 85%+.

---

## DECISION GATE (Week 6–7)
**This gate is the single highest-leverage change from the review — it changes execution order, not just hyperparameters. Do not start Phase 1 or Phase 2b before running it.**

Run on the Phase 2a SFT model:

- **Dutch comprehension check:** MMLU-Lite-nl (≥500 samples, decontaminated), XQuAD-nl, native-fluency spot-check.
- **Reasoning reliability check:** MATH-500 / GSM8K held-out accuracy + failure-mode breakdown — knowledge gap vs. reasoning-step error vs. format error.

Decide:

- Both acceptable → **ship RAG + SFT as v1, stop here.** Revisit Phase 1/2b only if production usage surfaces a real gap.
- Dutch score short → run **Phase 1** (CPT), scoped as below.
- Reasoning reliability short *and the failures are reasoning errors, not knowledge gaps* → run **Phase 2b** (GRPO), scoped as below.
- Both short → run Phase 1 first (it's the foundation), then run Phase 2b on the post-CPT SFT model.

---

## PHASE 1: BILINGUAL CONTINUAL PRETRAINING (CONDITIONAL)
**Only runs if the Decision Gate shows a Dutch comprehension gap RAG + SFT doesn't close.** Goal: adapt the base model to the Dutch/EN corpus and secure native comprehension — without wrecking English ability or scope-creeping into a tokenizer rebuild.

1.1 Corpus: 60–70% EN (FineWeb-Edu, SlimPajama reasoning-heavy, OpenWebMath) / 30–40% NL (Flemish/Dutch news, tech docs, forums, filtered) / 10–15% code. **Target cut to 10–20B tokens (down from 50–100B); quality over volume for NL — drop low-quality forum/web text rather than padding volume for size.** Store in Parquet: `[text, language, domain, quality_score]`.
   **Replay buffer (Vasquez):** blend 10–15% original Qwen3 pretraining-distribution data (general English) into every training batch — without it, English MMLU regresses.
1.2 Tokenizer check: compute Dutch STRR on held-out text. Target >80%. **If below: stop.** Do not retrain the tokenizer as a fallback — a new tokenizer is a new model, not a Phase 1 fix. Sub-80% STRR means re-scoping the entire project around a base model with native NL tokenizer support, not patching this one. (Tokenizer retraining is fully removed as a plan-B option.)
1.3 Training setup: base Qwen3-8B BF16, **Unsloth full-model continual pretraining (not QLoRA)** with gradient checkpointing — QLoRA doesn't inject the deep morphological knowledge this phase needs. Validate VRAM fit empirically before committing to a multi-day run (seq=2048, largest batch that fits, accumulate as needed). LR **5e-5** cosine (down from 2e-4 — CPT LR must be 5–10x lower than pretraining LR). Checkpoint every 2k steps.
1.4 Run pretraining over the 10–20B token budget. Re-baseline wall-clock once real full-model-CPT throughput is measured (slower per-token than QLoRA, but a smaller corpus). Monitor validation loss + Dutch token perplexity.
1.5 Eval at end of token budget: Dutch MMLU-Lite (**≥500 samples**, decontaminated), XQuAD-nl. Baseline Qwen3-8B ~50–60%. Target after Phase 1: 55–65%. If gap too large: extend token budget within the 10–20B range, or switch to Qwen3-14B base — do not reach for tokenizer retraining.

Deliverable: Qwen3-8B continual-pretrained (full-model, not LoRA), merged to BF16. Dutch MMLU-Lite +5–10pp vs. baseline, English MMLU not regressed (verified via replay-buffer holdout).

---

## PHASE 2B: RLVR WITH GRPO (CONDITIONAL)
**Only runs if the Decision Gate shows a reasoning-reliability gap, and failure analysis confirms it's reasoning errors rather than knowledge gaps.** Goal: reinforce correct reasoning via RL on verifiable tasks.

2b.1 Prepare RLVR dataset: MATH-500, GSM8K, code with unit tests. Only tasks with automated verifiable correctness. 5k–10k problems. 80/20 train/eval. Decontaminate per Global Requirements.
2b.2 Reward function, refined for less sparse gradient signal (Tanaka): math correct = +1 (symbolic match), wrong but correctly formatted = -0.5, wrong format = -1. Code correct = all unit tests pass = +1.
2b.3 GRPO setup: **G=8–16** rollouts per problem (up from 4–6, needed for stable advantage estimation) at temp 0.7–1.0, compute group advantage, LoRA policy gradient update on top rollouts. **Add an explicit length-penalty / token-normalized reward** — without it, sequence length drifts to the max as a known GRPO reward-hacking pathology. LR 1e-5, **KL penalty 0.1–0.2** (up from 0.05 — too low when starting from a strong SFT model, lets the policy drift too far too fast). Batch 2, epochs 2–3. Duration: 7–10 days (re-baseline given the larger G).
2b.4 Eval: held-out 2k problems, decontaminated. Target: SFT 70% → post-GRPO 80–85% on MATH. Check rollout diversity (no mode collapse) *and* check specifically for length-reward-hacking now that the length-penalty is in place.
2b.5 Merge RL LoRA → BF16 → Q4_K_M. Save as `qwen3-8b-cot-rlvr-q4.gguf`.

Deliverable: Qwen3-8B with long-CoT SFT + RLVR. MATH 80–85%+, GSM8K 90%+. Reliable verifiable reasoning.

---

## PHASE 3: VERIFICATION & GROUNDING
Goal: add self-consistency, DPO for subjective tasks, citations wired into RAG.

3.1 Self-consistency: N=5 inference passes per hard problem (temp 0.8), majority-vote on extracted answers. Test on MATH hold-out (≥500 samples where available, decontaminated — bumped from the original 20-sample test per Global Requirements): expect +3–8pp.
3.2 DPO (subjective tasks only): 500–1000 preference pairs for counseling/safety domains (chosen vs rejected). LR 5e-6, 1 epoch. Merge + quantize. *(Okafor flagged this as too few for counseling specifically, recommending 5k Dutch pairs — not in the net-8, not applied here; revisit if counseling-domain quality is a launch blocker.)*
3.3 Citation grounding: force LLM JSON output with doc_id citations. Eval: 20 retrieval tasks, target 80%+ cited docs relevant.
3.4 Integration test: 30 mixed tasks (10 reasoning, 10 retrieval, 10 counseling). Measure latency, accuracy, citation quality, safety. *(Okafor: this is not a substitute for a real safety/red-team evaluation — flagged, not expanded here; revisit before any external release.)*

Deliverable: full production pipeline — LLM + RAG + agentic search + self-consistency + citations. Interactive, grounded, verified Dutch/EN output.

---

## PHASE 4: EFFICIENCY & SCALE
Goal: longer context, faster inference, larger model capacity.

4.1 Q4_K_M + Q8 KV-cache: enables 32k context on Qwen3-14B at 11.1 GB VRAM. (Verified: Qwen3-14B uses GQA with 8 KV-heads, not 40 — the 11.1 GB figure is correct as originally computed.) Benchmark 50 prompts at 8k and 32k. *Implementation note (Chen): budget against ~10.2 GB effective VRAM (15% overhead buffer on the 12 GB card), and benchmark KoboldCpp alongside llama.cpp at 32k — FlashAttention in llama.cpp isn't always optimal at non-power-of-2 context lengths.*
4.2 Scale to Qwen3-14B: apply Phase 3 LoRA to 14B base (no full retrain). Q4_K_M inference = 7.9 GB + Q8 KV. Speed ~30–45 tok/sec. Repeat Phase 2a/2b benchmarks to compare 8B vs 14B gains.
4.3 Gemma 4 12B QAT option: test Unsloth Gemma 4 12B QAT Q4_0 (6.6 GB). Run same benchmark suite. Choose 14B or Gemma 4 12B based on Dutch reasoning + speed tradeoff.
4.4 ~~Gemma 4 26B A4B MoE option~~ **— removed as an interactive production candidate (Chen).** Realistic throughput with `-ot "exps=CPU"` is 5–12 tok/sec, not the originally estimated 20–35 — too slow for interactive use. Keep only as a non-interactive research footnote (e.g. offline batch generation), not a Phase 4 decision candidate.
4.5 JEPA / Mamba-hybrid (research watch): flag for integration if tooling matures. Do not pivot v1.

Deliverable: production model selected (14B dense or Gemma 4 12B QAT). Full benchmark comparison. Deployed locally via llama-server, fully auditable Polars/DuckDB/Parquet stack.

---

## METRICS ACROSS ALL PHASES

Phase 0: retrieval latency < 200ms, citations accurate.
Decision Gate: Dutch + reasoning measured on ≥500-sample, decontaminated evals before any further phase is greenlit.
Phase 1 (if triggered): Dutch MMLU-Lite +5–10pp vs Qwen3-8B base, English MMLU not regressed.
Phase 2a: MATH-500 70%+, GSM8K 85%+, Dutch math 60–70%.
Phase 2b (if triggered): MATH-500 80–85%+, GSM8K 90%+, no length-reward-hacking.
Phase 3: end-to-end 30 mixed tasks: latency + accuracy + citations all pass (not a safety sign-off — see 3.4 note).
Phase 4: 32k context stable, production model selected from {14B dense, Gemma 4 12B QAT}, speed 20–45 tok/sec.

---

## HARDWARE CONSTRAINTS SUMMARY

| Model | VRAM (Q4+Q8KV@8k) | Train VRAM (QLoRA r=16, seq=1024) | Fits? |
|---|---|---|---|
| Qwen3-8B | 5.8 GB | 6.3 GB | YES — workhorse |
| Qwen3-14B | 8.8 GB | 10.8 GB | YES — Phase 4 inference |
| Gemma4-12B QAT | 6.6 GB | ~8 GB (estimated) | YES — Phase 4 candidate |
| Gemma4-26B A4B | 9.1 GB GPU+CPU | not used for training | YES — inference only, **non-interactive / batch use only** (5–12 tok/sec, see 4.4) |
| Qwen3-32B | 18 GB | does not fit | NO |
| Mixtral 8x7B | 26 GB | does not fit | NO |

**Note:** the QLoRA training-VRAM column above does not apply to Phase 1, which now uses Unsloth full-model CPT instead of QLoRA (see 1.3) — validate that VRAM fit empirically before a multi-day run.

NVMe offload: technically possible via `--mmap`, but MoE random expert access = < 5 tok/sec. Not recommended for interactive use. Not needed: 12GB GPU + 32GB RAM covers all models through Phase 4.
