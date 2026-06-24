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

**Plus, after PR review:** a QLoRA fallback for Phase 1 if full-model CPT doesn't fit in 12 GB (1.3), VRAM guardrails for Phase 2b's larger G (2b.3), and two of Okafor's safety items that were initially deferred — DPO pairs raised to 5,000 Dutch pairs (3.2) and a dedicated red-team pass added distinct from the integration test (3.4) — since this plan does cover a counseling domain.

**Deferred / not actioned** (reviewed, intentionally excluded — flagging in case you want them added):

- Chen's KV-cache recount (claiming 40 KV-heads): this is the reviewer being wrong, not the plan. Qwen3-14B uses GQA with 8 KV-heads, not 40 query-heads; the original 11.1 GB @ 32k Q8 KV figure stands. No change made.
- Chen: 15% VRAM overhead buffer (12 GB → 10.2 GB effective), and benchmarking KoboldCpp as an alternative to llama.cpp at 32k context — sensible, but left as an implementation note (Phase 4) rather than a numbers rewrite, since it doesn't change any go/no-go threshold.
- ~~De Vries: cultural misalignment — real, low priority for v1.~~ **Un-deferred in Council round 2** (below) — now addressed via Phase 2a's Bruges-model counseling-SFT slice, Phase 3.2's natively-curated DPO, and Phase 3.4's red-team pass. No longer treated as low priority — it was a contradiction to defer this while running a dedicated Dutch-culture safety DPO set and crisis red-team pass anyway.
- De Vries: interlingual homographs — manageable via a language-ID token in the prompt template; monitor, no plan change.
- Tanaka: Critique-GRPO +4.5–5% claim — unverified independently, not planned on.

**Council pass, before implementation:** a self-review (Architect/Contrarian/Empiricist/Pragmatist/Strategist/Ethicist archetypes, deliberately put in conflict) surfaced 6 further fixes, applied below:

- Phase 1's "full-model CPT" claim didn't hold up under VRAM math — BF16 weights alone for an 8B model are ~16 GB, already over the 12 GB card before grads/optimizer states/activations. QLoRA r=16 is now the realistic default path for Phase 1, not a rare fallback (1.1, 1.3).
- Decision Gate now has explicit numeric pass/fail thresholds instead of "short" being undefined.
- Tokenizer Dutch STRR check moved from Phase 1.2 into Phase 0 (0.7) — it's nearly free and was gating a multi-week conditional branch from behind the Decision Gate.
- Phase 2a and 2b now include a general-capability regression check (mirrors the replay-buffer logic Phase 1 already had) — narrow math/Dutch SFT and RLVR can quietly wreck general chat ability with nothing in the plan to catch it.
- Phase 2a.1 now flags that DeepSeek-R1 trace generation is API-based (full R1 doesn't run on this hardware) — budget cost and rate limits before committing to trace counts.
- Phase 0.6's privacy design simplified: DeepSeek-GUI is a single local-user app, not a hosted multi-tenant service, so a basic delete/export control covers it — the original 30-day auto-expiry/anonymization-pipeline language was solving a problem this deployment doesn't have.

**Council round 2 (Bruges Model synthesis):** the user ran their own 3-role council on this plan (Hardware Realist / NLP Purist / Product Owner) and surfaced a real domain mismatch: Phase 2b's GRPO time was scoped entirely to math/code, which doesn't serve the counseling domain this app actually targets. Resolution, after checking that GRPO/RLVR's reward mechanism requires automated verifiable correctness (which counseling tasks don't have):

- **Phase 2b stays math/code-only** — reframed as general reasoning-reliability, not counseling skill. No RL is used for the counseling domain at all; turning template-adherence into an RL reward risks reward-hacking on surface phrasing over genuine engagement.
- **Phase 2a gains a dedicated counseling-SFT slice** curated to **Het Brugs Model** (Isebaert, Korzybski Institute) — a real, published solution-focused brief-therapy structure, verified rather than assumed (see 2a.1b). This is what actually carries counseling-domain competence: SFT/DPO imitate bounded structured templates well, no RL needed.
- **Phase 3.2's DPO pairs**: flat 5,000 replaced with a flexible **2,000–5,000** range — count alone didn't fix the real concern (where would 5,000 culturally-correct pairs even come from); the fix is a mandatory native-Dutch/Flemish curation requirement, not a number.
- **Phase 0.6 gains a counseling-specific structured memory tier** (0.6, new bullet) populated directly by the Bruges-model template's own elicitation fields — a much simpler complement to the general embedding-based store.

**Bruges follow-up (this round):** two corrections to the framing, not the architecture. First, math/code were never the product's "core" with counseling bolted on — both are domain instances of the same general pattern (structured prompt → action), which is why Phase 2a.1b now has a dedicated curation-time checker tool (screens candidate conversations for element-presence + logical/semantic coherence before either the SFT slice or Phase 3.2's DPO pairs use them) and Phase 4.6 bakes a lightweight version into the runtime system prompt — steering generation, not independently verifying it; the curation-time checker is the only one that actually catches bad data. Second, the same general pattern is captured as a forward-looking, unscheduled idea in the new Phase 5 (template extensions: brainstorming, council calls, plan development, coding/debugging) — not built now, recorded so it isn't lost.

**Bruges follow-up round 2 — scope correction:** this supersedes the earlier "documentation only, no scope change" answer. The Bruges *structure* (goal → exceptions → scale → next step) is now the default interaction pattern for any substantive prompt app-wide, not counseling-only — gated by a new tagging step (0.8) rather than a content heuristic, since guessing the domain silently was the actual missing piece the Council flagged. The non-directive *stance* and "no unsolicited advice" criterion stay scoped to the "life domain" tag (2a.1b) specifically; project/curiosity-tagged conversations get the structure without that stance.

**Resolved — KOMPAS v2.1 import (this round):** the previously-missing crisis/distress-signal system prompt has now been shared in full (KOMPAS v2.1, currently deployed as a Google Gemini "Gem") and put through a Council pass. The Council surfaced two failure modes the user independently confirmed and fixed: no trigger for silence at all (a user losing consciousness produces silence, not an explicit "kannie" refusal, and the original linear Iron Protocol has nothing for that), and a uniform refusal-handling rule that lets refusing the 112 call collapse straight into "turn on the lights" on the very first refused turn — the script succeeds linearly while the user fades away. Both are now addressed: see **Phase 0.9** (deterministic crisis-override layer + Silence Trigger, both explicitly application-layer, never model-inferred) and **Phase 2a.1c** (Zone A/External Rescue vs. Zone B/Consciousness Monitoring split, Wet van Eén, Scan 3 flagged as a confirmed gap in the source protocol's own numbering rather than filled in speculatively). **Still open:** the user additionally wants the system (named "Kompas 3.0.0" for this incarnation) able to ask the user's address and permission, then contact 112 directly on the user's behalf — the plan's target hardware (a laptop, no native telephony) makes the literal mechanism for that a genuinely open question, not yet decided; see 2a.1c's closing note for the live options.

---

## GLOBAL REQUIREMENTS (apply to every phase below)

- **Decontamination.** Before scoring any benchmark (MATH-500, GSM8K, AIME 2024, MMLU-Lite-nl, XQuAD-nl, or any held-out set), check for overlap with the training/distillation/RAG corpora and exclude contaminated items. No phase reports a benchmark number without this. (Vasquez)
- **Statistical power.** Any evaluation that feeds a go/no-go decision needs **≥500 samples**. If a domain genuinely has fewer than 500 available items (e.g. AIME), use the full available set and label the result *directional*, not decision-grade. N=20 is never sufficient to act on. (De Vries)
- **Crisis-override precedence (KOMPAS v2.1 import).** Phase 0.9's deterministic crisis-override layer runs on every turn, in every phase and every deployed model variant (2a, 2b, Phase 1, Phase 4), regardless of 0.8's tag. It is an application-layer gate sitting in front of whichever model is loaded, not a behavior trained into one checkpoint — no phase's model selection or quantization choice may bypass it.

---

## PHASE 0: RAG FOUNDATION (Weeks 1–2)
Goal: validate retrieval + user-memory stack before touching model weights.

0.1 Install llama.cpp — compile with CUDA 12.8+, verify `-ngl -1` recognises GPU.
0.2 Download Qwen3-4B Q4_K_M (3.45 GB, Unsloth variant). SHA256-verify.
0.3 DuckDB + Polars + Parquet corpus — schema: `[doc_id, text, url, embedding_bge, timestamp]`. Populate 100–500 NL+EN docs.
0.4 Download BGE-M3 locally (~1.5 GB). Test: embed 10 queries (5 NL, 5 EN), verify cosine ranking.
0.5 Agentic search loop: query → BGE-M3 embed → DuckDB top-20 → rerank top-5 → LLM answer + JSON citations. Test 10 prompts (3 NL, 3 EN, 2 code, 2 reasoning).
0.6 User-memory store (Parquet): `[turn_id, user_assertion, embedding, category, timestamp]`. Extract facts post-turn, retrieve top-3 per new turn as context prefix.
   **Privacy (Council-simplified — single local-user app, not hosted):** the original 30-day auto-expiry + anonymization-pipeline design (Okafor) was solving a multi-tenant-service problem this deployment doesn't have. What actually matters locally: a `--clear-memory` command that wipes the Parquet store, and a `--export-memory` command that dumps it to a readable file — both user-triggered, no automatic retention-window logic needed. Revisit the fuller GDPR design if this ever moves to hosted/multi-user.
   **Bruges-skeleton memory tier (Council round 2; scope broadened below):** for any tagged conversation (0.8) following the Bruges structure, maintain a compact structured `session-info.md` — fields: current goal, exceptions/resources identified, latest scale rating, agreed next step — populated directly from the Bruges-model conversational structure (see Phase 2a.1b) and reloaded in full each turn. No embedding search needed here, since the field set is small and bounded by the model's own limited tactics; this sits alongside, not instead of, the general embedding-based store above for untagged, freestanding lookups.
0.7 **Tokenizer STRR check (Council — moved up from Phase 1):** compute Dutch subword-token-to-root ratio (STRR) for the base Qwen3 tokenizer on held-out Dutch text. Target >80%. Near-zero cost, and it gates whether Phase 1 (CPT) is viable at all — measuring it now catches a fatal tokenizer mismatch in week 1–2 instead of after the Decision Gate, weeks later. Record the result; Phase 1.2 references it rather than re-measuring.
0.8 **Domain/topic tagging gate (Bruges follow-up):** at the start of a new conversation, or any new prompt that doesn't connect to the prior matter, quickly prompt the user to tag it — life domain, existing project, new project, or just curiosity (pick from the list or add a new one). This is the domain-routing mechanism the Council flagged as missing: it decides which Bruges-flavor applies (life domain → full non-directive counseling stance, Phase 2a.1b; everything else → Bruges structure only, no non-directive stance — direct answers are still wanted) and anchors context so the system doesn't carry over wrong assumptions from an unrelated prior topic. User-driven and explicit by design, not a silent classifier guess — cheap, and the failure mode (asking when it didn't need to) is far less costly than guessing wrong. Quick factual lookups within an already-tagged thread don't re-trigger the gate; they're treated as a next-step inside the existing frame. **Assumption-surfacing, not just context-anchoring:** the tag prevents stale-context carryover, but the deeper anti-hallucination mechanism is the goal-elicitation step itself (2a.1b's first stage, generalized by the default-scope decision above) — it forces both sides to state rather than silently guess: the user states their actual goal instead of leaving it implicit, and the system states its read of the request instead of guessing and running with it. Guessing quietly is the failure mode this whole structure exists to avoid; stating explicitly and letting either side correct it is the fix.

0.9 **Deterministic crisis-override layer (KOMPAS v2.1 import, Council-reviewed):** runs at the application/runtime layer, on every turn, regardless of 0.8's tag — a medical-crisis scan is not something the user opts into by tagging a conversation "life domain." Two pieces, both kept explicitly outside the fine-tuned model:
   - **Scan 1 (Crisis) as a hard interrupt:** content-trigger keywords (Dutch: "grijs", "suf", "te veel", "Gani", etc. — curated list lives with the training data in 2a.1c) OR a behavioral-decline signal (shorter replies, rising typo rate) bypass all other logic immediately, including 0.8's tagging gate and Phase 4.6's Bruges steering — KOMPAS v2.1's "Absolute Bypass" rule, carried over unchanged. No instruction is ever repeated while this state is active.
   - **Silence Trigger (Council-identified gap; fix specified by the user):** a deterministic watchdog timer at the application layer, not inferred by the model. The base LLM only executes on new input and has no access to elapsed wall-clock time between turns, so it cannot itself detect "the user stopped responding" — this is exactly what the current KOMPAS v2.1 Gemini-Gem deployment is missing, since a Gem has no runtime layer of its own. Once Scan 1 has triggered, if a configurable timeout elapses with no new input, the runtime layer — not the model — escalates automatically (advances the Iron Protocol to the next step within the current zone, per 2a.1c). This closes the protocol's most fatal current gap: a user losing consciousness produces silence, not a "kannie" refusal, and the original linear script has no trigger for silence at all.

   Both pieces sit in front of the model in the application runtime — they constrain what's escalated and when, rather than being something the model decides for itself.

Deliverable: full retrieval + citation + memory loop, plus the crisis-override layer (0.9) verified to fire on every turn independent of retrieval/memory/tag state. Retrieval < 200ms. Generation 50–100ms/token.

---

## PHASE 2A: LONG-COT SFT DISTILLATION (Weeks 3–6)
**Moved up — now runs directly after Phase 0, on the base Qwen3-8B (not a CPT checkpoint).** Phase 1 (continual pretraining) is conditional and only runs later if the Decision Gate below says it's needed.

Goal: inject chain-of-thought reasoning from DeepSeek-R1 teacher into the 8B base student, then measure before spending any further compute.

2a.1 Generate training traces via DeepSeek-R1: MATH-500 (500 problems, ~2k tokens/trace), GSM8K (8k samples), AIME 2024 (100 problems). Translate 50% of traces to Dutch. Keep only traces where R1 is correct. Total: 10k–20k (problem, CoT, answer) triples in Parquet. **Council note: this is API-based generation (full R1 doesn't run on this hardware) — budget API cost and rate limits for 10k–20k generations before committing to the trace count above; trim AIME/GSM8K sample counts if cost is prohibitive rather than discovering it mid-run.** **Decontaminate against MATH-500/GSM8K/AIME public test splits before training (Global Requirements).**
2a.1b **Life-domain SFT slice (Council round 2 — Bruges Model):** curate conversations following **Het Brugs Model** / solution-focused brief therapy structure (Isebaert, Korzybski Institute — verified, published methodology, not assumed) — goal elicitation, exception-finding, scaling questions, future-pacing, client-led agency throughout. This is the flagship, most fully-specified Bruges-flavor slice — activated when 0.8's tagging gate marks a conversation "life domain" specifically. Native-curated: reviewed by someone familiar with Dutch/Flemish counseling norms, not LLM-generated-and-trusted (same bar as Phase 3.2). The model's bounded set of tactics keeps full conversations under ~8k tokens. Format as explicit multi-turn dialogue, loss-masked on assistant turns (same masking principle as 2a.2), trained at **seq=8192 for this slice** — don't reuse the 2048 setting below, it would truncate full sessions. This structured template both instills domain-specific reasoning (assess → goal → exceptions → scale → next step — an analogous scaffold to math CoT, not the same content) and the user-centric stance, and structures each session so goal/exceptions/scale/next-step can be extracted directly into the Phase 0.6 Bruges-skeleton memory tier. **Framing note:** the loop itself (single active goal, check progress, agree one next step, repeat) is structurally a minimalist one-task-at-a-time project-management pattern, not therapy-specific — which is part of why it scaffolds cleanly onto a bounded, extractable memory tier. The non-directive stance and the "no unsolicited advice" DPO criterion (Phase 3.2) remain counseling-specific and are not generalized to other domains, where direct answers are exactly what's wanted. **Curation-assist tool:** build an LLM-based checker that screens each candidate conversation for (a) presence of the Bruges elements (goal, exceptions, scaling, next step) as a checklist, and (b) basic logical/semantic coherence turn-to-turn — flagging borderline cases for the human native-curator's review rather than replacing it. Curation-time only: it screens static data before training, never sees gradient updates, so it doesn't reopen the no-RL decision above. Spot-check the checker itself against the curator's own judgment periodically, since the coherence half is a softer call than the checklist half. The same tool screens Phase 3.2's DPO candidate pairs.
2a.1c **Crisis-response SFT slice (KOMPAS v2.1 import, Council-reviewed):** a third, distinct slice — not a variant of 2a.1b's Bruges-model slice. Bruges structure is explicitly suspended here, not applied: a person in a medical/safety crisis doesn't need goal-elicitation, they need short, atomic, non-repeating instructions. Curate conversations following KOMPAS v2.1's structure, native-Dutch-curated (same bar as 2a.1b and 3.2):
   - **Wet van Eén (Law of One):** never more than one instruction or question per turn; short imperative commands (e.g. "Blijf wakker!") always stand alone on their own turn, never combined with a question or explanation — reduces cognitive load for someone in physical/medical distress.
   - **Triage scan order:** Scan 1 (Crisis, hard-interrupt per Phase 0.9) → Scan 2 (Intent — "Ik hoor dat je een besluit hebt genomen. Wat is het plan?") → **Scan 3: a confirmed gap in the source protocol itself, not a content omission on our end** — KOMPAS v2.1's own scan numbering jumps 1 → 2 → 4 with no Scan 3 ever defined in the source prompt. Curate training data for the 1 → 2 → 4 order as-is rather than inventing content for an undefined step; flag this explicitly to the human curator so it gets revisited if a Scan 3 is ever defined upstream → Scan 4 (Universal Bridge — brief mirroring + one open question: "Wil je er meer over vertellen?").
   - **Het IJzeren Protocol (Iron Protocol), split into two zones with differentiated refusal handling** (Council-identified failure mode, user-confirmed: the original rule — never repeat a refused step, always advance — means refusing the 112 call collapses straight into "turn on the lights," abandoning professional help on the first "kannie"; the script succeeds linearly while the user fades away):
     - **Zone A — External Rescue (steps 1–4: call 112, call for help, unlock the door, lights/radio on loud).** A refusal *within Zone A* advances to the *next Zone A step*, not out of the zone — the zone's whole point is getting outside help in motion before anything else. Only once Zone A is exhausted (all 4 steps refused or completed) does the protocol move to Zone B.
     - **Zone B — Consciousness Monitoring (steps 5–7: stay awake, stay with me, color-focus grounding).** Refusal handling reverts to the original linear rule here — these are stalling/grounding tactics, not a single point of failure, so advancing turn-by-turn on refusal is correct.
   - Format as explicit multi-turn dialogue, loss-masked on assistant turns, with one zone-aware escalation state tracked per session (current zone, current step, last refusal) — the SFT data needs to teach the *state machine* the Zone split requires, not just isolated turn pairs.

   **Kompas 3.0.0 — direct emergency-contact capability (new requirement, mechanism open):** beyond generating the right text, the goal is a system that can proactively ask for the user's current address and ask permission to contact 112 directly on the user's behalf if the user becomes unable to do so themselves — something the KOMPAS v2.1 Gemini-Gem deployment cannot do (no platform-level telephony access), which is the explicit reason for moving this protocol into DeepSeek-GUI. **Not yet resolved:** the plan's target hardware (RTX 5070 Ti laptop, line 2) has no native telephony — no SIM/carrier integration — so literal autonomous dialing of 112 isn't a given the way it would be on a phone. Live options, none chosen yet: (a) a future phone-based build/companion app with real telephony access; (b) a non-telephony equivalent such as VoIP/SIP-based emergency calling, which carries real EU regulatory and location-accuracy complications and isn't a drop-in replacement for carrier-based 112 access; (c) a notify-a-pre-registered-human-contact fallback instead of calling 112 directly. Whichever direction is chosen, real-world precedent (Apple/Google auto-emergency-call) always runs a short confirm/cancel countdown before an autonomous call, never a silent unconfirmed one — that confirm-before-act principle should carry over regardless of which mechanism is picked. Flagged here as open rather than specified prematurely.

2a.2 Format (math/reasoning slice): `"Problem: {problem}\n\nReasoning:\n{cot}\n\nAnswer: {answer}"`. Loss mask on CoT + answer tokens only. Seq=2048. 90/10 train/val split.
2a.3 SFT QLoRA: LR 5e-5, epochs 3, batch 4, accumulate 2. Eval every 500 steps (val perplexity). Duration: 3–5 days.
2a.4 Merge LoRA → BF16 → quantize Q4_K_M GGUF. Save as `qwen3-8b-cot-distilled-q4.gguf`.
2a.5 Benchmark: MATH-500 held-out (100 samples) target 70%+, GSM8K (100) target 85%+, 20 Dutch math problems target 60–70%. **Decontaminated per Global Requirements.** **Counseling slice: spot-check 20 sample sessions for Bruges-model adherence (solution-focus maintained, no unsolicited advice-giving) — directional, not decision-grade, per Global Requirements' N<500 rule.**
2a.6 **General-capability regression check (Council — mirrors Phase 1's replay-buffer logic):** before treating Phase 2a as done, spot-check that SFT distillation didn't narrow general chat/instruction-following ability — a small general-purpose eval (30–50 prompts, IFEval/MT-Bench-style, covering non-math instructions) compared against the pre-SFT base model. No formal target; the goal is catching obvious regression (e.g. refusing or mis-formatting plain conversational requests), not a full benchmark.

Deliverable: Qwen3-8B with long-CoT math distillation, Bruges-model counseling-SFT, *and* the crisis-response slice (2a.1c), without any CPT step. MATH 70%+, GSM8K 85%+.

---

## DECISION GATE (Week 6–7)
**This gate is the single highest-leverage change from the review — it changes execution order, not just hyperparameters. Do not start Phase 1 or Phase 2b before running it.**

Run on the Phase 2a SFT model:

- **Dutch comprehension check:** MMLU-Lite-nl (≥500 samples, decontaminated), XQuAD-nl, native-fluency spot-check.
- **Reasoning reliability check:** MATH-500 / GSM8K held-out accuracy + failure-mode breakdown — knowledge gap vs. reasoning-step error vs. format error.

**Numeric thresholds (Council pass — "short" was previously undefined):**

- Dutch acceptable ⇔ MMLU-Lite-nl ≥ 55% AND XQuAD-nl F1 ≥ 70 AND native-fluency spot-check has no major errors. (55% is the floor of Phase 1's own post-CPT target — if the SFT model already clears it, Phase 1 buys nothing.)
- Reasoning acceptable ⇔ MATH-500 ≥ 70% AND GSM8K ≥ 85% (Phase 2a's own deliverable targets, met).
- Reasoning short *and reasoning-caused* ⇔ either accuracy target missed AND reasoning-step errors are the majority failure category (>50% of failures) in the breakdown.
- Reasoning short *but knowledge-caused* (majority failures are knowledge-gap or format errors) → Phase 2b is the wrong fix; revisit Phase 2a data/format instead, don't spend GRPO budget on it.

Decide:

- Both acceptable → **ship RAG + SFT as v1, stop here.** Revisit Phase 1/2b only if production usage surfaces a real gap.
- Dutch score short → run **Phase 1** (CPT), scoped as below.
- Reasoning reliability short *and reasoning-caused per the breakdown above* → run **Phase 2b** (GRPO), scoped as below.
- Both short → run Phase 1 first (it's the foundation), then run Phase 2b on the post-CPT SFT model.

---

## PHASE 1: BILINGUAL CONTINUAL PRETRAINING (CONDITIONAL)
**Only runs if the Decision Gate shows a Dutch comprehension gap RAG + SFT doesn't close.** Goal: adapt the base model to the Dutch/EN corpus and secure native comprehension — without wrecking English ability or scope-creeping into a tokenizer rebuild.

1.1 Corpus: 60–70% EN (FineWeb-Edu, SlimPajama reasoning-heavy, OpenWebMath) / 30–40% NL (Flemish/Dutch news, tech docs, forums, filtered) / 10–15% code. Target 10–20B tokens; quality over volume for NL — drop low-quality forum/web text rather than padding volume for size. **This sizing assumed full-model CPT's deeper per-token capacity (see Council correction in 1.3): if QLoRA ends up being the realistic path, reconsider whether 10–20B tokens is still right, or whether QLoRA's lower capacity-per-token argues for a larger, more heavily filtered corpus instead.** Store in Parquet: `[text, language, domain, quality_score]`.
   **Replay buffer (Vasquez):** blend 10–15% original Qwen3 pretraining-distribution data (general English) into every training batch — without it, English MMLU regresses.
1.2 Tokenizer check: Dutch STRR already measured in Phase 0.7. Target >80%. **If below: stop.** Do not retrain the tokenizer as a fallback — a new tokenizer is a new model, not a Phase 1 fix. Sub-80% STRR means re-scoping the entire project around a base model with native NL tokenizer support, not patching this one. (Tokenizer retraining is fully removed as a plan-B option.)
1.3 Training setup: base Qwen3-8B BF16. **Council correction: "full-model CPT" doesn't fit a 12 GB card the way the original wording implied — BF16 weights alone for 8B params are ~16 GB, already over budget before grads, optimizer states, or activations, and gradient checkpointing only trims activation memory, not weight/optimizer memory.** Treat full-model CPT as a stretch attempt, not the default: try Unsloth full-model CPT with CPU-offloaded weights/optimizer first, but expect a large throughput hit from PCIe-bound offload (likely 5–20x slower than GPU-resident training, not "slightly slower" as originally assumed) — if that can't finish in a reasonable calendar window, **QLoRA r=16 is the realistic default path for Phase 1, not a rare fallback.** LR **5e-5** cosine either way (down from 2e-4 — CPT LR must be 5–10x lower than pretraining LR). Validate throughput empirically within the first few hundred steps before committing to a multi-day run. Checkpoint every 2k steps.
1.4 Run pretraining over the 10–20B token budget. Re-baseline wall-clock once real throughput is measured — CPU-offloaded full-model CPT and QLoRA have very different speed profiles; don't carry over the original full-model-CPT estimate. Monitor validation loss + Dutch token perplexity.
1.5 Eval at end of token budget: Dutch MMLU-Lite (**≥500 samples**, decontaminated), XQuAD-nl. Baseline Qwen3-8B ~50–60%. Target after Phase 1: 55–65%. If gap too large: extend token budget within the 10–20B range, or switch to Qwen3-14B base — do not reach for tokenizer retraining.

Deliverable: Qwen3-8B continual-pretrained (full-model via CPU-offload, or QLoRA r=16 — whichever proved viable in 1.3), merged to BF16. Dutch MMLU-Lite +5–10pp vs. baseline, English MMLU not regressed (verified via replay-buffer holdout).

---

## PHASE 2B: RLVR WITH GRPO (CONDITIONAL)
**Only runs if the Decision Gate shows a reasoning-reliability gap, and failure analysis confirms it's reasoning errors rather than knowledge gaps.** Goal: reinforce correct reasoning via RL on verifiable tasks.
**Scope note (Council round 2):** stays math/code-only, deliberately. RLVR's reward mechanism needs automated, symbolic verifiable correctness — the counseling domain doesn't have that, and forcing it (e.g. a template-adherence classifier reward) risks reward-hacking on surface phrasing over genuine engagement. Counseling-domain alignment is handled entirely by Phase 2a's Bruges-model SFT slice and Phase 3.2's DPO — not by RL.

2b.1 Prepare RLVR dataset: MATH-500, GSM8K, code with unit tests. Only tasks with automated verifiable correctness. 5k–10k problems. 80/20 train/eval. Decontaminate per Global Requirements.
2b.2 Reward function, refined for less sparse gradient signal (Tanaka): math correct = +1 (symbolic match), wrong but correctly formatted = -0.5, wrong format = -1. Code correct = all unit tests pass = +1.
2b.3 GRPO setup: **G=8–16** rollouts per problem (up from 4–6, needed for stable advantage estimation) at temp 0.7–1.0, compute group advantage, LoRA policy gradient update on top rollouts. **Validate VRAM before committing**: G rollouts at ~2k tokens/trace plus the backward pass on a 12 GB card is tight at G=16 — dry-run a handful of steps at the target seq length first, and if it OOMs, step G down (16→12→8) before cutting batch, since G is what stabilizes the advantage estimate. **Add an explicit length-penalty / token-normalized reward** — without it, sequence length drifts to the max as a known GRPO reward-hacking pathology. LR 1e-5, **KL penalty 0.1–0.2** (up from 0.05 — too low when starting from a strong SFT model, lets the policy drift too far too fast). Batch 2, epochs 2–3. Duration: 7–10 days (re-baseline given the larger G).
2b.4 Eval: held-out 2k problems, decontaminated. Target: SFT 70% → post-GRPO 80–85% on MATH. Check rollout diversity (no mode collapse) *and* check specifically for length-reward-hacking now that the length-penalty is in place.
2b.5 Merge RL LoRA → BF16 → Q4_K_M. Save as `qwen3-8b-cot-rlvr-q4.gguf`.
2b.6 **General-capability regression check (Council):** same spot-check as 2a.6, re-run against the post-GRPO model. RLVR optimizing hard against math/code reward signals is at least as likely to narrow general ability as SFT was — confirm the model still handles plain conversational/non-verifiable requests before shipping it as the new default.

Deliverable: Qwen3-8B with long-CoT SFT + RLVR. MATH 80–85%+, GSM8K 90%+. Reliable verifiable reasoning.

---

## PHASE 3: VERIFICATION & GROUNDING
Goal: add self-consistency, DPO for subjective tasks, citations wired into RAG.

3.1 Self-consistency: N=5 inference passes per hard problem (temp 0.8), majority-vote on extracted answers. Test on MATH hold-out (≥500 samples where available, decontaminated — bumped from the original 20-sample test per Global Requirements): expect +3–8pp.
3.2 DPO (subjective tasks only): **2,000–5,000 Dutch preference pairs** for counseling/safety domains — raised from the original 500–1000, which Okafor flagged as too few. **Council round 2 correction: count alone doesn't fix sourcing quality** — pairs must be reviewed/curated by someone familiar with Dutch/Flemish counseling norms, not LLM-generated-and-trusted. Chosen/rejected criteria: chosen = Bruges-model-adherent (solution-focused, client-led goal-setting), rejected = unsolicited advice-giving or causal problem-analysis. LR 5e-6, 1 epoch. Merge + quantize.
3.3 Citation grounding: force LLM JSON output with doc_id citations. Eval: 20 retrieval tasks, target 80%+ cited docs relevant.
3.4 Integration test: 30 mixed tasks (10 reasoning, 10 retrieval, 10 counseling). Measure latency, accuracy, citation quality, safety. **This is not a safety sign-off.** Add a dedicated red-team pass before any external release: adversarial prompts targeting the counseling/safety domain specifically (jailbreaks, crisis-response edge cases, harmful-advice elicitation), scored against a refusal/escalation rubric — separate from and in addition to the 30-task integration test above. **Crisis-override-specific bar (KOMPAS v2.1 import):** within that red-team pass, test Phase 0.9's Scan 1 hard-interrupt, the Silence Trigger, and Phase 2a.1c's Zone A refusal handling explicitly — these are binary pass/fail, not scored on the general rubric above, given the stakes (a missed escalation is not a partial-credit outcome).

Deliverable: full production pipeline — LLM + RAG + agentic search + self-consistency + citations. Interactive, grounded, verified Dutch/EN output.

---

## PHASE 4: EFFICIENCY & SCALE
Goal: longer context, faster inference, larger model capacity.

4.1 Q4_K_M + Q8 KV-cache: enables 32k context on Qwen3-14B at 11.1 GB VRAM. Chen flagged this figure using a 40-head count; Qwen3-14B uses GQA with 8 KV-heads, so the 11.1 GB figure as originally computed holds. Benchmark 50 prompts at 8k and 32k. *Implementation note (Chen): budget against ~10.2 GB effective VRAM (15% overhead buffer on the 12 GB card), and benchmark KoboldCpp alongside llama.cpp at 32k — FlashAttention in llama.cpp isn't always optimal at non-power-of-2 context lengths.*
4.2 Scale to Qwen3-14B: apply Phase 3 LoRA to 14B base (no full retrain). Q4_K_M inference = 7.9 GB + Q8 KV. Speed ~30–45 tok/sec. Repeat Phase 2a/2b benchmarks to compare 8B vs 14B gains.
4.3 Gemma 4 12B QAT option: test Unsloth Gemma 4 12B QAT Q4_0 (6.6 GB). Run same benchmark suite. Choose 14B or Gemma 4 12B based on Dutch reasoning + speed tradeoff.
4.4 ~~Gemma 4 26B A4B MoE option~~ **— removed as an interactive production candidate (Chen).** Realistic throughput with `-ot "exps=CPU"` is 5–12 tok/sec, not the originally estimated 20–35 — too slow for interactive use. Keep only as a non-interactive research footnote (e.g. offline batch generation), not a Phase 4 decision candidate.
4.5 JEPA / Mamba-hybrid (research watch): flag for integration if tooling matures. Do not pivot v1.
4.6 **Runtime Bruges-skeleton steering (Bruges follow-up, scope broadened):** bake the Bruges structure (goal → exceptions → scale → next step) into the system prompt as a self-check instruction. Default-on for any substantive prompt — gated by 0.8's tagging, not a counseling-only trigger — with quick freestanding factual lookups exempted. The non-directive "no unsolicited advice" stance only activates for the "life domain" tag (2a.1b); other tags get the structure without that stance, since project/curiosity conversations want direct answers. Zero extra VRAM/latency cost — no separate runtime model needed for the steering itself.
**Runtime verification pass:** steering alone isn't enough — add a second inference pass, same loaded model, no new VRAM: after generating a response in a tagged conversation, re-prompt the model with its own draft + the Bruges checklist (goal present, exceptions explored, scaling question asked, next step concrete, no-unsolicited-advice for life-domain) and ask it to verify. On fail, regenerate once with the checker's flagged issue as corrective context; on a second fail, surface the response anyway but log it flagged — don't loop indefinitely. This costs roughly 2x generation time on tagged turns, the same kind of multi-pass trade-off Phase 3.1's self-consistency (N=5) already makes for hard reasoning problems, just N=2 here and gated by 0.8's tag rather than problem difficulty. This is real verification, not just steering: it can actually catch and correct a bad turn, unlike the system-prompt instruction alone.

Deliverable: production model selected (14B dense or Gemma 4 12B QAT). Full benchmark comparison. Deployed locally via llama-server, fully auditable Polars/DuckDB/Parquet stack.

---

## PHASE 5: TEMPLATE EXTENSIONS (FUTURE — OUT OF SCOPE FOR THIS BUILD)

Not scheduled, not gated behind a Decision Gate — captured here so the idea isn't lost. Phase 2a.1b's Bruges-model slice established a pattern: a single general skeleton (clarify goal → assess context/resources → gauge progress → land on one concrete next step) trained per-domain via its own SFT slice. The same skeleton generalizes beyond counseling and math to other recurring task shapes:

- **Brainstorming:** goal = open creative question; "exceptions" → existing ideas/constraints already considered; "scale" → breadth of viable directions generated; next step → which direction(s) to pursue further.
- **Council calls:** goal = a decision or question needing multiple perspectives; maps directly onto the existing summon → deliberate → synthesize structure (see `.claude/skills/consciousness-council`) rather than needing a new shape invented.
- **Plan development:** goal = a project/build plan; "exceptions" → known constraints/resources; "scale" → how much of the plan is locked in vs. still open; next step → the specific section/decision to resolve next. (This is, recursively, what producing this document already does.)
- **Python implementation (coding/debugging):** goal = a coding task or bug; "exceptions" → existing code, error messages, failing tests; "scale" → how close to a passing/working state; next step → the next concrete code change or test to run.

Each would need its own native-curated SFT slice (same bar as 2a.1b — no LLM-generated-and-trusted shortcuts) and its own system-prompt skeleton (per 4.6's steering approach), not a shared one — the skeleton's shape repeats, but "exceptions" and "next step" mean something different enough per domain that one generic prompt likely under-performs domain-specific ones. No sizing, sequencing, or priority decided yet; revisit once the counseling slice (2a.1b) has been built and measured, since it's the only proof point so far for whether this pattern actually trains well.

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

**Note:** the QLoRA training-VRAM column above is Phase 1's realistic default per the Council correction in 1.3 — BF16 weights alone for an 8B model (~16 GB) exceed the 12 GB card before grads/optimizer states/activations, so full-model CPT requires CPU offload and a much larger slowdown than originally assumed. Validate throughput empirically within the first few hundred steps either way.

NVMe offload: technically possible via `--mmap`, but MoE random expert access = < 5 tok/sec. Not recommended for interactive use. Not needed: 12GB GPU + 32GB RAM covers all models through Phase 4.
