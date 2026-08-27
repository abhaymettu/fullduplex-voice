# fullduplex-voice

**A tuned cascade answers in 386 ms on a laptop. The architectural floor everyone
assumes is there is mostly the endpointer doing nothing while it waits.**

This repo set out to prove a cascade can never be fast and that you need a native
speech-to-speech model instead. The measurements did not agree, so the headline
is the opposite of the pitch. What follows is what the runs actually show.

All timings are from runs on one machine (Apple M4 Pro, 24 GB, macOS) under one
definition of latency, stated once and used everywhere:

> **gap = onset of agent speech − offset of user speech, both silence-trimmed.**

That is `harness/exchange.py`'s definition in the sibling repo, vendored here as
`gap.py` and pinned to `aliveness-threshold@b7ccbb7` so the two sides stay
comparable. A number measured any other way is not in this table.

---

## The floor argument

The cascade is `audio → ASR → LM → TTS → audio`. Its stages cannot overlap
*each other*: the LM needs a transcript, TTS needs a sentence. That serial chain
is real and it is why people assume a floor.

But the chain is not what you wait for. An endpointer has to watch the user go
quiet for `HANGOVER` ms before it will call the turn over, and in the naive loop
**nothing runs during that window**. Speculation fills it: after `ARM` ms of
silence, start the pipeline on the audio captured so far. Everything between the
snapshot and the real endpoint is silence by definition, so nothing is guessed.

That gives a two-regime model, which `floor.py` states and tests:

```
gap  ≈  max(HANGOVER, ARM + W_total)  +  handoff
```

`W_total` is the summed on-clock time of ASR_final + LM_ttft + LM_sentence + TTS.
A cascade that does not speculate is the same formula with `ARM = HANGOVER`.

**Test 1 — sweep the hangover, and the gap does not move.** If the hangover were
the floor, cutting it would pay. It does not:

| hangover | gap median | exposed pipeline work | hangover + exposed |
|---|---|---|---|
| 120 ms | 511.7 ms | 385.4 ms | 505.4 ms |
| 150 ms | 510.7 ms | 351.9 ms | 501.9 ms |
| 250 ms | 515.4 ms | 250.8 ms | 500.8 ms |
| 350 ms | 498.2 ms | 134.4 ms | 484.4 ms |

Cutting the hangover by 230 ms bought 19 ms. Every millisecond taken off the
hangover reappears as an exposed millisecond of pipeline. n=20 per row.

**Test 2 — the model predicts the gap across a 2.3× range.** One formula, eight
runs from 348 ms to 797 ms, median residual **+15.3 ms** (the handoff to the
audio callback is ~3 ms of it):

| run | predicted | measured | error |
|---|---|---|---|
| fast, arm 40, tiny.en, hang 250 | 332.3 | 347.8 | +15.5 |
| fast, arm 40, tiny.en | 365.2 | 376.0 | +10.7 |
| fast, arm 80, tiny.en | 379.0 | 395.8 | +16.8 |
| fast, arm 80, base.en | 488.5 | 517.0 | +28.5 |
| baseline (no speculation) | 781.3 | 796.4 | +15.2 |
| baseline rep3 | 785.0 | 797.2 | +12.2 |

**Test 3a — the stages really are serial.** Without speculation the pipeline runs
start to finish after the endpoint, so its span is directly observable as
`gap − hangover`. If ASR, LM and TTS overlapped at all, that span would be shorter
than their summed work. It is not:

| run | sum of stage work | pipeline span | span / sum |
|---|---|---|---|
| baseline rep2 | 431.3 ms | 436.2 ms | **1.011** |
| baseline rep3 | 435.0 ms | 442.8 ms | **1.018** |

That is the dependency graph, measured: the LM cannot start without a transcript
and TTS cannot start without a sentence, so `W_total` is a real serial cost. You
cannot parallelise under it — only make the stages cheaper, or start them earlier.

**Test 3b — what is actually irreducible.** In the fastest configuration, per
turn (n=20, medians):

| stage | work on its own clock | charged to the gap | hidden? |
|---|---|---|---|
| ASR final | 111.5 ms | 0.0 ms | yes, inside the hangover |
| LM time-to-first-token | 98.2 ms | 0.3 ms | yes |
| LM to sentence end | 32.1 ms | 31.2 ms | **no** |
| TTS | 43.5 ms | 43.5 ms | **no** |
| **total** | **292.3 ms** | | |

So the floor for this stack is `ARM + W_total` = 40 + 292 = **332 ms**, against a
measured median of 347.8 ms. Not the hangover. The hangover is free real estate
that the naive loop wastes.

**What this does and does not license.** The stages are serial, so `W_total` is a
real cost and you cannot parallelise your way under it — you can only make the
stages individually cheaper (swapping `base.en` for `tiny.en` moved the median
from 506 to 386 ms) or start them earlier. `ARM` cannot go to zero: arming on
every inter-word pause launches pipelines that get thrown away.

> **Caveat, and it is a big one.** `ARM = 80 ms` works here because the test
> talker is a piper voice whose longest internal pause is 65 ms. A human pausing
> mid-sentence would be cut off. **386 ms is not validated on human speech** and
> should not be quoted as if it were.

---

## The side-by-side

Run `.venv/bin/python compare.py`. Cascade rows are recomputed from the sibling
repo's per-turn records, not quoted from its prose.

| system | n | median gap | IQR |
|---|---|---|---|
| cascade, speculative (arm 40, tiny.en, hang 250) | 20 | 347.8 ms | 338–450 |
| cascade, speculative (arm 80, tiny.en) | 40 | 386.2 ms | 371–442 |
| cascade, speculative (arm 80, base.en) | 40 | 506.3 ms | 487–533 |
| cascade, unoptimised | 100 | 807.4 ms | 779–895 |
| Moshi q4 (MLX) | — | *see below* | |

Published figures, **not measured here**, for scale only:

- GPT-4o voice: 320 ms average, 232 ms minimum ([OpenAI, May 2024](https://openai.com/index/hello-gpt-4o/))
- Moshi: 160 ms theoretical, 200 ms in practice ([Kyutai model card](http://kyutai.org/Moshi.pdf))

---

## Barge-in

Interrupting the robot is where the two architectures stop being comparable on a
latency axis and start being different kinds of system.

**The cascade cannot be interrupted.** This is structural, established by reading
`live/loop.py`, not by an interruption experiment:

- `run_turn` calls `player.play(y)` then `player.wait()`. `Player` holds the whole
  reply in one buffer and the callback drains it to completion. There is no stop
  path and no VAD running during playback.
- `capture()` builds a **fresh `queue.Queue()` and a fresh feeder thread per turn**.
  Audio the user produces while the agent is talking is not in the next turn's
  queue. It is discarded.

`cascade_bargein.py` measures the first half of that rather than asserting it: it
imports the sibling's real `Player`, plays real cascade replies on the same
BlackHole device, "interrupts" 900 ms in, and times how long the agent keeps
going.

| | n | median | IQR | min | max | stopped early |
|---|---|---|---|---|---|---|
| cascade keeps talking after interruption | 20 | **1173 ms** | 1060–1264 | 541 | 5916 | **0 / 20** |

Zero out of twenty stopped. And because `capture()` builds a fresh queue per
turn, the interrupting utterance is not merely late — it is discarded, so the
user has to say it again. `cascade_bargein.py --demo` re-checks the structural
claim against the sibling's current source and fails loudly if `Player` ever
grows a `stop()`.

This is a property of *this* cascade, not of cascades in principle: you can bolt
barge-in on with an always-on VAD and an explicit stop. The point is that it is a
bolt-on with its own detection latency, whereas in a full-duplex model being
interrupted is just the conversation continuing.

*Moshi's side of this comparison is below, from `moshi_run.py --bargein`.*

---

## Limitations

- **The prompts are synthetic.** Every cascade number and every Moshi number uses
  the same five piper-rendered prompts. That makes the comparison fair and makes
  neither of them a claim about human speech. See the `ARM` caveat above.
- **n is small.** 20–100 turns per configuration. Medians are stable across
  repeats; the IQRs are not narrow.
- **One machine, contended.** These ran on a laptop with other agents' jobs on it.
  `loadavg` is recorded per run.
- **The cascade is another agent's work.** It lives in
  `~/Desktop/Playground/aliveness-threshold` and was being actively optimised while
  this was written. Nothing here edits it; `gap.py` is vendored and pinned so the
  numbers do not move under me.
- **WER is not zero.** The fastest cascade config uses `tiny.en` and scores
  WER 0.033 against the prompts. Some of its speed is bought with accuracy.

## Layout

| file | what |
|---|---|
| `gap.py` | the latency definition, vendored and pinned. `--demo` self-checks it |
| `floor.py` | the three tests above, from the sibling's per-turn records |
| `cascade_baseline.py` | the cascade numbers, recomputed from raw records |
| `moshi_run.py` | runs Moshi frame by frame and measures the same gap |
| `compare.py` | the side-by-side table |
| `results/` | run records |
