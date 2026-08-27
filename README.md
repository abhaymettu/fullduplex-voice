# fullduplex-voice

**A tuned cascade answers in 386 ms on a laptop, and 499 ms on real human speech.
The architectural floor everyone assumes is there is mostly the endpointer sitting
on its hands, waiting out silence it could have been working through.**

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

> **Caveat.** Every number above is measured on synthetic piper prompts, so
> **386 ms is not validated on human speech** and should not be quoted as if it
> were.

### Does aggressive arming cut real speakers off? No. (tested)

The obvious worry is that `ARM = 80 ms` only survives because the piper test
talker's longest internal pause is 65 ms, and that a human pausing mid-sentence
would get truncated. **That was tested directly and it is false.** A sibling
project (`~/Desktop/Playground/expressive-s2s`) ran 24 held-out CREMA-D human
actor recordings through a serial control and the fast arm back to back. I
recomputed all of this from its raw per-turn records, `runs/h0-human-control.json`
and `runs/h1-human-fast.json`:

| arm | n | median gap | IQR | false endpoints |
|---|---|---|---|---|
| H0 serial control | 24 | 772.2 ms | 578–833 | **3 / 24** |
| H1 `--fast --arm 80` | 24 | 499.1 ms | 400–722 | **3 / 24** |

**Speculation added zero false endpoints.** It cut the median by 273 ms and
truncated nobody who was not already being truncated.

**The mechanism is that arming is not committing.** When a speaker resumes during
the pause, the speculative snapshot goes stale, the turn silently falls back to
serial, and the bad guess is thrown away. In H1, 14 of 24 turns were served
speculatively (median **441.8 ms**); the other 10 fell back (median **716.5 ms**,
against **779.6 ms** for those same clips in the serial control). That is why the
fast arm's p75 is essentially the serial number. Getting the guess wrong costs
CPU — **51 pipelines launched to serve 14 turns, 37 of them wasted** — not a
truncated user.

**What does truncate real speakers is the endpointer itself, at 12.5%,
independent of speculation.** The three losses in each arm are not near-misses:
they drop 1760 / 2270 / 1700 ms and transcribe as `"You"` or empty. Two of the
three clips are identical across both arms. So the real target for anyone
chasing this further is the endpointer, not the arming threshold.

> **A measurement trap worth repeating.** That repo's earlier "0 / 48 cut off"
> claim rested on a detector that flags `endpoint_hangover_ms < 0` — which
> *structurally cannot* see truncation, because cutting the buffer drags the
> measured speech offset earlier too, so the hangover stays positive. On the six
> genuinely truncated turns above it reads **+5.0, +6.7, +5.0, +30.6, +8.2,
> +31.6 ms**. Truncation has to be caught by comparing the measured offset
> against a reference offset, which is what the `truncated` flag above does.

Two things this does *not* license. The CREMA-D runs use a heavier stack than the
386 ms configuration above (`base.en` final, plus an emotion classifier inside
the hangover), so **499 ms and 386 ms are not the same measurement** and must not
be put in one column. And `ARM = 40 ms` — the 347.8 ms row — is *more* aggressive
than the 80 ms that was actually tested, so it stays out of the headline for the
right reason: **untested at that threshold**, not because arming is dangerous.

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

- **The prompts are synthetic.** Every cascade number and every Moshi number in
  the side-by-side uses the same five piper-rendered prompts. That makes the
  comparison fair and makes neither of them a claim about human speech. The one
  human-speech result here (24 CREMA-D recordings, above) comes from a different
  and heavier stack and is reported separately for that reason.
- **The endpointer truncates ~12.5% of real speakers**, in both the serial and
  speculative arms. That is a property of the endpointer, not of speculation, and
  it is the largest unfixed problem in the cascade.
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

## What went wrong getting Moshi to run

Reported because the cost is part of the result.

**The link was the binding constraint all night.** Several agents were pulling at
once. Measured: Cloudflare 166 KB/s, PyPI 41 KB/s, HuggingFace stalling to
25 KB/s. A 31 MB `mlx-metal` wheel could not be fetched at all; `mlx`, `piper`,
`onnxruntime` and `sounddevice` were copied out of a sibling venv instead (same
Python 3.12, same platform, zero bandwidth).

**Two `hf download` processes on the same repo corrupted the weights.** I started
a second one by accident. `huggingface_hub` fetches byte ranges in parallel, so a
partial file is *sparse*: regions that were written are correct, regions nobody
wrote yet read back as NUL. The tell was `du` reporting 34 MB for a file `ls`
called 1.1 GB.

The damage survived to the end and the model would not load:

```
RuntimeError: [json.exception.parse_error.101] parse error at line 1, column 34209:
invalid string: control character U+0000 (NUL) must be escaped; last read: '"depformer.sli<U+0000>'
```

A size check called the file complete — it was 4,820,474,823 bytes against a true
4,805,545,317, and **being larger than the target was itself the corruption
signature**, not evidence of success. `repair.py` fixed it without refetching
4.5 GB: pull the 168 KB header, parse it to confirm the true total, truncate the
junk tail, scan for NUL runs, and re-fetch only those ranges. **73.8% of the file
(3548 MB) was holes**, and their boundaries landed exactly on 1 GB marks — the
signature of the parallel chunker. Re-scanning holes on each run makes it
resumable for free.

Lesson worth keeping: for a large file over a bad link, **size is not a
completion check**. The check is that the container parses and the thing loads.

## Layout

| file | what |
|---|---|
| `gap.py` | the latency definition, vendored and pinned. `--demo` self-checks it |
| `floor.py` | the three tests above, from the sibling's per-turn records |
| `cascade_baseline.py` | the cascade numbers, recomputed from raw records |
| `human_speech.py` | the CREMA-D arming result, recomputed from raw records |
| `moshi_run.py` | runs Moshi frame by frame and measures the same gap |
| `compare.py` | the side-by-side table |
| `results/` | run records |

## Checks

Every module carries one self-check. Run them all:

```
for m in gap floor cascade_baseline cascade_bargein human_speech moshi_run; do .venv/bin/python $m.py --demo; done
```

`gap.py --demo` measures a synthetic 800 ms gap; `moshi_run.py --demo` a synthetic
500 ms one; `cascade_bargein.py --demo` re-reads the sibling's current source and
fails if `Player` ever grows a `stop()`.
