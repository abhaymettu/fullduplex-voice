"""Run Moshi (Kyutai, full-duplex speech-to-speech) on this laptop and measure
its response gap with the same definition the cascade is scored on.

Moshi has no turn structure. It consumes one 80 ms frame of user audio and emits
one 80 ms frame of its own audio, every step, forever. There is no endpointer,
no ASR, no TTS -- so there is nothing to overlap and nothing to speculate about.
The "gap" is simply: after the user's speech stops, how long until Moshi's
output stream contains speech.

That is measured with gap.py, which is the sibling cascade's own segmentation,
so the two numbers mean the same thing.

Two clocks, and both are reported, because for Moshi they can differ:

- **stream time** -- frame_index * 80 ms. This is the architectural latency: how
  far into the conversation the reply lands. Directly comparable to the
  cascade's gap, which ran in real time.
- **wall time** -- how long the machine actually took. If the real-time factor
  is above 1.0 the model cannot hold a live conversation on this hardware no
  matter how good its architectural latency is. Reported separately, never
  folded into the gap.

Usage:
    .venv/bin/python moshi_run.py --n 20 --out results/moshi-n20.json
    .venv/bin/python moshi_run.py --bargein --n 10 --out results/moshi-bargein.json
"""

from __future__ import annotations

import argparse, json, os, sys, time, statistics as s
from pathlib import Path

import numpy as np

import gap

SR_MOSHI = 24000
FRAME = 1920           # 80 ms at 24 kHz -- Mimi's frame, not a choice
FRAME_MS = 1000.0 * FRAME / SR_MOSHI
SIB = Path.home() / "Desktop/Playground/aliveness-threshold"
VOICE = SIB / "models/piper-live/en_US-lessac-medium.onnx"

# the sibling live loop's prompts, verbatim, so the stimulus matches
PROMPTS = [
    "What time do you close on Sunday?",
    "Is there parking near the entrance?",
    "How much does the annual pass cost?",
    "Can I bring a dog inside?",
    "Where do I pick up my order?",
]

MODELS = Path(__file__).parent / "models"


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)),
                     x.astype(np.float64)).astype(np.float32)


def render_prompts() -> dict[str, np.ndarray]:
    """The same piper voice the cascade was measured with, resampled to 24 kHz."""
    sys.path.insert(0, str(SIB))
    from harness import tts  # sibling's, unmodified
    v = tts.Voice("piper", name=str(VOICE))
    return {p: resample(v.synth(p), gap.SR, SR_MOSHI) for p in PROMPTS}


class Moshi:
    """Thin wrapper over the moshi_mlx streaming API. One step = one 80 ms frame."""

    def __init__(self, quantized: int = 4, max_steps: int = 4096):
        import mlx.core as mx, mlx.nn as nn, rustymimi, sentencepiece
        from moshi_mlx import models, utils
        self.mx, self.models, self.utils = mx, models, utils

        t0 = time.perf_counter()
        self.text_tok = sentencepiece.SentencePieceProcessor(
            str(MODELS / "tokenizer_spm_32k_3.model"))
        mx.random.seed(299792458)
        model = models.Lm(models.config_v0_1())
        model.set_dtype(mx.bfloat16)
        if quantized:
            nn.quantize(model, bits=quantized, group_size=32 if quantized == 4 else 64)
        model.load_weights(str(MODELS / "model.q4.safetensors"), strict=True)
        model.warmup()
        self.model = model
        self.max_steps = max_steps
        self.gen = models.LmGen(model=model, max_steps=max_steps,
                                text_sampler=utils.Sampler(),
                                audio_sampler=utils.Sampler(), check=False)
        self.mimi = rustymimi.StreamTokenizer(str(MODELS / "mimi.safetensors"))
        self.load_ms = (time.perf_counter() - t0) * 1000.0

    def reset(self):
        """Fresh conversation state between turns."""
        self.gen = self.models.LmGen(model=self.model, max_steps=self.max_steps,
                                     text_sampler=self.utils.Sampler(),
                                     audio_sampler=self.utils.Sampler(), check=False)

    def step(self, pcm: np.ndarray):
        """One 80 ms frame in, (pcm_out|None, text|None) out."""
        self.mimi.encode(pcm.astype(np.float32))
        enc = None
        while enc is None:
            enc = self.mimi.get_encoded()
            if enc is None:
                time.sleep(1e-3)  # rustymimi decodes on its own thread; a tight
                                  # spin starves it behind the GIL and hangs
        tokens = self.mx.array(enc).transpose(1, 0)[:, :8]
        text_token = self.gen.step(tokens)[0].item()
        text = None
        if text_token not in (0, 3):
            text = self.text_tok.id_to_piece(text_token).replace("▁", " ")
        out = None
        at = self.gen.last_audio_tokens()
        if at is not None:
            self.mimi.decode(np.array(at).astype(np.uint32))
            while out is None:
                out = self.mimi.get_decoded()
                if out is None:
                    time.sleep(1e-3)
        if out is None:
            return None, text
        # local.py asserts the decoded frame is (1920,); flatten and fit to the
        # frame either way, so one odd frame cannot silently shift the whole
        # output stream out of alignment with the input.
        o = np.asarray(out, dtype=np.float32).reshape(-1)
        if len(o) != FRAME:
            o = (o[:FRAME] if len(o) > FRAME
                 else np.pad(o, (0, FRAME - len(o))))
        return o, text


def run_turn(m: Moshi, prompt_pcm: np.ndarray, lead_ms=300.0, tail_ms=4000.0,
             bargein_at_ms=None, bargein_pcm=None):
    """Feed lead silence + prompt + trailing silence; capture Moshi's stream.

    Returns the input and output waveforms plus per-frame wall times. Nothing is
    measured here -- measurement is gap.py's job, on the waveforms.
    """
    m.reset()
    lead = np.zeros(int(lead_ms * SR_MOSHI / 1000), dtype=np.float32)
    tail = np.zeros(int(tail_ms * SR_MOSHI / 1000), dtype=np.float32)
    x = np.concatenate([lead, prompt_pcm, tail])

    # barge-in: the user starts talking again partway through Moshi's reply
    if bargein_at_ms is not None and bargein_pcm is not None:
        at = int(bargein_at_ms * SR_MOSHI / 1000)
        end = min(len(x), at + len(bargein_pcm))
        x[at:end] += bargein_pcm[: end - at]

    n_frames = len(x) // FRAME
    outs, step_ms, texts = [], [], []
    for i in range(n_frames):
        t0 = time.perf_counter()
        o, t = m.step(x[i * FRAME:(i + 1) * FRAME])
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        outs.append(o if o is not None else np.zeros(FRAME, dtype=np.float32))
        if t:
            texts.append(t)
    y = np.concatenate(outs)[: len(x)]
    return x[: n_frames * FRAME], y, step_ms, "".join(texts).strip()


def measure(x, y):
    """Gap = Moshi speech onset after user speech offset. gap.py's definition.

    Full duplex complicates this in a way the cascade cannot: Moshi may already
    be talking when the user stops. Taking "the first onset after the offset"
    would then skip the segment it is currently in the middle of and report the
    *next* utterance, badly overstating the gap. So an overlapping segment is
    detected and reported as such (gap 0, overlapping=True) rather than silently
    dropped. That is a real behaviour, not an error, and it is counted.
    """
    off = gap.speech_offset_ms(x, SR_MOSHI)
    if off is None:
        return None, None, False
    segs = gap.segments(y, SR_MOSHI, **gap.SEG_KW)
    if any(a <= off < b for a, b in segs):
        return 0.0, off, True          # already speaking when the user stopped
    after = [a for a, b in segs if a >= off]
    return ((after[0] - off) if after else None), off, False


def measure_bargein(x, y, bargein_at_ms):
    """How long Moshi keeps talking after the user starts interrupting.

    Comparable to cascade_bargein.py's number by construction: both are
    "ms of agent speech after the interrupting utterance begins".

    Returns (stop_latency_ms, interrupt_onset_ms, was_speaking). If Moshi was not
    actually talking at that moment there is nothing to interrupt, and the turn
    is flagged rather than scored -- otherwise a silent agent would look like an
    infinitely polite one.
    """
    ins = [a for a, b in gap.segments(x, SR_MOSHI, **gap.SEG_KW) if a >= bargein_at_ms - 50]
    if not ins:
        return None, None, False
    onset = ins[0]
    segs = gap.segments(y, SR_MOSHI, **gap.SEG_KW)
    active = [(a, b) for a, b in segs if a <= onset < b]
    if not active:
        return None, onset, False      # nothing to interrupt
    return active[0][1] - onset, onset, True


def stats(v):
    v = [a for a in v if a is not None]
    if not v:
        return {}
    q = s.quantiles(v, n=4) if len(v) > 3 else [min(v), s.median(v), max(v)]
    return dict(n=len(v), median=round(s.median(v), 1), iqr_lo=round(q[0], 1),
                iqr_hi=round(q[2], 1), min=round(min(v), 1), max=round(max(v), 1),
                mean=round(s.mean(v), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/moshi.json")
    ap.add_argument("--tail-ms", type=float, default=4000.0)
    ap.add_argument("--bargein", action="store_true")
    ap.add_argument("--bargein-at-ms", type=float, default=None,
                    help="ms into the stream to inject interrupting speech")
    args = ap.parse_args()

    prompts = render_prompts()
    m = Moshi()
    print(f"model loaded in {m.load_ms/1000:.1f}s", file=sys.stderr)

    interrupt = prompts[PROMPTS[3]] if args.bargein else None
    turns = []
    for i in range(args.n):
        p = PROMPTS[i % len(PROMPTS)]
        t0 = time.perf_counter()
        ba = args.bargein_at_ms
        if args.bargein and ba is None:
            # Moshi answers ~165 ms after the user stops and its replies are
            # short, so interrupting 900 ms in usually hits silence and measures
            # nothing. 400 ms lands ~235 ms into the reply, while it is talking.
            ba = 300 + len(prompts[p]) / SR_MOSHI * 1000 + 400
        x, y, step_ms, text = run_turn(m, prompts[p], tail_ms=args.tail_ms,
                                       bargein_at_ms=ba if args.bargein else None,
                                       bargein_pcm=interrupt)
        wall = (time.perf_counter() - t0) * 1000.0
        g, off, overlap = measure(x, y)
        stream_ms = len(x) / SR_MOSHI * 1000.0
        segs = gap.segments(y, SR_MOSHI, **gap.SEG_KW)
        turns.append(dict(
            turn=i, prompt=p, gap_ms=None if g is None else round(g, 1),
            user_offset_ms=None if off is None else round(off, 1),
            reply_text=text, rtf=round(wall / stream_ms, 3),
            wall_ms=round(wall, 1), stream_ms=round(stream_ms, 1),
            step_ms_median=round(s.median(step_ms), 2), n_frames=len(step_ms),
            out_speech_ms=round(sum(b - a for a, b in segs), 1),
            out_segments=len(segs), overlapping=overlap,
            bargein_at_ms=None if not args.bargein else round(ba, 1),
        ))
        if args.bargein:
            sl, on, spk = measure_bargein(x, y, ba)
            turns[-1].update(stop_latency_ms=None if sl is None else round(sl, 1),
                             interrupt_onset_ms=None if on is None else round(on, 1),
                             was_speaking_when_interrupted=spk)
        print(f"  turn {i}: gap={turns[-1]['gap_ms']} rtf={turns[-1]['rtf']} "
              f"text={text[:60]!r}", file=sys.stderr)
        # keep audio so the interruption can actually be listened to. In barge-in
        # mode keep every turn: whether the interruption lands while Moshi is
        # actually talking is not known until afterwards, and a demo of an
        # interruption that hit silence demonstrates nothing.
        if i == 0 or args.bargein:
            np.save(Path(args.out).with_suffix(f".turn{i}.user.npy"), x)
            np.save(Path(args.out).with_suffix(f".turn{i}.moshi.npy"), y)

    import mlx.core as _mx
    import resource as _res
    rec = dict(
        run_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cost=dict(
            weights_on_disk_mb=round(
                (MODELS / "model.q4.safetensors").stat().st_size / 1e6, 1),
            mimi_on_disk_mb=round((MODELS / "mimi.safetensors").stat().st_size / 1e6, 1),
            mlx_peak_mb=round(_mx.get_peak_memory() / 1e6, 1),
            mlx_active_mb=round(_mx.get_active_memory() / 1e6, 1),
            rss_peak_mb=round(_res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1e6, 1),
        ),
        model="kyutai/moshiko-mlx-q4 (4-bit, MLX)", frame_ms=FRAME_MS,
        sample_rate=SR_MOSHI, model_load_ms=round(m.load_ms, 1),
        loadavg=os.getloadavg(), n_turns=args.n, mode="bargein" if args.bargein else "turns",
        gap_definition=("agent speech onset - user speech offset, both silence-trimmed, "
                        "gap.py segments(merge_gap_ms=30, min_len_ms=20) -- vendored from "
                        "aliveness-threshold harness/audio.py@b7ccbb7"),
        gap_ms=stats([t["gap_ms"] for t in turns]),
        n_overlapping=sum(t["overlapping"] for t in turns),
        stop_latency_ms=(stats([t.get("stop_latency_ms") for t in turns])
                         if args.bargein else None),
        n_interrupted_while_speaking=(
            sum(bool(t.get("was_speaking_when_interrupted")) for t in turns)
            if args.bargein else None),
        n_silent=sum(t["gap_ms"] is None for t in turns),
        rtf=stats([t["rtf"] for t in turns]),
        turns=turns,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("gap_ms", "rtf", "model_load_ms")}, indent=1))


def demo():
    """One runnable check: measure() recovers a known gap from synthetic streams."""
    sr = SR_MOSHI
    tone = lambda ms: (0.3 * np.sin(2*np.pi*220*np.arange(int(ms*sr/1000))/sr)).astype(np.float32)
    sil = lambda ms: np.zeros(int(ms*sr/1000), dtype=np.float32)
    x = np.concatenate([sil(300), tone(600), sil(3000)])          # user stops at 900ms
    y = np.concatenate([sil(1400), tone(800), sil(1700)])         # moshi starts at 1400ms
    g, off, ov = measure(x, y)
    assert abs(off - 900) < 15, f"user offset {off}"
    assert abs(g - 500) < 15, f"gap {g}, want 500"
    assert not ov
    # a silent agent must report no gap rather than inventing one
    assert measure(x, np.zeros_like(y))[0] is None
    # an agent already talking when the user stops is gap 0, not the NEXT utterance
    y2 = np.concatenate([sil(600), tone(700), sil(600), tone(500), sil(2100)])
    g2, _, ov2 = measure(x, y2)
    assert ov2 and g2 == 0.0, f"overlap not detected: gap={g2} overlapping={ov2}"
    # barge-in: user speaks again at 2000ms while the agent is talking 1400-2600ms
    xb = np.concatenate([sil(300), tone(600), sil(1100), tone(500), sil(1500)])
    yb = np.concatenate([sil(1400), tone(1200), sil(1400)])
    sl, on, spk = measure_bargein(xb, yb, 2000)
    assert spk and abs(on - 2000) < 20 and abs(sl - 600) < 20, (sl, on, spk)
    # a silent agent has nothing to interrupt and must say so, not score 0
    assert measure_bargein(xb, np.zeros_like(yb), 2000) == (None, 2000.0, False)
    print(f"demo ok (500 ms gap measured as {g:.1f} ms)")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
