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
        return (np.asarray(out, dtype=np.float32) if out is not None else None), text


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
    """Gap = Moshi speech onset after user speech offset. gap.py's definition."""
    off = gap.speech_offset_ms(x, SR_MOSHI)
    if off is None:
        return None, None
    segs = gap.segments(y, SR_MOSHI, **gap.SEG_KW)
    after = [a for a, b in segs if a >= off]
    return (after[0] - off) if after else None, off


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
            ba = 300 + len(prompts[p]) / SR_MOSHI * 1000 + 900  # ~0.9s into the reply
        x, y, step_ms, text = run_turn(m, prompts[p], tail_ms=args.tail_ms,
                                       bargein_at_ms=ba if args.bargein else None,
                                       bargein_pcm=interrupt)
        wall = (time.perf_counter() - t0) * 1000.0
        g, off = measure(x, y)
        stream_ms = len(x) / SR_MOSHI * 1000.0
        segs = gap.segments(y, SR_MOSHI, **gap.SEG_KW)
        turns.append(dict(
            turn=i, prompt=p, gap_ms=None if g is None else round(g, 1),
            user_offset_ms=None if off is None else round(off, 1),
            reply_text=text, rtf=round(wall / stream_ms, 3),
            wall_ms=round(wall, 1), stream_ms=round(stream_ms, 1),
            step_ms_median=round(s.median(step_ms), 2), n_frames=len(step_ms),
            out_speech_ms=round(sum(b - a for a, b in segs), 1),
            out_segments=len(segs),
            bargein_at_ms=None if not args.bargein else round(ba, 1),
        ))
        print(f"  turn {i}: gap={turns[-1]['gap_ms']} rtf={turns[-1]['rtf']} "
              f"text={text[:60]!r}", file=sys.stderr)
        if args.out:
            np.save(Path(args.out).with_suffix(f".turn{i}.out.npy"), y) if i < 3 else None

    rec = dict(
        run_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        model="kyutai/moshiko-mlx-q4 (4-bit, MLX)", frame_ms=FRAME_MS,
        sample_rate=SR_MOSHI, model_load_ms=round(m.load_ms, 1),
        loadavg=os.getloadavg(), n_turns=args.n, mode="bargein" if args.bargein else "turns",
        gap_definition=("agent speech onset - user speech offset, both silence-trimmed, "
                        "gap.py segments(merge_gap_ms=30, min_len_ms=20) -- vendored from "
                        "aliveness-threshold harness/audio.py@b7ccbb7"),
        gap_ms=stats([t["gap_ms"] for t in turns]),
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
    g, off = measure(x, y)
    assert abs(off - 900) < 15, f"user offset {off}"
    assert abs(g - 500) < 15, f"gap {g}, want 500"
    # a silent agent must report no gap rather than inventing one
    assert measure(x, np.zeros_like(y))[0] is None
    print(f"demo ok (500 ms gap measured as {g:.1f} ms)")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
