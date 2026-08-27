"""The gap definition, vendored so both architectures are measured identically.

    gap = onset of agent speech - offset of user speech, both silence-trimmed

VENDORED, NOT AUTHORED HERE. `frame_rms`, `speech_mask` and `segments` below are
copied verbatim (constants included) from

    ~/Desktop/Playground/aliveness-threshold/harness/audio.py  @ b7ccbb7

which is another agent's repo and is being edited concurrently. Copied rather
than imported on purpose: this file must not change under me mid-measurement,
and the cascade numbers I compare against were produced by exactly this code.
If that repo's segmentation changes, these numbers stop being comparable to its
newer ones and this copy must be re-pinned deliberately.

The parameters that matter are the sibling's own measurement parameters:
SEG_KW = {"merge_gap_ms": 30.0, "min_len_ms": 20.0}. Both `exchange.measure_exchange`
and `live/loop.py` use those; so does everything here.
"""

from __future__ import annotations
import numpy as np

SR = 22050
FRAME_MS = 5.0
SEG_KW = {"merge_gap_ms": 30.0, "min_len_ms": 20.0}  # sibling live/loop.py


def samples(ms: float, sr: int = SR) -> int:
    return int(round(ms * sr / 1000.0))


def frame_rms(x, sr: int = SR, frame_ms: float = FRAME_MS):
    hop = samples(frame_ms, sr)
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[: n * hop].reshape(n, hop)
    return np.sqrt((f.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)


def speech_mask(x, sr=SR, rel_db=-35.0, abs_db=-55.0, frame_ms=FRAME_MS):
    r = frame_rms(x, sr, frame_ms)
    if r.size == 0 or r.max() <= 0:
        return np.zeros(r.size, dtype=bool)
    return (r >= r.max() * 10 ** (rel_db / 20.0)) & (r >= 10 ** (abs_db / 20.0))


def segments(x, sr=SR, merge_gap_ms=60.0, min_len_ms=30.0, frame_ms=FRAME_MS, **kw):
    """Speech segments as (start_ms, end_ms)."""
    m = speech_mask(x, sr, frame_ms=frame_ms, **kw)
    if not m.any():
        return []
    idx = np.flatnonzero(m)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    out = []
    for s, e in zip(starts, ends):
        s_ms, e_ms = s * frame_ms, (e + 1) * frame_ms
        if out and s_ms - out[-1][1] < merge_gap_ms:
            out[-1][1] = e_ms
        else:
            out.append([s_ms, e_ms])
    return [(a, b) for a, b in out if b - a >= min_len_ms]


# --- what the two architectures are scored on ---------------------------


def speech_offset_ms(x, sr=SR):
    """End of the last speech segment. The sibling's user-offset landmark."""
    segs = segments(x, sr, **SEG_KW)
    return segs[-1][1] if segs else None


def speech_onset_ms(x, sr=SR):
    """Start of the first speech segment. The agent-onset landmark."""
    segs = segments(x, sr, **SEG_KW)
    return segs[0][0] if segs else None


def demo():
    """One runnable check: a synthetic utterance with a known gap measures back."""
    sr = SR
    tone = lambda ms: (0.3 * np.sin(2 * np.pi * 220 * np.arange(samples(ms, sr)) / sr)
                       ).astype(np.float32)
    sil = lambda ms: np.zeros(samples(ms, sr), dtype=np.float32)
    # 200ms lead, 500ms speech, 800ms gap, 400ms speech
    x = np.concatenate([sil(200), tone(500), sil(800), tone(400)])
    segs = segments(x, sr, **SEG_KW)
    assert len(segs) == 2, f"expected 2 segments, got {segs}"
    measured = segs[1][0] - segs[0][1]
    assert abs(measured - 800) < 2 * FRAME_MS, f"gap measured {measured}, want 800"
    # offset/onset helpers agree with the segment ends
    assert abs(speech_offset_ms(x) - segs[-1][1]) < 1e-6
    assert abs(speech_onset_ms(x) - segs[0][0]) < 1e-6
    # silence has no landmarks, and says so rather than guessing
    assert speech_offset_ms(sil(500)) is None
    print(f"demo ok (800 ms gap measured as {measured:.1f} ms)")


if __name__ == "__main__":
    demo()
