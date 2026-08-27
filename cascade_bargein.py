"""Can the tuned cascade be interrupted? Measured with its own Player.

The claim in the README is structural: `live/loop.py`'s `run_turn` calls
`player.play(y)` then `player.wait()`, `Player` holds the whole reply in one
buffer, and `capture()` builds a fresh input queue per turn. So the agent cannot
stop early and the interrupting speech is discarded.

This measures the first half of that rather than asserting it. It imports the
sibling's real `Player` (not a copy) onto the same BlackHole device the cascade
was measured on, plays a real piper reply, "interrupts" at a chosen offset, and
times how long the agent keeps talking afterwards.

Nothing here edits the sibling repo. It only imports from it.

    .venv/bin/python cascade_bargein.py --n 10 --out results/cascade-bargein.json
"""
from __future__ import annotations
import argparse, json, sys, time, statistics as s
from pathlib import Path
import numpy as np

SIB = Path.home() / "Desktop/Playground/aliveness-threshold"
sys.path.insert(0, str(SIB))
VOICE = SIB / "models/piper-live/en_US-lessac-medium.onnx"

# Replies the cascade actually produced, taken from its own run records, so the
# durations here are the durations it really plays.
def real_replies(k=10):
    import glob
    out = []
    for f in sorted(glob.glob(str(SIB / "live/results/opt-fast-*tiny*.json"))):
        for t in json.load(open(f))["turns"]:
            # duplicates kept on purpose: repeated replies are what the cascade
            # really produces, so this is its real reply-duration distribution
            if t.get("reply"):
                out.append(t["reply"])
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--interrupt-at-ms", type=float, default=900.0,
                    help="ms into the agent's reply at which the user starts talking")
    ap.add_argument("--device", default="BlackHole 2ch")
    ap.add_argument("--out", default="results/cascade-bargein.json")
    args = ap.parse_args()

    from harness import tts, audio
    from live import loop as L

    voice = tts.Voice("piper", name=str(VOICE))
    player = L.Player(device=args.device)
    replies = real_replies(args.n)

    # the interrupting utterance, same voice, so the demo has a real user track
    interrupt_audio = voice.synth("Wait, actually, hold on.")

    turns = []
    for i, text in enumerate(replies):
        y = voice.synth(text)
        dur_ms = 1000.0 * len(y) / audio.SR
        player.play(y)
        t_play = time.perf_counter()
        while player.t_first is None:
            time.sleep(0.001)
        t_onset = player.t_first

        # the user starts talking `interrupt_at_ms` into the reply
        target = t_onset + args.interrupt_at_ms / 1000.0
        while time.perf_counter() < target:
            time.sleep(0.001)
        t_interrupt = time.perf_counter()

        # ...and we time how long the agent keeps going. Nothing is signalled to
        # it, because there is no channel to signal on: this is the point.
        player.wait(timeout=20.0)
        t_stop = time.perf_counter()

        stop_ms = (t_stop - t_interrupt) * 1000.0
        if i == 0:  # keep turn 0 so the interruption can be listened to
            import demo as _demo
            at = int(args.interrupt_at_ms * audio.SR / 1000)
            user = np.zeros(max(len(y), at + len(interrupt_audio)), dtype=np.float32)
            user[at:at + len(interrupt_audio)] = interrupt_audio
            _demo.write(user, y, "demo/cascade-bargein", sr=audio.SR)

        turns.append(dict(
            turn=i, reply=text, reply_audio_ms=round(dur_ms, 1),
            interrupt_at_ms=args.interrupt_at_ms,
            stop_latency_ms=round(stop_ms, 1),
            stopped_early=bool(stop_ms < dur_ms - args.interrupt_at_ms - 50),
        ))
        print(f"  turn {i}: reply {dur_ms:.0f}ms, interrupted at "
              f"{args.interrupt_at_ms:.0f}ms, kept talking {stop_ms:.0f}ms", file=sys.stderr)
    player.close()

    v = [t["stop_latency_ms"] for t in turns]
    q = s.quantiles(v, n=4) if len(v) > 3 else [min(v), s.median(v), max(v)]
    rec = dict(
        run_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        what="how long the cascade keeps talking after the user starts interrupting",
        device=player.device if hasattr(player, "device") else args.device,
        interrupt_at_ms=args.interrupt_at_ms,
        stop_latency_ms=dict(n=len(v), median=round(s.median(v), 1),
                             iqr_lo=round(q[0], 1), iqr_hi=round(q[2], 1),
                             min=round(min(v), 1), max=round(max(v), 1)),
        n_stopped_early=sum(t["stopped_early"] for t in turns),
        note=("Player has no stop path and capture() builds a fresh input queue "
              "per turn, so the interrupting speech is also discarded."),
        turns=turns,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("stop_latency_ms", "n_stopped_early")}, indent=1))


def demo():
    """One runnable check: a fresh queue per turn really does drop pending audio."""
    from live import loop as L
    import inspect, queue
    src = inspect.getsource(L.capture)
    assert "queue.Queue()" in src, "capture no longer builds its own queue; recheck the claim"
    assert not hasattr(L.Player, "stop"), "Player grew a stop(); the README claim is stale"
    # and wait() only returns when the buffer is exhausted
    assert "self.buf is not None" in inspect.getsource(L.Player.wait)
    print("demo ok: Player has no stop path, capture makes a fresh queue per turn")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
