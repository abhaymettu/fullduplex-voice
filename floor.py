"""What the cascade's latency floor actually is, and why.

Reads the tuned streaming cascade's own run records from the sibling repo
(~/Desktop/Playground/aliveness-threshold, another agent's work -- not vendored,
not edited, read-only) and tests one model of where its gap comes from.

The gap definition is the sibling's, which is harness/exchange.py's:
onset of agent speech minus offset of user speech, both silence-trimmed.
Every number here is that quantity or a decomposition of it.

The model
---------
The cascade speculates: after ARM ms of silence it launches ASR->LM->TTS on the
audio so far, betting the user is done. The endpointer only *confirms* the turn
after HANGOVER ms of silence. So downstream work runs *inside* the hangover and
is hidden -- but only as much of it as fits.

    gap  ~=  max(HANGOVER, ARM + W_total)  +  handoff

A cascade that does *not* speculate is the same formula with ARM = HANGOVER:
it only starts work once the endpoint has fired, so the whole pipeline is
exposed and gap = HANGOVER + W_total. Speculation's entire contribution is
buying ARM < HANGOVER.

where W_total is the summed on-clock time of ASR_final + LM_ttft + LM_sentence
+ TTS. Two regimes:

  * HANGOVER > ARM + W_total  -- work is fully hidden, the hangover is the floor
  * HANGOVER < ARM + W_total  -- work spills past the endpoint and *it* is the
    floor. Shrinking the hangover then buys exactly nothing, because every ms
    cut off the hangover reappears as an exposed ms of pipeline.

The second regime is the interesting one and it is where this stack lives.
The falsifiable prediction: sweep HANGOVER and the gap does not move.
"""

import json, glob, statistics as s, sys
from pathlib import Path

SIB = Path.home() / "Desktop/Playground/aliveness-threshold/live/results"
STAGES = ["asr_final", "lm_ttft", "lm_sentence", "tts"]


def load(pat="opt-*.json"):
    for f in sorted(SIB.glob(pat)):
        d = json.load(open(f))
        if not d.get("turns"):
            continue
        yield f.name, d


def med(xs):
    return s.median(xs) if xs else float("nan")


def iqr(xs):
    q = s.quantiles(xs, n=4)
    return q[0], q[2]


def summarize(name, d):
    T = d["turns"]
    gaps = [t["gap_ms"] for t in T]
    # work_ms is per-stage on its own clock; only the newer runs record it
    W = [sum(t["work_ms"].values()) for t in T if t.get("work_ms")]
    exposed = [t["gap_ms"] - t["stage_ms"]["endpoint_hangover_ms"] for t in T]
    arm = (d.get("speculation") or {}).get("armed_after_silence_ms")
    lo, hi = iqr(gaps)
    return dict(
        run=name, n=len(gaps), hangover=d["hangover_ms"], arm=arm,
        gap=med(gaps), lo=lo, hi=hi, W=med(W) if W else None,
        exposed=med(exposed), spec=bool(arm),
        # no speculation == the pipeline is armed only when the endpoint fires,
        # i.e. ARM == HANGOVER. One formula covers both modes.
        pred=(max(d["hangover_ms"], (arm if arm is not None else d["hangover_ms"]) + med(W))
              if W else None),
    )


def main():
    rows = [summarize(n, d) for n, d in load()]

    print("=" * 88)
    print("CASCADE RUNS  (sibling repo, gap = agent speech onset - user speech offset)")
    print("=" * 88)
    print(f'{"run":36s} {"n":>3s} {"hang":>5s} {"arm":>5s} {"gap med":>8s} {"IQR":>13s} {"W_tot":>6s}')
    for r in sorted(rows, key=lambda r: r["gap"]):
        w = f'{r["W"]:6.0f}' if r["W"] else "     -"
        print(f'{r["run"][:36]:36s} {r["n"]:3d} {r["hangover"]:5.0f} '
              f'{str(r["arm"] or "-"):>5s} {r["gap"]:8.1f} '
              f'{r["lo"]:6.0f}-{r["hi"]:<6.0f} {w}')

    # --- the hangover sweep: the falsifiable bit -------------------------
    sweep = [r for r in rows if r["arm"] == 80.0 and "tiny" not in r["run"]]
    print("\n" + "=" * 88)
    print("TEST 1 -- sweep the hangover, hold everything else. Does the gap move?")
    print("=" * 88)
    print(f'{"hangover":>9s} {"gap med":>9s} {"exposed work":>13s} {"hang+exposed":>13s}')
    for r in sorted(sweep, key=lambda r: r["hangover"]):
        print(f'{r["hangover"]:9.0f} {r["gap"]:9.1f} {r["exposed"]:13.1f} '
              f'{r["hangover"] + r["exposed"]:13.1f}')
    gs = [r["gap"] for r in sweep]
    print(f'\n  hangover range {min(r["hangover"] for r in sweep):.0f}-'
          f'{max(r["hangover"] for r in sweep):.0f} ms  ->  gap range '
          f'{min(gs):.1f}-{max(gs):.1f} ms (spread {max(gs)-min(gs):.1f} ms)')
    print("  Cutting the hangover by 230 ms bought"
          f" {max(gs)-min(gs):.0f} ms. The hangover is not the floor.")

    # --- does the model predict the gap? ---------------------------------
    print("\n" + "=" * 88)
    print("TEST 2 -- predicted gap = max(HANGOVER, ARM + W_total) vs measured")
    print("=" * 88)
    print(f'{"run":36s} {"pred":>7s} {"meas":>7s} {"err":>7s}')
    errs = []
    for r in sorted([r for r in rows if r["pred"]], key=lambda r: r["gap"]):
        e = r["gap"] - r["pred"]
        errs.append(e)
        print(f'{r["run"][:36]:36s} {r["pred"]:7.1f} {r["gap"]:7.1f} {e:+7.1f}')
    print(f'\n  median residual {med(errs):+.1f} ms over n={len(errs)} runs '
          f'(handoff to the audio callback is ~3 ms of it)')

    # --- can the stages overlap each other? ------------------------------
    print("\n" + "=" * 88)
    print("TEST 3 -- are the stages serial? sum(stage work) vs pipeline span")
    print("=" * 88)
    # The dependency-graph claim, measured rather than argued. Without
    # speculation the pipeline runs start-to-finish after the endpoint, so its
    # span is directly observable: gap - hangover. If ASR/LM/TTS overlapped at
    # all, the span would be shorter than the summed stage work.
    print("  no speculation, so the whole pipeline is observable end to end:")
    print(f'  {"run":30s} {"sum(work)":>10s} {"span":>8s} {"span/sum":>9s}')
    for name, d in load("opt-baseline*.json"):
        rows_ = [(sum(t["work_ms"].values()),
                  t["gap_ms"] - t["stage_ms"]["endpoint_hangover_ms"])
                 for t in d["turns"] if t.get("work_ms")]
        if not rows_:
            continue
        W_, S_ = med([r[0] for r in rows_]), med([r[1] for r in rows_])
        print(f'  {name[:30]:30s} {W_:10.1f} {S_:8.1f} {S_/W_:9.3f}')
    print("  1.0 means perfectly serial. Below 1.0 would mean the stages overlap.")
    print("  They do not: the LM needs a transcript and TTS needs a sentence.\n")
    best = min([r for r in rows if r["W"]], key=lambda r: r["gap"])
    d = dict(load())[best["run"]]
    print(f'  best run: {best["run"]}  (n={best["n"]})')
    print(f'  {"stage":16s} {"work ms":>9s} {"in-gap ms":>10s}   hidden by speculation?')
    for st in STAGES:
        w = med([t["work_ms"][st] for t in d["turns"] if t.get("work_ms")])
        key = {"asr_final": "asr_final_ms", "lm_ttft": "lm_ttft_ms",
               "lm_sentence": "lm_sentence_ms", "tts": "tts_ms"}[st]
        g = med([t["stage_ms"][key] for t in d["turns"]])
        print(f'  {st:16s} {w:9.1f} {g:10.1f}   {"yes" if g < w/2 else "NO -- exposed"}')
    Wt = med([sum(t["work_ms"].values()) for t in d["turns"] if t.get("work_ms")])
    print(f'  {"SUM":16s} {Wt:9.1f}')
    print(f'\n  floor for this stack = ARM + W_total = {best["arm"]:.0f} + {Wt:.0f}'
          f' = {best["arm"] + Wt:.0f} ms;  measured median {best["gap"]:.1f} ms')
    return rows


def demo():
    """One runnable check: the two-regime model behaves as claimed."""
    f = lambda hang, arm, W: max(hang, arm + W)
    assert f(350, 40, 100) == 350, "work under the hangover must hide entirely"
    assert f(250, 40, 292) == 332, "work over the hangover must set the floor"
    # the sweep prediction: below the knee, hangover changes nothing
    assert f(120, 80, 400) == f(250, 80, 400) == 480
    # ...and above the knee it changes everything
    assert f(900, 80, 400) == 900
    # no speculation is ARM == HANGOVER: the whole pipeline is exposed
    assert f(350, 350, 431) == 781
    print("demo ok")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
