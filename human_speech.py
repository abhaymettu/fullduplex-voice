"""Does aggressive arming truncate real speakers? Recomputed from raw records.

The README claims speculation adds no false endpoints on human speech, and that
what truncates people is the endpointer itself. Both claims come from another
agent's repo (~/Desktop/Playground/expressive-s2s, 24 held-out CREMA-D actor
recordings). This recomputes them from its per-turn JSON rather than quoting its
prose, the same way cascade_baseline.py treats the cascade.

Truncation is detected as `ref_speech_offset_ms - speech_offset_ms`, i.e. the
measured end of user speech landing earlier than the reference. It is NOT
detected as `endpoint_hangover_ms < 0`: cutting the buffer drags the measured
offset earlier too, so the hangover stays positive on genuinely truncated turns.
That trap is what this file exists to keep nailed down.
"""
import json, statistics as s, sys
from pathlib import Path

SRC = Path.home() / "Desktop/Playground/expressive-s2s/runs"
ARMS = {"H0 serial control": "h0-human-control.json",
        "H1 fast, arm 80": "h1-human-fast.json"}
LOST_MS = 100.0  # below this, offset jitter rather than a cut


def load(f):
    return json.load(open(SRC / f))


def truncated(d):
    return [(t["label"], t["ref_speech_offset_ms"] - t["speech_offset_ms"],
             t["stage_ms"].get("endpoint_hangover_ms"), t["transcript"])
            for t in d["turns"]
            if t["ref_speech_offset_ms"] - t["speech_offset_ms"] > LOST_MS]


def table():
    out = {}
    for name, f in ARMS.items():
        d = load(f)
        g = [t["gap_ms"] for t in d["turns"] if t.get("gap_ms")]
        q = s.quantiles(g, n=4)
        out[name] = dict(n=len(g), median=round(s.median(g), 1),
                         iqr_lo=round(q[0], 1), iqr_hi=round(q[2], 1),
                         false_endpoints=d["endpointing"]["false_endpoints"],
                         truncated=len(truncated(d)),
                         spec=d["speculation"])
    return out


def main():
    for k, v in table().items():
        sp = v["spec"]
        print(f'{k:20s} n={v["n"]:3d} med={v["median"]:6.1f} '
              f'IQR {v["iqr_lo"]:.0f}-{v["iqr_hi"]:<4.0f} '
              f'false_endpoints={v["false_endpoints"]} truncated={v["truncated"]} '
              f'spec={sp["turns_served_speculatively"]}/{v["n"]} '
              f'launched={sp["pipelines_launched"]}')

    d1 = load(ARMS["H1 fast, arm 80"])
    h0 = {t["label"]: t["gap_ms"] for t in load(ARMS["H0 serial control"])["turns"]}
    spec = [t["gap_ms"] for t in d1["turns"] if t["speculated"]]
    fell = [t for t in d1["turns"] if not t["speculated"]]
    print(f'\n  arming is not committing:')
    print(f'    served speculatively : n={len(spec):2d} med={s.median(spec):6.1f}')
    print(f'    fell back to serial  : n={len(fell):2d} '
          f'med={s.median([t["gap_ms"] for t in fell]):6.1f}')
    print(f'    ...same clips in H0  :        '
          f'med={s.median([h0[t["label"]] for t in fell if t["label"] in h0]):6.1f}')

    print(f'\n  truncated turns, and what a hangover<0 detector would have said:')
    for name, f in ARMS.items():
        for lab, lost, hang, tr in truncated(load(f)):
            print(f'    {name[:2]} {lab:26s} lost {lost:6.0f} ms  '
                  f'hangover {hang:+6.1f} ms  {tr[:28]!r}')


def demo():
    """The two claims the README rests on, as assertions."""
    t = table()
    h0, h1 = t["H0 serial control"], t["H1 fast, arm 80"]
    assert h0["false_endpoints"] == h1["false_endpoints"] == 3, (h0, h1)
    assert h0["truncated"] == h1["truncated"] == 3, "truncation count changed"
    assert h1["median"] < h0["median"] - 200, "speculation stopped paying"
    # the trap: every truncated turn still reports a POSITIVE hangover, so a
    # hangover<0 detector sees none of them
    for f in ARMS.values():
        for lab, lost, hang, _ in truncated(load(f)):
            assert hang > 0, f"{lab} hangover {hang} -- detector story changed"
    # speculation wastes work rather than committing to a bad guess
    sp = h1["spec"]
    assert sp["pipelines_launched"] > sp["turns_served_speculatively"]
    print(f'demo ok: arm80 added 0 false endpoints '
          f'({h0["median"]:.0f} -> {h1["median"]:.0f} ms), '
          f'{h1["truncated"]}/{h1["n"]} truncated in both arms, all with hangover > 0')


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
