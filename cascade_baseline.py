"""Freeze the cascade numbers Moshi is compared against.

Pooled from the sibling repo's own run records (another agent's work, read-only)
at aliveness-threshold@0d202a6. Recomputed here rather than copied from its
prose, so the comparison rests on the raw per-turn records.

Gap definition is the sibling's, which is gap.py's: agent speech onset minus
user speech offset, both silence-trimmed.
"""
import json, statistics as s
from pathlib import Path

SIB = Path.home() / "Desktop/Playground/aliveness-threshold/live/results"

CONFIGS = {
    "cascade, unoptimised": [
        "opt-baseline-n20.json", "opt-baseline-blackhole-n20.json",
        "opt-baseline-newcode-n20.json", "opt-baseline-rep2-n20.json",
        "opt-baseline-rep3-n20.json"],
    "cascade, speculative (arm 80, base.en)": [
        "opt-fast-arm80-n20.json", "opt-fast-arm80-rep2-n20.json"],
    "cascade, speculative (arm 80, tiny.en)": [
        "opt-fast-arm80-tiny-n20.json", "opt-fast-tiny-rep2-n20.json"],
    "cascade, speculative (arm 40, tiny.en, hang 250)": [
        "opt-fast-tiny-arm40-hang250-n20.json"],
}


def stats(gaps):
    q = s.quantiles(gaps, n=4)
    return dict(n=len(gaps), median=round(s.median(gaps), 1),
                iqr_lo=round(q[0], 1), iqr_hi=round(q[2], 1),
                p90=round(sorted(gaps)[int(0.9 * len(gaps)) - 1], 1),
                under_400=sum(g < 400 for g in gaps))


def load(files):
    gaps, wers, fe = [], [], 0
    for f in files:
        d = json.load(open(SIB / f))
        gaps += [t["gap_ms"] for t in d["turns"]]
        wers += [t["wer"] for t in d["turns"] if t.get("wer") is not None]
        fe += (d.get("endpointing") or {}).get("false_endpoints", 0)
    r = stats(gaps)
    r["wer"] = round(s.mean(wers), 4) if wers else None
    r["false_endpoints"] = fe
    return r


def table():
    return {k: load(v) for k, v in CONFIGS.items()}


def demo():
    t = table()
    b = t["cascade, unoptimised"]
    f = t["cascade, speculative (arm 80, tiny.en)"]
    assert b["n"] == 100 and abs(b["median"] - 807.4) < 1, b
    assert f["n"] == 40 and abs(f["median"] - 386.2) < 1, f
    # speculation must actually beat the serial baseline, not just look different
    assert f["median"] < b["median"] / 2
    print("demo ok")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        for k, v in table().items():
            print(f'{k:50s} n={v["n"]:3d} med={v["median"]:6.1f} '
                  f'IQR {v["iqr_lo"]:.0f}-{v["iqr_hi"]:.0f} wer={v["wer"]} '
                  f'false_endpoints={v["false_endpoints"]}')
