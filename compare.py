"""The side-by-side table: tuned cascade vs Moshi, same gap definition, same machine.

Cascade numbers are recomputed from the sibling repo's per-turn records
(cascade_baseline.py). Moshi numbers come from results/moshi-*.json, produced by
moshi_run.py on this machine. Nothing here is quoted from prose.

Published figures for other systems (GPT-4o voice, Moshi's own claimed latency)
are listed separately and labelled as published, never as measurements taken here.
"""
import json, statistics as s, sys
from pathlib import Path
import cascade_baseline

R = Path(__file__).parent / "results"

# Published, NOT measured here. Cited so the measured numbers have a scale.
PUBLISHED = {
    "GPT-4o voice (OpenAI, published)":
        "320 ms average, 232 ms minimum -- openai.com/index/hello-gpt-4o, May 2024",
    "Moshi (Kyutai, published)":
        "160 ms theoretical, 200 ms in practice -- kyutai.org/Moshi.pdf model card",
}


def moshi_rows():
    out = {}
    for f in sorted(R.glob("moshi-*.json")):
        if "bargein" in f.name:
            continue
        d = json.load(open(f))
        g = d.get("gap_ms") or {}
        if g:
            out[f"Moshi q4 MLX ({f.stem.replace('moshi-','')})"] = dict(
                n=g["n"], median=g["median"], iqr_lo=g["iqr_lo"], iqr_hi=g["iqr_hi"],
                rtf=(d.get("rtf") or {}).get("median"))
    return out


def main():
    print("=" * 96)
    print("MEASURED ON THIS MACHINE  (Apple M4 Pro, 24 GB, macOS)")
    print("gap = agent speech onset - user speech offset, both silence-trimmed (gap.py)")
    print("=" * 96)
    print(f'{"system":52s} {"n":>4s} {"median":>8s} {"IQR":>15s} {"RTF":>6s}')
    rows = []
    for k, v in cascade_baseline.table().items():
        rows.append((k, v["n"], v["median"], v["iqr_lo"], v["iqr_hi"], None))
    for k, v in moshi_rows().items():
        rows.append((k, v["n"], v["median"], v["iqr_lo"], v["iqr_hi"], v["rtf"]))
    if not moshi_rows():
        print("  (no Moshi runs in results/ yet)")
    for k, n, m, lo, hi, rtf in sorted(rows, key=lambda r: r[2]):
        print(f'{k:52s} {n:4d} {m:8.1f} {lo:7.0f}-{hi:<7.0f} '
              f'{"-" if rtf is None else f"{rtf:6.2f}"}')

    # --- barge-in -------------------------------------------------------
    print("\n" + "=" * 96)
    print("BARGE-IN -- how long the agent keeps talking after the user interrupts")
    print("=" * 96)
    cb = R / "cascade-bargein.json"
    mb = R / "moshi-bargein.json"
    print(f'{"system":52s} {"n":>4s} {"median":>8s} {"IQR":>15s} {"stopped early":>14s}')
    if cb.exists():
        d = json.load(open(cb)); v = d["stop_latency_ms"]
        print(f'{"cascade (tuned, piper reply)":52s} {v["n"]:4d} {v["median"]:8.1f} '
              f'{v["iqr_lo"]:7.0f}-{v["iqr_hi"]:<7.0f} '
              f'{d["n_stopped_early"]:>6d} / {v["n"]:<5d}')
    if mb.exists():
        d = json.load(open(mb))
        v = d.get("stop_latency_ms") or {}
        if v:
            print(f'{"Moshi q4 MLX":52s} {v["n"]:4d} {v["median"]:8.1f} '
                  f'{v["iqr_lo"]:7.0f}-{v["iqr_hi"]:<7.0f} '
                  f'{d.get("n_stopped_early",0):>6d} / {v["n"]:<5d}')
    else:
        print("  (no Moshi barge-in run yet)")

    print("\nPUBLISHED FIGURES (not measured here, cited for scale)")
    for k, v in PUBLISHED.items():
        print(f"  {k:36s} {v}")


if __name__ == "__main__":
    main()
