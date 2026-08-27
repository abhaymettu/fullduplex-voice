"""Repair model.q4.safetensors in place: patch the header, truncate the tail,
find the NUL holes and re-fetch only those byte ranges.

Why this exists: two `hf download` processes briefly wrote the same blob at the
same time. huggingface_hub fetches ranges in parallel, so a partial file is
sparse -- the regions that were written are correct, the regions nobody wrote yet
read back as NUL. Re-downloading 4.5 GB over a contended link to fix a few holes
would be the expensive way to do this.

The end-to-end check is not a checksum (safetensors has none): it is that the
header parses, every tensor's declared range is hole-free, and the model actually
loads with strict=True and generates.

    .venv/bin/python repair.py --scan          # map holes, change nothing
    .venv/bin/python repair.py --fix           # patch header, truncate, refetch
"""
import json, struct, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

F = Path("models/model.q4.safetensors")
URL = "https://huggingface.co/kyutai/moshiko-mlx-q4/resolve/main/model.q4.safetensors"
TOTAL = 4805545317
MIN_HOLE = 4096          # shorter zero runs are plausibly real weights
CHUNK = 1 << 26          # 64 MB scan window
# One stream tops out around 90 KB/s on this link; several in parallel measured
# ~3x that, so holes are split and fetched concurrently. 16 MB keeps each
# request small enough that a stalled one is cheap to retry and never parks a
# gigabyte in RAM.
SUB = 1 << 24
WORKERS = 6


def fetch(start, end, retries=8):
    """Inclusive byte range -> bytes, with retries. HTTP 206 or bust."""
    for _ in range(retries):
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "600", "--connect-timeout", "20",
             "--speed-limit", "15000", "--speed-time", "30",
             "-r", f"{start}-{end}", URL],
            capture_output=True)
        if r.returncode == 0 and len(r.stdout) == end - start + 1:
            return r.stdout
    raise RuntimeError(f"could not fetch {start}-{end}")


def header():
    with open(F, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return n, json.loads(fh.read(n))


def holes(lo=0, hi=TOTAL):
    """Runs of >= MIN_HOLE zero bytes in [lo, hi). Returns [(start, end_exclusive)]."""
    out, run = [], None
    with open(F, "rb") as fh:
        fh.seek(lo)
        pos = lo
        while pos < hi:
            buf = np.frombuffer(fh.read(min(CHUNK, hi - pos)), dtype=np.uint8)
            if not buf.size:
                break
            nz = np.flatnonzero(buf)
            # stitch a run that spans the window boundary
            first = nz[0] if nz.size else buf.size
            if run is not None:
                if first > 0 or not nz.size:
                    run = (run[0], pos + first)
                if nz.size:
                    if run[1] - run[0] >= MIN_HOLE:
                        out.append(run)
                    run = None
            if nz.size:
                gaps = np.flatnonzero(np.diff(nz) > MIN_HOLE)
                for g in gaps:
                    out.append((pos + int(nz[g]) + 1, pos + int(nz[g + 1])))
                tail = pos + int(nz[-1]) + 1
                run = (tail, pos + buf.size) if tail < pos + buf.size else None
            elif run is None:
                run = (pos, pos + buf.size)
            pos += buf.size
    if run is not None and run[1] - run[0] >= MIN_HOLE:
        out.append(run)
    return [(a, b) for a, b in out if b - a >= MIN_HOLE]


def main():
    if "--fix" in sys.argv:
        # 1. clean header, 2. drop the junk tail
        hdr = fetch(0, 167580)
        with open(F, "r+b") as fh:
            fh.seek(0); fh.write(hdr)
        if F.stat().st_size != TOTAL:
            with open(F, "r+b") as fh:
                fh.truncate(TOTAL)
            print(f"truncated to {TOTAL}")

    n, j = header()
    print(f"header OK, {len(j)} tensors, data starts at {8+n}")

    h = holes()
    tot = sum(b - a for a, b in h)
    print(f"{len(h)} holes, {tot/1e6:.1f} MB ({100*tot/TOTAL:.2f}% of the file)")
    for a, b in h[:10]:
        print(f"  {a:12d} - {b:12d}  {(b-a)/1e6:8.2f} MB")
    if len(h) > 10:
        print(f"  ... and {len(h)-10} more")

    if "--fix" in sys.argv and h:
        jobs = [(a, min(a + SUB, b)) for a, b in h for a in range(a, b, SUB)]
        tot_mb = sum(b - a for a, b in jobs) / 1e6
        print(f"refetching {len(jobs)} chunks, {tot_mb:.0f} MB, {WORKERS} at a time",
              flush=True)
        lock, done = threading.Lock(), [0]

        def one(job):
            a, b = job
            data = fetch(a, b - 1)
            with lock:                      # one writer, so no second sparse file
                with open(F, "r+b") as fh:
                    fh.seek(a); fh.write(data)
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(jobs):
                    print(f"  {done[0]}/{len(jobs)} chunks "
                          f"({100*done[0]/len(jobs):.0f}%)", flush=True)

        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(one, jobs))
        left = holes()
        print(f"after repair: {len(left)} holes remain, "
              f"{sum(b-a for a,b in left)/1e6:.1f} MB")
    return h


if __name__ == "__main__":
    main()
