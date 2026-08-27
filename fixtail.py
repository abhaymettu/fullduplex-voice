import repair, threading, hashlib
from concurrent.futures import ThreadPoolExecutor
START = 4_623_000_000
jobs = [(a, min(a + repair.SUB, repair.TOTAL)) for a in range(START, repair.TOTAL, repair.SUB)]
print(f"refetching tail {(repair.TOTAL-START)/1e6:.0f} MB in {len(jobs)} chunks", flush=True)
lock, done = threading.Lock(), [0]
def one(j):
    a, b = j
    d = repair.fetch(a, b - 1)
    with lock:
        with open(repair.F, "r+b") as fh:
            fh.seek(a); fh.write(d)
        done[0] += 1
        print(f"  {done[0]}/{len(jobs)}", flush=True)
with ThreadPoolExecutor(8) as ex:
    list(ex.map(one, jobs))
h = hashlib.sha256()
with open(repair.F, "rb") as fh:
    for blk in iter(lambda: fh.read(1 << 24), b""):
        h.update(blk)
want = "7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c"
print("sha256", h.hexdigest())
print("MATCH" if h.hexdigest() == want else "STILL WRONG")
