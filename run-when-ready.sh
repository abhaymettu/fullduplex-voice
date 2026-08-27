#!/bin/bash
# Wait for the weights, then run the benchmark. The pull is the long pole tonight
# and it is rude to make it wait on me noticing.
cd "$(dirname "$0")"
F=models/model.q4.safetensors; WANT=4805545317
while [ "$(stat -f%z $F 2>/dev/null || echo 0)" -lt "$WANT" ]; do sleep 30; done
echo "$(date +%H:%M:%S) weights complete, verifying load"
.venv/bin/python -c "
import moshi_run, time
t0=time.perf_counter(); m=moshi_run.Moshi(); print(f'LOAD OK {m.load_ms/1000:.1f}s')
" || { echo "LOAD FAILED"; exit 1; }
echo "$(date +%H:%M:%S) running 20 turns"
.venv/bin/python moshi_run.py --n 20 --out results/moshi-n20.json
echo "$(date +%H:%M:%S) running barge-in"
.venv/bin/python moshi_run.py --bargein --n 10 --out results/moshi-bargein.json
echo "$(date +%H:%M:%S) done"
