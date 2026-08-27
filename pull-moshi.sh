#!/bin/bash
# Resume model.q4.safetensors until the byte count matches. The link is
# heavily contended tonight and both hf-cli and plain curl get cut mid-stream,
# so this just keeps resuming rather than trying to be clever.
F=models/model.q4.safetensors
WANT=4805545317
URL=https://huggingface.co/kyutai/moshiko-mlx-q4/resolve/main/model.q4.safetensors
for i in $(seq 1 400); do
  HAVE=$(stat -f%z "$F" 2>/dev/null || echo 0)
  [ "$HAVE" -ge "$WANT" ] && { echo "$(date +%H:%M:%S) complete: $HAVE"; exit 0; }
  echo "$(date +%H:%M:%S) try $i, have $HAVE / $WANT ($((HAVE*100/WANT))%)"
  # bail out of a stalled stream fast (<20KB/s for 30s) and resume, rather
  # than sitting in a dead connection until max-time.
  curl -sL -C - --connect-timeout 20 --speed-limit 20000 --speed-time 30 -o "$F" "$URL"
  sleep 2
done
echo "gave up"; exit 1
