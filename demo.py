"""Write a small, listenable recording of an interruption.

Stereo on purpose: user on the left, agent on the right. Mixed to mono you cannot
tell who stopped talking, which is the entire thing being demonstrated.

    .venv/bin/python demo.py results/moshi-bargein.turn0 demo/moshi-bargein
"""
import subprocess, sys
from pathlib import Path
import numpy as np
import soundfile as sf

SR = 24000


def write(user: np.ndarray, agent: np.ndarray, out: str, sr: int = SR):
    n = max(len(user), len(agent))
    pad = lambda a: np.pad(a, (0, n - len(a)))
    st = np.stack([pad(user) * 0.8, pad(agent) * 0.8], axis=1).astype(np.float32)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".wav")
    sf.write(wav, np.clip(st, -1, 1), sr)
    # m4a so it is small enough to commit; afconvert ships with macOS
    m4a = out.with_suffix(".m4a")
    r = subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "48000",
                        str(wav), str(m4a)], capture_output=True)
    if r.returncode == 0:
        wav.unlink()
        return m4a
    return wav


def main():
    stem, out = sys.argv[1], sys.argv[2]
    u = np.load(f"{stem}.user.npy"); a = np.load(f"{stem}.moshi.npy")
    p = write(u, a, out)
    print(f"{p}  {p.stat().st_size/1024:.0f} KB  {len(u)/SR:.1f}s")


if __name__ == "__main__":
    main()
