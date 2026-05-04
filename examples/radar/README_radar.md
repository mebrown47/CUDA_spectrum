# Radar Signal Characterization — Blind Analysis Example

This example demonstrates blind characterization of an **unknown radar signal**
using the periodicity analysis pipeline. No prior knowledge of the signal type
was used — the pipeline reveals the structure from the IQ data alone.

**Source:** SigIDWiki — "Low VHF radar-like irregular IQ sample" (~30.74 MHz)
**Files:** Two captures, one minute apart, same signal

---
## Purpose

Evaluate the Python scripts in this repo using various and varying RF signal types.

---
## The Signal

Two IQ captures from SigIDWiki, labeled "irregular" — suggesting non-standard
or non-uniform pulse timing:

| File | Center Freq | Duration | Sample Rate |
|------|------------|----------|-------------|
| `audio_30741521Hz_10-53-28_26-04-2024.wav` | 30.7415 MHz | 9.37 s | 96 kHz |
| `audio_30745856Hz_10-54-29_26-04-2024.wav` | 30.7458 MHz | 14.45 s | 96 kHz |

Format: int16 stereo (L=I, R=Q), standard SDR IQ convention.

---

## Step 1 — Convert to cf32

```bash
python3 -c "
import numpy as np

files = [
    ('audio_30741521Hz_10-53-28_26-04-2024.wav', 'radar_30741.cf32'),
    ('audio_30745856Hz_10-54-29_26-04-2024.wav', 'radar_30745.cf32'),
]

for wav, cf32 in files:
    with open(wav, 'rb') as f:
        f.seek(44)
        data = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
    with open(cf32, 'wb') as f:
        f.write(data.tobytes())
    print(f'{wav}: {len(data)//2:,} IQ pairs, {len(data)/2/96000:.2f}s')
"
```

---

## Step 2 — Spectral Overview (512K-point FFT)

```bash
./cuda_spectrum -f 524288 -n 524288 -w blackman-harris --db -l 1 \
    < radar_30741.cf32 > radar_30741_512K.csv

./cuda_spectrum -f 524288 -n 524288 -w blackman-harris --db -l 1 \
    < radar_30745.cf32 > radar_30745_512K.csv
```

At 96000 / 524288 = **0.18 Hz bins** across 96 kHz span.

```bash
python3 spectrum_plot.py radar_30741_512K.csv \
    --fft 524288 --rate 96000 --center 30.7415e6 \
    --title "Unknown VHF Radar — 30.741 MHz" \
    --save radar_30741.png

python3 spectrum_plot.py radar_30745_512K.csv \
    --fft 524288 --rate 96000 --center 30.7458e6 \
    --title "Unknown VHF Radar — 30.745 MHz" \
    --save radar_30745.png
```

**What you see:**
- Gaussian-shaped spectral envelope ~30 kHz wide — the integrated radar chirp bandwidth
- Dense irregular comb structure throughout — pulse rate sidebands
- Non-uniform sideband spacing — "irregular" PRF
- Antenna scan visible as asymmetric envelope shape between captures

The comb structure is the key observation: unlike Kontayner's perfectly uniform
FMCW sidebands, these are clustered at multiple spacings — the signature of
**staggered PRF**.

---

## Step 3 — Power Envelope Extraction

For pulsed radar the transmitter switches on and off at the PRF. The power
envelope method captures this directly:

```bash
./cuda_spectrum -f 4096 -n 4096 -w blackman-harris --db -o binary \
    < radar_30741.cf32 > radar_30741_4K.bin

./cuda_spectrum -f 4096 -n 4096 -w blackman-harris --db -o binary \
    < radar_30745.cf32 > radar_30745_4K.bin

python3 extract_power_envelope.py radar_30741_4K.bin \
    --fft 4096 --rate 96000 --center 30.7415e6 \
    --lo 30.725e6 --hi 30.760e6 \
    --out radar_30741_envelope.csv

python3 extract_power_envelope.py radar_30745_4K.bin \
    --fft 4096 --rate 96000 --center 30.7458e6 \
    --lo 30.725e6 --hi 30.760e6 \
    --out radar_30745_envelope.csv
```

---

## Step 4 — Periodicity Analysis

```bash
python3 find_periodicity.py radar_30741_envelope.csv \
    --plot radar_30741_periodicity.png

python3 find_periodicity.py radar_30745_envelope.csv \
    --plot radar_30745_periodicity.png
```

**Results — 30.741 MHz capture:**

| Period (ms) | Rel Power | Notes |
|-------------|-----------|-------|
| 90.8 ms | 0.068 | PRF component 1 |
| 101.7 ms | 0.117 | PRF component 2 — most stable |
| 107.3 ms | 0.083 | PRF component 3 |
| 117.3 ms | 0.095 | PRF component 4 |
| 139.7 ms | 0.077 | PRF component 5 |
| 286.1 ms | 0.144 | Stagger cycle period |

**Results — 30.745 MHz capture (one minute later):**

| Period (ms) | Rel Power | Notes |
|-------------|-----------|-------|
| 94.1 ms | 0.381 | PRF component 1 |
| 101.5 ms | 0.420 | PRF component 2 — most stable |
| 112.6 ms | 0.312 | PRF component 3 |
| 131.3 ms | 0.377 | PRF component 4 |
| 149.7 ms | 0.360 | PRF component 5 |
| 224.8 ms | 0.447 | Stagger cycle period |

---

## Cross-Capture Validation

The **101.5 ms period matches to within 0.3 ms** between captures taken one
minute apart at slightly different center frequencies.

| Period | Capture 1 | Capture 2 | Difference |
|--------|-----------|-----------|------------|
| PRF component 2 | 101.747 ms | 101.482 ms | 0.265 ms |

Both captures show the same cluster of 4-5 stagger periods in the 90-150 ms
range. The antenna scan is apparent in the power envelope — power drifting down
in capture 1 (beam scanning away), rising then leveling in capture 2 (beam
scanning in from a different angle one minute later).

---

## Conclusion

The SigIDWiki "irregular" label may be misleading. The pipeline suggests this is
**not** an irregular signal — it is a **systematically staggered PRF** radar:

- **Signal class:** Pulsed radar with staggered PRF
- **Frequency:** ~30.74 MHz (low VHF, OTH-capable band)
- **PRF components:** 5 stagger periods, ~90–150 ms range (6.7–11 Hz)
- **Most stable period:** ~101.5 ms (confirmed across two independent captures)
- **Stagger cycle:** ~225–286 ms (one complete sequence of all stagger periods)
- **Antenna scan:** Apparently visible in power envelope, beam dwell ~9-14 seconds

---

## Signal Classification Summary

| Method | TETRA (TDMA) | Kontayner (FMCW) | This signal (Pulsed) |
|--------|-------------|-----------------|---------------------|
| Power envelope periodicity | ✓ Primary | ✗ Not applicable | ✓ Primary |
| Spectral sideband spacing | Secondary | ✓ Primary | Secondary |
| Signature | Single period + harmonics | Uniform comb | Multiple stagger periods |

**Key principle:** The analysis method must match the signal class.
- **TDMA/Pulsed** → power envelope periodicity
- **FMCW** → spectral sideband spacing
- **Unknown** → try both, spectral shape guides which applies

---

## Pipeline

```bash
# Convert
python3 convert.py capture.wav capture.cf32

# Spectrum
cuda_spectrum -f 524288 -o binary < capture.cf32 > capture.bin

# Visualize
python3 spectrum_plot.py capture.bin --binary --fft 524288 --rate 96000 --center FREQ

# Characterize
cuda_spectrum -f 4096 -o binary < capture.cf32 > capture_4K.bin
python3 extract_power_envelope.py capture_4K.bin --fft 4096 --rate 96000 \
    --center FREQ --lo LO --hi HI | \
python3 find_periodicity.py --plot periodicity.png
```

