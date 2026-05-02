# cuda_spectrum

GPU-accelerated spectrum analyzer. Reads interleaved float32 IQ samples from stdin, computes large FFTs on the GPU via cuFFT, outputs power spectrum to stdout and/or ZMQ.


## Build

```bash
nvcc -O2 -o cuda_spectrum main.cpp cuda_processing.cu -lcufft -lzmq
```

**Dependencies:**
- CUDA toolkit (nvcc + cuFFT) — [nvidia.com/cuda](https://developer.nvidia.com/cuda-downloads)
- libzmq: `sudo apt install libzmq3-dev`

Tested on NVIDIA RTX 30xx series GPU. Any CUDA-capable GPU should work.

## Input format

Interleaved float32 IQ: `I0 Q0 I1 Q1 …`

Most SDR tools output int16 or int8. Convert with Python or csdr before piping:

```bash
# From SDR++ WAV (int16, skip 44-byte header)
python3 -c "
import numpy as np, sys
with open('capture.wav','rb') as f:
    f.seek(44)
    data = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
sys.stdout.buffer.write(data.tobytes())
" | ./cuda_spectrum [options]

# From RTL-SDR (int8 via rtl_sdr)
rtl_sdr -f 435e6 -s 10e6 - | csdr convert_u8_f | ./cuda_spectrum [options]
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-n N` | 1024 | IQ pairs per frame (input samples) |
| `-f N` | 4096 | FFT size, must be power of 2 |
| `-w TYPE` | blackman-harris | Window: `none hann hamming blackman blackman-harris flat-top` |
| `-o FORMAT` | csv | Output format: `csv binary json` |
| `--db` | off | Output raw dBFS instead of normalized 0–1 |
| `-l N` | 0 | Limit to N frames (0 = unlimited) |
| `-q` | off | Quiet mode (suppress stderr) |
| `--zmq ENDPOINT` | — | Stream via ZMQ (e.g. `tcp://*:5555`) |
| `--no-stdout` | off | Disable stdout (ZMQ-only mode) |

Set `-n` equal to `-f` for maximum coherent integration per frame. Use `-n` < `-f` for zero-padded interpolation.

## Output formats

**CSV** (default): `FRAME_START,N` / one value per line / `FRAME_END,N`

**Binary** (`-o binary`): 4-byte int32 frame number + N × float32 spectrum values. Recommended for multi-frame captures — dramatically faster than CSV at large FFT sizes.

**ZMQ**: Binary float32 vector, one message per frame.

## Examples

```bash
# Sanity check — Gaussian noise through a 4K FFT
python3 -c "
import numpy as np, sys
sys.stdout.buffer.write(np.random.randn(8192).astype(np.float32).tobytes())
" | ./cuda_spectrum -n 4096 -f 4096 -l 1

# RF survey — 16M-point single frame, 0.6 Hz bins at 10 Msps
./cuda_spectrum -f 16777216 -n 16777216 -w blackman-harris --db -l 1 \
    < capture.cf32 > spectrum_16M.csv

# Multi-frame waterfall — binary output for speed
./cuda_spectrum -f 4194304 -n 4194304 -w blackman-harris --db -o binary \
    < capture.cf32 > frames.bin

# GPS L1 — live via HackRF + ZMQ to plotter
hackrf_transfer -f 1575420000 -s 20000000 -g 40 -l 32 -a 1 -r - \
    | csdr convert_s8_f \
    | ./cuda_spectrum -f 4194304 -n 4194304 --zmq tcp://*:5555 --no-stdout -q
```

## FFT size guide

| FFT size | Bins at 10 Msps | Integration |
|----------|-----------------|-------------|
| 1M (1048576) | 9.5 Hz | 0.10 s |
| 4M (4194304) | 2.38 Hz | 0.42 s |
| 16M (16777216) | 0.60 Hz | 1.68 s |
| 64M (67108864) | 0.15 Hz | 6.71 s |

Larger FFT = finer frequency resolution = more processing gain for weak signals.

## Plotting

`rf_plot.py` reads both CSV and binary output, produces PSD + waterfall PNG:

```bash
# Single frame from CSV
python3 rf_plot.py spectrum_16M.csv --fft 16777216 --rate 10e6 --center 435e6

# Multi-frame waterfall from binary
python3 rf_plot.py frames.bin --binary --fft 4194304 --rate 10e6 --center 435e6

# Zoom (no re-processing — resolution is already there)
python3 rf_plot.py frames.bin --binary --fft 4194304 --rate 10e6 --center 435e6 \
    --zoom 434.5 435.5
    
```

## Analysis (no plotting)

`analyze_spectrum.py` reads binary output and prints statistics — no matplotlib required:

```bash
python3 analyze_spectrum.py frames.bin \
    --fft-size 4194304 \
    --samp-rate 10e6 \
    --center-freq 435e6

# Limit to first 50 frames
python3 analyze_spectrum.py frames.bin \
    --fft-size 4194304 \
    --samp-rate 10e6 \
    --center-freq 435e6 \
    --max-frames 50
```

Outputs per-frame statistics (average power, peak count) and top signals with frequency offset from center. Useful for quick triage before plotting.

Requires: `numpy scipy`
Requires: `numpy matplotlib scipy`
