#!/usr/bin/env python3
"""
Lightweight FFT output analyzer
Reads binary output and shows statistics without heavy visualization
"""

import numpy as np
import argparse
import sys
from scipy.signal import find_peaks

def analyze_binary_spectrum(filepath, fft_size, sample_rate, center_freq, max_frames=100):
    """Analyze binary spectrum output"""
    
    frame_size = 4 + fft_size * 4  # 4-byte frame number + fft_size floats
    
    print(f"\nAnalyzing: {filepath}")
    print(f"FFT Size: {fft_size:,} points")
    print(f"Resolution: {sample_rate/fft_size:.3f} Hz/bin")
    print(f"Frame size: {frame_size:,} bytes")
    print("-" * 60)
    
    frames_analyzed = 0
    peak_counts = []
    avg_power = []
    peak_positions = []
    
    try:
        with open(filepath, 'rb') as f:
            while frames_analyzed < max_frames:
                # Read frame number
                frame_num_bytes = f.read(4)
                if len(frame_num_bytes) < 4:
                    break
                
                frame_num = np.frombuffer(frame_num_bytes, dtype=np.int32)[0]
                
                # Read spectrum data
                spectrum_bytes = f.read(fft_size * 4)
                if len(spectrum_bytes) < fft_size * 4:
                    break
                
                spectrum = np.frombuffer(spectrum_bytes, dtype=np.float32)
                
                # Calculate statistics
                avg_power.append(np.mean(spectrum))
                
                # Find peaks (prominence threshold)
                peaks, properties = find_peaks(spectrum, prominence=0.2, height=0.6)
                peak_counts.append(len(peaks))
                
                # Store peak positions (in Hz from center)
                if len(peaks) > 0:
                    # Convert bin numbers to frequencies
                    freqs = (np.arange(fft_size) - fft_size/2) * (sample_rate / fft_size) + center_freq
                    peak_freqs = freqs[peaks]
                    peak_powers = spectrum[peaks]
                    
                    # Sort by power and keep top 10
                    sorted_idx = np.argsort(peak_powers)[::-1][:10]
                    frame_peaks = [(peak_freqs[i], peak_powers[i]) for i in sorted_idx]
                    peak_positions.append(frame_peaks)
                
                frames_analyzed += 1
                
                # Progress indicator
                if frames_analyzed % 10 == 0:
                    print(f"Processed {frames_analyzed} frames...", end='\r')
        
        print(f"\nProcessed {frames_analyzed} frames total")
        print("-" * 60)
        
        if frames_analyzed == 0:
            print("No frames found in file")
            return
        
        # Display statistics
        print(f"\nStatistics over {frames_analyzed} frames:")
        print(f"  Average power: {np.mean(avg_power):.3f} (normalized)")
        print(f"  Power std dev: {np.std(avg_power):.3f}")
        print(f"  Average peaks detected: {np.mean(peak_counts):.1f}")
        print(f"  Peak count range: {np.min(peak_counts)}-{np.max(peak_counts)}")
        
        # Show consistent peaks (appear in >50% of frames)
        if peak_positions:
            print(f"\nTop signals (from last frame):")
            last_peaks = peak_positions[-1]
            
            for i, (freq, power) in enumerate(last_peaks[:8], 1):
                doppler_hz = freq - center_freq
                doppler_khz = doppler_hz / 1e3
                
                if abs(doppler_khz) >= 1:
                    offset_str = f"{doppler_khz:+.2f} kHz"
                else:
                    offset_str = f"{doppler_hz:+.0f} Hz"
                
                print(f"  {i}. {freq/1e6:.6f} MHz ({offset_str}) - Power: {power:.3f}")
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
    except Exception as e:
        print(f"Error analyzing file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze binary spectrum output'
    )
    parser.add_argument(
        'input_file',
        help='Binary spectrum file to analyze'
    )
    parser.add_argument(
        '--fft-size',
        type=int,
        required=True,
        help='FFT size used'
    )
    parser.add_argument(
        '--samp-rate',
        type=float,
        default=20e6,
        help='Sample rate in Hz (default: 20 MHz)'
    )
    parser.add_argument(
        '--center-freq',
        type=float,
        default=1575.42e6,
        help='Center frequency in Hz (default: 1575.42 MHz for GPS L1)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum frames to analyze (default: 100)'
    )
    
    args = parser.parse_args()
    
    analyze_binary_spectrum(
        args.input_file,
        args.fft_size,
        args.samp_rate,
        args.center_freq,
        args.max_frames
    )

if __name__ == '__main__':
    main()

