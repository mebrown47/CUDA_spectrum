#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// ============================================================================
// PERSISTENT STATE (avoids per-frame allocations)
// ============================================================================
static cufftComplex* d_fftInput = nullptr;
static cufftComplex* d_fftOutput = nullptr;
static float* d_powerSpectrum = nullptr;
static float* d_window = nullptr;  // Window in global memory (unlimited size)
static cufftHandle fftPlan = 0;
static int currentFftSize = 0;
static int currentInputSamples = 0;
static cudaStream_t computeStream = nullptr;

// ============================================================================
// WINDOW FUNCTION TYPES
// ============================================================================
typedef enum {
    WINDOW_NONE = 0,
    WINDOW_HANN,
    WINDOW_HAMMING,
    WINDOW_BLACKMAN,
    WINDOW_BLACKMAN_HARRIS,
    WINDOW_FLAT_TOP
} WindowType;

// ============================================================================
// HOST-SIDE WINDOW GENERATION
// ============================================================================
void generateWindow(float* h_window, int size, WindowType type) {
    const double PI = 3.14159265358979323846;
    const double TWO_PI = 2.0 * PI;
    
    for (int i = 0; i < size; i++) {
        double x = (double)i / (double)(size - 1);
        
        switch (type) {
            case WINDOW_NONE:
                h_window[i] = 1.0f;
                break;
                
            case WINDOW_HANN:
                h_window[i] = 0.5f * (1.0f - cos(TWO_PI * x));
                break;
                
            case WINDOW_HAMMING:
                h_window[i] = 0.54f - 0.46f * cos(TWO_PI * x);
                break;
                
            case WINDOW_BLACKMAN:
                h_window[i] = 0.42f - 0.5f * cos(TWO_PI * x) + 0.08f * cos(4.0 * PI * x);
                break;
                
            case WINDOW_BLACKMAN_HARRIS:
                // 4-term Blackman-Harris - excellent sidelobe suppression
                h_window[i] = 0.35875f 
                            - 0.48829f * cos(TWO_PI * x)
                            + 0.14128f * cos(4.0 * PI * x)
                            - 0.01168f * cos(6.0 * PI * x);
                break;
                
            case WINDOW_FLAT_TOP:
                // Flat-top window - best amplitude accuracy
                h_window[i] = 0.21557895f
                            - 0.41663158f * cos(TWO_PI * x)
                            + 0.277263158f * cos(4.0 * PI * x)
                            - 0.083578947f * cos(6.0 * PI * x)
                            + 0.006947368f * cos(8.0 * PI * x);
                break;
                
            default:
                h_window[i] = 1.0f;
        }
    }
}

// ============================================================================
// KERNELS
// ============================================================================

// Combined interleave + windowing + zero-padding kernel
__global__ void interleaveWindowZeroPadKernel(
    const float* __restrict__ d_interleaved, 
    cufftComplex* __restrict__ d_complex,
    const float* __restrict__ window,  // Window array passed as parameter
    int input_samples, 
    int fft_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_samples) {
        // Data region: apply window and convert to complex
        float window_val = window[idx];
        d_complex[idx].x = d_interleaved[2 * idx] * window_val;
        d_complex[idx].y = d_interleaved[2 * idx + 1] * window_val;
    } else if (idx < fft_size) {
        // Zero-padding region
        d_complex[idx].x = 0.0f;
        d_complex[idx].y = 0.0f;
    }
}

// Power spectrum with FFT shift and normalization
__global__ void powerSpectrumKernel(
    const cufftComplex* __restrict__ fftData, 
    float* __restrict__ powerData, 
    int size,
    float referenceLevel, 
    float dynamicRange,
    bool doFftShift
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Calculate power (magnitude squared)
        float re = fftData[idx].x;
        float im = fftData[idx].y;
        float power = re * re + im * im;
        
        // Log scale with adjustable reference and range
        float db = 10.0f * log10f(power + 1e-10f) - referenceLevel;
        
        // Normalize to [0,1] range - centered around 0.5
        float normalized = db / dynamicRange + 0.5f;
        normalized = fmaxf(0.0f, fminf(1.0f, normalized));
        
        // Apply FFT shift (center DC) if requested
        // Avoid modulo to prevent artifacts - use conditional instead
        int out_idx;
        if (doFftShift) {
            out_idx = (idx < size/2) ? (idx + size/2) : (idx - size/2);
        } else {
            out_idx = idx;
        }
        powerData[out_idx] = normalized;
    }
}

// Raw dB output kernel (no normalization)
__global__ void powerSpectrumDbKernel(
    const cufftComplex* __restrict__ fftData, 
    float* __restrict__ powerData, 
    int size,
    float referenceLevel,
    bool doFftShift
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float re = fftData[idx].x;
        float im = fftData[idx].y;
        float power = re * re + im * im;
        
        // Return raw dB value
        float db = 10.0f * log10f(power + 1e-10f) - referenceLevel;
        
        // Apply FFT shift (center DC) if requested
        // Avoid modulo to prevent artifacts - use conditional instead
        int out_idx;
        if (doFftShift) {
            out_idx = (idx < size/2) ? (idx + size/2) : (idx - size/2);
        } else {
            out_idx = idx;
        }
        powerData[out_idx] = db;
    }
}

// ============================================================================
// PIPELINE MANAGEMENT API
// ============================================================================

extern "C" {

// Initialize the pipeline with given parameters
int initPipeline(int input_samples, int fft_size, WindowType windowType) {
    // Check if FFT size is power of 2
    if ((fft_size & (fft_size - 1)) != 0) {
        fprintf(stderr, "Error: fft_size must be a power of 2\n");
        return -1;
    }
    
    // Validate input samples <= FFT size
    if (input_samples > fft_size) {
        fprintf(stderr, "Error: input_samples (%d) cannot exceed fft_size (%d)\n", 
                input_samples, fft_size);
        return -1;
    }
    
    // Clean up existing resources if reinitializing
    if (currentFftSize > 0) {
        if (d_fftInput) cudaFree(d_fftInput);
        if (d_fftOutput) cudaFree(d_fftOutput);
        if (d_powerSpectrum) cudaFree(d_powerSpectrum);
        if (d_window) cudaFree(d_window);
        if (fftPlan) cufftDestroy(fftPlan);
        if (computeStream) cudaStreamDestroy(computeStream);
    }
    
    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc(&d_fftInput, fft_size * sizeof(cufftComplex));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating d_fftInput: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc(&d_fftOutput, fft_size * sizeof(cufftComplex));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating d_fftOutput: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc(&d_powerSpectrum, fft_size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating d_powerSpectrum: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Allocate window in global memory (unlimited size!)
    err = cudaMalloc(&d_window, input_samples * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating d_window: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Create CUDA stream for async operations
    err = cudaStreamCreate(&computeStream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error creating CUDA stream: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Create cuFFT plan
    cufftResult fftErr = cufftPlan1d(&fftPlan, fft_size, CUFFT_C2C, 1);
    if (fftErr != CUFFT_SUCCESS) {
        fprintf(stderr, "Error creating cuFFT plan: %d\n", fftErr);
        return -1;
    }
    
    // Associate FFT plan with stream
    cufftSetStream(fftPlan, computeStream);
    
    // Generate and upload window function to global memory
    float* h_window = new float[input_samples];
    generateWindow(h_window, input_samples, windowType);
    cudaMemcpy(d_window, h_window, input_samples * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_window;
    
    // Store current configuration
    currentFftSize = fft_size;
    currentInputSamples = input_samples;
    
    return 0;
}

// Clean up all resources
void cleanupPipeline() {
    if (d_fftInput) { cudaFree(d_fftInput); d_fftInput = nullptr; }
    if (d_fftOutput) { cudaFree(d_fftOutput); d_fftOutput = nullptr; }
    if (d_powerSpectrum) { cudaFree(d_powerSpectrum); d_powerSpectrum = nullptr; }
    if (d_window) { cudaFree(d_window); d_window = nullptr; }
    if (fftPlan) { cufftDestroy(fftPlan); fftPlan = 0; }
    if (computeStream) { cudaStreamDestroy(computeStream); computeStream = nullptr; }
    currentFftSize = 0;
    currentInputSamples = 0;
}

// Process a single frame (synchronous)
// d_input: device pointer to interleaved I/Q float data
// h_output: host pointer for output spectrum
// outputDb: if true, output raw dB values; if false, output normalized [0,1]
// doFftShift: if true, shift DC to center
void processFrame(
    float* d_input, 
    float* h_output, 
    float referenceLevel, 
    float dynamicRange,
    bool outputDb,
    bool doFftShift
) {
    if (currentFftSize == 0) {
        fprintf(stderr, "Error: Pipeline not initialized. Call initPipeline() first.\n");
        return;
    }
    
    int threads = 256;
    int blocks = (currentFftSize + threads - 1) / threads;
    
    // 1. Interleave, window, and zero-pad
    interleaveWindowZeroPadKernel<<<blocks, threads, 0, computeStream>>>(
        d_input, d_fftInput, d_window, currentInputSamples, currentFftSize
    );
    
    // 2. Execute FFT (stream already set during init)
    cufftExecC2C(fftPlan, d_fftInput, d_fftOutput, CUFFT_FORWARD);
    
    // 3. Calculate power spectrum
    if (outputDb) {
        powerSpectrumDbKernel<<<blocks, threads, 0, computeStream>>>(
            d_fftOutput, d_powerSpectrum, currentFftSize, referenceLevel, doFftShift
        );
    } else {
        powerSpectrumKernel<<<blocks, threads, 0, computeStream>>>(
            d_fftOutput, d_powerSpectrum, currentFftSize, referenceLevel, dynamicRange, doFftShift
        );
    }
    
    // 4. Copy result back to host
    cudaMemcpyAsync(h_output, d_powerSpectrum, currentFftSize * sizeof(float), 
                    cudaMemcpyDeviceToHost, computeStream);
    
    // 5. Synchronize
    cudaStreamSynchronize(computeStream);
}

// Get current configuration
int getFftSize() { return currentFftSize; }
int getInputSamples() { return currentInputSamples; }

// Legacy API compatibility wrapper
void processSpectrumPipeline(float* d_input, float* h_output, int input_samples, int fft_size, 
                            float referenceLevel, float dynamicRange) {
    // Auto-initialize if needed (for backwards compatibility)
    if (currentFftSize != fft_size || currentInputSamples != input_samples) {
        initPipeline(input_samples, fft_size, WINDOW_BLACKMAN_HARRIS);
    }
    processFrame(d_input, h_output, referenceLevel, dynamicRange, false, true);
}

} // extern "C"

