/**
 * cuda_spectrum - GPU-accelerated spectrum analyzer with ZMQ support
 * 
 * Usage: 
 *   cuda_spectrum [options] < input.raw > output.csv
 *   cuda_spectrum --zmq tcp://*:5555 -f 524288 -n 131072 < input.raw
 * 
 * Reads interleaved I/Q float32 samples from stdin, computes FFT-based
 * power spectrum on GPU, outputs via stdout and/or ZMQ.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <getopt.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <zmq.h>

// ============================================================================
// CONFIGURATION DEFAULTS
// ============================================================================
struct Config {
    int inputSamples = 1024;       // Number of IQ pairs per frame
    int fftSize = 4096;            // FFT size (must be power of 2)
    float referenceLevel = 0.0f;   // dBFS reference level
    float dynamicRange = 80.0f;    // Dynamic range in dB
    int windowType = 4;            // 0=none, 1=hann, 2=hamming, 3=blackman, 4=blackman-harris, 5=flat-top
    int outputFormat = 0;          // 0=csv, 1=binary, 2=json
    bool outputDb = false;         // Output raw dB instead of normalized
    bool noFftShift = false;       // Disable FFT shift (don't center DC)
    bool quiet = false;            // Suppress progress output
    bool showHelp = false;
    int frameLimit = 0;            // 0 = unlimited
    
    // ZMQ options
    bool useZmq = false;
    std::string zmqEndpoint = "tcp://*:5555";
    bool noStdout = false;         // Disable stdout (ZMQ only)
};

// ============================================================================
// EXTERNAL CUDA FUNCTIONS
// ============================================================================
extern "C" {
    int initPipeline(int input_samples, int fft_size, int windowType);
    void cleanupPipeline();
    void processFrame(float* d_input, float* h_output, 
                     float referenceLevel, float dynamicRange,
                     bool outputDb, bool doFftShift);
    int getFftSize();
    int getInputSamples();
}

// ============================================================================
// CLI ARGUMENT PARSING
// ============================================================================
void printUsage(const char* progName) {
    std::cerr << R"(
cuda_spectrum - GPU-accelerated spectrum analyzer with ZMQ support

USAGE:
    )" << progName << R"( [OPTIONS] < input.raw > output.csv

OPTIONS:
    -n, --input-samples N   IQ pairs per frame (default: 1024)
    -f, --fft-size N        FFT size, must be power of 2 (default: 4096)
    -r, --ref-level DB      Reference level in dB (default: 0.0)
    -d, --dynamic-range DB  Dynamic range in dB (default: 80.0)
    -w, --window TYPE       Window function (default: blackman-harris)
                            Types: none, hann, hamming, blackman, 
                                   blackman-harris, flat-top
    -o, --output FORMAT     Output format (default: csv)
                            Formats: csv, binary, json
    --db                    Output raw dB values (not normalized 0-1)
    --no-fft-shift          Don't center DC (leave at bin 0)
    -l, --limit N           Process only N frames (0 = unlimited)
    -q, --quiet             Suppress progress messages
    
    --zmq ENDPOINT          Enable ZMQ streaming (e.g., tcp://*:5555)
    --no-stdout             Disable stdout (ZMQ only mode)
    
    -h, --help              Show this help message

EXAMPLES:
    # Basic usage with rtl_sdr
    rtl_sdr -f 100e6 -s 2.4e6 - | cuda_spectrum > spectrum.csv

    # High-resolution FFT (512K)
    cuda_spectrum -n 131072 -f 524288 < iq.raw > spectrum.csv

    # ZMQ streaming to GNU Radio (4K FFT)
    rtl_sdr - | cuda_spectrum --zmq tcp://*:5555

    # ZMQ streaming with mega-FFT (512K)
    rtl_sdr - | cuda_spectrum --zmq tcp://*:5555 -f 524288 -n 131072

    # Both ZMQ and CSV output
    rtl_sdr - | cuda_spectrum --zmq tcp://*:5555 > recording.csv

    # ZMQ only (no CSV, maximum speed)
    rtl_sdr - | cuda_spectrum --zmq tcp://*:5555 --no-stdout --quiet

INPUT FORMAT:
    Interleaved float32 I/Q samples: I0, Q0, I1, Q1, ...
    Each frame requires (input-samples * 2 * 4) bytes.

OUTPUT FORMATS:
    csv:    FRAME_START,N / value per line / FRAME_END,N
    binary: 4-byte frame count + fft_size * 4-byte floats
    json:   {"frame": N, "spectrum": [...]}
    zmq:    Binary float32 vector stream

)";
}

int parseWindowType(const char* name) {
    if (strcmp(name, "none") == 0) return 0;
    if (strcmp(name, "hann") == 0) return 1;
    if (strcmp(name, "hamming") == 0) return 2;
    if (strcmp(name, "blackman") == 0) return 3;
    if (strcmp(name, "blackman-harris") == 0) return 4;
    if (strcmp(name, "flat-top") == 0) return 5;
    return -1;
}

int parseOutputFormat(const char* name) {
    if (strcmp(name, "csv") == 0) return 0;
    if (strcmp(name, "binary") == 0) return 1;
    if (strcmp(name, "json") == 0) return 2;
    return -1;
}

bool parseArgs(int argc, char* argv[], Config& cfg) {
    static struct option longOptions[] = {
        {"input-samples", required_argument, 0, 'n'},
        {"fft-size",      required_argument, 0, 'f'},
        {"ref-level",     required_argument, 0, 'r'},
        {"dynamic-range", required_argument, 0, 'd'},
        {"window",        required_argument, 0, 'w'},
        {"output",        required_argument, 0, 'o'},
        {"db",            no_argument,       0, 'D'},
        {"no-fft-shift",  no_argument,       0, 'S'},
        {"limit",         required_argument, 0, 'l'},
        {"quiet",         no_argument,       0, 'q'},
        {"zmq",           required_argument, 0, 'z'},
        {"no-stdout",     no_argument,       0, 'N'},
        {"help",          no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int optionIndex = 0;
    
    while ((opt = getopt_long(argc, argv, "n:f:r:d:w:o:l:z:qh", longOptions, &optionIndex)) != -1) {
        switch (opt) {
            case 'n':
                cfg.inputSamples = atoi(optarg);
                break;
            case 'f':
                cfg.fftSize = atoi(optarg);
                break;
            case 'r':
                cfg.referenceLevel = atof(optarg);
                break;
            case 'd':
                cfg.dynamicRange = atof(optarg);
                break;
            case 'w':
                cfg.windowType = parseWindowType(optarg);
                if (cfg.windowType < 0) {
                    std::cerr << "Error: Unknown window type: " << optarg << std::endl;
                    return false;
                }
                break;
            case 'o':
                cfg.outputFormat = parseOutputFormat(optarg);
                if (cfg.outputFormat < 0) {
                    std::cerr << "Error: Unknown output format: " << optarg << std::endl;
                    return false;
                }
                break;
            case 'D':
                cfg.outputDb = true;
                break;
            case 'S':
                cfg.noFftShift = true;
                break;
            case 'l':
                cfg.frameLimit = atoi(optarg);
                break;
            case 'q':
                cfg.quiet = true;
                break;
            case 'z':
                cfg.useZmq = true;
                cfg.zmqEndpoint = optarg;
                break;
            case 'N':
                cfg.noStdout = true;
                break;
            case 'h':
                cfg.showHelp = true;
                return true;
            default:
                return false;
        }
    }
    
    // Validate FFT size is power of 2
    if ((cfg.fftSize & (cfg.fftSize - 1)) != 0) {
        std::cerr << "Error: FFT size must be a power of 2" << std::endl;
        return false;
    }
    
    // Validate input samples <= FFT size
    if (cfg.inputSamples > cfg.fftSize) {
        std::cerr << "Error: input-samples cannot exceed fft-size" << std::endl;
        return false;
    }
    
    return true;
}

// ============================================================================
// OUTPUT FORMATTERS
// ============================================================================
void outputFrameCsv(const float* spectrum, int size, int frameNum) {
    std::cout << "FRAME_START," << frameNum << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << spectrum[i] << std::endl;
    }
    std::cout << "FRAME_END," << frameNum << std::endl;
    std::cout.flush();
}

void outputFrameBinary(const float* spectrum, int size, int frameNum) {
    fwrite(&frameNum, sizeof(int), 1, stdout);
    fwrite(spectrum, sizeof(float), size, stdout);
    fflush(stdout);
}

void outputFrameJson(const float* spectrum, int size, int frameNum) {
    std::cout << "{\"frame\":" << frameNum << ",\"spectrum\":[";
    for (int i = 0; i < size; i++) {
        if (i > 0) std::cout << ",";
        std::cout << spectrum[i];
    }
    std::cout << "]}" << std::endl;
    std::cout.flush();
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    Config cfg;
    
    if (!parseArgs(argc, argv, cfg)) {
        printUsage(argv[0]);
        return 1;
    }
    
    if (cfg.showHelp) {
        printUsage(argv[0]);
        return 0;
    }
    
    // Calculate buffer sizes
    const int inputBufferSize = cfg.inputSamples * 2;
    const size_t inputByteSize = inputBufferSize * sizeof(float);
    
    // Set stdin to binary mode
    std::ios::sync_with_stdio(false);
    std::cin.tie(NULL);
    
    // ========================================================================
    // ZMQ SETUP (if enabled)
    // ========================================================================
    void* zmq_context = nullptr;
    void* zmq_publisher = nullptr;
    
    if (cfg.useZmq) {
        zmq_context = zmq_ctx_new();
        if (!zmq_context) {
            std::cerr << "Error: Could not create ZMQ context" << std::endl;
            return 1;
        }
        
        zmq_publisher = zmq_socket(zmq_context, ZMQ_PUB);
        if (!zmq_publisher) {
            std::cerr << "Error: Could not create ZMQ socket" << std::endl;
            zmq_ctx_destroy(zmq_context);
            return 1;
        }
        
        if (zmq_bind(zmq_publisher, cfg.zmqEndpoint.c_str()) != 0) {
            std::cerr << "Error: Could not bind ZMQ socket to " << cfg.zmqEndpoint << std::endl;
            zmq_close(zmq_publisher);
            zmq_ctx_destroy(zmq_context);
            return 1;
        }
        
        if (!cfg.quiet) {
            std::cerr << "ZMQ publisher bound to " << cfg.zmqEndpoint << std::endl;
        }
        
        // Give subscribers time to connect
        usleep(100000);
    }
    
    // ========================================================================
    // ALLOCATE PINNED HOST MEMORY
    // ========================================================================
    float* h_input = nullptr;
    float* h_output = nullptr;
    
    cudaError_t err = cudaHostAlloc(&h_input, inputByteSize, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cerr << "Error: Could not allocate pinned input memory: " 
                  << cudaGetErrorString(err) << std::endl;
        if (cfg.useZmq) {
            zmq_close(zmq_publisher);
            zmq_ctx_destroy(zmq_context);
        }
        return 1;
    }
    
    err = cudaHostAlloc(&h_output, cfg.fftSize * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cerr << "Error: Could not allocate pinned output memory: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFreeHost(h_input);
        if (cfg.useZmq) {
            zmq_close(zmq_publisher);
            zmq_ctx_destroy(zmq_context);
        }
        return 1;
    }
    
    // ========================================================================
    // ALLOCATE DEVICE INPUT MEMORY
    // ========================================================================
    float* d_input = nullptr;
    err = cudaMalloc(&d_input, inputByteSize);
    if (err != cudaSuccess) {
        std::cerr << "Error: Could not allocate GPU input memory: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFreeHost(h_input);
        cudaFreeHost(h_output);
        if (cfg.useZmq) {
            zmq_close(zmq_publisher);
            zmq_ctx_destroy(zmq_context);
        }
        return 1;
    }
    
    // ========================================================================
    // INITIALIZE CUDA PIPELINE
    // ========================================================================
    if (initPipeline(cfg.inputSamples, cfg.fftSize, cfg.windowType) != 0) {
        std::cerr << "Error: Failed to initialize CUDA pipeline" << std::endl;
        cudaFree(d_input);
        cudaFreeHost(h_input);
        cudaFreeHost(h_output);
        if (cfg.useZmq) {
            zmq_close(zmq_publisher);
            zmq_ctx_destroy(zmq_context);
        }
        return 1;
    }
    
    if (!cfg.quiet) {
        std::cerr << "cuda_spectrum initialized:" << std::endl;
        std::cerr << "  Input samples: " << cfg.inputSamples << std::endl;
        std::cerr << "  FFT size:      " << cfg.fftSize << std::endl;
        std::cerr << "  Frame size:    " << inputByteSize << " bytes" << std::endl;
        std::cerr << "  Window:        " << cfg.windowType << std::endl;
        std::cerr << "  Ref level:     " << cfg.referenceLevel << " dB" << std::endl;
        std::cerr << "  Dynamic range: " << cfg.dynamicRange << " dB" << std::endl;
        if (cfg.useZmq) {
            std::cerr << "  ZMQ endpoint:  " << cfg.zmqEndpoint << std::endl;
        }
        if (!cfg.noStdout) {
            std::cerr << "  Stdout:        enabled" << std::endl;
        }
        std::cerr << "Ready to process..." << std::endl;
    }
    
    // ========================================================================
    // MAIN PROCESSING LOOP
    // ========================================================================
    int frameCount = 0;
    
    while (true) {
        // Check frame limit
        if (cfg.frameLimit > 0 && frameCount >= cfg.frameLimit) {
            break;
        }
        
        // Read frame from stdin
        std::cin.read(reinterpret_cast<char*>(h_input), inputByteSize);
        std::streamsize bytesRead = std::cin.gcount();
        
        if (bytesRead < static_cast<std::streamsize>(inputByteSize)) {
            if (bytesRead > 0 && !cfg.quiet) {
                std::cerr << "Warning: Partial frame (" << bytesRead 
                          << " bytes), skipping." << std::endl;
            }
            break;
        }
        
        frameCount++;
        
        // Copy to device
        cudaMemcpy(d_input, h_input, inputByteSize, cudaMemcpyHostToDevice);
        
        // Process frame
        processFrame(d_input, h_output, 
                    cfg.referenceLevel, cfg.dynamicRange,
                    cfg.outputDb, !cfg.noFftShift);
        
        // Output via ZMQ
        if (cfg.useZmq) {
            int rc = zmq_send(zmq_publisher, h_output, 
                             cfg.fftSize * sizeof(float), 0);
            if (rc == -1 && !cfg.quiet) {
                std::cerr << "Warning: ZMQ send failed" << std::endl;
            }
        }
        
        // Output to stdout (if not disabled)
        if (!cfg.noStdout) {
            switch (cfg.outputFormat) {
                case 0:
                    outputFrameCsv(h_output, cfg.fftSize, frameCount);
                    break;
                case 1:
                    outputFrameBinary(h_output, cfg.fftSize, frameCount);
                    break;
                case 2:
                    outputFrameJson(h_output, cfg.fftSize, frameCount);
                    break;
            }
        }
    }
    
    if (!cfg.quiet) {
        std::cerr << "\nProcessed " << frameCount << " frames." << std::endl;
    }
    
    // ========================================================================
    // CLEANUP
    // ========================================================================
    cleanupPipeline();
    cudaFree(d_input);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    
    if (cfg.useZmq) {
        zmq_close(zmq_publisher);
        zmq_ctx_destroy(zmq_context);
    }
    
    return 0;
}

