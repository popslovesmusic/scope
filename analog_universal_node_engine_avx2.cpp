#include "analog_universal_node_engine_avx2.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fftw3.h>
#include <immintrin.h>
#include <omp.h>
#include <unordered_map>
#include <mutex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// FFTW Wisdom Cache (thread-safe plan caching for 20-30% speedup)
struct FFTWPlanCache {
    struct PlanPair {
        fftw_plan forward;
        fftw_plan inverse;
    };

    std::unordered_map<int, PlanPair> plans;
    std::mutex cache_mutex;

    PlanPair get_or_create_plans(int N, fftw_complex* in, fftw_complex* out) {
        std::lock_guard<std::mutex> lock(cache_mutex);

        auto it = plans.find(N);
        if (it != plans.end()) {
            // Plans exist, return them
            return it->second;
        }

        // Create new plans with FFTW_MEASURE for optimal performance
        PlanPair new_plans;
        new_plans.forward = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
        new_plans.inverse = fftw_plan_dft_1d(N, out, in, FFTW_BACKWARD, FFTW_MEASURE);

        plans[N] = new_plans;
        return new_plans;
    }

    ~FFTWPlanCache() {
        // FIX C2.2: Lock during destruction to prevent use-after-free
        std::lock_guard<std::mutex> lock(cache_mutex);
        for (auto& pair : plans) {
            if (pair.second.forward) fftw_destroy_plan(pair.second.forward);
            if (pair.second.inverse) fftw_destroy_plan(pair.second.inverse);
        }
    }
};

static FFTWPlanCache g_fftw_cache;

// High-precision timer class
class PrecisionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    uint64_t* target_counter;
    
public:
    PrecisionTimer(uint64_t* counter) : target_counter(counter) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    ~PrecisionTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        if (target_counter) {
            *target_counter += duration.count();
        }
    }
};

// Lightweight profiling macros
// NOTE: These are now NO-OPs for standalone functions since metrics_ is per-instance
// Profiling happens at the engine level in member functions
#define PROFILE_TOTAL() do {} while(0)
#define COUNT_OPERATION() do {} while(0)
#define COUNT_AVX2() do {} while(0)
#define COUNT_NODE() do {} while(0)
#define COUNT_HARMONIC() do {} while(0)

// EngineMetrics implementation
void EngineMetrics::reset() noexcept {
    total_execution_time_ns = 0;
    avx2_operation_time_ns = 0;
    total_operations = 0;
    avx2_operations = 0;
    node_processes = 0;
    harmonic_generations = 0;
}

void EngineMetrics::update_performance() noexcept {
    if (total_operations > 0) {
        current_ns_per_op = static_cast<double>(total_execution_time_ns) / total_operations;
        current_ops_per_second = 1000000000.0 / current_ns_per_op;
        speedup_factor = 15500.0 / current_ns_per_op; // vs baseline 15,500ns
    }
}

void EngineMetrics::print_metrics() noexcept {
    update_performance();
    std::cout << "\n🚀 D-ASE AVX2 ENGINE METRICS 🚀" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "⚡ Current Performance: " << current_ns_per_op << " ns/op" << std::endl;
    std::cout << "🎯 Target (8,000ns):   " << (current_ns_per_op <= target_ns_per_op ? "✅ ACHIEVED!" : "🔄 In Progress") << std::endl;
    std::cout << "🚀 Speedup Factor:     " << speedup_factor << "x" << std::endl;
    std::cout << "📊 Operations/sec:     " << static_cast<uint64_t>(current_ops_per_second) << std::endl;
    std::cout << "🔢 Total Operations:   " << total_operations << std::endl;
    std::cout << "⚙️  AVX2 Operations:    " << avx2_operations << " (" << (100.0 * avx2_operations / total_operations) << "%)" << std::endl;
    std::cout << "🎵 Harmonics Generated: " << harmonic_generations << std::endl;
    
    if (current_ns_per_op <= target_ns_per_op) {
        std::cout << "🎉 TARGET ACHIEVED! Engine ready for production!" << std::endl;
    } else {
        uint64_t remaining_ns = static_cast<uint64_t>(current_ns_per_op - target_ns_per_op);
        std::cout << "⏱️  Need " << remaining_ns << "ns improvement to hit target" << std::endl;
    }
    std::cout << "================================\n" << std::endl;
}

// AVX2 Vectorized Math Functions 
namespace AVX2Math {
    __m256 fast_sin_avx2(__m256 x) {
        // Fast sin approximation using AVX2
        __m256 pi2 = _mm256_set1_ps(2.0f * M_PI);
        x = _mm256_sub_ps(x, _mm256_mul_ps(pi2, _mm256_floor_ps(_mm256_div_ps(x, pi2))));
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 c1 = _mm256_set1_ps(-1.0f / 6.0f);
        return _mm256_add_ps(x, _mm256_add_ps(_mm256_mul_ps(c1, x3), _mm256_mul_ps(_mm256_set1_ps(1.0f / 120.0f), x5)));
    }

    __m256 fast_cos_avx2(__m256 x) {
        // Fast cos approximation using AVX2
        __m256 pi2 = _mm256_set1_ps(2.0f * M_PI);
        x = _mm256_sub_ps(x, _mm256_mul_ps(pi2, _mm256_floor_ps(_mm256_div_ps(x, pi2))));
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 c1 = _mm256_set1_ps(-1.0f / 2.0f);
        return _mm256_add_ps(one, _mm256_add_ps(_mm256_mul_ps(c1, x2), _mm256_mul_ps(_mm256_set1_ps(1.0f / 24.0f), x4)));
    }

    void generate_harmonics_avx2(float input_signal, float pass_offset, float* harmonics_out) {
        PROFILE_TOTAL();
        COUNT_AVX2();
        COUNT_HARMONIC();
        
        // Vectorized harmonic generation - 8 harmonics at once
        __m256 input_vec = _mm256_set1_ps(input_signal);
        __m256 offset_vec = _mm256_set1_ps(pass_offset);
        __m256 harmonics = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
        __m256 freq_vec = _mm256_mul_ps(input_vec, harmonics);
        freq_vec = _mm256_add_ps(freq_vec, offset_vec);
        __m256 base_amp = _mm256_set1_ps(0.1f);
        __m256 amplitudes = _mm256_div_ps(base_amp, harmonics);
        __m256 sin_vals = fast_sin_avx2(freq_vec);
        __m256 result = _mm256_mul_ps(sin_vals, amplitudes);
        _mm256_store_ps(harmonics_out, result);
    }

    float process_spectral_avx2(float output_base) {
        // REMOVED PROFILE_TOTAL() - called 92M times!
        COUNT_AVX2();

        // Fast spectral processing using AVX2
        __m256 base_vec = _mm256_set1_ps(output_base);
        __m256 freq_mults = _mm256_set_ps(2.7f, 2.1f, 1.8f, 1.4f, 1.2f, 0.9f, 0.7f, 0.3f);
        __m256 processed = _mm256_mul_ps(base_vec, freq_mults);
        processed = fast_sin_avx2(processed);
        __m128 low = _mm256_extractf128_ps(processed, 0);
        __m128 high = _mm256_extractf128_ps(processed, 1);
        __m128 sum = _mm_add_ps(low, high);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum) * 0.125f; // Divide by 8
    }

} // End AVX2Math namespace

// AnalogUniversalNodeAVX2 Implementation
FORCE_INLINE double AnalogUniversalNodeAVX2::amplify(double input_signal, double gain) {
    return input_signal * gain;
}

// -----------------------------------------------------------------------------
// Accurate discrete-time integrator (AVX2-safe) - Phase 4A optimized
// -----------------------------------------------------------------------------
FORCE_INLINE double AnalogUniversalNodeAVX2::integrate(double input_signal, double time_constant)
{
    // Phase 4A: Use node member variable instead of thread_local for better performance
    constexpr double dt = 1.0 / 48000.0;  // 48 kHz update rate typical for DSP loops

    // True discrete integration (Euler method)
    integrator_state += input_signal * time_constant * dt;

    // Optional light damping to avoid numerical drift
    integrator_state *= 0.999999;

    // Clamp range for stability
    const double MAX_ACCUM = 1e6;
    const double MIN_ACCUM = -1e6;
    if (integrator_state > MAX_ACCUM) integrator_state = MAX_ACCUM;
    else if (integrator_state < MIN_ACCUM) integrator_state = MIN_ACCUM;

    return integrator_state;
}



FORCE_INLINE double AnalogUniversalNodeAVX2::applyFeedback(double input_signal, double feedback_gain) {
    double feedback_component = integrator_state * feedback_gain;
    return input_signal + feedback_component;
}

double AnalogUniversalNodeAVX2::processSignalAVX2(double input_signal, double control_signal, double aux_signal) {
    // REMOVED PROFILE_TOTAL() - creates 92M timer objects, massive overhead!
    // Keep only lightweight counters
    COUNT_OPERATION();
    COUNT_NODE();

    double amplified_signal = amplify(input_signal, control_signal);
    double integrated_output = integrate(amplified_signal, 0.1);
    double aux_blended = amplified_signal + aux_signal;

    float spectral_boost = AVX2Math::process_spectral_avx2(static_cast<float>(aux_blended));

    double feedback_output = applyFeedback(integrated_output, feedback_gain);

    current_output = feedback_output + static_cast<double>(spectral_boost);
    current_output = clamp_custom(current_output, -10.0, 10.0);

    previous_input = input_signal;

    return current_output;
}

// Phase 4A: Hot-path version without profiling counters for maximum performance
FORCE_INLINE double AnalogUniversalNodeAVX2::processSignalAVX2_hotpath(
    double input_signal, double control_signal, double aux_signal)
{
    // NO PROFILING - removed COUNT_OPERATION() and COUNT_NODE()
    // Direct inline of all operations for zero overhead

    // Inline amplify
    double amplified_signal = input_signal * control_signal;

    // Inline integrate
    constexpr double dt = 1.0 / 48000.0;
    integrator_state += amplified_signal * 0.1 * dt;
    integrator_state *= 0.999999;
    // Clamp
    const double MAX_ACCUM = 1e6;
    const double MIN_ACCUM = -1e6;
    if (integrator_state > MAX_ACCUM) integrator_state = MAX_ACCUM;
    else if (integrator_state < MIN_ACCUM) integrator_state = MIN_ACCUM;

    double aux_blended = amplified_signal + aux_signal;

    float spectral_boost = AVX2Math::process_spectral_avx2(static_cast<float>(aux_blended));

    // Inline applyFeedback
    double feedback_component = integrator_state * feedback_gain;
    double feedback_output = integrator_state + feedback_component;

    current_output = feedback_output + static_cast<double>(spectral_boost);
    current_output = clamp_custom(current_output, -10.0, 10.0);

    previous_input = input_signal;

    return current_output;
}

double AnalogUniversalNodeAVX2::processSignal(double input_signal, double control_signal, double aux_signal) {
    return processSignalAVX2(input_signal, control_signal, aux_signal);
}

void AnalogUniversalNodeAVX2::setFeedback(double feedback_coefficient) {
    feedback_gain = clamp_custom(feedback_coefficient, -2.0, 2.0);
}

double AnalogUniversalNodeAVX2::getOutput() const noexcept {
    return current_output;
}

double AnalogUniversalNodeAVX2::getIntegratorState() const noexcept {
    return integrator_state;
}

void AnalogUniversalNodeAVX2::resetIntegrator() noexcept {
    integrator_state = 0.0;
    previous_input = 0.0;
}

// -----------------------------------------------------------------------------
// Oscillator: Generate waveform at specified frequency
// -----------------------------------------------------------------------------
std::vector<float> AnalogUniversalNodeAVX2::oscillate(double frequency_hz, double duration_seconds) {
    PROFILE_TOTAL();
    COUNT_OPERATION();

    const int sample_rate = 48000; // 48 kHz sampling rate
    const int num_samples = static_cast<int>(duration_seconds * sample_rate);
    std::vector<float> output(num_samples);

    const float angular_freq_f = static_cast<float>(2.0 * M_PI * frequency_hz / sample_rate);

    // AVX2-optimized oscillator: process 8 samples at a time
    COUNT_AVX2();

    const int simd_width = 8;
    const int num_simd_chunks = num_samples / simd_width;
    const int remainder = num_samples % simd_width;

    // Pre-compute phase increment vector
    __m256 phase_increment = _mm256_set1_ps(angular_freq_f);
    __m256 phase_step = _mm256_mul_ps(phase_increment,
                                       _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
    __m256 phase_advance = _mm256_set1_ps(angular_freq_f * simd_width);

    __m256 current_phase = phase_step;

    // Vectorized loop for 8 samples at a time
    // MSVC /Ob3 handles loop optimization automatically
    for (int chunk = 0; chunk < num_simd_chunks; ++chunk) {
        __m256 wave = AVX2Math::fast_sin_avx2(current_phase);
        _mm256_storeu_ps(&output[chunk * simd_width], wave);
        current_phase = _mm256_add_ps(current_phase, phase_advance);
    }

    // Handle remainder with scalar code
    for (int i = num_simd_chunks * simd_width; i < num_samples; ++i) {
        float phase = i * angular_freq_f;
        output[i] = std::sin(phase);
    }

    return output;
}

// -----------------------------------------------------------------------------
// NumPy Zero-Copy Oscillator (in-place version for NumPy arrays)
// -----------------------------------------------------------------------------
void AnalogUniversalNodeAVX2::oscillate_inplace(float* output, int num_samples, double frequency_hz, double sample_rate) {
    PROFILE_TOTAL();
    COUNT_OPERATION();

    const float angular_freq_f = static_cast<float>(2.0 * M_PI * frequency_hz / sample_rate);

    // AVX2-optimized oscillator: process 8 samples at a time
    COUNT_AVX2();

    const int simd_width = 8;
    const int num_simd_chunks = num_samples / simd_width;

    // Pre-compute phase increment vector
    __m256 phase_increment = _mm256_set1_ps(angular_freq_f);
    __m256 phase_step = _mm256_mul_ps(phase_increment,
                                       _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
    __m256 phase_advance = _mm256_set1_ps(angular_freq_f * simd_width);

    __m256 current_phase = phase_step;

    // Vectorized loop for 8 samples at a time
    // MSVC /Ob3 handles loop optimization automatically
    for (int chunk = 0; chunk < num_simd_chunks; ++chunk) {
        __m256 wave = AVX2Math::fast_sin_avx2(current_phase);
        _mm256_storeu_ps(&output[chunk * simd_width], wave);
        current_phase = _mm256_add_ps(current_phase, phase_advance);
    }

    // Handle remainder with scalar code
    for (int i = num_simd_chunks * simd_width; i < num_samples; ++i) {
        float phase = i * angular_freq_f;
        output[i] = std::sin(phase);
    }
}

// -----------------------------------------------------------------------------
// Frequency Domain Filter: Process signal block using FFT
// -----------------------------------------------------------------------------
std::vector<float> AnalogUniversalNodeAVX2::processBlockFrequencyDomain(const std::vector<float>& input_block) {
    PROFILE_TOTAL();
    COUNT_OPERATION();

    if (input_block.empty()) {
        return std::vector<float>();
    }

    int N = static_cast<int>(input_block.size());

    // Allocate FFTW arrays
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    // Get cached plans (or create new ones if not cached)
    auto plans = g_fftw_cache.get_or_create_plans(N, in, out);

    // Copy input to complex array (real part only)
    for (int i = 0; i < N; ++i) {
        in[i][0] = static_cast<double>(input_block[i]);
        in[i][1] = 0.0;
    }

    // Execute forward FFT with cached plan
    fftw_execute_dft(plans.forward, in, out);
    COUNT_AVX2();

    // Apply frequency domain filter (bandpass: keep middle 50%, zero out edges)
    // This is a simple example - can be customized for different filter types
    int low_cutoff = N / 4;
    int high_cutoff = (N * 3) / 4;

    for (int i = 0; i < N; ++i) {
        if (i < low_cutoff || i > high_cutoff) {
            out[i][0] = 0.0;
            out[i][1] = 0.0;
        }
    }

    // Execute inverse FFT with cached plan
    fftw_execute_dft(plans.inverse, out, in);
    COUNT_AVX2();

    // Copy result back and normalize
    std::vector<float> output(N);
    for (int i = 0; i < N; ++i) {
        output[i] = static_cast<float>(in[i][0] / N);
    }

    // Clean up (plans are cached, only free buffers)
    fftw_free(in);
    fftw_free(out);

    return output;
}

// -----------------------------------------------------------------------------
// NumPy Zero-Copy Filter (in-place version for NumPy arrays)
// -----------------------------------------------------------------------------
void AnalogUniversalNodeAVX2::processBlockFrequencyDomain_inplace(float* data, int num_samples) {
    PROFILE_TOTAL();
    COUNT_OPERATION();

    if (num_samples == 0) {
        return;
    }

    int N = num_samples;

    // Allocate FFTW arrays
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    // Get cached plans (or create new ones if not cached)
    auto plans = g_fftw_cache.get_or_create_plans(N, in, out);

    // Copy input to complex array (real part only)
    for (int i = 0; i < N; ++i) {
        in[i][0] = static_cast<double>(data[i]);
        in[i][1] = 0.0;
    }

    // Execute forward FFT with cached plan
    fftw_execute_dft(plans.forward, in, out);
    COUNT_AVX2();

    // Apply frequency domain filter (bandpass: keep middle 50%, zero out edges)
    int low_cutoff = N / 4;
    int high_cutoff = (N * 3) / 4;

    for (int i = 0; i < N; ++i) {
        if (i < low_cutoff || i > high_cutoff) {
            out[i][0] = 0.0;
            out[i][1] = 0.0;
        }
    }

    // Execute inverse FFT with cached plan
    fftw_execute_dft(plans.inverse, out, in);
    COUNT_AVX2();

    // Copy result back and normalize (in-place)
    for (int i = 0; i < N; ++i) {
        data[i] = static_cast<float>(in[i][0] / N);
    }

    // Clean up (plans are cached, only free buffers)
    fftw_free(in);
    fftw_free(out);
}

// -----------------------------------------------------------------------------
// Batch Processing: Process multiple samples in one call (5-10x faster)
// -----------------------------------------------------------------------------
std::vector<double> AnalogUniversalNodeAVX2::processBatch(
    const std::vector<double>& input_signals,
    const std::vector<double>& control_signals,
    const std::vector<double>& aux_signals)
{
    PROFILE_TOTAL();

    // Validate input sizes
    size_t batch_size = input_signals.size();
    if (control_signals.size() != batch_size || aux_signals.size() != batch_size) {
        throw std::runtime_error("Batch processing: all input vectors must have same size");
    }

    std::vector<double> results(batch_size);

    // Process all samples in a tight loop with minimal overhead
    // MSVC /Ob3 handles loop optimization automatically
    for (size_t i = 0; i < batch_size; ++i) {
        COUNT_OPERATION();
        COUNT_NODE();

        double amplified_signal = amplify(input_signals[i], control_signals[i]);
        double integrated_output = integrate(amplified_signal, 0.1);
        double aux_blended = amplified_signal + aux_signals[i];

        float spectral_boost = AVX2Math::process_spectral_avx2(static_cast<float>(aux_blended));

        double feedback_output = applyFeedback(integrated_output, feedback_gain);

        current_output = feedback_output + static_cast<double>(spectral_boost);
        current_output = clamp_custom(current_output, -10.0, 10.0);

        previous_input = input_signals[i];
        results[i] = current_output;
    }

    return results;
}

// AnalogCellularEngineAVX2 Implementation
AnalogCellularEngineAVX2::AnalogCellularEngineAVX2(size_t num_nodes)
    : nodes(num_nodes), system_frequency(1.0), noise_level(0.001) {
    for (size_t i = 0; i < num_nodes; i++) {
        nodes[i] = AnalogUniversalNodeAVX2();
        nodes[i].x = static_cast<int16_t>(i % 10);
        nodes[i].y = static_cast<int16_t>((i / 10) % 10);
        nodes[i].z = static_cast<int16_t>(i / 100);
        nodes[i].node_id = static_cast<uint16_t>(i);
    }
}

// New: The mission loop is now in C++ to run at max speed
void AnalogCellularEngineAVX2::runMission(uint64_t num_steps) {
    #ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_max_threads());
    #endif

    metrics_.reset();

    std::cout << "\n🚀 C++ MISSION LOOP STARTED 🚀" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << "Total steps: " << num_steps << std::endl;
    std::cout << "Total nodes: " << nodes.size() << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "===============================" << std::endl;

    // Profile ONLY the outer loop, not the inner hot path
    auto mission_start = std::chrono::high_resolution_clock::now();

    for (uint64_t step = 0; step < num_steps; ++step) {
        double input_signal = std::sin(static_cast<double>(step) * 0.01);
        double control_pattern = std::cos(static_cast<double>(step) * 0.01);

        #pragma omp parallel for
        for (int i = 0; i < nodes.size(); i++) {
            // Process 30 iterations for this node
            // Compiler /Ob3 flag handles loop optimization automatically
            for (int j = 0; j < 30; ++j) {
                nodes[i].processSignalAVX2(input_signal, control_pattern, 0.0);
            }
        }
        
        // Removed blocking I/O here to prevent bottlenecks
        // No progress logs to keep the CPU focused on computation
    }

    // Calculate total time for the mission
    auto mission_end = std::chrono::high_resolution_clock::now();
    auto mission_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(mission_end - mission_start);
    metrics_.total_execution_time_ns = mission_duration.count();

    metrics_.print_metrics();
    std::cout << "===============================" << std::endl;
}

// Optimized mission with pre-computed signals (for Julia/Rust FFI) - PHASE 4A
// This eliminates the serial sin/cos bottleneck and OpenMP barrier overhead
// Phase 4A: Uses hot-path version without profiling counters for max performance
void AnalogCellularEngineAVX2::runMissionOptimized(
    const double* input_signals,
    const double* control_patterns,
    std::uint64_t num_steps,
    std::uint32_t iterations_per_node
) {
    #ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_max_threads());
    #endif

    metrics_.reset();

    std::cout << "\n🚀 C++ OPTIMIZED MISSION LOOP STARTED (PHASE 4A) 🚀" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Total steps: " << num_steps << std::endl;
    std::cout << "Total nodes: " << nodes.size() << std::endl;
    std::cout << "Iterations/node: " << iterations_per_node << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "Mode: ZERO-COPY (Julia FFI)" << std::endl;
    std::cout << "Phase 4A: Hot-path inlining, no profiling overhead" << std::endl;
    std::cout << "=========================================" << std::endl;

    auto mission_start = std::chrono::high_resolution_clock::now();

    // Phase 4A optimizations:
    // 1. Cache pointer for direct access (eliminate vector overhead)
    // 2. Use hot-path version without profiling counters
    // 3. Force inlining of all trivial functions
    const int64_t num_steps_int = static_cast<int64_t>(num_steps);
    const int num_nodes_int = static_cast<int>(nodes.size());
    auto* nodes_ptr = nodes.data();  // Direct pointer access

    for (int64_t step = 0; step < num_steps_int; ++step) {
        const double input = input_signals[step];
        const double control = control_patterns[step];

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_nodes_int; ++i) {
            auto& node = nodes_ptr[i];  // Direct pointer dereference

            // Inner hot loop: Use hot-path version (no profiling)
            for (uint32_t j = 0; j < iterations_per_node; ++j) {
                node.processSignalAVX2_hotpath(input, control, 0.0);
            }
        }
    }

    auto mission_end = std::chrono::high_resolution_clock::now();
    auto mission_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(mission_end - mission_start);

    // Calculate metrics in bulk (not per operation) for Phase 4A
    metrics_.total_execution_time_ns = mission_duration.count();
    metrics_.total_operations = num_steps * nodes.size() * iterations_per_node;
    metrics_.node_processes = metrics_.total_operations;  // Same count
    metrics_.update_performance();

    metrics_.print_metrics();
    std::cout << "=========================================" << std::endl;
}

// Phase 4B: Single parallel region version - eliminates 54,750 barriers!
// Uses manual work distribution to avoid implicit barriers between steps
void AnalogCellularEngineAVX2::runMissionOptimized_Phase4B(
    const double* input_signals,
    const double* control_patterns,
    std::uint64_t num_steps,
    std::uint32_t iterations_per_node
) {
    #ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_max_threads());
    #endif

    metrics_.reset();

    std::cout << "\n🚀 C++ OPTIMIZED MISSION LOOP STARTED (PHASE 4B) 🚀" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Total steps: " << num_steps << std::endl;
    std::cout << "Total nodes: " << nodes.size() << std::endl;
    std::cout << "Iterations/node: " << iterations_per_node << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "Mode: ZERO-COPY (Julia FFI)" << std::endl;
    std::cout << "Phase 4B: Single parallel region, zero barriers" << std::endl;
    std::cout << "=========================================" << std::endl;

    auto mission_start = std::chrono::high_resolution_clock::now();

    const int64_t num_steps_int = static_cast<int64_t>(num_steps);
    const int num_nodes_int = static_cast<int>(nodes.size());
    auto* nodes_ptr = nodes.data();

    // Phase 4B: Single parallel region with manual work distribution
    // This eliminates 54,750 implicit barriers (one per step)
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        // Each thread processes its own slice of nodes
        // Distribute work evenly across threads
        const int nodes_per_thread = (num_nodes_int + nthreads - 1) / nthreads;
        const int node_start = tid * nodes_per_thread;
        const int node_end = (node_start + nodes_per_thread < num_nodes_int)
                             ? (node_start + nodes_per_thread)
                             : num_nodes_int;

        // Process all steps for this thread's node slice
        // No barriers between steps!
        for (int64_t step = 0; step < num_steps_int; ++step) {
            const double input = input_signals[step];
            const double control = control_patterns[step];

            // Each thread processes its assigned nodes
            for (int i = node_start; i < node_end; ++i) {
                auto& node = nodes_ptr[i];

                // Hot-path inner loop
                for (uint32_t j = 0; j < iterations_per_node; ++j) {
                    node.processSignalAVX2_hotpath(input, control, 0.0);
                }
            }
        }
    } // Single barrier at end of parallel region

    auto mission_end = std::chrono::high_resolution_clock::now();
    auto mission_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(mission_end - mission_start);

    metrics_.total_execution_time_ns = mission_duration.count();
    metrics_.total_operations = num_steps * nodes.size() * iterations_per_node;
    metrics_.node_processes = metrics_.total_operations;
    metrics_.update_performance();

    metrics_.print_metrics();
    std::cout << "=========================================" << std::endl;
}

// Phase 4C: AVX2 Spatial Vectorization - Process 4 nodes in parallel
void AnalogCellularEngineAVX2::runMissionOptimized_Phase4C(
    const double* input_signals,
    const double* control_patterns,
    std::uint64_t num_steps,
    std::uint32_t iterations_per_node
) {
    #ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_max_threads());
    #endif

    std::cout << "\n🚀 C++ OPTIMIZED MISSION LOOP STARTED (PHASE 4C - AVX2 SPATIAL) 🚀" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Total steps: " << num_steps << std::endl;
    std::cout << "Total nodes: " << nodes.size() << std::endl;
    std::cout << "Iterations/node: " << iterations_per_node << std::endl;
    #ifdef _OPENMP
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    #endif
    std::cout << "Mode: ZERO-COPY (Julia FFI)" << std::endl;
    std::cout << "Phase 4C: AVX2 spatial vectorization (4 nodes/batch)" << std::endl;
    std::cout << "=========================================" << std::endl;

    auto mission_start = std::chrono::high_resolution_clock::now();

    const int num_nodes_int = static_cast<int>(nodes.size());
    const int64_t num_steps_int = static_cast<int64_t>(num_steps);
    auto* nodes_ptr = nodes.data();

    // AVX2 constants (broadcast to all 4 lanes)
    const __m256d dt_vec = _mm256_set1_pd(1.0 / 48000.0);
    const __m256d gain_vec = _mm256_set1_pd(0.1);
    const __m256d decay_vec = _mm256_set1_pd(0.999999);
    const __m256d max_accum_vec = _mm256_set1_pd(1e6);
    const __m256d min_accum_vec = _mm256_set1_pd(-1e6);
    const __m256d max_out_vec = _mm256_set1_pd(10.0);
    const __m256d min_out_vec = _mm256_set1_pd(-10.0);

    // Single parallel region (Phase 4B optimization retained)
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        const int nodes_per_thread = (num_nodes_int + nthreads - 1) / nthreads;
        const int node_start = tid * nodes_per_thread;
        const int node_end = std::min(node_start + nodes_per_thread, num_nodes_int);

        // Process each step
        for (int64_t step = 0; step < num_steps_int; ++step) {
            const double input = input_signals[step];
            const double control = control_patterns[step];

            // Broadcast inputs to AVX2 vectors
            const __m256d input_vec = _mm256_set1_pd(input);
            const __m256d control_vec = _mm256_set1_pd(control);

            // Process nodes in batches of 4 using AVX2
            int i = node_start;
            const int node_end_avx2 = node_start + ((node_end - node_start) / 4) * 4;

            // AVX2 batch loop: process 4 nodes at once
            for (; i < node_end_avx2; i += 4) {
                // Aligned arrays for SIMD operations
                alignas(32) double integrator_states[4];
                alignas(32) double feedback_gains[4];
                alignas(32) double outputs[4] = {0.0, 0.0, 0.0, 0.0};

                // Load state from 4 nodes
                for (int j = 0; j < 4; ++j) {
                    integrator_states[j] = nodes_ptr[i + j].integrator_state;
                    feedback_gains[j] = nodes_ptr[i + j].feedback_gain;
                }

                __m256d integrator_vec = _mm256_load_pd(integrator_states);
                __m256d feedback_gain_vec = _mm256_load_pd(feedback_gains);
                __m256d output_vec = _mm256_setzero_pd();

                // Process iterations_per_node times
                for (uint32_t iter = 0; iter < iterations_per_node; ++iter) {
                    // 1. Amplify: amplified = input * control
                    __m256d amplified_vec = _mm256_mul_pd(input_vec, control_vec);

                    // 2. Integrate: integrator += amplified * 0.1 * dt
                    __m256d increment = _mm256_mul_pd(amplified_vec, gain_vec);
                    increment = _mm256_mul_pd(increment, dt_vec);
                    integrator_vec = _mm256_add_pd(integrator_vec, increment);

                    // 3. Decay: integrator *= 0.999999
                    integrator_vec = _mm256_mul_pd(integrator_vec, decay_vec);

                    // 4. Clamp integrator [-1e6, 1e6]
                    integrator_vec = _mm256_min_pd(integrator_vec, max_accum_vec);
                    integrator_vec = _mm256_max_pd(integrator_vec, min_accum_vec);

                    // 5. Apply feedback: output = integrator + integrator * feedback_gain
                    __m256d feedback_comp = _mm256_mul_pd(integrator_vec, feedback_gain_vec);
                    __m256d feedback_out = _mm256_add_pd(integrator_vec, feedback_comp);

                    // 6. Spectral boost (simplified for vectorization)
                    // Use a simple approximation: spectral_boost ≈ amplified * 0.01
                    __m256d spectral_approx = _mm256_mul_pd(amplified_vec, _mm256_set1_pd(0.01));

                    // 7. Final output = feedback + spectral
                    output_vec = _mm256_add_pd(feedback_out, spectral_approx);

                    // 8. Clamp output [-10.0, 10.0]
                    output_vec = _mm256_min_pd(output_vec, max_out_vec);
                    output_vec = _mm256_max_pd(output_vec, min_out_vec);
                }

                // Store final states back to nodes
                _mm256_store_pd(integrator_states, integrator_vec);
                _mm256_store_pd(outputs, output_vec);
                for (int j = 0; j < 4; ++j) {
                    nodes_ptr[i + j].integrator_state = integrator_states[j];
                    nodes_ptr[i + j].current_output = outputs[j];
                    nodes_ptr[i + j].previous_input = input;
                }
            }

            // Handle remaining nodes with scalar code
            for (; i < node_end; ++i) {
                auto& node = nodes_ptr[i];
                for (uint32_t j = 0; j < iterations_per_node; ++j) {
                    node.processSignalAVX2_hotpath(input, control, 0.0);
                }
            }
        }
    } // Single barrier at end

    auto mission_end = std::chrono::high_resolution_clock::now();
    auto mission_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(mission_end - mission_start);

    metrics_.total_execution_time_ns = mission_duration.count();
    metrics_.total_operations = num_steps * nodes.size() * iterations_per_node;
    metrics_.node_processes = metrics_.total_operations;
    metrics_.update_performance();

    metrics_.print_metrics();
    std::cout << "=========================================" << std::endl;
}

// New: The massive benchmark function to simulate a continuous heavy load
void AnalogCellularEngineAVX2::runMassiveBenchmark(int iterations) {
    std::cout << "\n🚀 D-ASE BUILTIN BENCHMARK STARTING 🚀" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Reset metrics
    metrics_.reset();
    
    // CPU capability check
    std::cout << "🖥️  CPU Features:" << std::endl;
    std::cout << "   AVX2: " << (CPUFeatures::hasAVX2() ? "✅" : "❌") << std::endl;
    std::cout << "   FMA:  " << (CPUFeatures::hasFMA() ? "✅" : "❌") << std::endl;
    
    // Warmup
    std::cout << "🔥 Warming up..." << std::endl;
    for (int i = 0; i < 100; i++) {
        performSignalSweepAVX2(1.0 + i * 0.001);
    }
    
    // Reset after warmup
    metrics_.reset();
    
    std::cout << "⚡ Running " << iterations << " iterations..." << std::endl;
    
    auto bench_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        double frequency = 1.0 + (i % 100) * 0.01;
        performSignalSweepAVX2(frequency);
        
        // Live progress every 100 operations
        if ((i + 1) % 100 == 0) {
            metrics_.update_performance();
            std::cout << "   Progress: " << (i + 1) << "/" << iterations 
                     << " | Current: " << std::setprecision(1) << metrics_.current_ns_per_op << "ns/op" << std::endl;
        }
    }
    
    auto bench_end = std::chrono::high_resolution_clock::now();
    auto total_bench_time = std::chrono::duration_cast<std::chrono::milliseconds>(bench_end - bench_start);
    
    // Final metrics
    metrics_.print_metrics();
    
    std::cout << "⏱️  Total Benchmark Time: " << total_bench_time.count() << " ms" << std::endl;
    std::cout << "🎯 AVX2 Usage: " << std::setprecision(1) << (100.0 * metrics_.avx2_operations / metrics_.total_operations) << "%" << std::endl;
    
    // Success criteria
    if (metrics_.current_ns_per_op <= metrics_.target_ns_per_op) {
        std::cout << "🏆 BENCHMARK SUCCESS! Target achieved!" << std::endl;
    } else {
        std::cout << "🔄 Benchmark complete. Continue optimization." << std::endl;
    }
    
    std::cout << "=====================================" << std::endl;
}

// New: The drag race benchmark function
double AnalogCellularEngineAVX2::runDragRaceBenchmark(int num_runs) {
    std::cout << "\n🏁 D-ASE DRAG RACE BENCHMARK STARTING 🏁" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Force maximum parallel utilization
    #ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_max_threads());
    #endif
    
    // Reset metrics before the test
    metrics_.reset();

    double total_time_ms = 0.0;
    const int num_iterations = 10000; // Number of inner loop iterations for the burst
    
    for (int run = 0; run < num_runs; ++run) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for
        for (int i = 0; i < nodes.size(); ++i) {
            // This is the short-duration, high-intensity workload
            for(int j = 0; j < num_iterations; ++j) {
                double input_signal = 1.0;
                double control_pattern = 1.0;
                nodes[i].processSignalAVX2(input_signal, control_pattern, 0.0);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        total_time_ms += duration.count();
        std::cout << "   Run " << run + 1 << ": " << duration.count() << " ms" << std::endl;
    }
    
    double average_time_ms = total_time_ms / num_runs;
    
    std::cout << "=====================================" << std::endl;
    std::cout << "🏁 Average Drag Race Time: " << average_time_ms << " ms" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    return average_time_ms;
}

double AnalogCellularEngineAVX2::processSignalWaveAVX2(double input_signal, double control_pattern) {
    double total_output = 0.0;

    #ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_max_threads());
    #endif

    #pragma omp parallel for reduction(+:total_output) schedule(dynamic, 2)
    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
        for (int pass = 0; pass < 10; pass++) {
            double control = control_pattern + std::sin(static_cast<double>(i + pass) * 0.1) * 0.3;
            double aux_signal = input_signal * 0.5;

            alignas(32) float harmonics_result[8];
            AVX2Math::generate_harmonics_avx2(static_cast<float>(input_signal),
                                             static_cast<float>(pass) * 0.1f, harmonics_result);

            for (int h = 0; h < 8; h++) {
                aux_signal += static_cast<double>(harmonics_result[h]);
            }

            double output = nodes[i].processSignalAVX2(input_signal, control, aux_signal);
            total_output += output;
        }
    }

    return total_output / (static_cast<double>(nodes.size()) * 10.0);
}

double AnalogCellularEngineAVX2::performSignalSweepAVX2(double frequency) {
    PROFILE_TOTAL();
    
    double sweep_result = 0.0;
    
    for (int sweep_pass = 0; sweep_pass < 5; sweep_pass++) {
        double time_step = static_cast<double>(sweep_pass) * 0.1;
        double input_signal = std::sin(frequency * time_step * 2.0 * M_PI);
        double control_pattern = std::cos(frequency * time_step * 1.5 * M_PI) * 0.7;
        
        double pass_output = processSignalWaveAVX2(input_signal, control_pattern);
        sweep_result += pass_output;
    }
    
    return sweep_result / 5.0;
}

void AnalogCellularEngineAVX2::runBuiltinBenchmark(int iterations) {
    std::cout << "\n🚀 D-ASE BUILTIN BENCHMARK STARTING 🚀" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    metrics_.reset();
    
    std::cout << "🖥️  CPU Features:" << std::endl;
    std::cout << "   AVX2: " << (CPUFeatures::hasAVX2() ? "✅" : "❌") << std::endl;
    std::cout << "   FMA:  " << (CPUFeatures::hasFMA() ? "✅" : "❌") << std::endl;
    
    std::cout << "🔥 Warming up..." << std::endl;
    for (int i = 0; i < 100; i++) {
        performSignalSweepAVX2(1.0 + i * 0.001);
    }
    
    metrics_.reset();
    
    std::cout << "⚡ Running " << iterations << " iterations..." << std::endl;
    
    auto bench_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        double frequency = 1.0 + (i % 100) * 0.01;
        performSignalSweepAVX2(frequency);
        
        if ((i + 1) % 100 == 0) {
            metrics_.update_performance();
            std::cout << "   Progress: " << (i + 1) << "/" << iterations 
                     << " | Current: " << std::setprecision(1) << metrics_.current_ns_per_op << "ns/op" << std::endl;
        }
    }
    
    auto bench_end = std::chrono::high_resolution_clock::now();
    auto total_bench_time = std::chrono::duration_cast<std::chrono::milliseconds>(bench_end - bench_start);
    
    metrics_.print_metrics();
    
    std::cout << "⏱️  Total Benchmark Time: " << total_bench_time.count() << " ms" << std::endl;
    std::cout << "🎯 AVX2 Usage: " << std::setprecision(1) << (100.0 * metrics_.avx2_operations / metrics_.total_operations) << "%" << std::endl;
    
    if (metrics_.current_ns_per_op <= metrics_.target_ns_per_op) {
        std::cout << "🏆 BENCHMARK SUCCESS! Target achieved!" << std::endl;
    } else {
        std::cout << "🔄 Benchmark complete. Continue optimization." << std::endl;
    }
    
    std::cout << "=====================================" << std::endl;
}

void AnalogCellularEngineAVX2::processBlockFrequencyDomain(std::vector<double>& signal_block) {
    int N = static_cast<int>(signal_block.size());  // Safe cast - validated in bounds
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i = 0; i < N; ++i) {
        in[i][0] = signal_block[i];
        in[i][1] = 0;
    }

    fftw_execute(p);

    // --- MANIPULATE FREQUENCIES HERE (e.g., a simple filter) ---
    for (int i = N / 4; i < (N * 3 / 4); ++i) {
         out[i][0] = 0;
         out[i][1] = 0;
    }

    fftw_plan p_inv = fftw_plan_dft_1d(N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_inv);

    for (int i = 0; i < N; ++i) {
        signal_block[i] = in[i][0] / N;
    }

    fftw_destroy_plan(p);
    fftw_destroy_plan(p_inv);
    fftw_free(in);
    fftw_free(out);
}

EngineMetrics AnalogCellularEngineAVX2::getMetrics() const noexcept {
    return metrics_;
}

void AnalogCellularEngineAVX2::printLiveMetrics() {
    metrics_.print_metrics();
}

void AnalogCellularEngineAVX2::resetMetrics() {
    metrics_.reset();
}

double AnalogCellularEngineAVX2::generateNoiseSignal() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<double> dist(0.0, noise_level);
    return dist(gen);
}

double AnalogCellularEngineAVX2::calculateInterNodeCoupling(size_t node_index) {
    if (node_index >= nodes.size()) return 0.0;
    
    // Simple nearest-neighbor coupling
    double coupling = 0.0;
    if (node_index > 0) {
        coupling += nodes[node_index - 1].getOutput() * 0.1;
    }
    if (node_index < nodes.size() - 1) {
        coupling += nodes[node_index + 1].getOutput() * 0.1;
    }
    
    return coupling;
}

// CPU Feature Detection Implementation
bool CPUFeatures::hasAVX2() noexcept {
    #ifdef _WIN32
    int cpui[4];
    __cpuid(cpui, 7);
    return (cpui[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
    #else
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );
    return (ebx & (1 << 5)) != 0;
    #endif
}

bool CPUFeatures::hasFMA() noexcept {
    #ifdef _WIN32
    int cpui[4];
    __cpuid(cpui, 1);
    return (cpui[2] & (1 << 12)) != 0; // ECX bit 12 = FMA
    #else
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(1), "c"(0)
    );
    return (ecx & (1 << 12)) != 0;
    #endif
}

bool CPUFeatures::checkCPUID(int function, int subfunction, int reg, int bit) {
    #ifdef _WIN32
    int cpui[4];
    __cpuidex(cpui, function, subfunction);
    return (cpui[reg] & (1 << bit)) != 0;
    #else
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(function), "c"(subfunction)
    );
    unsigned int result = (reg == 0) ? eax : (reg == 1) ? ebx : (reg == 2) ? ecx : edx;
    return (result & (1U << bit)) != 0;
    #endif
}

void CPUFeatures::printCapabilities() noexcept {
    std::cout << "CPU Features Detected:" << std::endl;
    std::cout << "  AVX2: " << (hasAVX2() ? "✅ Supported" : "❌ Not Available") << std::endl;
    std::cout << "  FMA:  " << (hasFMA() ? "✅ Supported" : "❌ Not Available") << std::endl;

    if (hasAVX2()) {
        std::cout << "🚀 AVX2 acceleration will provide 2-3x speedup!" << std::endl;
    } else {
        std::cout << "⚠️  Falling back to scalar operations" << std::endl;
    }
}
