#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

// ============================================================================
// ALIGNED ALLOCATOR (64-byte cache-line alignment for AVX2 optimization)
// ============================================================================
template<typename T, std::size_t Alignment = 64>
class aligned_allocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    template<typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept = default;

    template<typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;

        size_type alignment = Alignment;
        size_type size = n * sizeof(T);

        void* ptr = nullptr;

#ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
#else
        if (posix_memalign(&ptr, alignment, size) != 0) {
            ptr = nullptr;
        }
#endif

        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        if (p) {
#ifdef _WIN32
            _aligned_free(p);
#else
            free(p);
#endif
        }
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new(p) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p) {
        p->~U();
    }
};

template<typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator==(const aligned_allocator<T1, A1>&, const aligned_allocator<T2, A2>&) noexcept {
    return A1 == A2;
}

template<typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator!=(const aligned_allocator<T1, A1>&, const aligned_allocator<T2, A2>&) noexcept {
    return A1 != A2;
}

// ============================================================================
// CPU FEATURES
// ============================================================================
struct CPUFeatures {
    static bool hasAVX2() noexcept;
    static bool hasFMA() noexcept;
    static bool checkCPUID(int function, int subfunction, int reg, int bit);
    static void printCapabilities() noexcept;
};

// ============================================================================
// ENGINE METRICS (original snake_case version)
// ============================================================================
struct EngineMetrics {
    uint64_t total_execution_time_ns = 0;
    uint64_t avx2_operation_time_ns  = 0;
    long long total_operations     = 0;
    long long avx2_operations      = 0;
    long long node_processes       = 0;
    long long harmonic_generations = 0;

    double current_ns_per_op       = 0.0;
    double target_ns_per_op        = 8000.0;
    double speedup_factor          = 0.0;
    double current_ops_per_second  = 0.0;

    double avg_time_ns             = 0.0;
    double throughput_gflops       = 0.0;

    // legacy method declarations
    void reset() noexcept;
    void update_performance() noexcept;
    void print_metrics() noexcept;
};

// ============================================================================
// ANALOG UNIVERSAL NODE AVX2
// ============================================================================
// Force inline macros for Phase 4A optimization
#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline __attribute__((always_inline))
#endif

class AnalogUniversalNodeAVX2 {
public:
    double integrator_state = 0.0;
    double previous_input   = 0.0;
    double current_output   = 0.0;
    double feedback_gain    = 0.0;
    int    node_id          = 0;

    // Spatial coordinates for node positioning
    int16_t x = 0;
    int16_t y = 0;
    int16_t z = 0;

    AnalogUniversalNodeAVX2() = default;

    // Phase 4A: Force inline trivial functions
    FORCE_INLINE double amplify(double input_signal, double gain);
    FORCE_INLINE double integrate(double input_signal, double time_constant);
    FORCE_INLINE double applyFeedback(double input_signal, double feedback_gain);

    double processSignal(double input_signal, double control_signal, double aux_signal);
    double processSignalAVX2(double input_signal, double control_signal, double aux_signal);

    // Phase 4A: Hot-path version without profiling overhead
    FORCE_INLINE double processSignalAVX2_hotpath(double input_signal, double control_signal, double aux_signal);

    // Oscillator: generates waveform at given frequency
    std::vector<float> oscillate(double frequency_hz, double duration_seconds);

    // Frequency domain filter: processes signal block using FFT
    std::vector<float> processBlockFrequencyDomain(const std::vector<float>& input_block);

    // NumPy zero-copy versions (for Python integration)
    void oscillate_inplace(float* output, int num_samples, double frequency_hz, double sample_rate);
    void processBlockFrequencyDomain_inplace(float* data, int num_samples);

    // Batch processing: reduces Python call overhead by 5-10x
    std::vector<double> processBatch(const std::vector<double>& input_signals,
                                      const std::vector<double>& control_signals,
                                      const std::vector<double>& aux_signals);

    void setFeedback(double feedback_coefficient);
    double getOutput() const noexcept;
    double getIntegratorState() const noexcept;
    void resetIntegrator() noexcept;

    // Helper function for clamping values
    template<typename T>
    static inline T clamp_custom(T value, T min_val, T max_val) {
        return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
    }
};

// ============================================================================
// ANALOG CELLULAR ENGINE AVX2
// ============================================================================
class AnalogCellularEngineAVX2 {
private:
    // Use 64-byte aligned allocator for cache-line optimization
    std::vector<AnalogUniversalNodeAVX2, aligned_allocator<AnalogUniversalNodeAVX2, 64>> nodes;
    double system_frequency;
    double noise_level;

    // FIX C2.1: Per-instance metrics instead of global static
    // This prevents data races when multiple engines run concurrently
    EngineMetrics metrics_;

public:
    explicit AnalogCellularEngineAVX2(std::size_t num_nodes);

    // Mission and benchmark functions
    void runMission(std::uint64_t steps);

    // Optimized mission with pre-computed signals (for Julia/Rust FFI)
    // Eliminates serial sin/cos bottleneck, achieves ~90% CPU utilization
    void runMissionOptimized(const double* input_signals,
                            const double* control_patterns,
                            std::uint64_t num_steps,
                            std::uint32_t iterations_per_node = 30);

    // Phase 4B: Single parallel region, reduced barriers
    void runMissionOptimized_Phase4B(const double* input_signals,
                                     const double* control_patterns,
                                     std::uint64_t num_steps,
                                     std::uint32_t iterations_per_node = 30);

    // Phase 4C: AVX2 spatial vectorization (process 4 nodes in parallel)
    void runMissionOptimized_Phase4C(const double* input_signals,
                                     const double* control_patterns,
                                     std::uint64_t num_steps,
                                     std::uint32_t iterations_per_node = 30);

    void runMassiveBenchmark(int iterations);
    double runDragRaceBenchmark(int num_runs);
    void runBuiltinBenchmark(int iterations);

    // Signal processing functions (return double, not void)
    double performSignalSweepAVX2(double frequency);
    double processSignalWaveAVX2(double input_signal, double control_pattern);

    // Frequency domain processing (takes std::vector<double>&, not const std::vector<float>&)
    void processBlockFrequencyDomain(std::vector<double>& signal_block);

    // Helper functions
    double calculateInterNodeCoupling(std::size_t node_index);
    void printLiveMetrics();
    void resetMetrics();
    double generateNoiseSignal();

    // Node output access
    std::vector<double> getNodeOutputs() const {
        std::vector<double> outputs;
        outputs.reserve(nodes.size());
        for (const auto& node : nodes) {
            outputs.push_back(node.current_output);
        }
        return outputs;
    }

    // Metrics access
    EngineMetrics getMetrics() const noexcept;
};
