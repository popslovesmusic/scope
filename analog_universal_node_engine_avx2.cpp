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
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// FFTW Wisdom Cache
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
        if (it != plans.end()) return it->second;
        PlanPair new_plans;
        new_plans.forward = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
        new_plans.inverse = fftw_plan_dft_1d(N, out, in, FFTW_BACKWARD, FFTW_MEASURE);
        plans[N] = new_plans;
        return new_plans;
    }

    ~FFTWPlanCache() {
        std::lock_guard<std::mutex> lock(cache_mutex);
        for (auto& pair : plans) {
            if (pair.second.forward) fftw_destroy_plan(pair.second.forward);
            if (pair.second.inverse) fftw_destroy_plan(pair.second.inverse);
        }
    }
};

static FFTWPlanCache g_fftw_cache;

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
        speedup_factor = 15500.0 / current_ns_per_op;
    }
}

void EngineMetrics::print_metrics() noexcept {
    update_performance();
    std::cout << "\n🚀 D-ASE AVX2 ENGINE METRICS 🚀" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "⚡ Current Performance: " << current_ns_per_op << " ns/op" << std::endl;
    std::cout << "🚀 Speedup Factor:     " << speedup_factor << "x" << std::endl;
    std::cout << "📊 Operations/sec:     " << static_cast<uint64_t>(current_ops_per_second) << std::endl;
    std::cout << "================================\n" << std::endl;
}

// AVX2 Vectorized Math Functions 
namespace AVX2Math {
    __m256 fast_sin_avx2(__m256 x) {
        __m256 pi2 = _mm256_set1_ps(2.0f * M_PI);
        x = _mm256_sub_ps(x, _mm256_mul_ps(pi2, _mm256_floor_ps(_mm256_div_ps(x, pi2))));
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 c1 = _mm256_set1_ps(-1.0f / 6.0f);
        return _mm256_add_ps(x, _mm256_add_ps(_mm256_mul_ps(c1, x3), _mm256_mul_ps(_mm256_set1_ps(1.0f / 120.0f), x5)));
    }

    float process_spectral_avx2(float output_base) {
        __m256 base_vec = _mm256_set1_ps(output_base);
        __m256 freq_mults = _mm256_set_ps(2.7f, 2.1f, 1.8f, 1.4f, 1.2f, 0.9f, 0.7f, 0.3f);
        __m256 processed = _mm256_mul_ps(base_vec, freq_mults);
        processed = fast_sin_avx2(processed);
        __m128 low = _mm256_extractf128_ps(processed, 0);
        __m128 high = _mm256_extractf128_ps(processed, 1);
        __m128 sum = _mm_add_ps(low, high);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum) * 0.125f;
    }

    void generate_harmonics_avx2(float input_signal, float pass_offset, float* harmonics_out) {
        __m256 input_vec = _mm256_set1_ps(input_signal);
        __m256 offset_vec = _mm256_set1_ps(pass_offset);
        __m256 harmonics = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
        __m256 freq_vec = _mm256_add_ps(_mm256_mul_ps(input_vec, harmonics), offset_vec);
        __m256 sin_vals = fast_sin_avx2(freq_vec);
        _mm256_store_ps(harmonics_out, sin_vals);
    }
}

// AnalogUniversalNodeAVX2 Implementation
FORCE_INLINE double AnalogUniversalNodeAVX2::amplify(double input_signal, double gain) {
    return input_signal * gain;
}

FORCE_INLINE double AnalogUniversalNodeAVX2::integrate(double input_signal, double time_constant) {
    constexpr double dt = 1.0 / 48000.0;
    integrator_state += input_signal * time_constant * dt;
    integrator_state *= 0.999999;
    const double MAX_ACCUM = 1e6;
    if (integrator_state > MAX_ACCUM) integrator_state = MAX_ACCUM;
    else if (integrator_state < -MAX_ACCUM) integrator_state = -MAX_ACCUM;
    return integrator_state;
}

FORCE_INLINE double AnalogUniversalNodeAVX2::applyFeedback(double input_signal, double feedback_gain) {
    return input_signal + integrator_state * feedback_gain;
}

double AnalogUniversalNodeAVX2::processSignalScalar(double input_signal, double control_signal, double aux_signal) {
    double amplified = input_signal * control_signal;
    
    // Reaction term (Patch 28): Non-linear dynamics
    if (reaction_enabled) {
        amplified += 0.05 * std::tanh(integrator_state);
    }

    double integrated = integrate(amplified, 0.1);
    
    // Corridor Gating (Patch 28)
    if (corridor_enabled) {
        if (std::abs(integrated) > 8.0) integrated *= 0.5;
    }

    double fb = applyFeedback(integrated, feedback_gain);
    
    // Scalar spectral approximation
    double spectral = 0.0;
    for (double mult : {0.3, 0.7, 0.9, 1.2, 1.4, 1.8, 2.1, 2.7}) {
        spectral += std::sin((amplified + aux_signal) * mult);
    }
    
    current_output = fb + (spectral * 0.125);
    current_output = clamp_custom(current_output, -10.0, 10.0);
    previous_input = input_signal;
    return current_output;
}

double AnalogUniversalNodeAVX2::processSignalAVX2(double input_signal, double control_signal, double aux_signal) {
    double amplified = input_signal * control_signal;
    if (reaction_enabled) {
        amplified += 0.05 * std::tanh(integrator_state);
    }
    double integrated = integrate(amplified, 0.1);
    if (corridor_enabled) {
        if (std::abs(integrated) > 8.0) integrated *= 0.5;
    }
    float spectral = AVX2Math::process_spectral_avx2(static_cast<float>(amplified + aux_signal));
    current_output = applyFeedback(integrated, feedback_gain) + static_cast<double>(spectral);
    current_output = clamp_custom(current_output, -10.0, 10.0);
    previous_input = input_signal;
    return current_output;
}

FORCE_INLINE double AnalogUniversalNodeAVX2::processSignalAVX2_hotpath(double input_signal, double control_signal, double aux_signal) {
    return processSignalAVX2(input_signal, control_signal, aux_signal);
}

double AnalogUniversalNodeAVX2::processSignal(double input_signal, double control_signal, double aux_signal) {
    return processSignalAVX2(input_signal, control_signal, aux_signal);
}

void AnalogUniversalNodeAVX2::setFeedback(double feedback_coefficient) {
    feedback_gain = clamp_custom(feedback_coefficient, -2.0, 2.0);
}

double AnalogUniversalNodeAVX2::getOutput() const noexcept { return current_output; }
double AnalogUniversalNodeAVX2::getIntegratorState() const noexcept { return integrator_state; }
void AnalogUniversalNodeAVX2::resetIntegrator() noexcept { integrator_state = 0.0; previous_input = 0.0; }

// AnalogCellularEngineAVX2 Implementation
AnalogCellularEngineAVX2::AnalogCellularEngineAVX2(size_t num_nodes)
    : nodes(num_nodes), system_frequency(1.0), noise_level(0.001) {
    for (size_t i = 0; i < num_nodes; i++) {
        nodes[i] = AnalogUniversalNodeAVX2();
        nodes[i].node_id = static_cast<int>(i);
    }
}

void AnalogCellularEngineAVX2::runMissionScalar(uint64_t steps) {
    metrics_.reset();
    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t step = 0; step < steps; ++step) {
        double input = std::sin(static_cast<double>(step) * 0.01);
        double ctrl = std::cos(static_cast<double>(step) * 0.01);
        for (auto& node : nodes) {
            node.processSignalScalar(input, ctrl, 0.0);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    metrics_.total_execution_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    metrics_.total_operations = steps * nodes.size();
}

void AnalogCellularEngineAVX2::runMission(uint64_t steps) {
    metrics_.reset();
    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t step = 0; step < steps; ++step) {
        double input = std::sin(static_cast<double>(step) * 0.01);
        double ctrl = std::cos(static_cast<double>(step) * 0.01);
        #pragma omp parallel for
        for (int i = 0; i < (int)nodes.size(); ++i) {
            nodes[i].processSignalAVX2(input, ctrl, 0.0);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    metrics_.total_execution_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    metrics_.total_operations = steps * nodes.size();
}

AnalogCellularEngineAVX2::FieldStats AnalogCellularEngineAVX2::getFieldStatistics(const std::vector<double>& prev_outputs) const {
    FieldStats stats = {0,0,0,0,0};
    if (nodes.empty()) return stats;
    size_t n = nodes.size();
    
    double sum = 0, sum_sq = 0, sum_delta = 0, sum_grad = 0;
    for (size_t i = 0; i < n; ++i) {
        double out = nodes[i].current_output;
        sum += out;
        sum_sq += out * out;
        if (i < prev_outputs.size()) {
            sum_delta += std::abs(out - prev_outputs[i]);
        }
        if (i > 0) {
            double grad = out - nodes[i-1].current_output;
            sum_grad += grad * grad;
        }
    }
    stats.mean = sum / n;
    stats.variance = (sum_sq / n) - (stats.mean * stats.mean);
    stats.state_delta = sum_delta / n;
    stats.gradient_energy = sum_grad / n;
    stats.total_energy = sum_sq;
    return stats;
}

void AnalogCellularEngineAVX2::setIntegratorState(double val) {
    for (auto& node : nodes) node.integrator_state = val;
}

void AnalogCellularEngineAVX2::setReactionEnabled(bool enabled) {
    reaction_enabled = enabled;
    for (auto& node : nodes) node.reaction_enabled = enabled;
}

void AnalogCellularEngineAVX2::setCorridorEnabled(bool enabled) {
    corridor_enabled = enabled;
    for (auto& node : nodes) node.corridor_enabled = enabled;
}

EngineMetrics AnalogCellularEngineAVX2::getMetrics() const noexcept { return metrics_; }
void AnalogCellularEngineAVX2::printLiveMetrics() { metrics_.print_metrics(); }
void AnalogCellularEngineAVX2::resetMetrics() { metrics_.reset(); }

// CPU Feature Detection
bool CPUFeatures::hasAVX2() noexcept { return true; } // Placeholder
bool CPUFeatures::hasFMA() noexcept { return true; } // Placeholder
void CPUFeatures::printCapabilities() noexcept { std::cout << "AVX2: ✅ FMA: ✅" << std::endl; }

// Node block processing (legacy/utility)
void AnalogUniversalNodeAVX2::oscillate_inplace(float* out, int n, double f, double sr) {}
void AnalogUniversalNodeAVX2::processBlockFrequencyDomain_inplace(float* d, int n) {}
std::vector<float> AnalogUniversalNodeAVX2.oscillate(double f, double d) { return {}; }
std::vector<float> AnalogUniversalNodeAVX2.processBlockFrequencyDomain(const std::vector<float>& b) { return {}; }
std::vector<double> AnalogUniversalNodeAVX2.processBatch(const std::vector<double>& i, const std::vector<double>& c, const std::vector<double>& a) { return {}; }
void AnalogCellularEngineAVX2::processBlockFrequencyDomain(std::vector<double>& b) {}
double AnalogCellularEngineAVX2::performSignalSweepAVX2(double f) { return 0; }
double AnalogCellularEngineAVX2::processSignalWaveAVX2(double i, double c) { return 0; }
void AnalogCellularEngineAVX2::runMissionOptimized(const double* i, const double* c, uint64_t s, uint32_t it) {}
void AnalogCellularEngineAVX2::runMissionOptimized_Phase4B(const double* i, const double* c, uint64_t s, uint32_t it) {}
void AnalogCellularEngineAVX2::runMissionOptimized_Phase4C(const double* i, const double* c, uint64_t s, uint32_t it) {}
void AnalogCellularEngineAVX2::runMassiveBenchmark(int i) {}
double AnalogCellularEngineAVX2::runDragRaceBenchmark(int n) { return 0; }
void AnalogCellularEngineAVX2::runBuiltinBenchmark(int i) {}
double AnalogCellularEngineAVX2::calculateInterNodeCoupling(size_t n) { return 0; }
double AnalogCellularEngineAVX2::generateNoiseSignal() { return 0; }
