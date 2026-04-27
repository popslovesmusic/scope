
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "analog_universal_node_engine_avx2.h"

namespace py = pybind11;

PYBIND11_MODULE(engine_bridge, m) {
    m.doc() = "Bridge for AnalogUniversalNodeEngineAVX2";

    py::class_<EngineMetrics>(m, "EngineMetrics")
        .def_readwrite("total_execution_time_ns", &EngineMetrics::total_execution_time_ns)
        .def_readwrite("total_operations", &EngineMetrics::total_operations)
        .def_readwrite("current_ns_per_op", &EngineMetrics::current_ns_per_op)
        .def_readwrite("speedup_factor", &EngineMetrics::speedup_factor);

    py::class_<AnalogCellularEngineAVX2>(m, "AnalogCellularEngineAVX2")
        .def(py::init<size_t>())
        .def("runMission", &AnalogCellularEngineAVX2::runMission)
        .def("runMissionOptimized", [](AnalogCellularEngineAVX2& self, py::array_t<double> inputs, py::array_t<double> controls, uint32_t iterations) {
            auto r_inputs = inputs.unchecked<1>();
            auto r_controls = controls.unchecked<1>();
            if (r_inputs.size() != r_controls.size()) {
                throw std::runtime_error("Input and control arrays must have same size");
            }
            self.runMissionOptimized(r_inputs.data(0), r_controls.data(0), r_inputs.size(), iterations);
        })
        .def("getNodeOutputs", &AnalogCellularEngineAVX2::getNodeOutputs)
        .def("getMetrics", &AnalogCellularEngineAVX2::getMetrics)
        .def("resetMetrics", &AnalogCellularEngineAVX2::resetMetrics)
        .def("processSignalWaveAVX2", &AnalogCellularEngineAVX2::processSignalWaveAVX2);
}
