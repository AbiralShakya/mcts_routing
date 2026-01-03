/**
 * C++ bindings for nextpnr router integration.
 *
 * This file provides pybind11 bindings to call nextpnr's router2
 * directly from Python, avoiding subprocess overhead.
 *
 * Build requirements:
 * - nextpnr source (linked as library)
 * - pybind11
 * - Boost (for nextpnr)
 *
 * Build command (example):
 * c++ -O3 -Wall -shared -std=c++17 -fPIC \
 *     $(python3 -m pybind11 --includes) \
 *     -I/path/to/nextpnr/common \
 *     -I/path/to/nextpnr/xilinx \
 *     bindings.cpp \
 *     -o _nextpnr_bindings$(python3-config --extension-suffix) \
 *     -L/path/to/nextpnr/build -lnextpnr-xilinx
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// TODO: Include nextpnr headers when building
// #include "nextpnr.h"
// #include "router2.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>

namespace py = pybind11;

/**
 * Routing result structure matching Python RoutingResult.
 */
struct CppRoutingResult {
    bool success;
    double score;
    int wirelength;
    double congestion;
    bool timing_met;
    double slack;
    double runtime_ms;
    std::string error_message;
};

/**
 * Cell placement data.
 */
struct CellPlacement {
    std::string name;
    int x;
    int y;
    std::string bel_name;
};

/**
 * Net data for routing.
 */
struct NetData {
    int net_id;
    std::string name;
    std::vector<std::pair<int, int>> pin_positions;  // (x, y) pairs
};

/**
 * Route a placement using nextpnr's router2.
 *
 * This is a stub implementation. The actual implementation requires:
 * 1. Building nextpnr as a library
 * 2. Creating a Context with the chip database
 * 3. Loading the placement into the context
 * 4. Running router2
 * 5. Extracting metrics
 */
CppRoutingResult route_placement(
    const std::vector<CellPlacement>& cells,
    const std::vector<NetData>& nets,
    int grid_width,
    int grid_height,
    const std::string& chipdb_path = ""
) {
    auto start = std::chrono::high_resolution_clock::now();

    CppRoutingResult result;
    result.success = false;
    result.score = 0.0;
    result.wirelength = 0;
    result.congestion = 0.0;
    result.timing_met = false;
    result.slack = 0.0;

    /*
     * TODO: Actual nextpnr integration
     *
     * // Create context
     * Context ctx(chip);
     *
     * // Load placement
     * for (const auto& cell : cells) {
     *     BelId bel = ctx.getBelByName(cell.bel_name);
     *     CellInfo* ci = ctx.getCell(cell.name);
     *     ctx.bindBel(bel, ci, STRENGTH_USER);
     * }
     *
     * // Run router
     * Router2 router(&ctx);
     * bool route_ok = router.route();
     *
     * // Extract metrics
     * result.success = route_ok;
     * result.wirelength = ctx.getTotalWirelength();
     * result.congestion = router.getMaxCongestion();
     * ...
     */

    // Stub: Return failure with message
    result.error_message = "C++ bindings not yet implemented - use subprocess mode";

    auto end = std::chrono::high_resolution_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

/**
 * Estimate wirelength without routing (fast proxy).
 *
 * Uses HPWL (half-perimeter wirelength) as a quick estimate.
 */
int estimate_wirelength(
    const std::vector<CellPlacement>& cells,
    const std::vector<NetData>& nets
) {
    int total_wl = 0;

    // Build cell position lookup
    std::unordered_map<std::string, std::pair<int, int>> cell_pos;
    for (const auto& cell : cells) {
        cell_pos[cell.name] = {cell.x, cell.y};
    }

    // Compute HPWL for each net
    for (const auto& net : nets) {
        if (net.pin_positions.size() < 2) continue;

        int min_x = INT_MAX, max_x = INT_MIN;
        int min_y = INT_MAX, max_y = INT_MIN;

        for (const auto& [x, y] : net.pin_positions) {
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }

        total_wl += (max_x - min_x) + (max_y - min_y);
    }

    return total_wl;
}

/**
 * Check for placement overlaps (DRC).
 */
bool check_placement_valid(
    const std::vector<CellPlacement>& cells,
    int grid_width,
    int grid_height
) {
    std::unordered_map<std::string, bool> occupied;

    for (const auto& cell : cells) {
        // Check bounds
        if (cell.x < 0 || cell.x >= grid_width ||
            cell.y < 0 || cell.y >= grid_height) {
            return false;
        }

        // Check overlap
        std::string key = std::to_string(cell.x) + "_" + std::to_string(cell.y);
        if (occupied.count(key)) {
            return false;  // Overlap
        }
        occupied[key] = true;
    }

    return true;
}

/**
 * pybind11 module definition.
 */
PYBIND11_MODULE(_nextpnr_bindings, m) {
    m.doc() = "C++ bindings for nextpnr router integration";

    // Result structure
    py::class_<CppRoutingResult>(m, "RoutingResult")
        .def(py::init<>())
        .def_readwrite("success", &CppRoutingResult::success)
        .def_readwrite("score", &CppRoutingResult::score)
        .def_readwrite("wirelength", &CppRoutingResult::wirelength)
        .def_readwrite("congestion", &CppRoutingResult::congestion)
        .def_readwrite("timing_met", &CppRoutingResult::timing_met)
        .def_readwrite("slack", &CppRoutingResult::slack)
        .def_readwrite("runtime_ms", &CppRoutingResult::runtime_ms)
        .def_readwrite("error_message", &CppRoutingResult::error_message);

    // Cell placement
    py::class_<CellPlacement>(m, "CellPlacement")
        .def(py::init<>())
        .def_readwrite("name", &CellPlacement::name)
        .def_readwrite("x", &CellPlacement::x)
        .def_readwrite("y", &CellPlacement::y)
        .def_readwrite("bel_name", &CellPlacement::bel_name);

    // Net data
    py::class_<NetData>(m, "NetData")
        .def(py::init<>())
        .def_readwrite("net_id", &NetData::net_id)
        .def_readwrite("name", &NetData::name)
        .def_readwrite("pin_positions", &NetData::pin_positions);

    // Functions
    m.def("route_placement", &route_placement,
          "Route a placement using nextpnr router2",
          py::arg("cells"),
          py::arg("nets"),
          py::arg("grid_width"),
          py::arg("grid_height"),
          py::arg("chipdb_path") = "");

    m.def("estimate_wirelength", &estimate_wirelength,
          "Estimate wirelength using HPWL",
          py::arg("cells"),
          py::arg("nets"));

    m.def("check_placement_valid", &check_placement_valid,
          "Check if placement is valid (no overlaps, in bounds)",
          py::arg("cells"),
          py::arg("grid_width"),
          py::arg("grid_height"));
}
