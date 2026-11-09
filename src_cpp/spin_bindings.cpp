// ============================================================================
//  spin_bindings.cpp  --  Python interface to spin::Qudit (via pybind11)
// ----------------------------------------------------------------------------
//  Build the Python module "spinlib" (one line, from this folder):
//
//    c++ -O2 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) -isystem /usr/include/eigen3 spin_bindings.cpp -o spinlib$(python3-config --extension-suffix)
//
//  Then, in Python (from the same folder):
//
//    import spinlib
//    q = spinlib.Qudit(41, "CSS+X")
//    print(q.avgSx(), q.varSy())
//
//  Eigen <-> NumPy: every StateVector / RealVector / Matrix is converted
//  automatically to / from a NumPy array (pybind11/eigen.h).
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>     // Eigen vectors/matrices <-> NumPy arrays
#include <pybind11/complex.h>   // std::complex<double> <-> Python complex
#include <pybind11/iostream.h>  // redirect std::cout to Python's sys.stdout

#include "qudit.hpp"

namespace py = pybind11;
using spin::Qudit;

PYBIND11_MODULE(spinlib, m) {
    m.doc() = "Single spin J (qudit) of dimension N = 2J+1";

    // Methods returning Qudit& (preparations, modifiers) must return a REFERENCE to the
    // same Python object, otherwise Python would receive a copy and chaining such as
    // q.prepareNamed("CSS+Z").applyPulseXPiHalf() would act on that copy.
    const auto self = py::return_value_policy::reference_internal;

    py::class_<Qudit>(m, "Qudit")
        // a.) Constructor
        .def(py::init<int, const std::string&>(), py::arg("N"), py::arg("init") = "CSS+Z")

        // b.) State preparations
        .def("prepareNamed",         &Qudit::prepareNamed,         py::arg("label"),       self)
        .def("prepareDickeZ",        &Qudit::prepareDickeZ,        py::arg("k"),           self)
        .def("prepareDickeX",        &Qudit::prepareDickeX,        py::arg("k"),           self)
        .def("prepareDickeY",        &Qudit::prepareDickeY,        py::arg("k"),           self)
        .def("prepareReference",     &Qudit::prepareReference,                             self)
        .def("prepareCSS",           &Qudit::prepareCSS,           py::arg("theta"), py::arg("phi"), self)
        .def("prepareEquatorialCSS", &Qudit::prepareEquatorialCSS, py::arg("q"),           self)

        // c.) Getters
        .def("dim",        &Qudit::dim)
        .def("spinJ",      &Qudit::spinJ)
        .def("spinM",      &Qudit::spinM, py::arg("k"))
        .def("state",      &Qudit::state)     // returns a NumPy array (copy of the state)
        .def("norm",       &Qudit::norm)
        .def("amplitude",  &Qudit::amplitude, py::arg("k"))
        .def("overlapCSS", &Qudit::overlapCSS, py::arg("theta"), py::arg("phi"))
        .def("avgSpin",    &Qudit::avgSpin)   // NumPy array: <S_x>, <S_y>, <S_z>
        .def("covSpin",    &Qudit::covSpin)   // NumPy 3 x 3 array: covariances
        .def("husimiQ",    &Qudit::husimiQ, py::arg("nTheta") = 61, py::arg("nPhi") = 121)

        // d.) Modifiers
        .def("normalize",         &Qudit::normalize,         self)
        .def("setState",          &Qudit::setState,          py::arg("amplitudes"), py::arg("normalizeIt") = true, self)
        .def("applyRotation",     &Qudit::applyRotation,     py::arg("U"), self)        
        .def("applyPulseZ",       &Qudit::applyPulseZ,       py::arg("phi"), self)
        .def("applyPulseX",       &Qudit::applyPulseX,       py::arg("phi"), self)
        .def("applyPulseY",       &Qudit::applyPulseY,       py::arg("phi"), self)
        .def("applyPulseXPi",     &Qudit::applyPulseXPi,     self)
        .def("applyPulseYPi",     &Qudit::applyPulseYPi,     self)
        .def("applyPulseXPiHalf", &Qudit::applyPulseXPiHalf, self)
        .def("applyPulseYPiHalf", &Qudit::applyPulseYPiHalf, self)
        .def("evolveDiagonal",    &Qudit::evolveDiagonal,    py::arg("energies"), py::arg("t"), self)
        .def("evolve",            &Qudit::evolve,            py::arg("E"), py::arg("P"), py::arg("t"), self)

        // e.) Input / output
        // printSpin writes to C++'s std::cout, which Jupyter does not show: redirect it to Python.
        .def("printSpin", &Qudit::printSpin, py::arg("name") = "qudit",
             py::call_guard<py::scoped_ostream_redirect>())
        .def("save", &Qudit::save, py::arg("fileName"))
        .def("load", &Qudit::load, py::arg("fileName"), self);
}
