# QSqueeG

Welcome to the **Quantum Squeezing for Gravity** project!

This repository collects numerical tools to simulate a single collective spin $J$ (a *qudit* of dimension
$N = 2J+1$): state preparation, pulses, time evolution under spin Hamiltonians, spin moments and squeezing,
and visualisation on the Bloch sphere. The same physics is available in two independent implementations:

* **C++** (`src_cpp/`): the header-only class `spin::Qudit`, built on Eigen, usable from C++ or from Python
  through the `spinlib` module;
* **Julia** (`src_jul/`): the function library `SpinLibrary.jl`, working with plain vectors and matrices.

## Repository layout

```
QSqueeG/
├── src_cpp/
│   ├── qudit.hpp            # class spin::Qudit (double precision)
│   ├── quditType.hpp        # same class with the alias spin::Real = long double (extended precision)
│   └── spin_bindings.cpp    # pybind11 bindings -> Python module "spinlib"
├── src_jul/
│   └── SpinLibrary.jl       # states, spin operators, tensor products, diagonalisation and evolution
├── test/
│   ├── test_qudit.cpp       # checks of spin::Qudit against exact results
│   └── test_eigen.cpp       # minimal check that Eigen is installed and works
├── examples/
│   ├── notebook_cpy_playground.ipynb   # spin::Qudit from Python: tests, Bloch sphere, dynamics
│   ├── notebook_jul_playground.ipynb   # SpinLibrary.jl: tests, Bloch sphere, dynamics
│   └── notebook_jul_spinDynamics.ipynb # first example: one-axis twisting in Julia
├── notebook_jul_QSqueezeG.ipynb        # working notebook: analytical calculations for 2 atoms, squeezing
├── docs/
└── Project.toml
```

## Conventions

Both libraries work in the eigenbasis of $S_z$, ordered from the **highest** to the **lowest** projection $m$,
so the first basis vector is the coherent state along $+z$, $|J, J\rangle = (1, 0, \dots, 0)$:

| | index | projection |
|---|---|---|
| C++ / Python (`spin::Qudit`, `spinlib`) | $k = 0, \dots, N-1$ | $m = J - k$ |
| Julia (`SpinLibrary.jl`) | $k = 1, \dots, 2S+1$ | $m = S + 1 - k$ |

Times are in units with $\hbar = 1$: a state evolves as $|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle$, and a
pulse of angle $\alpha$ about the axis $a$ is $e^{-i\alpha S_a}$.

---

## C++ library: `spin::Qudit`

`spin::Qudit` stores the pure state of one spin as $N$ complex amplitudes. It is header-only: just
`#include "qudit.hpp"`. The only dependency is [Eigen 3](https://eigen.tuxfamily.org) (also header-only).

| Group | Methods |
|---|---|
| Construction | `Qudit(N, "CSS+Z")`: the label is any of `CSS±X`, `CSS±Y`, `CSS±Z` |
| State preparation | `prepareNamed(label)`, `prepareCSS(theta, phi)`, `prepareEquatorialCSS(q)`, `prepareDickeZ(k)`, `prepareDickeX(k)`, `prepareDickeY(k)` |
| Pulses | `applyPulseZ(phi)`, `applyPulseXPi()`, `applyPulseYPi()`, `applyPulseXPiHalf()`, `applyPulseYPiHalf()`, `applyRotation(U)` |
| Evolution | `evolveDiagonal(E, t)` for a Hamiltonian diagonal in the $S_z$ basis; `evolve(E, P, t)` for $H = P\,\mathrm{diag}(E)\,P^\dagger$ |
| Spin moments | `avgSx/Sy/Sz()`, `avgSx2/Sy2/Sz2()`, `avgSplus()`, `avgSminus()`, `avgSplus2()`, `varSx/Sy/Sz()` |
| State access | `dim()`, `spinJ()`, `spinM(k)`, `state()`, `amplitude(k)`, `norm()`, `normalize()`, `setState(psi)` |
| Bloch sphere | `overlapCSS(theta, phi)`, `husimiQ(nTheta, nPhi)`: Husimi function $Q = \lvert\langle\theta,\varphi\vert\psi\rangle\rvert^2$ on a grid |
| Output | `printSpin(name)`, `save(fileName)`, `load(fileName)` |

State preparations, pulses and evolutions return a reference to the qudit, so they can be chained:

```cpp
#include "qudit.hpp"

int main() {
    spin::Qudit q(41, "CSS+Z");                    // N = 41, i.e. J = 20
    q.applyPulseYPiHalf().applyPulseZ(0.3);        // to the equator, then precession by 0.3 rad
    q.printSpin("after the pulses");               // norm, <S> and variances
}
```

Numerical notes:
* π and π/2 pulses are computed exactly (vector reversal, Wigner matrix $d^J(\pi/2)$ from a stable
  three-term recurrence), not by diagonalisation;
* variances are computed as $\|(S_a - \langle S_a\rangle)\psi\|^2$, which avoids cancellations and is never negative;
* `quditType.hpp` is the same class with every real number of type `spin::Real = long double`
  (about 19 significant digits on x86 Linux instead of 16, at the price of slower products).

### Build and test

Install Eigen (e.g. `sudo apt install libeigen3-dev` on Debian/Ubuntu, `brew install eigen` on macOS), then:

```bash
cd test
g++ -std=c++17 -Wall -Wextra -O2 -isystem /usr/include/eigen3 test_qudit.cpp -o qudit_test.out
./qudit_test.out
```

`-isystem` (instead of `-I`) hides the harmless warnings coming from Eigen's own headers.
The test prints each quantity next to its exact value: coherent and Dicke states, pulse sequences,
one-axis twisting, non-diagonal evolution and save/load.

### From Python: the `spinlib` module

`spin_bindings.cpp` exposes `spin::Qudit` to Python with [pybind11](https://pybind11.readthedocs.io);
Eigen vectors and matrices become NumPy arrays. Build the module once (needs `pip install pybind11` and the
Python development headers, e.g. `python3-dev`):

```bash
cd src_cpp
c++ -O2 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) -isystem /usr/include/eigen3 \
    spin_bindings.cpp -o spinlib$(python3-config --extension-suffix)
```

and use it from the same folder (or after adding `src_cpp` to `sys.path`):

```python
import spinlib
q = spinlib.Qudit(41, "CSS+X")
print(q.avgSx(), q.varSy())                  # 20.0, 10.0
Q = q.husimiQ()                              # 61 x 121 NumPy array
```

The compiled file is specific to one Python version and platform: rebuild it on each machine.
The notebook `examples/notebook_cpy_playground.ipynb` does the build for you (section 1).

---

## Julia library: `SpinLibrary.jl`

A collection of plain functions, loaded with `include("src_jul/SpinLibrary.jl")`; it only needs the standard
library `LinearAlgebra`.

| Group | Functions |
|---|---|
| States | `init_cssX(S)`, `init_cssXm(S)`, `init_css(S, theta, phi)`, `init_Dickez(S, k)`, `init_Dickex(S, k, baseX(S))`, `norme(psi)` |
| Spin operators | `Sx(S)`, `Sy(S)`, `Sz(S)`, `Splus(S)`, `Sminus(S)`, `identite(n)` |
| Several spins | `prod_tens_vec`, `prod_tens_op` (and the complex versions `...C`), `ham_part_lin_diff`, `ham_part_quad_diff` |
| Evolution | `vp, P, Pm = diagonalisation(H)`, then `evolution(vp, P, Pm, psi, t)`; `evol_diag(E, psi, t)` for a diagonal $H$ |

```julia
using LinearAlgebra
include("src_jul/SpinLibrary.jl")

S = 10
vp, P, Pm = diagonalisation(Sz(S) * Sz(S))           # one-axis twisting H = Sz^2
psi = evolution(vp, P, Pm, init_cssX(S), 0.3)        # state at t = 0.3
println(real(dot(psi, Sx(S) * psi)))                 # <Sx> = S cos(t)^(2S-1)
```

---

## Example notebooks

The two playground notebooks have the same structure, so the C++ and Julia implementations can be
compared side by side:

1. **Setup**: packages and library (for Python, also the build of `spinlib`);
2. **Tests**: coherent and Dicke states, pulse sequences, diagonal and non-diagonal evolution, save/load,
   each compared with its exact value;
3. **Bloch sphere**: the Husimi function $Q(\theta,\varphi)$ drawn on the sphere, with the mean spin
   (interactive Plotly figures in Python, CairoMakie in Julia);
4. **Dynamics**: curves of the spin moments together with the state on the sphere for
   * Rabi rotation $H = \Omega S_x$,
   * one-axis twisting $H = S_z^2$, from the coherent state to squeezing and cat states,
   * twisting and turning $H = \chi S_z^2 + \Omega S_x$, with the Wineland squeezing parameter
     $\xi^2 = 2J\,\mathrm{Var}_{\min}/\langle S_x\rangle^2$,
   * Ramsey interferometry and its fringes.

The notebooks are meant to be run from the `examples/` folder: they find the libraries at `../src_cpp` and
`../src_jul`.

| Notebook | Requirements |
|---|---|
| `notebook_cpy_playground.ipynb` | Python 3 with NumPy, Matplotlib, Plotly, pybind11; Eigen; a C++17 compiler |
| `notebook_jul_playground.ipynb` | Julia ≥ 1.10 with IJulia; installs CairoMakie, LaTeXStrings and JLD2 if missing |

## Authors

Dario Cafasso, Youssef Trifa
