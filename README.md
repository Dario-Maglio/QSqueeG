# QSqueeG

Welcome to the **Quantum Squeezing for Gravity** project!

[![Open notebook_cpy_playground in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dario-Maglio/QSqueeG/blob/main/examples/notebook_cpy_playground.ipynb)
&nbsp;`notebook_cpy_playground`: the `spin::Qudit` library from Python

[![Open notebook_cpy_QSqueezeG in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dario-Maglio/QSqueeG/blob/main/notebook_cpy_QSqueezeG.ipynb)
&nbsp;`notebook_cpy_QSqueezeG`: atom interferometer with internal clocks in gravity

## Goal

The project studies the interplay between the mass-energy equivalence and the superposition principle with
atomic ensembles and collective spin techniques (pulses, interferometry, squeezing). Its central piece is the
notebook **`notebook_cpy_QSqueezeG.ipynb`**, which simulates exactly a Mach–Zehnder interferometer for $N$
bosonic two-level atoms whose internal states, ground $g$ and excited $e$, form an optical clock (⁸⁷Sr):

1. all the atoms enter the same port, in $g$;
2. a first beam splitter puts them in a superposition of two arms, held at heights $z_d$ and $z_u$;
3. within each arm, the clocks undergo a Ramsey sequence: $\pi/2$ pulse, free evolution, $\pi/2$ pulse;
4. a second beam splitter recombines the arms, and the atoms are counted.

The same simulation reproduces the predictions of two reference works, translated to the notation of the
notebook and checked against their equations:

* Zych *et al.*, Nat. Commun. **2**, 505 (2011): loss of interferometric visibility due to proper time
  (Fig. 2, Eq. 12);
* Paczos, Foo & Zych, Quantum (2025), arXiv:2406.19037: Ramsey spectroscopy of clocks trapped at two heights
  (Fig. 3, Eqs. 1 and 5), with the comparison between a superposition and a mixture of heights.

The notebook is written in Python on top of the C++ spin library `spin::Qudit` (described below), which it
imports as the module `spinlib`; the library and its tests are presented in `examples/notebook_cpy_playground.ipynb`.

> **Work in progress.** A parallel implementation in Julia (`src_jul/SpinLibrary.jl` and the `notebook_jul_*`
> notebooks) is also in progress, and does not yet cover the interferometer of `notebook_cpy_QSqueezeG.ipynb`.

The interferometer acts on two degrees of freedom at once, and each one shows its own fringe:

| | observable | fringe | requires |
|---|---|---|---|
| internal (clock) | $P(N_g)$, atoms in $g$ | Ramsey fringe, at the detuning $\bar\omega$ of the clocks from the laser | superposition of $g$ and $e$ |
| external (path) | $P(N^d)$, atoms leaving the down port | gravitational fringe, at $\Delta\Omega = m g\,\Delta z/\hbar$ | superposition of the two heights |

Both fringes are modulated by the same envelope, the **visibility**
$\mathcal V(T) = \cos(\Delta\omega T/2)$, where $\Delta\omega = \omega_0 g\,\Delta z/c^2$ is the gravitational
redshift between the arms: the internal energy gravitates (mass–energy equivalence), so the clocks at the
two heights tick at different rates. The notebook shows that

* the envelope alone does not certify a superposition: in $P(N_g)$ it also appears for atoms placed at random
  at either height, in $P(N^d)$ for clocks in $g$ or $e$ at random;
* each observable must therefore be complemented by the other degree of freedom; the port-resolved ground
  population $D_g$ combines the two, vanishing without path coherence and losing its Ramsey modulation without
  internal coherence;
* with the physical values (⁸⁷Sr, $\Delta z = 10\ \mu$m) the three time scales are very far apart: the
  gravitational fringe has a period of $47\ \mu$s, the Ramsey fringe of $4$ s, and the visibility vanishes
  only after about $12$ days.



## Repository layout

```
QSqueeG/
├── notebook_cpy_QSqueezeG.ipynb        # main notebook: atom interferometer with internal clocks in gravity
├── notebook_jul_QSqueezeG.ipynb        # working notebook (Julia): analytical calculations for 2 atoms, squeezing
├── src_cpp/
│   ├── qudit.hpp                       # header-only class spin::Qudit
│   └── spin_bindings.cpp               # pybind11 bindings -> Python module "spinlib"
├── src_jul/
│   └── SpinLibrary.jl                  # states, spin operators, tensor products, diagonalisation and evolution
├── test/
│   ├── test_qudit.cpp                  # checks of spin::Qudit against exact results
│   └── test_eigen.cpp                  # minimal check that Eigen is installed and works
├── examples/
│   ├── notebook_cpy_playground.ipynb   # spin::Qudit from Python: build, tests, Bloch sphere, dynamics
│   ├── notebook_jul_playground.ipynb   # SpinLibrary.jl: tests, Bloch sphere, dynamics
│   └── notebook_jul_spinDynamics.ipynb # one-axis twisting in Julia
└── Project.toml
```


## The C++ library `spin::Qudit`

The class `spin::Qudit` stores the pure state of one spin $J$ as $Z = 2J+1$ complex amplitudes, in the eigenbasis 
of $J_z$ ordered from the highest to the lowest projection: index $k = 0, \dots, Z-1$ is $|J, m = J - k\rangle$.
It is header-only (`#include "qudit.hpp"`); the only dependency is [Eigen 3](https://eigen.tuxfamily.org).
Times are in units with $\hbar = 1$, and a pulse of angle $\alpha$ about the axis $a$ is $e^{-i\alpha J_a}$.

| Group | Methods |
|---|---|
| Construction | `Qudit(Z, "CSS+Z")`: the label is any of `CSS±X`, `CSS±Y`, `CSS±Z` |
| State preparation | `prepareNamed(label)`, `prepareCSS(theta, phi)`, `prepareEquatorialCSS(q)`, `prepareDickeZ(k)`, `prepareDickeX(k)`, `prepareDickeY(k)`, `prepareReference()` |
| Pulses | `applyPulseX(phi)`, `applyPulseY(phi)`, `applyPulseZ(phi)`, exact `applyPulseXPi()`, `applyPulseYPi()`, `applyPulseXPiHalf()`, `applyPulseYPiHalf()`, `applyRotation(U)` |
| Evolution | `evolveDiagonal(E, t)` for $H$ diagonal in $J_z$; `evolve(E, P, t)` for $H = P\,\mathrm{diag}(E)\,P^\dagger$; `squeezeOAT(chi)` $= e^{-i\chi J_z^2}$; `squeezeTACT(chi)` $= e^{-i\chi\{J_x,J_y\}/2}$ |
| Spin moments | `avgSpin()`: mean spin $(\langle J_x\rangle, \langle J_y\rangle, \langle J_z\rangle)$; `covSpin()`: $3\times3$ covariance matrix, variances on the diagonal |
| State access | `dim()`, `spinJ()`, `spinM(k)`, `state()`, `amplitude(k)`, `norm()`, `normalize()`, `setState(psi)`, `Dy()` |
| Bloch sphere | `overlapCSS(theta, phi)`, `husimiQ(nTheta, nPhi)`: Husimi function $Q = \lvert\langle\theta,\varphi\vert\psi\rangle\rvert^2$ on a grid |
| Static | `wignerDy(Z)`: the matrix $e^{-i\pi J_y/2}$ for any dimension; `cssModuli(Z, theta)`: moduli of a coherent state, whose squares are the binomial law |
| Output | `printSpin(name)`, `save(fileName)`, `load(fileName)` |

Preparations, pulses and evolutions return a reference to the qudit, so they can be chained:

```cpp
#include "qudit.hpp"

int main() {
    spin::Qudit q(41, "CSS+Z");                    // Z = 41, i.e. J = 20
    q.applyPulseYPiHalf().applyPulseZ(0.3);        // to the equator, then precession by 0.3 rad
    q.printSpin("after the pulses");               // norm, <J> and variances
}
```

Numerical notes: $\pi$ and $\pi/2$ pulses are exact (vector reversal, and the Wigner matrix $d^J(\pi/2)$ from a
stable three-term recurrence, accurate to $\sim 10^{-15}$ and valid up to $Z = 2045$); the covariance matrix is computed
from $(J_a - \langle J_a\rangle)\psi$, which avoids cancellations; binomial coefficients go through
`lgamma`, so coherent states do not overflow for large $J$.

### Build and test

Install Eigen (`sudo apt install libeigen3-dev` on Debian/Ubuntu, `brew install eigen` on macOS), then:

```bash
cd test
g++ -std=c++17 -Wall -Wextra -O2 -isystem /usr/include/eigen3 test_qudit.cpp -o qudit_test.out
./qudit_test.out
```

### From Python: the `spinlib` module

`spin_bindings.cpp` exposes `spin::Qudit` to Python with [pybind11](https://pybind11.readthedocs.io); Eigen
vectors and matrices become NumPy arrays. The build command is the one in the Colab cell above, run inside
`src_cpp/`. Then:

```python
import spinlib
q = spinlib.Qudit(41, "CSS+X")
print(q.avgSpin()[0], q.covSpin()[1, 1])     # <J_x> = 20.0, Var(J_y) = 10.0
Q = q.husimiQ()                              # 61 x 121 NumPy array
```

## The Julia library `SpinLibrary.jl`

A collection of plain functions, loaded with `include("src_jul/SpinLibrary.jl")`; it only needs the standard
library `LinearAlgebra`. Indices run over $k = 1, \dots, 2S+1$, with $m = S + 1 - k$.

| Group | Functions |
|---|---|
| States | `init_cssX(S)`, `init_cssXm(S)`, `init_css(S, theta, phi)`, `init_Dickez(S, k)`, `init_Dickex(S, k, baseX(S))`, `norme(psi)` |
| Spin operators | `Sx(S)`, `Sy(S)`, `Sz(S)`, `Splus(S)`, `Sminus(S)`, `identite(n)` |
| Several spins | `prod_tens_vec`, `prod_tens_op` (and the complex versions `...C`), `ham_part_lin_diff`, `ham_part_quad_diff` |
| Evolution | `vp, P, Pm = diagonalisation(H)`, then `evolution(vp, P, Pm, psi, t)`; `evol_diag(E, psi, t)` for a diagonal $H$ |

## Authors

Dario Cafasso, Youssef Trifa
