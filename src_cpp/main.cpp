// ============================================================================
//  main.cpp  --  test of spin::Qudit
//  Build:  g++ -std=c++17 -Wall -Wextra -O2 -isystem /usr/include/eigen3 main.cpp -o qudit_demo.out
//  Run:    ./qudit_demo.out
// ============================================================================

#include "qudit.hpp"

#include <cmath>
#include <cstdio>     // std::remove
#include <iomanip>    // std::setprecision, std::setw
#include <iostream>

int main() {
    const int N = 121;
    const int steps = 12;
    const double J = 0.5 * (N - 1);
    const double pi = 3.14159265358979323846;
    std::cout << "N = " << N << ", J = " << J << "\n" << std::fixed << std::setprecision(4);

    // 1) Named states: <S> = J along the named axis, variance J/2 in the two other directions.
    for (const std::string label : {"CSS+Z", "CSS-X", "CSS+Y"}) spin::Qudit(N, label).printSpin(label);

    // 2) Generic CSS: expected <S> = J (sin t cos p, sin t sin p, cos t).
    const double theta = 0.7, phi = 1.1;
    spin::Qudit q(N);
    q.prepareCSS(theta, phi).printSpin("CSS(0.7,1.1)");
    std::cout << "    expected:  <Sx> = " << J * std::sin(theta) * std::cos(phi) << "   <Sy> = "
              << J * std::sin(theta) * std::sin(phi) << "   <Sz> = " << J * std::cos(theta) << "\n";

    // 3) Dicke states with k = 3, i.e. m = J - 3 along each axis: zero variance along that axis.
    q.prepareDickeZ(3).printSpin("DickeZ k=3");
    q.prepareDickeX(3).printSpin("DickeX k=3");
    q.prepareDickeY(3).printSpin("DickeY k=3");

    // 4) Pulses on CSS+Z (or CSS+X): expected final directions in brackets.
    std::cout << "\nPulses (expected direction of <S>):\n";
    q.prepareNamed("CSS+Z").applyPulseXPiHalf();  std::cout << "  pi/2 X on +Z -> <Sy> = " << q.avgSy() << "  [-J]\n";
    q.prepareNamed("CSS+Z").applyPulseYPiHalf();  std::cout << "  pi/2 Y on +Z -> <Sx> = " << q.avgSx() << "  [+J]\n";
    q.prepareNamed("CSS+Z").applyPulseXPi();      std::cout << "  pi   X on +Z -> <Sz> = " << q.avgSz() << "  [-J]\n";
    q.prepareNamed("CSS+X").applyPulseYPi();      std::cout << "  pi   Y on +X -> <Sx> = " << q.avgSx() << "  [-J]\n";
    q.prepareNamed("CSS+X").applyPulseZ(pi / 2);  std::cout << "  pi/2 Z on +X -> <Sy> = " << q.avgSy() << "  [+J]\n";

    // 5) Diagonal evolution: one-axis twisting H = chi Sz^2 from CSS+X.
    //    Exact result: <Sx>(t) = J cos(chi t)^(2J-1).
    std::cout << "\nOne-axis twisting H = chi Sz^2 (diagonal, evolveDiagonal):\n";
    const double chi = 0.5, dt = 0.02;
    spin::RealVector E(N);
    for (int k = 0; k < N; k++) E(k) = chi * q.spinM(k) * q.spinM(k);
    q.prepareNamed("CSS+X");
    for (int step = 1; step <= steps; step++) {
        q.evolveDiagonal(E, dt);
        const double t = step * dt;
        std::cout << "  t = " << t << " :  <Sx> = " << q.avgSx() << "   exact = " << J * std::pow(std::cos(chi * t), 2 * J - 1)
                  << "   Var(Sy) = " << q.varSy() << "\n";
    }

    // 6) NON-diagonal evolution: H = Omega Sx (Rabi rotation about x) from CSS+Z.
    //    H is built as an N x N matrix and diagonalised ONCE; evolve() then works for any t.
    //    Exact result: <Sz>(t) = J cos(Omega t),   Var(Sz)(t) = (J/2) sin^2(Omega t).
    std::cout << "\nRabi rotation H = Omega Sx (non-diagonal, evolve):\n";
    const double Omega = 1.0;
    spin::Matrix Sx = spin::Matrix::Zero(N, N);
    for (int k = 0; k + 1 < N; k++) {
        const double m = J - k;
        Sx(k, k + 1) = Sx(k + 1, k) = 0.5 * std::sqrt(J * (J + 1) - m * (m - 1));   // <J,m|S_x|J,m-1>
    }
    const Eigen::SelfAdjointEigenSolver<spin::Matrix> solver(Omega * Sx);

    std::cout << "      t      <Sz>     exact    Var(Sz)    exact\n";
    for (int step = 0; step <= steps; step++) {
        const double t = step * pi / 8;                              // from 0 to a pi rotation
        q.prepareNamed("CSS+Z").evolve(solver.eigenvalues(), solver.eigenvectors(), t);
        const double s = std::sin(Omega * t);
        std::cout << std::setw(7) << t << std::setw(10) << q.avgSz() << std::setw(10) << J * std::cos(Omega * t)
                  << std::setw(10) << q.varSz() << std::setw(10) << 0.5 * J * s * s << "\n";
    }

    // 7) Save / load round trip.
    q.prepareCSS(theta, phi).save("q_test.txt");
    spin::Qudit loaded(N);
    loaded.load("q_test.txt");
    std::cout << std::scientific << "\nsave/load: max |difference| = "
              << (loaded.state() - q.state()).cwiseAbs().maxCoeff() << "\n";

    // 8) Errors are reported with exceptions.
    try { spin::Qudit(N + 3).load("q_test.txt"); } catch (const std::exception& e) { std::cout << "expected error: " << e.what() << "\n"; }
    try { spin::Qudit(N, "CSS+W");                } catch (const std::exception& e) { std::cout << "expected error: " << e.what() << "\n"; }
    try { q.prepareDickeZ(N);                     } catch (const std::exception& e) { std::cout << "expected error: " << e.what() << "\n"; }

    std::remove("q_test.txt");
    return 0;
}
