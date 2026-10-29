// ============================================================================
//  main.cpp  --  test of spin::Qudit
//  Build:  g++ -std=c++17 -Wall -Wextra -O2 -isystem /usr/include/eigen3 main.cpp -o qudit_demo.out
//  Run:    ./qudit_demo.out
// ============================================================================

#include "qudit.hpp"

#include <cmath>
#include <cstdio>     // std::remove
#include <iomanip>    // std::setprecision
#include <iostream>

int main() {
    const int N = 1;
    const double J = 0.5 * (N - 1);
    std::cout << "N = " << N << ", J = " << J << "\n" << std::fixed << std::setprecision(3);

    // 1) Named states (the default constructor gives CSS+Z).
    for (const std::string label : {"CSS+Z", "CSS-Z", "CSS+X", "CSS-X", "CSS+Y", "CSS-Y"}) spin::Qudit(N, label).printSpin(label);

    // 2) Generic CSS and Dicke state: expected <S> = J (sin t cos p, sin t sin p, cos t).
    const double theta = 0.7, phi = 1.1;
    spin::Qudit q(N);
    q.prepareCSS(theta, phi).printSpin("CSS(0.7,1.1)");
    std::cout << "expected     :  <S0> = 1.000   <Sx> = " << J * std::sin(theta) * std::cos(phi) << "   <Sy> = "
              << J * std::sin(theta) * std::sin(phi) << "   <Sz> = " << J * std::cos(theta) << "\n";
    q.prepareDicke(N - 1).printSpin("Dicke k=N-1 ");

    // 3) One-axis twisting H = chi Sz^2 from CSS+X. Exact result: <Sx>(t) = J cos(chi t)^(2J-1).
    const double chi = 0.5, dt = 0.02;   // <Sx> decays on a time ~ 1/(chi sqrt(J))
    spin::RealVector E(N);
    for (int k = 0; k < N; k++) E(k) = chi * q.spinM(k) * q.spinM(k);
    q.prepareNamed("CSS+X");
    for (int step = 1; step <= 6; step++) {
        const double t = step * dt;
        q.evolveDiagonal(E, dt);
        std::cout << "OAT t = " << t << " :  <Sx> = " << q.avgSpin()[1]
                  << "   exact = " << J * std::pow(std::cos(chi * t), 2 * J - 1) << "\n";
    }

    // 4) Save / load round trip.
    q.prepareCSS(theta, phi).save("q_test.txt");
    spin::Qudit loaded(N);
    loaded.load("q_test.txt");
    std::cout << std::scientific << "save/load: max |difference| = "
              << (loaded.state() - q.state()).cwiseAbs().maxCoeff() << "\n";

    // 5) Errors are reported with exceptions.
    try { spin::Qudit(N + 3).load("q_test.txt"); } catch (const std::exception& e) { std::cout << "expected error: " << e.what() << "\n"; }
    try { spin::Qudit(N, "CSS+W");                } catch (const std::exception& e) { std::cout << "expected error: " << e.what() << "\n"; }

    std::remove("q_test.txt");
    return 0;
}
