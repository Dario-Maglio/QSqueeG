// g++ -std=c++17 -I/usr/include/eigen3 test_eigen.cpp -o test.out

#include <Eigen/Dense>
#include <iostream>

int main() {
    // A 2x2 Hermitian matrix: the Pauli matrix sigma_y.
    Eigen::MatrixXcd H(2, 2);
    H << 0.0, std::complex<double>(0, -1),
         std::complex<double>(0, 1), 0.0;

    // Diagonalise it. SelfAdjointEigenSolver is for Hermitian matrices:
    // eigenvalues are real and sorted in increasing order.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H);
    std::cout << "eigenvalues:\n" << solver.eigenvalues() << "\n";    // -1, 1
    std::cout << "eigenvectors:\n" << solver.eigenvectors() << "\n";  // columns = P
}