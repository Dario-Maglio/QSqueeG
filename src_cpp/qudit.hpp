// ============================================================================
//  qudit.hpp   --   header-only definition of the class spin::Qudit
// ----------------------------------------------------------------------------
//  The class stores the (pure) quantum state of a single spin J, i.e. a
//  vector in a Hilbert space of dimension N = 2J + 1.
//
//  To use the class, simply write #include "qudit.hpp"
//
//  Requires Eigen 3 (header-only linear algebra library). Compile with e.g.
//      g++ -std=c++17 -Wall -Wextra -O2 -isystem /usr/include/eigen3 main.cpp
//  (-isystem instead of -I hides the harmless warnings coming from Eigen.)
//
//  ---------------------------------------------------------------------------
//  BASIS CONVENTION
//  ---------------------------------------------------------------------------
//  We work in the eigenbasis of S_z, ordered from the HIGHEST to the LOWEST
//  projection m:
//
//      index k :   0       1        2       ...    N-1
//      state   : |J,J>  |J,J-1>  |J,J-2>    ...  |J,-J>
//
//  i.e.  m = J - k   <=>   k = J - m.
//
//  Hence the default state |J,J> is the vector (1, 0, 0, ..., 0).
// ============================================================================

#pragma once  // prevents the header from being included twice in one unit






// ============================================================================
//  Includes and type aliases
// ============================================================================
#include <cmath>        // std::sqrt, std::cos, std::pow, std::lgamma, ...
#include <complex>      // std::complex<double>, std::polar, std::norm
#include <algorithm>    // std::min, std::max

#include <Eigen/Dense>  // Eigen::VectorXcd, Eigen::VectorXd, Eigen::Array, ...

#include <string>       // std::string, std::to_string
#include <fstream>      // std::ifstream, std::ofstream (file I/O)
#include <sstream>      // std::istringstream (parsing one line at a time)
#include <iostream>     // std::cout to print the expectation values of the spin
#include <stdexcept>    // std::invalid_argument, std::out_of_range, std::runtime_error




namespace spin {        // everything we write lives in the namespace "spin"

// Short alias: a complex number made of two doubles.
using Complex = std::complex<double>;

// Short alias: a spin vector = <J^mu> with mu = 0,x,y,z.
// Eigen::Vector4d has its size FIXED at compile time (no memory allocation).
//using SpinVector = Eigen::Vector4d;

// Short alias: a column of booleans whose size is chosen at run time.
// Array instead of Vector: it is a plain table, no linear algebra.
using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

// Short alias: a vector of N real numbers (e.g. the energies of H).
// Eigen::VectorXd = double vector whose size is chosen at run time.
using RealVector = Eigen::VectorXd;

// Short alias: a state vector = list of N complex amplitudes.
// Eigen::VectorXcd = complex<double> vector whose size is chosen at run time.
using StateVector = Eigen::VectorXcd;

// Short alias: a complex N x N matrix (e.g. a rotation or evolution operator).
using Matrix = Eigen::MatrixXcd;




// ============================================================================
//  Class Qudit
// ============================================================================
//  a.) Constructor
//        Qudit(N, init = "CSS+Z")         N = 2J+1; init = named state (see below)
//
//  b.) State preparations  (overwrite the state, return *this for chaining)
//        prepareNamed(label)              "CSS+Z", "CSS-Z", "CSS+X", "CSS-X", "CSS+Y", "CSS-Y"
//        prepareDickeZ(k)                 |J,m>   along z,  m = J - k
//        prepareDickeX(k)                 |J,m>_x along x,  m = J - k
//        prepareDickeY(k)                 |J,m>_y along y,  m = J - k
//        prepareCSS(theta, phi)           coherent spin state along (theta, phi)
//        prepareEquatorialCSS(q)          CSS along +x, +y, -x, -y for q = 0, 1, 2, 3 (exact phases)
//
//  c.) Getters  (read-only, const)
//        dim(), spinJ(), spinM(k)         N, J, and m = J - k
//        state(), amplitude(k), norm()    state vector, single amplitude, Euclidean norm
//        avgSx(), avgSy(), avgSz()        first moments <S_a>
//        avgSplus(), avgSminus()          <S_+>, <S_->
//        avgSx2(), avgSy2(), avgSz2()     second moments <S_a^2>
//        avgSplus2()                      <S_+^2>
//        varSx(), varSy(), varSz()        variances Var(S_a) = <S_a^2> - <S_a>^2
//
//  d.) Modifiers  (change the state, return *this for chaining)
//        normalize()                      rescale to unit norm
//        setState(v, normalizeIt = true)  replace the state with a given vector
//        applyPulseZ(phi)                 exp(-i phi S_z), exact for multiples of pi/2
//        applyPulseXPi(), applyPulseYPi() exact pi pulses (vector reversal)
//        applyPulseXPiHalf()              pi/2 pulse about x  (uses Dy_ and two z pulses)
//        applyPulseYPiHalf()              pi/2 pulse about y  (uses Dy_)
//        applyRotation(U)                 psi -> U psi, U unitary N x N
//        evolveDiagonal(E, t)             exp(-i H t) for H diagonal in S_z, diagonal entries E
//        evolve(E, P, t)                  exp(-i H t) for H = P diag(E) P^dagger
//
//  e.) Input / output
//        printSpin(name = "qudit")        norm, <S_a> and Var(S_a) for a = x, y, z
//        save(fileName), load(fileName)   plain text file, full double precision
//
//  f.) Private data members and helpers
//        N_, psi_, Dy_                    dimension, state, Wigner matrix exp(-i pi/2 S_y)
//        phase_, kPi_, kHeader_           powers of i, pi, data file header
//        checkIndex(k)                    throws if k is not in 0..N-1
//        Cplus(J, m), Cminus(J, m)        ladder matrix elements of S_+ and S_-
//        logSqrtBinomial(n, k)            log sqrt(C(n, k)), stable for large n
//        buildDy()                        Risbo recursion for d^J(pi/2)
// ============================================================================
class Qudit {
public:
    // ========================================================================
    //  a.) Constructor
    // ========================================================================
    // N    : dimension of the Hilbert space, N = 2J + 1  (must be >= 1).
    //        N = 2 is a spin-1/2 (qubit), N = 3 a spin-1 (qutrit), ...
    //
    // init : OPTIONAL label of the initial state. Accepted labels:
    //          "CSS+Z"  -> |J, J>   (spin pointing along +z)  [DEFAULT]
    //          "CSS-Z"  -> |J,-J>   (spin pointing along -z)
    //          "CSS+X"  -> coherent spin state along +x
    //          "CSS-X"  -> coherent spin state along -x
    //          "CSS+Y"  -> coherent spin state along +y
    //          "CSS-Y"  -> coherent spin state along -y
    //        Any other string throws std::invalid_argument.
    //
    // To start from an arbitrary vector, construct and then call setState();
    // to start from a file, construct and then call load().
    // ------------------------------------------------------------------------
    
    // The "explicit" forbids silent conversions. The member initialiser 
    // list ( : N_(N), psi_(...) ) initialises the data members first.
    // StateVector::Zero(n) creates a vector of n zeros. (Careful: with Eigen,
    // StateVector(n, value) would NOT mean "n copies of value".)
    explicit Qudit(int N, const std::string& init = "CSS+Z")
        : N_(N), psi_(StateVector::Zero(N > 0 ? N : 0)) {
        // Check the physical size N.
        if (N < 1) throw std::invalid_argument("Qudit: invalid dimension " + std::to_string(N) + ".");
        // Fill psi_ according to the requested label.
        prepareNamed(init);
        // Build the rotation matrix exp(-i pi/2 S_y).
        Dy_ = buildDy();
    }



    // ========================================================================
    //  b.) State preparations
    //  They overwrite the current state and return a reference to the object 
    //  itself (*this). Calls can be chained: q.prepareDickeZ(2).save("dicke");
    // ========================================================================

    // Prepare one of the named states listed in the constructor ("CSS+X"...).
    Qudit& prepareNamed(const std::string& label) {
        // Exact states along +/- z are set as Dicke states.
        if (label == "CSS+Z") return prepareDickeZ(0);
        if (label == "CSS-Z") return prepareDickeZ(N_ - 1);
        // Equatorial states: theta = pi/2, phi selects the direction.
        if (label == "CSS+X") return prepareEquatorialCSS(0);
        if (label == "CSS+Y") return prepareEquatorialCSS(1);
        if (label == "CSS-X") return prepareEquatorialCSS(2);
        if (label == "CSS-Y") return prepareEquatorialCSS(3);
        // Unknown label error.
        throw std::invalid_argument("Qudit: unknown initial state \"" + label + "\".");
    }

    // Dicke state |J, m> along z, selected by its INDEX k = J - m, with
    // 0 <= k <= N-1.  k = 0 gives |J,J>, k = N-1 gives |J,-J>.
    Qudit& prepareDickeZ(int k) {
        // Check that k is a valid spin k = J - m.
        checkIndex(k);
        // Set every amplitude to 0...
        psi_.setZero();
        // ... and put a single 1 at position k.
        psi_(k) = Complex(1.0, 0.0);
        // return the object itself (allows chaining)
        return *this;  
    }

    // Dicke state |J,m_x> along x in the computational basis.
    Qudit& prepareDickeX(int k) { 
        // Just rotate DickeZ around the Y axis by PiHalf.
        return prepareDickeZ(k).applyPulseYPiHalf(); 
    }
    
    // Dicke state |J,m_y> along y in the computational basis.
    Qudit& prepareDickeY(int k) { 
        // Further rotate DickeX around the Z axis by PiHalf.
        return prepareDickeX(k).applyPulseZ(kPi_ / 2); 
    }

    // Coherent Spin State (CSS) |theta, phi>, 
    // i.e. the spin fully polarised along the unit vector
    //
    //      n = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)).
    //
    // Angles are in radians: theta in [0, pi] (polar), phi in [0, 2pi)
    // (azimuthal). Explicitly, with k = J - m:
    //
    //   <J,m|theta,phi> = sqrt( C(2J, k) ) cos(theta/2)^(2J-k)
    //                                       sin(theta/2)^k  exp(i k phi)
    //
    // where C(2J,k) is the binomial coefficient. This is the state obtained
    // by rotating |J,J> so that <S_n> = J. It is normalised by construction.
    Qudit& prepareCSS(double theta, double phi) {
        // Check the polar angle is between zero and pi.
        if (theta < 0. || theta > kPi_) throw std::invalid_argument("Qudit: invalid angle theta.");

        const int twoJ = N_ - 1;                  // 2J, an integer
        const double c = std::cos(0.5 * theta);   // cos(theta/2)
        const double s = std::sin(0.5 * theta);   // sin(theta/2)        

        for (int k = 0; k < N_; k++) {
            // Real modulus: sqrt(C(2J,k)) cos^(2J-k) sin^k.
            double logMod = logSqrtBinomial(twoJ, k);
            // Manage the edges to avoid NaN.
            if (k > 0)        logMod += k * std::log(s);
            if (twoJ - k > 0) logMod += (twoJ - k) * std::log(c);
            // Phase exp(i k phi), built with std::polar(r, angle) = r e^{i angle}.
            psi_(k) = std::polar(std::exp(logMod), k * phi);
        }
        return *this;
    }

    // CSS on the equator (theta = pi/2) with phi = q * pi/2:
    //   q = 0 -> +X,  q = 1 -> +Y,  q = 2 -> -X,  q = 3 -> -Y.
    // Amplitudes: sqrt( C(2J,k) / 2^(2J) ) * i^(q k).
    // The phases i^(qk) are taken from a table, so they are EXACT
    // (no sin/cos of pi, hence no 1e-16 spurious real/imaginary parts).
    Qudit& prepareEquatorialCSS(int q) {
        // Check the q value is valid.
        if (q < 0 || q > 3) throw std::invalid_argument("Qudit: unknown equatorial preparation.");
        const int twoJ = N_ - 1;
        // Compute the amplitudes.
        for (int k = 0; k < N_; k++) {
            const int kk = std::min(k, twoJ - k); 
            // Modulus is computed once per symmetric pair |psi_k| == |psi_{2J-k}|.
            const double modulus = std::exp(logSqrtBinomial(twoJ, kk) - 0.5 * twoJ * std::log(2.0));
            psi_(k) = modulus * phase_[(q * k) % 4];
        }
        return *this;
    }


    
    // ========================================================================
    //  c.) Getters (read-only information)
    // ========================================================================

    //  The "const" promises that these methods do not modify the qudit.
    int dim() const { return N_; }                   // N = 2J + 1

    double spinJ() const { return 0.5 * (N_ - 1); }  // J = (N - 1) / 2

    // Projection m associated with the basis index k (m = J - k).
    double spinM(int k) const { return spinJ() - k; }

    // Full state vector (read-only reference: no copy is made).
    const StateVector& state() const { return psi_; }

    // Euclidean norm sqrt( sum_k |psi_k|^2 ). Should be 1 for a physical state.
    double norm() const { return psi_.norm(); }

    // Single amplitude <J, J-k | psi>, with bounds checking.
    Complex amplitude(int k) const {
        checkIndex(k);
        return psi_(k);
    }

    // Compute the expectation value of S^plus = S^x + i S^y.
    Complex avgSplus() const {
        const double J = spinJ();
        Complex sPlus(0.0, 0.0);
        for (int k = 1; k < N_; k++) sPlus += std::conj(psi_(k - 1)) * Cplus(J,spinM(k)) * psi_(k);
        return sPlus;
    }

    // Compute the expectation value of S^minus = S^x - i S^y.
    Complex avgSminus() const {
        const double J = spinJ();
        Complex sMinus(0.0, 0.0);
        for (int k = 0; k < N_ - 1; k++) sMinus += std::conj(psi_(k + 1)) * Cminus(J, spinM(k)) * psi_(k);
        return sMinus;
    }

    // Return <S_x> = Re<S_+>   and   <S_y> = Im<S_+>.
    double avgSx() const { return avgSplus().real(); }
    double avgSy() const { return avgSplus().imag(); }

    // Compute the expectation value of S^z.
    double avgSz() const {
        double sz = 0.0;
        for (int k = 0; k < N_; k++) sz += spinM(k) * std::norm(psi_(k));
        return sz;
    }

    // Compute <S_z^2> = sum_m m^2 |psi_m|^2.
    double avgSz2() const {
        double sz2 = 0.0;
        for (int k = 0; k < N_; k++) sz2 += spinM(k) * spinM(k) * std::norm(psi_(k));
        return sz2;
    }

    // Compute <S_+^2> = sum_m conj(psi_{m+2}) <J,m+2|S_+|J,m+1><J,m+1|S_+|J,m> psi_m.
    Complex avgSplus2() const {
        const double J = spinJ();
        Complex sPlus2(0.0, 0.0);
        for (int k = 2; k < N_; k++) sPlus2 += std::conj(psi_(k - 2)) * Cplus(J, spinM(k) + 1) * Cplus(J, spinM(k)) * psi_(k);
        return sPlus2;
    }

    // Compute <S_x^2> = (J(J+1) - <S_z^2> + Re<S_+^2> ) / 2.
    double avgSx2() const {
        const double J = spinJ();
        return 0.5 * (J * (J + 1) * psi_.squaredNorm() - avgSz2() + avgSplus2().real());
    }

    // Compute <S_y^2> = (J(J+1) - <S_z^2> - Re<S_+^2> ) / 2.
    double avgSy2() const {
        const double J = spinJ();
        return 0.5 * (J * (J + 1) * psi_.squaredNorm() - avgSz2() - avgSplus2().real());
    }

    // Variances Var(S_a) = <S_a^2> - <S_a>^2 (for a normalised state).
    double varSx() const { 
        const double s = avgSx(); 
        return std::max(0.0, avgSx2() - s * s);
    }
    double varSy() const { 
        const double s = avgSy(); 
        return std::max(0.0, avgSy2() - s * s); 
    }
    double varSz() const { 
        const double s = avgSz(); 
        return std::max(0.0, avgSz2() - s * s); 
    }





    // ========================================================================
    //  d.) Modifiers
    // ========================================================================

    // Rescale the state to unit norm (throws if the state is the zero vector).
    Qudit& normalize() {
        const double n = norm();
        // Check the vector is a valid state.
        if (n == 0.0) throw std::runtime_error("Qudit::normalize: the state is the zero vector");
        // Eigen divides every component by n.
        psi_ /= n;   
        return *this;
    }

    // Replace the state with a user-given vector (must have size N).
    Qudit& setState(const StateVector& amplitudes, bool normalizeIt = true) {
        // Check the size of amplitudes. 
        if (amplitudes.size() != N_) throw std::invalid_argument("Qudit::setState: vector size differs from N");
        // Fill and normalize.
        psi_ = amplitudes;
        if (normalizeIt) normalize();
        return *this;
    }

    // Pulse about z by phi:  exp(-i phi S_z)|J,m> = exp(-i phi m)|J,m>,  m = J - k.
    Qudit& applyPulseZ(double phi) {
        // Written as exp(-i phi J) * exp(i phi k): the first factor is a GLOBAL phase...
        const Complex global = std::polar(1.0, -phi * spinJ());
        // ... and the second carries the relative phases. For phi = q pi/2...
        const double q = 2 * (phi / kPi_);
        // ... these phases are powers of i when q is an integer:
        const long qInt = std::lround(q);
        // Hence, check whether the approximation holds before multiplication. 
        if (std::abs(q - static_cast<double>(qInt)) <= 1e-15 * std::abs(q)) {
            // In this case, take q mod 4 in 0..3, also for q < 0
            const int r = static_cast<int>(((qInt % 4) + 4) % 4);   
            for (int k = 0; k < N_; k++) psi_(k) *= global * phase_[(r * k) % 4];
        } else {
            for (int k = 0; k < N_; k++) psi_(k) *= global * std::polar(1.0, phi * k);
        }
        return *this;
    }

    // Pulse about y by Pi:   exp(-i pi S_y) |J,m> = (-1)^(J-m)   |J,-m>
    Qudit& applyPulseYPi() {
        StateVector out(N_);
        // Since k = J - m, m -> -m means k -> 2J-k: send psi(k) to position N-1-k.
        for (int k = 0; k < N_; k++) out(N_ - 1 - k) = (k % 2 == 0 ? 1.0 : -1.0) * psi_(k);
        psi_.swap(out);
        return *this;
    }

    // Pulse about y by Pi/2: send z into x axis.
    Qudit& applyPulseYPiHalf() { 
        // Apply Dy = exp{-i pi S_y /2} to psi.
        return applyRotation(Dy_); 
    }  

    // Pulse about x by Pi:   exp(-i pi S_x) |J,m> = exp(-i pi J) |J,-m>
    Qudit& applyPulseXPi() {
        StateVector out(N_);
        const Complex phaseX = std::polar(1.0, -kPi_ * spinJ());
        // Since k = J - m, m -> -m means k -> 2J-k: send psi(k) to position N-1-k.
        for (int k = 0; k < N_; k++) out(N_ - 1 - k) =  phaseX * psi_(k);
        psi_.swap(out);
        return *this;
    }

    // Pulse about x by Pi/2: send y into z axis.
    Qudit& applyPulseXPiHalf() { 
        // The rightmost factor acts first, so the calls are in reverse order:
        // applyPulseZ(phi) = exp(-i phi S_z), hence +pi/2 first and -pi/2 last.        
        return applyPulseZ(kPi_ / 2).applyPulseYPiHalf().applyPulseZ(-kPi_ / 2); 
    }

    // Apply a unitary N x N matrix:   psi -> U psi.
    Qudit& applyRotation(const Matrix& U) {
        // Check the matrix size.
        if (U.rows() != N_ || U.cols() != N_) throw std::invalid_argument("Qudit::applyRotation: U must be N x N");
        // Compute a new vector: psi_ is untouched for now.
        StateVector out = U * psi_;   
        // Check the rotation is norm preserving.
        if (std::abs(out.norm() - psi_.norm()) > 1e-10 * psi_.norm()) throw std::runtime_error("Qudit::applyRotation: U does not preserve the norm (not unitary?)");
        // Finally, update the state.
        psi_.swap(out);
        return *this;
    }    

    // Evolve for a time t under a Hamiltonian DIAGONAL in the S_z basis:
    //     psi_k -> exp(-i E_k t) psi_k,   with E_k = <J,J-k| H |J,J-k>.
    // energies : the N diagonal elements E_k of H, in the same order as psi_.
    // Exact for any t (no time step); the norm is preserved.
    Qudit& evolveDiagonal(const RealVector& energies, double t) {
        // Check the size of energies.
        if (energies.size() != N_) throw std::invalid_argument("Qudit::evolveDiagonal: number of energies differs from N");
        // Multiply each entry by the corresponding phase.
        for (int k = 0; k < N_; k++) {
            // std::polar(1.0, x) = exp(i x): a pure phase of modulus 1.
            psi_(k) *= std::polar(1.0, -t * energies(k));
        }
        return *this;
    }

    // Evolve for a time t under H = P diag(E) P^dagger (E: eigenvalues, P: eigenvectors as columns):
    //     exp(-i H t) = P diag(exp(-i E t)) P^dagger
    // i.e. rotate into the eigenbasis of H, evolve diagonally there, rotate back.
    Qudit& evolve(const RealVector& E, const Matrix& P, double t) {
        // Check E before changing anything (evolveDiagonal would check it only after the first rotation).
        if (E.size() != P.rows()) throw std::invalid_argument("Qudit::evolve: different number of eigenvalues and eigenvectors.");
        return applyRotation(P.adjoint()).evolveDiagonal(E, t).applyRotation(P);
    }



    // ========================================================================
    //  e.) Input / output
    // ========================================================================

    // Print the norm, the mean spin components and their variances, one axis per line.
    void printSpin(const std::string& name = "qudit") const {
        // Format (human readable):
        //     name :  <psi|psi> = 1
        //         <Sx> = 0         Var(Sx) = 10
        //         <Sy> = 0         Var(Sy) = 10
        //         <Sz> = 20        Var(Sz) = 0
        //
        // The number format follows the stream settings chosen by the caller,
        // e.g. std::cout << std::fixed << std::setprecision(4) before calling it.
        std::cout << name << " :  <psi|psi> = " << psi_.squaredNorm() << "\n"
                  << "    <Sx> = " << avgSx() << "\tVar(Sx) = " << varSx() << "\n"
                  << "    <Sy> = " << avgSy() << "\tVar(Sy) = " << varSy() << "\n"
                  << "    <Sz> = " << avgSz() << "\tVar(Sz) = " << varSz() << "\n";
    }

    // Save the state to a plain text file named fileName.
    void save(const std::string& fileName) const {
        // Try to open (and truncate) the file...
        std::ofstream out(fileName);  
        // ... and check everything went smooth.
        if (!out) throw std::runtime_error("Qudit::save: cannot open \"" + fileName + "\"");

        // Write 17 significant digits: enough to read back exactly the same.
        out.precision(17);
        // Format (human readable):
        //
        //     # spin::Qudit state: N 6
        //     # k   Re(psi_k)   Im(psi_k)
        //     0  0.17677669529663687  0
        //     1  0.39528470752104744  0
        //     ...
        //
        out << kHeader_ << N_ << "\n";
        out << "# k   Re(psi_k)   Im(psi_k)\n";
        for (int k = 0; k < N_; k++) {
            const Complex& a = psi_(k);
            out << k << "  " << a.real() << "  " << a.imag() << "\n";
        }
        out.close();  // flushes the buffer to disk and closes the file

        // Check if any error occurred while writing or closing.
        if (!out) throw std::runtime_error("Qudit::save: error while writing \"" + fileName + "\"");
    }

    // Load a state previously written by save(), overwriting the current one.
    // The N written in the file must coincide with this qudit's N.
    Qudit& load(const std::string& fileName) {
        // Try to open the file...
        std::ifstream in(fileName);
        // ... and check everything went smooth.
        if (!in) throw std::runtime_error("Qudit::load: cannot open \"" + fileName + "\"");

        // Introduce a switch to check the header...
        bool headerRead = false;
        // ... and a checklist to verify that every entry is filled exactly once.
        BoolArray seen = BoolArray::Constant(N_, false);        
        // Create a local vector instead of starting by overwriting psi_.
        StateVector amplitudes = StateVector::Zero(N_);

        // Start reading the file contents.
        std::string line;
        while (std::getline(in, line)) {
            const auto first = line.find_first_not_of(" \t\r");

            // Skip empty lines.
            if (first == std::string::npos) continue;

            // The first non-empty line must be the header: read N from it.
            if (!headerRead) {
                // line.rfind(header, 0) != 0  means "first line is not the header".
                if (line.rfind(kHeader_, 0) != 0) throw std::runtime_error("Qudit::load: wrong header in \"" + fileName + "\"");

                std::istringstream iss(line.substr(kHeader_.size()));  // the text after "N "
                int fileN = 0;

                // Read N from file...
                if (!(iss >> fileN)) throw std::runtime_error("Qudit::load: cannot read N in \"" + fileName + "\"");
                // ... and compare it with the current one.
                if (fileN != N_) throw std::runtime_error("Qudit::load: N in \"" + fileName + "\" differs from the qudit's N");
                // Switch on and skip header reading from next iteration.
                headerRead = true;
                continue;
            }

            // Any later line starting with # is a comment.
            if (line[first] == '#') continue;

            // Data line: "k  re  im".
            std::istringstream iss(line);
            int k = 0;
            double re = 0.0, im = 0.0;
            // Load data from line:
            if (!(iss >> k >> re >> im)) throw std::runtime_error("Qudit::load: malformed line \"" + line + "\"");
            // Check index (seen(k) is evaluated only if k is in range, thanks to ||):
            if (k < 0 || k >= N_ || seen(k)) throw std::runtime_error("Qudit::load: invalid or duplicated index " + std::to_string(k));
            // Store amplitude...
            amplitudes(k) = Complex(re, im);
            // ... and switch the checklist entry.
            seen(k) = true;
        }

        // After reading, check there was something inside...
        if (!headerRead) throw std::runtime_error("Qudit::load: empty file \"" + fileName + "\"");
        // ... and that every entry was filled: all() is true only if every element is true.
        if (!seen.all()) throw std::runtime_error("Qudit::load: some amplitudes are missing in \"" + fileName + "\"");

        // All checks passed: exchange the two vectors' internal pointers.
        psi_.swap(amplitudes);
        return *this;
    }



private:
    // ========================================================================
    //  f.) Private data members and helpers.
    // ========================================================================
    // "static" members are properties of the class, i.e. do not belong to an object.

    int N_;            // dimension of the Hilbert space, N = 2J + 1 (fixed)

    StateVector psi_;  // amplitudes psi_(k) = <J, J-k | psi>, k = 0..N-1

    Matrix Dy_;        // Wigner matrix Dy_ = exp(-i pi/2 S_y)

     // Powers of i: i^0, i^1, i^2, i^3 (exact phases for multiples of pi/2).
    static constexpr Complex phase_[4] = {Complex(1, 0), Complex(0, 1), Complex(-1, 0), Complex(0, -1)};

    // Pi with full double precision (std::numbers::pi exists only from C++20).
    static constexpr double kPi_ = 3.14159265358979323846;

    // Header for data files
    static inline const std::string kHeader_ = "# spin::Qudit state: N ";

    // Throws std::out_of_range if k is not a valid basis index.
    void checkIndex(int k) const {
        // The message is built inside throw: only when the error happens, not at every call.
        if (k < 0 || k >= N_) throw std::out_of_range("Qudit: basis index " + std::to_string(k) + " out of range");
    }

    // Matrix element of the raising operator: <J,m+1| S_+ |J,m> = sqrt( J(J+1) - m(m+1)).
    static double Cplus(double J, double m) {
        return std::sqrt(J * (J + 1) - m * (m + 1));
    }

    // Matrix element of the lowering operator: <J,m-1| S_- |J,m> = sqrt( J(J+1) - m(m-1)).
    static double Cminus(double J, double m) {
        return std::sqrt(J * (J + 1) - m * (m - 1));
    }

    // log of sqrt( C(n, k) ) with C(n, k) = n! / (k! (n-k)!).
    //
    // Computing the factorials directly would overflow already for n ~ 170, 
    // so we use the logarithm of the Gamma function, lgamma(x+1) = log(x!):
    //     log C(n,k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    // and return the log for further simplifications.
    static double logSqrtBinomial(int n, int k) {
        return 0.5 * (std::lgamma(n + 1.0)
                    - std::lgamma(k + 1.0)
                    - std::lgamma(n - k + 1.0));
    }

    // Wigner matrix Dy_ = exp(-i pi/2 S_y), i.e. the rotation by pi/2 about y, with elements
    //     Dy_(k', k) = d^J_{m'm}(pi/2) = <J,m'| exp(-i pi/2 S_y) |J,m>,   m' = J - k',  m = J - k.
    // It depends only on N: built once in the constructor. Memory: 16 N^2 bytes.
    // Build d^J(pi/2) with the exact recursion in J (Risbo's recursion).
    // (The closed Wigner formula is an alternating sum that loses all digits for N ~ 100.)
    // A spin J = n/2 is the symmetric state of n spins 1/2. Adding one spin 1/2 to a spin (n-1)/2,
    //     |n/2, a> = sqrt((n-a)/n) |(n-1)/2, a>|up> + sqrt(a/n) |(n-1)/2, a-1>|down>,   a = J - m,
    // and rotating each spin 1/2 with d^{1/2}(pi/2) = [[c, -s], [s, c]], c = s = 1/sqrt(2), gives
    //     d^{(n)}(a',a) = [ sqrt((n-a')(n-a)) c d^{(n-1)}(a',  a  ) - sqrt((n-a') a) s d^{(n-1)}(a',  a-1)
    //                     + sqrt(a'(n-a))    s d^{(n-1)}(a'-1,a  ) + sqrt(a' a)    c d^{(n-1)}(a'-1,a-1) ] / n.
    // Start from d^{(0)} = (1) (spin 0) and repeat up to n = N-1. No cancellations: accuracy ~1e-13.
    // Cost O(N^3): ~0.02 s for N = 200, ~2 s for N = 1000.
    Matrix buildDy() const {
        // Table of square roots sqrt(0), ..., sqrt(N), used by every step.
        RealVector sq(N_ + 1);
        for (int i = 0; i <= N_; i++) sq(i) = std::sqrt(static_cast<double>(i));

        Eigen::MatrixXd d = Eigen::MatrixXd::Ones(1, 1);   // spin 0: d = (1)
        for (int n = 1; n < N_; n++) {                       // from spin (n-1)/2 to spin n/2
            Eigen::MatrixXd next = Eigen::MatrixXd::Zero(n + 1, n + 1);
            const double c = std::sqrt(0.5) / n;             // cos(pi/4) / n = sin(pi/4) / n
            // Each old element d(a, b) contributes to its 4 neighbours in the new matrix.
            // (b is the outer loop: Eigen stores matrices by columns, so a runs through memory.)
            for (int b = 0; b < n; b++) for (int a = 0; a < n; a++) {
                const double t = c * d(a, b);
                next(a, b)         += sq(n - a) * sq(n - b) * t;   // up   -> up   :  +cos
                next(a, b + 1)     -= sq(n - a) * sq(b + 1) * t;   // down -> up   :  -sin
                next(a + 1, b)     += sq(a + 1) * sq(n - b) * t;   // up   -> down :  +sin
                next(a + 1, b + 1) += sq(a + 1) * sq(b + 1) * t;   // down -> down :  +cos
            }
            d.swap(next);
        }
        return d.cast<Complex>();   // d^J(pi/2) is real; stored as complex to be used with applyRotation
    }
};

}  // namespace spin
