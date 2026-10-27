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
#include <cmath>        // std::sqrt, std::cos, std::sin, std::pow, std::lgamma, ...
#include <complex>      // std::complex<double>, std::polar, std::norm
#include <Eigen/Dense>  // Eigen::VectorXcd, Eigen::Vector4d, Eigen::VectorXd, Eigen::Array

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
using SpinVector = Eigen::Vector4d;

// Short alias: a column of booleans whose size is chosen at run time.
// Array instead of Vector: it is a plain table, no linear algebra.
using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

// Short alias: a vector of N real numbers (e.g. the energies of H).
// Eigen::VectorXd = double vector whose size is chosen at run time.
using RealVector = Eigen::VectorXd;

// Short alias: a state vector = list of N complex amplitudes.
// Eigen::VectorXcd = complex<double> vector whose size is chosen at run time.
using StateVector = Eigen::VectorXcd;





// ============================================================================
//  Class Qudit
// ============================================================================
//     a.) Constructor
//     b.) State preparations (Dicke, CSS, named states)
//     c.) Getters (read-only information)
//     d.) Modifiers (normalize, setState)
//     e.) Input / output (save, load)
//     f.) Private data members and helpers
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
    }



    // ========================================================================
    //  b.) State preparations
    //  They overwrite the current state and return a reference to the object 
    //  itself (*this). Calls can be chained: q.prepareDicke(2).save("dicke");
    // ========================================================================

    // Prepare one of the named states listed in the constructor ("CSS+X"...).
    Qudit& prepareNamed(const std::string& label) {
        // Exact states along +/- z are set as Dicke states.
        if (label == "CSS+Z") return prepareDicke(0);
        if (label == "CSS-Z") return prepareDicke(N_ - 1);
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
    Qudit& prepareDicke(int k) {
        // Check that k is a valid spin k = J - m.
        checkIndex(k);
        // Set every amplitude to 0...
        psi_.setZero();
        // ... and put a single 1 at position k.
        psi_(k) = Complex(1.0, 0.0);
        // return the object itself (allows chaining)
        return *this;  
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

        for (int k = 0; k < N_; ++k) {
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
        // Check the phi angle is well-defined.
        if (q < 0 || q > 3) throw std::invalid_argument("Qudit: unknown equatorial preparation.");

        const int twoJ = N_ - 1;
        const Complex phase[4] = {Complex(1, 0), Complex(0, 1), Complex(-1, 0), Complex(0, -1)};
        
        for (int k = 0; k < N_; k++) {
            // Modulus computed once per symmetric pair |psi_k| == |psi_{2J-k}|.
            const int kk = ((k <= twoJ - k) ? k : twoJ - k);   // symmetric partner
            const double modulus = std::exp(logSqrtBinomial(twoJ, kk) - 0.5 * twoJ * std::log(2.0));
            psi_(k) = modulus * phase[(q * k) % 4];
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

    // Single amplitude <J, J-k | psi>, with bounds checking.
    Complex amplitude(int k) const {
        checkIndex(k);
        return psi_(k);
    }

    // Euclidean norm sqrt( sum_k |psi_k|^2 ). Should be 1 for a physical state.
    double norm() const { return psi_.norm(); }

    // Compute the expectation value of S^mu with mu = 0,x,y,z and S^0 = 1.
    SpinVector avgSpin() const {
        const double J = spinJ();
        double s0 = 0.0;
        double sz = 0.0;
        Complex sPlus(0.0, 0.0);

        for (int k = 0; k < N_; k++) {
            const double m = spinM(k);
            const double p = std::norm(psi_(k));   // |psi_k|^2
            //sum squared amplitudes
            s0 += p;
            // sum S_z components
            sz += m * p;
            // sum S_+ components
            if (k > 0) sPlus += std::sqrt(J * (J + 1) - m * (m + 1)) * std::conj(psi_(k - 1)) * psi_(k);
        }
        // Vector4d has a constructor taking its 4 components.
        return SpinVector(s0, sPlus.real(), sPlus.imag(), sz);
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



    // ========================================================================
    //  e.) Input / output
    // ========================================================================

    // Print the average spin of the qudit
    void printSpin(const std::string& name = "qudit") const {
        const SpinVector s = avgSpin();

        std::cout << name << " :  <S0> = " << s[0]
                << "   <Sx> = " << s[1]
                << "   <Sy> = " << s[2]
                << "   <Sz> = " << s[3] << "\n";
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

    int N_;            // dimension of the Hilbert space, N = 2J + 1 (fixed)

    StateVector psi_;  // amplitudes psi_(k) = <J, J-k | psi>, k = 0..N-1

    // pi with full double precision (std::numbers::pi exists only from C++20).
    // "static" because it does not use N_ or psi_: it is a property of the class.
    static constexpr double kPi_ = 3.14159265358979323846;

    // header for data files
    static inline const std::string kHeader_ = "# spin::Qudit state: N ";

    // Throws std::out_of_range if k is not a valid basis index.
    void checkIndex(int k) const {
        // The message is built inside throw: only when the error happens, not at every call.
        if (k < 0 || k >= N_) throw std::out_of_range("Qudit: basis index " + std::to_string(k) + " out of range");
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

};

}  // namespace spin
