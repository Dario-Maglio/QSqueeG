// ============================================================================
//  qudit.hpp   --   header-only definition of the class spin::Qudit
// ----------------------------------------------------------------------------
//  The class stores the (pure) quantum state of a single spin J, i.e. a
//  vector in a Hilbert space of dimension Z = 2J + 1.
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
//      index k :   0       1        2       ...    Z-1
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

#include <Eigen/Dense>  // Eigen::VectorXcd, Eigen::VectorXd, Eigen::Array, ...

#include <string>       // std::string, std::to_string
#include <fstream>      // std::ifstream, std::ofstream (file I/O)
#include <sstream>      // std::istringstream (parsing one line at a time)
#include <iostream>     // std::cout to print the expectation values of the spin
#include <stdexcept>    // std::invalid_argument, std::out_of_range, std::runtime_error




namespace spin {        // everything we write lives in the namespace "spin"

// Short alias: a complex number made of two doubles.
using Complex = std::complex<double>;

// Short alias: a column of booleans whose size is chosen at run time.
// Array instead of Vector: it is a plain table, no linear algebra.
using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

// Short alias: a vector of Z real numbers (e.g. the energies of H).
// Eigen::VectorXd = double vector whose size is chosen at run time.
using RealVector = Eigen::VectorXd;

// Short alias: a state vector = list of Z complex amplitudes.
// Eigen::VectorXcd = complex<double> vector whose size is chosen at run time.
using StateVector = Eigen::VectorXcd;

// Short alias: a real matrix (e.g. a function sampled on a grid).
using RealMatrix = Eigen::MatrixXd;

// Short alias: a complex Z x Z matrix (e.g. a rotation or evolution operator).
using Matrix = Eigen::MatrixXcd;

// Short aliases: a real 3-vector and a real 3 x 3 matrix, one entry per axis x, y, z
using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;






// ============================================================================
//  Class Qudit
// ============================================================================
//  a.) Constructor
//        Qudit(Z, init="CSS+Z")           Z = 2J+1; init = named state (see below)
//
//  b.) State preparations  (overwrite the state, return *this for chaining)
//        prepareNamed(label)              "CSS+Z", "CSS-Z", "CSS+X", "CSS-X", "CSS+Y", "CSS-Y"
//        prepareDickeZ(k)                 |J,m>   along z,  m = J - k
//        prepareDickeX(k)                 |J,m>_x along x,  m = J - k
//        prepareDickeY(k)                 |J,m>_y along y,  m = J - k
//        prepareReference()               reference state |o> = sum_m |J,m>_y / sqrt(2J+1)
//        prepareCSS(theta, phi)           coherent spin state along (theta, phi)
//        prepareEquatorialCSS(q)          CSS along +x, +y, -x, -y for q = 0, 1, 2, 3 (exact phases)
//
//  c.) Getters  (read-only, const)
//        dim(), spinJ(), spinM(k)         Z, J, and m = J - k
//        state(), norm(), amplitude(k)    state vector, Euclidean norm, single amplitude
//        overlapCSS(theta, phi)           single amplitude with the corresponding CSS
//        avgSpin()                        mean spin vector (<S_x>, <S_y>, <S_z>)
//        covSpin()                        3 x 3 covariance matrix, variances on the diagonal
//        husimiQ(ntheta=61, nphi=121)     Husimi Q function as a ntheta x nphi matrix
//
//  d.) Modifiers  (change the state, return *this for chaining)
//        normalize()                      rescale to unit norm
//        setState(v, normalizeIt=true)    replace the state with a given vector
//        applyRotation(U)                 psi -> U psi, U unitary Z x Z
//        applyPulseZ(phi)                 exp(-i phi S_z), exact for multiples of pi/2
//        applyPulseX(phi)                 exp(-i phi S_x), uses Dy_ twice and one z pulse
//        applyPulseY(phi)                 exp(-i phi S_y), applyPulseX between two z pulses
//        applyPulseXPi(), applyPulseYPi() exact pi pulses (vector reversal)
//        applyPulseXPiHalf()              pi/2 pulse about x, uses Dy_ and two z pulses
//        applyPulseYPiHalf()              pi/2 pulse about y, just applyRotation(Dy_)
//        evolveDiagonal(E, t)             exp(-i H t) for H diagonal in S_z, diagonal entries E
//        evolve(E, P, t)                  exp(-i H t) for H = P diag(E) P^dagger
//
//  e.) Input / output
//        printSpin(name="qudit")          norm, <S_a> and Var(S_a) for a = x, y, z
//        save(fileName), load(fileName)   plain text file, full double precision
//
//  f.) Private data members and helpers
//        Z_, psi_, Dy_                    dimension, state, Wigner matrix exp(-i pi/2 S_y)
//        kPrecision_                      precision required by the save and load methods
//        phase_, kPi_, kHeader_           powers of i, pi, data file header
//        Cplus(J, m), Cmnus(J, m)         ladder matrix elements of S_+ and S_-
//        logSqrtBinomial(n, k)            log sqrt(C(n, k)), stable for large n
//        applyS(axis)                     S_a psi for axis = 1, 2, 3 (x, y, z)
//        cssModuli(theta)                 compute the modulus of CSS amplitudes
//        buildDy()                        ladder recurrence for d^J(pi/2)
// ============================================================================
class Qudit {
public:
    // ========================================================================
    //  a.) Constructor
    // ========================================================================
    // Z    : dimension of the Hilbert space, Z = 2J + 1  (must be >= 1).
    //        Z = 2 is a spin-1/2 (qubit), Z = 3 a spin-1 (qutrit), ...
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
    // list ( : Z_(Z), psi_(...), ... ) initialises the data members first.
    // StateVector::Zero(n) creates a vector of n zeros.
    explicit Qudit(int Z, const std::string& init = "CSS+Z")
        : Z_(Z), psi_(StateVector::Zero(Z > 0 ? Z : 0)) {
        // Check the physical size Z.
        if (Z < 1) throw std::invalid_argument("Qudit: invalid dimension " + std::to_string(Z) + ".");
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
        if (label == "CSS-Z") return prepareDickeZ(Z_ - 1);
        // Equatorial states: theta = pi/2, phi selects the direction.
        if (label == "CSS+X") return prepareEquatorialCSS(0);
        if (label == "CSS+Y") return prepareEquatorialCSS(1);
        if (label == "CSS-X") return prepareEquatorialCSS(2);
        if (label == "CSS-Y") return prepareEquatorialCSS(3);
        // Unknown label error.
        throw std::invalid_argument("Qudit: unknown initial state \"" + label + "\".");
    }

    // Dicke state |J, m>, with m = J - k for 0 <= k <= Z-1.  
    // Applying Jplus = Jx + i Jy sends k to k - 1.
    // Setting k = 0 gives |J,J>, k = Z-1 gives |J,-J>.
    Qudit& prepareDickeZ(int k) {
        // Check that k is a valid spin k = J - m.
        if (k < 0 || k >= Z_) throw std::out_of_range("Qudit: basis index " + std::to_string(k) + " out of range");
        // Set every amplitude to 0...
        psi_.setZero();
        // ... and put a single 1 at position k.
        psi_(k) = Complex(1.0, 0.0);
        // Return the object itself (allows chaining)
        return *this;  
    }

    // Dicke state |J,m>_x, with m = J - k in the cyclic frame (y, z, x):
    // Applying Jplus_(x) = Jy + i Jz sends k to k - 1.
    Qudit& prepareDickeX(int k) {
        prepareDickeZ(k);
        // Apply pi/2 pulse about z, i.e. phase i^k on |J,J-k>... 
        psi_(k) = phase_[k % 4];
        // ... then a pi/2 pulse about y.
        return applyPulseYPiHalf();
    }
    
    // Dicke state |J,m>_y, with m = J - k and phases in the cyclic frame (z, x, y):
    // Applying Jplus_(y) = Jz + i Jx sends k to k - 1.
    Qudit& prepareDickeY(int k) {
        prepareDickeZ(k);
        // Apply pi pulse about z, i.e. phase (-1)^k = i^(2k) on |J,J-k>...
        psi_(k) = phase_[(2 * k) % 4];
        // ... then a pi/2 pulse about y and a pi/2 pulse about z.
        return applyPulseYPiHalf().applyPulseZ(kPi_ / 2);
    }

    // Reference state of the qudit ruler: |o> = sum_m |J,m>_y / sqrt(2J+1).
    // Its mean spin points to +z, with <S_z>/J -> pi/4 for large J.
    Qudit& prepareReference() {
        const double amplitude = 1.0 / std::sqrt(static_cast<double>(Z_));
        // Signed sum of the z Dicke states: phase (-1)^k = i^(2k), as in prepareDickeY...
        for (int k = 0; k < Z_; k++) psi_(k) = amplitude * phase_[(2 * k) % 4];
        // ... then a pi/2 pulse about y and a pi/2 pulse about z.
        return applyPulseYPiHalf().applyPulseZ(kPi_ / 2);
    }

    // Coherent Spin State (CSS) |theta, phi>, fully polarised along the unit vector
    //      n = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)).
    // Angles are in radians: theta in [0, pi] (polar), phi in [0, 2pi) (azimuthal).
    Qudit& prepareCSS(double theta, double phi) {
        // Explicitly, with k = J - m, we have:
        //   <J,m|theta,phi> = sqrt( C(2J, k) ) cos(theta/2)^(2J-k)
        //                                      sin(theta/2)^k  exp(i k phi)
        // where C(2J,k) is the binomial coefficient. This is the state obtained
        // by rotating |J,J> so that <S_n> = J. It is normalised by construction.
        const RealVector r = cssModuli(theta);
        // Moduli r times the phases exp(i k phi) with std::polar(r, angle).
        for (int k = 0; k < Z_; k++) psi_(k) = std::polar(r(k), k * phi);
        return *this;
    }

    // CSS on the equator (theta = pi/2) with phi = q * pi/2:
    //   q = 0 -> +X,  q = 1 -> +Y,  q = 2 -> -X,  q = 3 -> -Y.
    // Avoids numerical errors of prepareCSS for these states.
    Qudit& prepareEquatorialCSS(int q) {
        // Check the q value is valid.
        if (q < 0 || q > 3) throw std::invalid_argument("Qudit: unknown equatorial preparation.");
        const int twoJ = Z_ - 1;
        // Compute log of 2^J, the same for every k.
        const double logNorm = 0.5 * twoJ * std::log(2.0);
        // Compute the amplitudes in a symmetric way.
        for (int k = 0; k <= twoJ / 2; k++) {
            const double modulus = std::exp(logSqrtBinomial(twoJ, k) - logNorm);
            // The phases i^(qk) are taken from a table, so they are EXACT
            // (no sin/cos of pi, hence no 1e-16 spurious real/imaginary parts).
            psi_(k)        = modulus * phase_[(q * k) % 4];
            psi_(twoJ - k) = modulus * phase_[(q * (twoJ - k)) % 4];
        }
        return *this;
    }


    
    // ========================================================================
    //  c.) Getters (read-only information)
    // ========================================================================

    //  The "const" promises that these methods do not modify the qudit.
    int dim() const { return Z_; }                   // Z = 2J + 1

    double spinJ() const { return 0.5 * (Z_ - 1); }  // J = (Z - 1) / 2

    // Projection m associated with the basis index k (m = J - k).
    double spinM(int k) const { return spinJ() - k; }

    // Full state vector (read-only reference: no copy is made).
    const StateVector& state() const { return psi_; }

    // Euclidean norm sqrt( sum_k |psi_k|^2 ). Should be 1 for a physical state.
    double norm() const { return psi_.norm(); }

    // Single amplitude <J, J-k | psi>, with bounds checking.
    Complex amplitude(int k) const {
        if (k < 0 || k >= Z_) throw std::out_of_range("Qudit: basis index " + std::to_string(k) + " out of range");
        return psi_(k);
    }

    // Single amplitude <theta, phi | psi>, with checks on theta in cssModuli.
    Complex overlapCSS(double theta, double phi) const {
        // Compute r_k = |<J,J-k|theta,phi>|, which only depends on theta.
        const RealVector r = cssModuli(theta); 
        // The sum is a polynomial in z = exp(-i phi):  sum_k r_k psi_k z^k.  
        // Horner's rule counts from the highest power down, with b_k = r_k psi_k:
        //         overlap = b_0 + z (b_1 + z (b_2 + ...))  
        // only Z multiplications and additions, no powers z^k and no sin/cos.
        const Complex z = std::polar(1.0, -phi);
        // Define b_{Z-1} as...
        Complex overlap = r(Z_ - 1) * psi_(Z_ - 1);
        // ... and iterate until k=0.
        for (int k = Z_ - 2; k >= 0; k--) overlap = overlap * z + r(k) * psi_(k);
        return overlap;
    }

    // Mean spin vector (<S_x>, <S_y>, <S_z>).
    Vector3 avgSpin() const {
        const double J = spinJ();
        // <S_+> = <S_x> + i <S_y>, with S_+ taking index k to k - 1 (m to m + 1).
        Complex sPlus(0.0, 0.0);
        for (int k = 1; k < Z_; k++) sPlus += std::conj(psi_(k - 1)) * Cplus(J, spinM(k)) * psi_(k);
        // <S_z>, pairing m and -m: symmetric evaluation helps against rounding errors.
        double sz = 0.0;
        for (int k = 0; k < Z_ / 2; k++) sz += spinM(k) * (std::norm(psi_(k)) - std::norm(psi_(Z_ - 1 - k)));
        return Vector3(sPlus.real(), sPlus.imag(), sz);
    }

    // Covariance matrix of the spin, with the variances Var(S_a) on the diagonal:
    //     C_ab = < (dS_a dS_b + dS_b dS_a) / 2 >,   with dS_a = S_a - <S_a>   (a, b = x, y, z).
    // The symmetrisation makes C real and symmetric even though S_a and S_b do not commute.
    Matrix3 covSpin() const {
        Matrix dS(Z_, 3);
        // First, compute the mean spin <S_a> = (<S_x>, <S_y>, <S_z>).
        const Vector3 mean = avgSpin();
        // Subtracting the mean BEFORE any product is what keeps the result precise
        for (int a = 0; a < 3; a++) dS.col(a) = applyS(a + 1) - mean(a) * psi_;
        // In our notation (dS^dagger * dS)_ab =  <psi| dS_a dS_b |psi>.
        // The symmetrised average is then just the real part.
        return (dS.adjoint() * dS).real();
    }


    // Husimi Q function on a (theta, phi) grid, for plotting the state on the Bloch sphere:
    RealMatrix husimiQ(int nTheta=61, int nPhi=121) const {
        // Check the grid dimension is valid.
        if (nTheta < 2 || nPhi < 2) throw std::invalid_argument("Qudit::husimiQ: the grid needs at least 2 x 2 points");
        // Compute Q(theta, phi) = |<theta, phi | psi>|^2, with
        //     theta_i = pi * i / (nTheta - 1),   i = 0 .. nTheta-1   (from +z, theta = 0, to -z, theta = pi),
        //     phi_j   = 2 pi * j / (nPhi - 1),   j = 0 .. nPhi-1     (the last j gives 2pi to close the sphere).
        // Values of Q in [0, 1]: (2J+1)/(4 pi) ∫ Q(theta,phi) sin(theta) dtheta dphi = 1.
        RealMatrix Q(nTheta, nPhi);
        for (int i = 0; i < nTheta; i++) {
            // Compute r_k = |<J,J-k|theta,phi>|, which only depends on theta.
            const RealVector r = cssModuli(kPi_ * i / (nTheta - 1));
            for (int j = 0; j < nPhi; j++) {
                const Complex z = std::polar(1.0, -2.0 * kPi_ * j / (nPhi - 1));
                // To compute sum_k b_k z^k with with b_k = r_k psi_k and z = exp(-i phi_j),
                // we use Horner's rule as in overlapCSS. First, define b_{Z-1} as...
                Complex overlap = r(Z_ - 1) * psi_(Z_ - 1);
                // ... and then iterate until k=0.
                for (int k = Z_ - 2; k >= 0; k--) overlap = overlap * z + r(k) * psi_(k);
                // This gives the point (i,j) on the grid as:
                Q(i, j) = std::norm(overlap);
            }
        }
        // Check the normalization over the Bloch sphere with:
        //     1 approx C sum_{i=0}^{ntheta-1} sin(theta_i) sum_{j=0}^{nphi−2} Q_{i,j}
        // where C = (2J+1)pi / (2(ntheta−1)(nphi−1)), and we neglect the last j.
        return Q;
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

    // Replace the state with a user-given vector (must have size Z).
    Qudit& setState(const StateVector& amplitudes, bool normalizeIt = true) {
        // Check the size of amplitudes. 
        if (amplitudes.size() != Z_) throw std::invalid_argument("Qudit::setState: vector size differs from Z");
        // Fill and normalize.
        psi_ = amplitudes;
        if (normalizeIt) normalize();
        return *this;
    }

    // Apply a unitary Z x Z matrix:   psi -> U psi.
    Qudit& applyRotation(const Matrix& U) {
        // Check the matrix size.
        if (U.rows() != Z_ || U.cols() != Z_) throw std::invalid_argument("Qudit::applyRotation: U must be Z x Z");
        // Compute a new vector: psi_ is untouched for now.
        StateVector out = U * psi_;   
        // Check the rotation is norm preserving.
        if (std::abs(out.norm() - psi_.norm()) > 1e-10 * psi_.norm()) throw std::runtime_error("Qudit::applyRotation: U does not preserve the norm (not unitary?)");
        // Finally, update the state.
        psi_.swap(out);
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
            for (int k = 0; k < Z_; k++) psi_(k) *= global * phase_[(r * k) % 4];
        } else {
            for (int k = 0; k < Z_; k++) psi_(k) *= global * std::polar(1.0, phi * k);
        }
        return *this;
    }

    // Pulse about x by phi: exp(-i phi S_x) = R_y(pi/2) R_z(phi) R_y(-pi/2),
    // with R_a(phi) = exp(-i phi S_a) and R_y(-pi/2) = Dy^dagger.
    Qudit& applyPulseX(double phi) {
        // The rightmost factor acts first, so the calls are in reverse order.
        return applyRotation(Dy_.adjoint()).applyPulseZ(phi).applyRotation(Dy_);
    }

    // Pulse about y by phi: exp(-i phi S_y) = R_z(pi/2) R_x(phi) R_z(-pi/2).
    Qudit& applyPulseY(double phi) {
        return applyPulseZ(-kPi_ / 2).applyPulseX(phi).applyPulseZ(kPi_ / 2);
    } 

    // Pulse about x by Pi:   exp(-i pi S_x) |J,m> = exp(-i pi J) |J,-m>
    Qudit& applyPulseXPi() {
        StateVector out(Z_);
        const Complex phaseX = std::polar(1.0, -kPi_ * spinJ());
        // Since k = J - m, m -> -m means k -> 2J-k: send psi(k) to position Z-1-k.
        for (int k = 0; k < Z_; k++) out(Z_ - 1 - k) =  phaseX * psi_(k);
        psi_.swap(out);
        return *this;
    }

    // Pulse about y by Pi:   exp(-i pi S_y) |J,m> = (-1)^(J-m)   |J,-m>
    Qudit& applyPulseYPi() {
        StateVector out(Z_);
        // Since k = J - m, m -> -m means k -> 2J-k: send psi(k) to position Z-1-k.
        for (int k = 0; k < Z_; k++) out(Z_ - 1 - k) = (k % 2 == 0 ? 1.0 : -1.0) * psi_(k);
        psi_.swap(out);
        return *this;
    }

    // Pulse about x by Pi/2: sends y into z and z into -y.
    Qudit& applyPulseXPiHalf() { 
        // applyPulseZ(phi) = exp(-i phi S_z), hence +pi/2 first and -pi/2 last.        
        return applyPulseZ(kPi_ / 2).applyRotation(Dy_).applyPulseZ(-kPi_ / 2); 
    }

    // Pulse about y by Pi/2: sends z into x and x into -z.
    Qudit& applyPulseYPiHalf() { 
        // Apply Dy = exp{-i pi S_y /2} to psi.
        return applyRotation(Dy_); 
    }  

    // Evolve for a time t under a Hamiltonian DIAGONAL in the S_z basis:
    //     psi_k -> exp(-i E_k t) psi_k,   with E_k = <J,J-k| H |J,J-k>.
    // energies : the Z diagonal elements E_k of H, in the same order as psi_.
    Qudit& evolveDiagonal(const RealVector& energies, double t) {
        // Check the size of energies.
        if (energies.size() != Z_) throw std::invalid_argument("Qudit::evolveDiagonal: number of energies differs from Z");
        // Multiply each entry by the corresponding phase.
        for (int k = 0; k < Z_; k++) {
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
        // The number format follows the stream settings chosen by the caller,
        // e.g. std::cout << std::fixed << std::setprecision(4) before calling it.
        const Vector3 s = avgSpin();
        const Matrix3 C = covSpin();
        const char axis[3] = {'x', 'y', 'z'};
        std::cout << name << " :  <psi|psi> = " << psi_.squaredNorm() << "\n";
        for (int a = 0; a < 3; a++) std::cout << "    <S" << axis[a] << "> = " << s(a) << "\tVar(S" << axis[a] << ") = " << C(a, a) << "\n";
    }

    // Save the state to a plain text file named fileName.
    void save(const std::string& fileName) const {
        // Try to open (and truncate) the file...
        std::ofstream out(fileName);  
        // ... and check everything went smooth.
        if (!out) throw std::runtime_error("Qudit::save: cannot open \"" + fileName + "\"");
        // Write kPrecision_ significant digits: enough to read back exactly the same.
        out.precision(kPrecision_);
        // Format (human readable):
        //     # spin::Qudit state: Z 6
        //     # k   Re(psi_k)   Im(psi_k)
        //     0  0.17677669529663687  0
        //     1  0.39528470752104744  0
        //     ...
        // Change precision when changing format!
        out << kHeader_ << Z_ << "\n";
        out << "# k   Re(psi_k)   Im(psi_k)\n";
        for (int k = 0; k < Z_; k++) {
            const Complex& a = psi_(k);
            out << k << "  " << a.real() << "  " << a.imag() << "\n";
        }
        out.close();
        // Check if any error occurred while writing or closing.
        if (!out) throw std::runtime_error("Qudit::save: error while writing \"" + fileName + "\"");
    }

    // Load a state previously written by save(), overwriting the current one.
    // The Z written in the file must coincide with this qudit's Z.
    Qudit& load(const std::string& fileName) {
        // Try to open the file...
        std::ifstream in(fileName);
        // ... and check everything went smooth.
        if (!in) throw std::runtime_error("Qudit::load: cannot open \"" + fileName + "\"");

        // Introduce a switch to check the header...
        bool headerRead = false;
        // ... and a checklist to verify that every entry is filled exactly once.
        BoolArray seen = BoolArray::Constant(Z_, false);        
        // Create a local vector instead of starting by overwriting psi_.
        StateVector amplitudes = StateVector::Zero(Z_);

        // Start reading the file contents.
        std::string line;
        while (std::getline(in, line)) {
            const auto first = line.find_first_not_of(" \t\r");

            // Skip empty lines.
            if (first == std::string::npos) continue;

            // The first non-empty line must be the header: read Z from it.
            if (!headerRead) {
                // line.rfind(header, 0) != 0  means "first line is not the header".
                if (line.rfind(kHeader_, 0) != 0) throw std::runtime_error("Qudit::load: wrong header in \"" + fileName + "\"");

                std::istringstream iss(line.substr(kHeader_.size()));  // the text after "Z "
                int fileN = 0;

                // Read Z from file...
                if (!(iss >> fileN)) throw std::runtime_error("Qudit::load: cannot read Z in \"" + fileName + "\"");
                // ... and compare it with the current one.
                if (fileN != Z_) throw std::runtime_error("Qudit::load: Z in \"" + fileName + "\" differs from the qudit's Z");
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
            if (k < 0 || k >= Z_ || seen(k)) throw std::runtime_error("Qudit::load: invalid or duplicated index " + std::to_string(k));
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

    int Z_;            // dimension of the Hilbert space, Z = 2J + 1 (fixed)

    StateVector psi_;  // amplitudes psi_(k) = <J, J-k | psi>, k = 0..Z-1

    Matrix Dy_;        // Wigner matrix Dy_ = exp(-i pi/2 S_y)

    // Significant digits written by save(): 17 are enough to get the same double.
    static constexpr int kPrecision_ = 17;

     // Powers of i: i^0, i^1, i^2, i^3 (exact phases for multiples of pi/2).
    static constexpr Complex phase_[4] = {Complex(1, 0), Complex(0, 1), Complex(-1, 0), Complex(0, -1)};

    // Pi with full double precision (std::numbers::pi exists only from C++20).
    static constexpr double kPi_ = 3.14159265358979323846;

    // Header for data files
    static inline const std::string kHeader_ = "# spin::Qudit state: Z ";

    // Matrix element of the raising operator: <J,m+1| S_+ |J,m> = sqrt( J(J+1) - m(m+1)).
    static double Cplus(double J, double m) {
        return std::sqrt(J * (J + 1) - m * (m + 1));
    }

    // Matrix element of the lowering operator: <J,m-1| S_- |J,m> = sqrt( J(J+1) - m(m-1)).
    static double Cmnus(double J, double m) {
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

    // S_a psi for a = 1 (x), 2 (y), 3 (z), without building any matrix.
    // (S_+ psi)_k = Cplus(J, m_{k+1}) psi_{k+1}   (S_+ raises m: takes index k+1 to k),
    // (S_- psi)_k = Cmnus(J, m_{k-1}) psi_{k-1}   (S_- lowers m: takes index k-1 to k),
    // S_x = (S_+ + S_-)/2,   S_y = (S_+ - S_-)/(2i),   S_z diagonal with entries m.
    StateVector applyS(int axis) const {
        if (axis < 1 || axis > 3) throw std::invalid_argument("Qudit::applyS: axis must be 1, 2 or 3");
        const double J = spinJ();
        StateVector out(Z_);
        for (int k = 0; k < Z_; k++) {
            if (axis == 3) {
                out(k) = spinM(k) * psi_(k);
            } else {
                // Manage the edges while computing S_+ psi and S_- psi.
                const Complex down = (k > 0)      ? Cmnus(J, spinM(k - 1)) * psi_(k - 1) : Complex(0.0, 0.0);
                const Complex up   = (k + 1 < Z_) ? Cplus(J, spinM(k + 1)) * psi_(k + 1) : Complex(0.0, 0.0);
                // Compute the S_x or S_y k-component.
                out(k) = (axis == 1) ? 0.5 * (up + down) : Complex(0.0, -0.5) * (up - down);
            }
        }
        return out;
    }

    // Moduli of the coherent spin state amplitudes:
    //      |<J,m|theta,phi>| = sqrt(C(2J,k)) cos(theta/2)^(2J-k) sin(theta/2)^k,
    // with k = J - m, computed through logarithms (no overflow for large J).
    RealVector cssModuli(double theta) const {
        // Check the polar angle is between zero and pi (otherwise cos(theta/2) < 0 and its log is NaN).
        if (theta < 0. || theta > kPi_) throw std::invalid_argument("Qudit: invalid angle theta.");
        const int twoJ = Z_ - 1;                  // 2J, an integer
        const double c = std::cos(0.5 * theta);   // cos(theta/2)
        const double s = std::sin(0.5 * theta);   // sin(theta/2)
        RealVector r(Z_);
        for (int k = 0; k < Z_; k++) {
            double logMod = logSqrtBinomial(twoJ, k);
            // Manage the edges to avoid 0 * log(0) = NaN at the poles.
            if (k > 0)        logMod += k * std::log(s);
            if (twoJ - k > 0) logMod += (twoJ - k) * std::log(c);
            r(k) = std::exp(logMod);
        }
        return r;
    }

    // Wigner matrix Dy_ = exp(-i pi/2 S_y): rotation by pi/2 about y, built once in the constructor.
    //     Dy_(a, k) = <J,m_a| exp(-i pi/2 S_y) |J,m_k>,   m_a = J - a,  m_k = J - k   (real numbers).
    // Cost O(Z^2), memory 16 Z^2 bytes, accuracy ~1e-15 (Z = 801: 2.5e-15), valid for Z <= 2045.
    Matrix buildDy() const {
        const int twoJ = Z_ - 1;
        const double J = spinJ();
        // The smallest number used below is 2^-J: it must be a normal double, i.e. J <= 1022.
        if (twoJ / 2 > 1022) throw std::invalid_argument("Qudit: Z > 2045 is too large for the double-precision Dy");
        
        // Edge rows. Their moduli are known in closed form (as for the equatorial CSS):
        //     D(0, k) = (-1)^k e_k,   D(2J, k) = e_k,   e_k = sqrt(C(2J,k)) / 2^J.
        // e_k is built as a product, e_k = e_{k-1} sqrt((2J-k+1)/k), from e_0 = 2^-J 
        // given exactly by std::ldexp(x, n) = x * 2^n. This is the reason for J <= 1022.
        RealVector e(Z_);
        e(0) = std::ldexp(twoJ % 2 == 1 ? std::sqrt(0.5) : 1.0, -(twoJ / 2));
        // 2^-J = x * 2^-floor(J), x = sqrt(1/2) for half-integer J, 1 for integer J.
        for (int k = 1; k <= twoJ / 2; k++) e(k) = e(k - 1) * std::sqrt((twoJ - k + 1) / static_cast<double>(k));
        for (int k = 0; k <= twoJ / 2; k++) e(twoJ - k) = e(k);      // symmetry e_k = e_{2J-k}
        // Through logSqrtBinomial, the 1e-16 * lgamma(2J) ~ 1e-13 error of the exponent would become 1e-13 after exp.

        // Inner rows: Dy rotates S_z into S_x, i.e. S_x Dy = Dy S_z, so column k is the eigenvector of S_x
        // with eigenvalue m_k (the Dicke state |J, m_k>_x). S_x only couples rows a and a+1, through
        // <J,m_a|S_+|J,m_a - 1> = Cmnus(J, m_a), which gives a three-term recurrence for each column:
        //     ( Cmnus(J, m_{a-1}) D(a-1,k) + Cmnus(J, m_a) D(a+1,k) ) / 2 = m_k D(a,k).
        // It is run INWARD from both edges (the stable direction: near the edges each column grows inward),
        // and the two halves meet in the middle row.
        RealMatrix D(Z_, Z_);
        const int mid = twoJ / 2;                   // rows 0..mid from the top, mid+1..2J from the bottom
        for (int k = 0; k < Z_; k++) {
            const double m = spinM(k);
            D(0, k)    = (k % 2 == 0) ? e(k) : -e(k);
            D(twoJ, k) = e(k);
            // From the top:     D(a+1) = ( 2 m D(a) - Cmnus(J, m_{a-1}) D(a-1) ) / Cmnus(J, m_a)
            for (int a = 0; a < mid; a++) {
                const double above = (a > 0) ? Cmnus(J, spinM(a - 1)) * D(a - 1, k) : 0.0;
                D(a + 1, k) = (2 * m * D(a, k) - above) / Cmnus(J, spinM(a));
            }
            // From the bottom:  D(a-1) = ( 2 m D(a) - Cmnus(J, m_a) D(a+1) ) / Cmnus(J, m_{a-1})
            for (int a = twoJ; a > mid + 1; a--) {
                const double below = (a < twoJ) ? Cmnus(J, spinM(a)) * D(a + 1, k) : 0.0;
                D(a - 1, k) = (2 * m * D(a, k) - below) / Cmnus(J, spinM(a - 1));
            }
        }
        // d^J(pi/2) is real; stored as complex to be used with applyRotation.
        return D.cast<Complex>();
    }
};

}  // namespace spin
