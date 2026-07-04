// NimbleCAS wavelet transforms across the modern family (ROADMAP §7 signal layer).
// @author Olumuyiwa Oluwasanmi
//
// Conforms to config/cpp_details.txt: C++23 modules, `import std`, trailing return
// types, no owning raw pointers, std::expected error handling (no exceptions),
// [[nodiscard]] throughout.
//
// ===========================================================================
// HONESTY BOUNDARY — read this before trusting a result.
// ===========================================================================
//   * ONLY the Haar transform (and its lifting realisation) is EXACT over the
//     rationals Q. It uses the UNNORMALISED Haar filters — the sum/difference
//     pair — so the 1/sqrt(2) orthonormal scaling is factored out and applied as
//     an integer (x2) rescale in the lifting. Reconstruction is EXACT: every
//     coefficient is a Rational and haar_idwt(haar_dwt(x)) == x bit-for-bit.
//   * EVERY OTHER family (Daubechies dbN, Symlets symN, Coiflets coifN,
//     biorthogonal bior/rbio) has IRRATIONAL filter coefficients. Those paths are
//     NUMERICAL (double). Perfect reconstruction holds to floating-point round-off,
//     NOT exactly. We do not — and must not — claim exactness for them.
//   * The continuous wavelets (Morlet, Ricker/Mexican-hat, Meyer, Shannon, complex
//     Gaussian) and the CWT are NUMERICAL (double / std::complex<double>).
//   * The biorthogonal transforms here are realised through their LIFTING
//     factorisation (the JPEG2000-style CDF 5/3 and CDF 9/7). Lifting is perfect-
//     reconstruction BY CONSTRUCTION (every predict/update step is exactly
//     invertible), so the round-trip is PR to round-off; for the 5/3 whose lifting
//     weights are dyadic (1/2, 1/4) the double round-trip is in fact bit-exact.
//     The tabulated bior filter PAIRS are exposed via biorthogonal_filters() for
//     inspection / sum-rule checks, but the transform is driven by the lifting.
//   * Directional multiscale systems — CURVELETS and SHEARLETS — need a 2D framing
//     (parabolic scaling, shear/rotation, wedge tiling of the frequency plane) that
//     is out of scope for this first module. They are HONEST not_implemented stubs;
//     they are NOT faked.
//
// Railway-oriented (Rule 32): every fallible entry point returns Result<T>. An
// odd-length signal where an even length is required -> domain_error; a mismatched
// approx/detail pairing -> domain_error; an unknown wavelet order -> not_implemented;
// a non-positive CWT scale -> domain_error.

export module nimblecas.wavelets;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

// ===========================================================================
// Internal (module-linkage) helpers — not exported.
// ===========================================================================
namespace nimblecas::wavelets::detail {

inline constexpr double pi_v = std::numbers::pi;

// Non-negative periodic index: ((a mod n) + n) mod n, returned as an index.
[[nodiscard]] auto pmod(std::int64_t a, std::int64_t n) -> std::size_t {
    const std::int64_t r = ((a % n) + n) % n;
    return static_cast<std::size_t>(r);
}

// Integer power of a double base (m >= 0). ipow(x,0) == 1, including 0^0 == 1.
[[nodiscard]] auto ipow(double base, int m) -> double {
    double acc = 1.0;
    for (int i = 0; i < m; ++i) {
        acc *= base;
    }
    return acc;
}

[[nodiscard]] auto is_power_of_two(std::size_t n) -> bool {
    return n != 0 && (n & (n - 1)) == 0;
}

// Quadrature-mirror high-pass from an orthogonal low-pass: g[k] = (-1)^k h[L-1-k].
[[nodiscard]] auto qmf(std::span<const double> lo) -> std::vector<double> {
    const std::size_t L = lo.size();
    std::vector<double> hi(L);
    for (std::size_t k = 0; k < L; ++k) {
        const double sign = (k % 2 == 0) ? 1.0 : -1.0;
        hi[k] = sign * lo[L - 1 - k];
    }
    return hi;
}

// One periodic analysis step (convolve + decimate by 2). Requires n even.
// a[i] = sum_k lo[k] x[(2i+k) mod n];  d[i] = sum_k hi[k] x[(2i+k) mod n].
[[nodiscard]] auto analysis_step(std::span<const double> x, std::span<const double> lo,
                                 std::span<const double> hi)
    -> std::pair<std::vector<double>, std::vector<double>> {
    const std::size_t n = x.size();
    const std::size_t L = lo.size();
    const std::size_t m = n / 2;
    std::vector<double> a(m, 0.0);
    std::vector<double> d(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        double sa = 0.0;
        double sd = 0.0;
        for (std::size_t k = 0; k < L; ++k) {
            const std::size_t idx = pmod(static_cast<std::int64_t>(2 * i + k),
                                         static_cast<std::int64_t>(n));
            sa = std::fma(lo[k], x[idx], sa);
            sd = std::fma(hi[k], x[idx], sd);
        }
        a[i] = sa;
        d[i] = sd;
    }
    return {std::move(a), std::move(d)};
}

// The transpose (adjoint / scatter) of analysis_step. For an ORTHONORMAL filter
// bank the analysis operator W is orthogonal (W^T W = I on the DFT grid for even n),
// so this transpose is exactly the inverse — perfect reconstruction to round-off.
[[nodiscard]] auto synthesis_step(std::span<const double> a, std::span<const double> d,
                                  std::span<const double> lo, std::span<const double> hi)
    -> std::vector<double> {
    const std::size_t m = a.size();
    const std::size_t n = 2 * m;
    const std::size_t L = lo.size();
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t k = 0; k < L; ++k) {
            const std::size_t idx = pmod(static_cast<std::int64_t>(2 * i + k),
                                         static_cast<std::int64_t>(n));
            x[idx] += lo[k] * a[i] + hi[k] * d[i];
        }
    }
    return x;
}

// ---- Daubechies scaling (low-pass) coefficients, dec_lo, sum == sqrt(2) ----
// db1 and db2 are produced from closed form (exact transcription-free); db3..db10
// are the standard published tables (the same constants PyWavelets ships).
[[nodiscard]] auto daubechies_lo(int order) -> std::optional<std::vector<double>> {
    if (order == 1) {  // Haar
        const double c = 1.0 / std::numbers::sqrt2;
        return std::vector<double>{c, c};
    }
    if (order == 2) {  // closed form: (1±√3)/(4√2), (3±√3)/(4√2)
        const double r3 = std::sqrt(3.0);
        const double den = 4.0 * std::numbers::sqrt2;
        return std::vector<double>{(1.0 - r3) / den, (3.0 - r3) / den,
                                   (3.0 + r3) / den, (1.0 + r3) / den};
    }
    switch (order) {
        case 3:
            return std::vector<double>{
                0.035226291882100656, -0.08544127388224149, -0.13501102001039084,
                0.4598775021193313, 0.8068915093133388, 0.3326705529509569};
        case 4:
            return std::vector<double>{
                -0.010597401784997278, 0.032883011666982945, 0.030841381835986965,
                -0.18703481171888114, -0.02798376941698385, 0.6308807679295904,
                0.7148465705525415, 0.23037781330885523};
        case 5:
            return std::vector<double>{
                0.003335725285001549, -0.012580751999015526, -0.006241490213011705,
                0.07757149384006515, -0.03224486958502952, -0.24229488706619015,
                0.13842814590110342, 0.7243085284385744, 0.6038292697974729,
                0.160102397974125};
        case 6:
            return std::vector<double>{
                -0.00107730108499558, 0.004777257511010651, 0.0005538422009938016,
                -0.031582039318031156, 0.02752286553001629, 0.09750160558707936,
                -0.12976686756709563, -0.22626469396516913, 0.3152503517092432,
                0.7511339080215775, 0.4946238903983854, 0.11154074335008017};
        case 7:
            return std::vector<double>{
                0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274,
                0.012550998556013784, -0.01657454163101562, -0.03802993693503463,
                0.0806126091510659, 0.07130921926705004, -0.22403618499416572,
                -0.14390600392910627, 0.4697822874053586, 0.7291320908465551,
                0.39653931948230575, 0.07785205408506236};
        case 8:
            return std::vector<double>{
                -0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771,
                -0.00487035299301066, 0.008746094047015655, 0.013981027917015516,
                -0.04408825393106472, -0.01736930100202211, 0.128747426620186,
                0.00047248457399797254, -0.2840155429624281, -0.015829105256023893,
                0.5853546836548691, 0.6756307362980128, 0.3128715909144659,
                0.05441584224308161};
        case 9:
            return std::vector<double>{
                3.9347319995026124e-05, -0.0002519631889981789, 0.00023038576399541288,
                0.0018476468829611268, -0.004281503681904723, -0.004723204757894831,
                0.022361662123515244, 0.00025094711499193845, -0.06763282905952399,
                0.030725681478322865, 0.14854074933476008, -0.09684078322087904,
                -0.29327378327258685, 0.13319738582208895, 0.6572880780366389,
                0.6048231236767786, 0.24383467463766728, 0.03807794736316728};
        case 10:
            return std::vector<double>{
                -1.326420300235487e-05, 9.358867000108985e-05, -0.0001164668549943862,
                -0.0006858566950046825, 0.00199240529499085, 0.0013953517469940798,
                -0.010733175482979604, 0.0036065535669883944, 0.03321267405893324,
                -0.02945753682194567, -0.07139414716586077, 0.09305736460380659,
                0.12736934033574265, -0.19594627437659665, -0.24984642432648865,
                0.2811723436605715, 0.6884590394525921, 0.5272011889309198,
                0.18817680007762133, 0.026670057900950818};
        default:
            return std::nullopt;
    }
}

// ---- Symlet scaling (low-pass) coefficients, dec_lo (sym2==db2, sym3==db3) ----
[[nodiscard]] auto symlet_lo(int order) -> std::optional<std::vector<double>> {
    if (order == 2 || order == 3) {
        return daubechies_lo(order);
    }
    switch (order) {
        case 4:
            return std::vector<double>{
                -0.07576571478927333, -0.02963552764599851, 0.49761866763201545,
                0.8037387518059161, 0.29785779560527736, -0.09921954357684722,
                -0.012603967262037833, 0.0322231006040427};
        case 5:
            return std::vector<double>{
                0.027333068345077982, 0.029519490925774643, -0.039134249302383094,
                0.1993975339773936, 0.7234076904024206, 0.6339789634582119,
                0.01660210576452232, -0.17532808990845047, -0.021101834024758855,
                0.019538882735286728};
        case 6:
            return std::vector<double>{
                0.015404109327027373, 0.0034907120842174702, -0.11799011114819057,
                -0.048311742585633, 0.4910559419267466, 0.787641141030194,
                0.3379294217276218, -0.07263752278646252, -0.021060292512300564,
                0.04472490177066578, 0.0017677118642428036, -0.007800708325034148};
        case 7:
            return std::vector<double>{
                0.002681814568257878, -0.0010473848886829163, -0.01263630340325193,
                0.03051551316596357, 0.0678926935013727, -0.049552834937127255,
                0.017441255086855827, 0.5361019170917628, 0.767764317003164,
                0.2886296317515146, -0.14004724044296152, -0.10780823770381774,
                0.004010244871533663, 0.010268176708511255};
        case 8:
            return std::vector<double>{
                -0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298,
                0.007607487324917605, -0.1432942383508097, -0.061273359067658524,
                0.4813596512583722, 0.7771857517005235, 0.3644418948353314,
                -0.05194583810770904, -0.027219029917056003, 0.049137179673607506,
                0.003808752013890615, -0.01495225833704823, -0.0003029205147213668,
                0.0018899503327594609};
        default:
            return std::nullopt;
    }
}

// ---- Coiflet scaling (low-pass) coefficients, dec_lo (coif1..coif3) ----
[[nodiscard]] auto coiflet_lo(int order) -> std::optional<std::vector<double>> {
    switch (order) {
        case 1:
            return std::vector<double>{
                -0.01565572813546454, -0.0727326195128539, 0.38486484686420286,
                0.8525720202122554, 0.3378976624578092, -0.07273261951285821};
        case 2:
            return std::vector<double>{
                -0.0007205494453645122, -0.0018232088707029932, 0.0056114348193944995,
                0.023680171946334084, -0.0594344186464569, -0.0764885990783064,
                0.41700518442169254, 0.8127236354455423, 0.3861100668211622,
                -0.06737255472196302, -0.04146493678175915, 0.016387336463522112};
        case 3:
            return std::vector<double>{
                -3.459977283621256e-05, -7.098330313814125e-05, 0.0004662169601128863,
                0.0011175187708906016, -0.0025745176887502236, -0.00900797613666158,
                0.015880544863615904, 0.03455502757306163, -0.08230192710688598,
                -0.07179982161931202, 0.42848347637761874, 0.7937772226256206,
                0.4051769024096169, -0.06112339000267287, -0.0657719112818555,
                0.023452696141836267, 0.007782596427325418, -0.003793512864491014};
        default:
            return std::nullopt;
    }
}

// Meyer frequency window nu-smoothed magnitude |hat_psi(w)| (phase e^{iw/2} dropped).
[[nodiscard]] auto meyer_window(double w) -> double {
    const double a = std::abs(w);
    const double lo = 2.0 * pi_v / 3.0;
    const double mid = 4.0 * pi_v / 3.0;
    const double hi = 8.0 * pi_v / 3.0;
    if (a < lo || a > hi) {
        return 0.0;
    }
    const auto nu = [](double x) {
        return x * x * x * x * (35.0 - 84.0 * x + 70.0 * x * x - 20.0 * x * x * x);
    };
    if (a <= mid) {
        return std::sin(pi_v / 2.0 * nu(3.0 * a / (2.0 * pi_v) - 1.0));
    }
    return std::cos(pi_v / 2.0 * nu(3.0 * a / (4.0 * pi_v) - 1.0));
}

}  // namespace nimblecas::wavelets::detail

// ===========================================================================
// Public API.
// ===========================================================================
export namespace nimblecas::wavelets {

// ---------------------------------------------------------------------------
// HAAR — EXACT over the rationals Q (unnormalised sum/difference filters).
// ---------------------------------------------------------------------------
// Single decimated level: approx[i] = x[2i] + x[2i+1] (sum, the low band),
// detail[i] = x[2i] - x[2i+1] (difference, the high band). Keeping the 1/sqrt(2)
// factor out of the filters is exactly what makes the transform stay in Q.
struct HaarLevel {
    std::vector<Rational> approx;
    std::vector<Rational> detail;
};

[[nodiscard]] auto haar_dwt(std::span<const Rational> signal) -> Result<HaarLevel> {
    const std::size_t n = signal.size();
    if (n == 0 || (n % 2) != 0) {
        return make_error<HaarLevel>(MathError::domain_error);  // need a non-empty even length
    }
    HaarLevel out;
    out.approx.reserve(n / 2);
    out.detail.reserve(n / 2);
    for (std::size_t i = 0; i < n / 2; ++i) {
        auto s = signal[2 * i].add(signal[2 * i + 1]);
        if (!s) {
            return make_error<HaarLevel>(s.error());
        }
        auto d = signal[2 * i].subtract(signal[2 * i + 1]);
        if (!d) {
            return make_error<HaarLevel>(d.error());
        }
        out.approx.push_back(*s);
        out.detail.push_back(*d);
    }
    return out;
}

// Exact inverse: x[2i] = (approx[i] + detail[i]) / 2, x[2i+1] = (approx[i] - detail[i]) / 2.
// The /2 is exact in Q, so haar_idwt(haar_dwt(x)) == x exactly.
[[nodiscard]] auto haar_idwt(std::span<const Rational> approx, std::span<const Rational> detail)
    -> Result<std::vector<Rational>> {
    if (approx.size() != detail.size()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    const Rational two = Rational::from_int(2);
    const std::size_t m = approx.size();
    std::vector<Rational> x(2 * m);
    for (std::size_t i = 0; i < m; ++i) {
        auto sum = approx[i].add(detail[i]);  // = 2 x[2i]
        if (!sum) {
            return make_error<std::vector<Rational>>(sum.error());
        }
        auto diff = approx[i].subtract(detail[i]);  // = 2 x[2i+1]
        if (!diff) {
            return make_error<std::vector<Rational>>(diff.error());
        }
        auto lo = sum->divide(two);
        if (!lo) {
            return make_error<std::vector<Rational>>(lo.error());
        }
        auto hi = diff->divide(two);
        if (!hi) {
            return make_error<std::vector<Rational>>(hi.error());
        }
        x[2 * i] = *lo;
        x[2 * i + 1] = *hi;
    }
    return x;
}

// Multi-level (Mallat pyramid): recursively transform the approximation band.
// details[0] is the finest (level-1) detail; approx is the final coarse band.
struct HaarDecomposition {
    std::vector<std::vector<Rational>> details;  // details[0] = finest ... last = coarsest
    std::vector<Rational> approx;                // final approximation band
};

[[nodiscard]] auto haar_dwt_multi(std::span<const Rational> signal, std::size_t levels)
    -> Result<HaarDecomposition> {
    if (levels == 0 || signal.empty()) {
        return make_error<HaarDecomposition>(MathError::domain_error);
    }
    // The signal length must be divisible by 2^levels (each level halves it).
    if (levels >= 63 || (signal.size() % (std::size_t{1} << levels)) != 0) {
        return make_error<HaarDecomposition>(MathError::domain_error);
    }
    HaarDecomposition out;
    std::vector<Rational> current(signal.begin(), signal.end());
    for (std::size_t l = 0; l < levels; ++l) {
        auto lvl = haar_dwt(current);
        if (!lvl) {
            return make_error<HaarDecomposition>(lvl.error());
        }
        out.details.push_back(std::move(lvl->detail));
        current = std::move(lvl->approx);
    }
    out.approx = std::move(current);
    return out;
}

[[nodiscard]] auto haar_idwt_multi(const HaarDecomposition& decomp)
    -> Result<std::vector<Rational>> {
    std::vector<Rational> current = decomp.approx;
    for (std::size_t l = decomp.details.size(); l-- > 0;) {
        auto rec = haar_idwt(current, decomp.details[l]);
        if (!rec) {
            return rec;
        }
        current = std::move(*rec);
    }
    return current;
}

// ---------------------------------------------------------------------------
// HAAR via the LIFTING scheme (second generation) — EXACT over Q.
// ---------------------------------------------------------------------------
// Split / predict / update with a final integer rescale so the output is BIT-FOR-BIT
// identical to the sum/difference filter bank above:
//   split:   e = x[2i], o = x[2i+1]
//   predict: o <- o - e            (=> b - a)
//   update:  e <- e + o/2          (=> (a + b) / 2)
//   rescale: approx = 2 e = a + b,  detail = -o = a - b
// Every step is exactly invertible in Q, so this is perfect reconstruction and,
// by design, haar_lifting_forward == haar_dwt exactly.
[[nodiscard]] auto haar_lifting_forward(std::span<const Rational> signal) -> Result<HaarLevel> {
    const std::size_t n = signal.size();
    if (n == 0 || (n % 2) != 0) {
        return make_error<HaarLevel>(MathError::domain_error);
    }
    const Rational two = Rational::from_int(2);
    HaarLevel out;
    out.approx.reserve(n / 2);
    out.detail.reserve(n / 2);
    for (std::size_t i = 0; i < n / 2; ++i) {
        const Rational a = signal[2 * i];
        const Rational b = signal[2 * i + 1];
        auto o = b.subtract(a);  // predict: o = b - a
        if (!o) {
            return make_error<HaarLevel>(o.error());
        }
        auto half = o->divide(two);  // o/2
        if (!half) {
            return make_error<HaarLevel>(half.error());
        }
        auto e = a.add(*half);  // update: e = (a + b)/2
        if (!e) {
            return make_error<HaarLevel>(e.error());
        }
        auto approx = e->multiply(two);  // rescale: 2e = a + b
        if (!approx) {
            return make_error<HaarLevel>(approx.error());
        }
        auto detail = o->negate();  // rescale: -o = a - b
        if (!detail) {
            return make_error<HaarLevel>(detail.error());
        }
        out.approx.push_back(*approx);
        out.detail.push_back(*detail);
    }
    return out;
}

[[nodiscard]] auto haar_lifting_inverse(std::span<const Rational> approx,
                                        std::span<const Rational> detail)
    -> Result<std::vector<Rational>> {
    if (approx.size() != detail.size()) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    const Rational two = Rational::from_int(2);
    const std::size_t m = approx.size();
    std::vector<Rational> x(2 * m);
    for (std::size_t i = 0; i < m; ++i) {
        auto e = approx[i].divide(two);  // undo rescale: e = approx/2 = (a+b)/2
        if (!e) {
            return make_error<std::vector<Rational>>(e.error());
        }
        auto o = detail[i].negate();  // undo rescale: o = -detail = b - a
        if (!o) {
            return make_error<std::vector<Rational>>(o.error());
        }
        auto half = o->divide(two);  // o/2
        if (!half) {
            return make_error<std::vector<Rational>>(half.error());
        }
        auto a = e->subtract(*half);  // undo update: a = e - o/2
        if (!a) {
            return make_error<std::vector<Rational>>(a.error());
        }
        auto b = o->add(*a);  // undo predict: b = o + a
        if (!b) {
            return make_error<std::vector<Rational>>(b.error());
        }
        x[2 * i] = *a;
        x[2 * i + 1] = *b;
    }
    return x;
}

// ---------------------------------------------------------------------------
// NUMERICAL orthogonal filter banks (Daubechies / Symlet / Coiflet).
// ---------------------------------------------------------------------------
// analysis_lo/analysis_hi are the decomposition filters; synthesis_lo/synthesis_hi
// are the reconstruction filters used by the transpose-convolution synthesis. For an
// ORTHOGONAL wavelet the synthesis filters equal the analysis filters (the synthesis
// is the adjoint of analysis). Coefficients are IRRATIONAL — this is numerical.
struct FilterBank {
    std::vector<double> analysis_lo;
    std::vector<double> analysis_hi;
    std::vector<double> synthesis_lo;
    std::vector<double> synthesis_hi;
};

enum class Family : std::uint8_t { daubechies, symlet, coiflet };

// Build an orthogonal filter bank from a low-pass analysis filter (dec_lo).
[[nodiscard]] auto make_orthogonal(std::vector<double> lo) -> FilterBank {
    std::vector<double> hi = detail::qmf(lo);
    FilterBank fb;
    fb.analysis_lo = lo;
    fb.analysis_hi = hi;
    fb.synthesis_lo = std::move(lo);  // orthogonal: synthesis == analysis (adjoint)
    fb.synthesis_hi = std::move(hi);
    return fb;
}

[[nodiscard]] auto daubechies(int order) -> Result<FilterBank> {
    auto lo = detail::daubechies_lo(order);
    if (!lo) {
        return make_error<FilterBank>(MathError::not_implemented);
    }
    return make_orthogonal(std::move(*lo));
}

[[nodiscard]] auto symlet(int order) -> Result<FilterBank> {
    auto lo = detail::symlet_lo(order);
    if (!lo) {
        return make_error<FilterBank>(MathError::not_implemented);
    }
    return make_orthogonal(std::move(*lo));
}

[[nodiscard]] auto coiflet(int order) -> Result<FilterBank> {
    auto lo = detail::coiflet_lo(order);
    if (!lo) {
        return make_error<FilterBank>(MathError::not_implemented);
    }
    return make_orthogonal(std::move(*lo));
}

[[nodiscard]] auto orthogonal_filters(Family fam, int order) -> Result<FilterBank> {
    switch (fam) {
        case Family::daubechies: return daubechies(order);
        case Family::symlet:     return symlet(order);
        case Family::coiflet:    return coiflet(order);
    }
    return make_error<FilterBank>(MathError::not_implemented);
}

// ---- sum-rule / orthogonality diagnostics (all NUMERICAL) ----

// Low-pass sum rule: Sum h_k should equal sqrt(2) for a normalised scaling filter.
[[nodiscard]] auto lowpass_sum(std::span<const double> h) -> double {
    double s = 0.0;
    for (double v : h) {
        s += v;
    }
    return s;
}

// Vanishing-moment test on the LOW-PASS filter as the task states it:
//   Sum_k (-1)^k k^m h_k == 0 for m = 0 .. p-1 (p vanishing moments).
[[nodiscard]] auto lowpass_alternating_moment(std::span<const double> h, int m) -> double {
    double s = 0.0;
    for (std::size_t k = 0; k < h.size(); ++k) {
        const double sign = (k % 2 == 0) ? 1.0 : -1.0;
        s += sign * detail::ipow(static_cast<double>(k), m) * h[k];
    }
    return s;
}

// Equivalent high-pass moment: Sum_k k^m g_k, which vanishes for m = 0 .. p-1.
[[nodiscard]] auto highpass_moment(std::span<const double> g, int m) -> double {
    double s = 0.0;
    for (std::size_t k = 0; k < g.size(); ++k) {
        s += detail::ipow(static_cast<double>(k), m) * g[k];
    }
    return s;
}

// Double-shift autocorrelation of the low-pass minus the Kronecker delta:
//   ( Sum_k h_k h_{k+2m} ) - delta_{m,0}. Should be ~0 for an orthonormal filter.
[[nodiscard]] auto orthogonality_defect(std::span<const double> h, int m) -> double {
    double s = 0.0;
    const auto L = static_cast<std::int64_t>(h.size());
    for (std::int64_t k = 0; k < L; ++k) {
        const std::int64_t j = k + 2 * m;
        if (j >= 0 && j < L) {
            s += h[static_cast<std::size_t>(k)] * h[static_cast<std::size_t>(j)];
        }
    }
    return s - (m == 0 ? 1.0 : 0.0);
}

// ---------------------------------------------------------------------------
// NUMERICAL decimated DWT / IDWT (periodic Mallat filter bank).
// ---------------------------------------------------------------------------
struct DwtLevel {
    std::vector<double> approx;
    std::vector<double> detail;
};

[[nodiscard]] auto dwt(std::span<const double> signal, const FilterBank& fb) -> Result<DwtLevel> {
    const std::size_t n = signal.size();
    if (n == 0 || (n % 2) != 0) {
        return make_error<DwtLevel>(MathError::domain_error);
    }
    auto [a, d] = detail::analysis_step(signal, fb.analysis_lo, fb.analysis_hi);
    return DwtLevel{.approx = std::move(a), .detail = std::move(d)};
}

[[nodiscard]] auto idwt(std::span<const double> approx, std::span<const double> detail_band,
                        const FilterBank& fb) -> Result<std::vector<double>> {
    if (approx.size() != detail_band.size()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    return detail::synthesis_step(approx, detail_band, fb.synthesis_lo, fb.synthesis_hi);
}

struct Decomposition {
    std::vector<std::vector<double>> details;  // finest .. coarsest
    std::vector<double> approx;
};

[[nodiscard]] auto dwt_multi(std::span<const double> signal, const FilterBank& fb,
                             std::size_t levels) -> Result<Decomposition> {
    if (levels == 0 || signal.empty()) {
        return make_error<Decomposition>(MathError::domain_error);
    }
    if (levels >= 63 || (signal.size() % (std::size_t{1} << levels)) != 0) {
        return make_error<Decomposition>(MathError::domain_error);
    }
    Decomposition out;
    std::vector<double> current(signal.begin(), signal.end());
    for (std::size_t l = 0; l < levels; ++l) {
        auto lvl = dwt(current, fb);
        if (!lvl) {
            return make_error<Decomposition>(lvl.error());
        }
        out.details.push_back(std::move(lvl->detail));
        current = std::move(lvl->approx);
    }
    out.approx = std::move(current);
    return out;
}

[[nodiscard]] auto idwt_multi(const Decomposition& decomp, const FilterBank& fb)
    -> Result<std::vector<double>> {
    std::vector<double> current = decomp.approx;
    for (std::size_t l = decomp.details.size(); l-- > 0;) {
        auto rec = idwt(current, decomp.details[l], fb);
        if (!rec) {
            return rec;
        }
        current = std::move(*rec);
    }
    return current;
}

// ---------------------------------------------------------------------------
// STATIONARY / undecimated wavelet transform (SWT, a-trous) — NUMERICAL.
// ---------------------------------------------------------------------------
// No decimation: at level L the filters are dilated by inserting 2^(L-1)-1 zeros
// between taps (the "a trous" holes). Both output bands keep the FULL input length
// n — the defining length-invariance of the SWT.
struct SwtLevel {
    std::vector<double> approx;
    std::vector<double> detail;
};

[[nodiscard]] auto swt(std::span<const double> signal, const FilterBank& fb,
                       std::size_t level = 1) -> Result<SwtLevel> {
    const std::size_t n = signal.size();
    if (n == 0 || level == 0 || level >= 31) {
        return make_error<SwtLevel>(MathError::domain_error);
    }
    const std::int64_t dilation = std::int64_t{1} << (level - 1);
    const std::size_t L = fb.analysis_lo.size();
    SwtLevel out;
    out.approx.assign(n, 0.0);
    out.detail.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sa = 0.0;
        double sd = 0.0;
        for (std::size_t k = 0; k < L; ++k) {
            const std::size_t idx = detail::pmod(
                static_cast<std::int64_t>(i) + static_cast<std::int64_t>(k) * dilation,
                static_cast<std::int64_t>(n));
            sa = std::fma(fb.analysis_lo[k], signal[idx], sa);
            sd = std::fma(fb.analysis_hi[k], signal[idx], sd);
        }
        out.approx[i] = sa;
        out.detail[i] = sd;
    }
    return out;
}

// ---------------------------------------------------------------------------
// WAVELET PACKET decomposition — NUMERICAL.
// ---------------------------------------------------------------------------
// Full binary tree: at each level BOTH the approximation and the detail band are
// split again. Returns the 2^level leaf subbands, left-to-right in natural order,
// each of length n / 2^level.
[[nodiscard]] auto wavelet_packet(std::span<const double> signal, const FilterBank& fb,
                                  std::size_t level) -> Result<std::vector<std::vector<double>>> {
    if (level == 0 || signal.empty()) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    if (level >= 63 || (signal.size() % (std::size_t{1} << level)) != 0) {
        return make_error<std::vector<std::vector<double>>>(MathError::domain_error);
    }
    std::vector<std::vector<double>> bands;
    bands.emplace_back(signal.begin(), signal.end());
    for (std::size_t l = 0; l < level; ++l) {
        std::vector<std::vector<double>> next;
        next.reserve(bands.size() * 2);
        for (const auto& band : bands) {
            auto [a, d] = detail::analysis_step(band, fb.analysis_lo, fb.analysis_hi);
            next.push_back(std::move(a));
            next.push_back(std::move(d));
        }
        bands = std::move(next);
    }
    return bands;
}

// ---------------------------------------------------------------------------
// BIORTHOGONAL transforms via the LIFTING factorisation — NUMERICAL, PR by design.
// ---------------------------------------------------------------------------
// CDF 5/3 (bior2.2 / LeGall, the JPEG2000 reversible filter). Periodic lifting:
//   predict: d[i] = o[i] - (e[i] + e[i+1]) / 2
//   update:  s[i] = e[i] + (d[i-1] + d[i]) / 4
// Every step is exactly invertible; the weights are dyadic so the double round-trip
// is bit-exact. This is a genuine BIORTHOGONAL (not orthogonal) wavelet.
[[nodiscard]] auto cdf53_forward(std::span<const double> signal) -> Result<DwtLevel> {
    const std::size_t n = signal.size();
    if (n == 0 || (n % 2) != 0) {
        return make_error<DwtLevel>(MathError::domain_error);
    }
    const std::size_t m = n / 2;
    std::vector<double> e(m);
    std::vector<double> o(m);
    for (std::size_t i = 0; i < m; ++i) {
        e[i] = signal[2 * i];
        o[i] = signal[2 * i + 1];
    }
    std::vector<double> d(m);
    for (std::size_t i = 0; i < m; ++i) {
        d[i] = o[i] - 0.5 * (e[i] + e[(i + 1) % m]);  // predict
    }
    std::vector<double> s(m);
    for (std::size_t i = 0; i < m; ++i) {
        s[i] = e[i] + 0.25 * (d[(i + m - 1) % m] + d[i]);  // update
    }
    return DwtLevel{.approx = std::move(s), .detail = std::move(d)};
}

[[nodiscard]] auto cdf53_inverse(std::span<const double> approx, std::span<const double> detail_band)
    -> Result<std::vector<double>> {
    if (approx.size() != detail_band.size()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::size_t m = approx.size();
    std::vector<double> e(m);
    std::vector<double> o(m);
    for (std::size_t i = 0; i < m; ++i) {
        e[i] = approx[i] - 0.25 * (detail_band[(i + m - 1) % m] + detail_band[i]);  // undo update
    }
    for (std::size_t i = 0; i < m; ++i) {
        o[i] = detail_band[i] + 0.5 * (e[i] + e[(i + 1) % m]);  // undo predict
    }
    std::vector<double> x(2 * m);
    for (std::size_t i = 0; i < m; ++i) {
        x[2 * i] = e[i];
        x[2 * i + 1] = o[i];
    }
    return x;
}

// Tabulated bior/rbio analysis+synthesis filter PAIRS for inspection (sum-rule /
// documentation). Currently the CDF 5/3 (bior2.2) pair; other orders are honestly
// reported as not_implemented rather than faked. Note these are provided as data;
// the working biorthogonal transform is cdf53_forward/inverse (the lifting).
[[nodiscard]] auto biorthogonal_filters(int recon_order, int decon_order) -> Result<FilterBank> {
    if (recon_order == 2 && decon_order == 2) {  // bior2.2 == CDF 5/3
        constexpr double h = 0.35355339059327373;  // 1/(2 sqrt(2))
        constexpr double q = 0.17677669529663687;  // 1/(4 sqrt(2))
        constexpr double t = 1.0606601717798212;   // 3/(2 sqrt(2))
        FilterBank fb;
        fb.analysis_lo = {0.0, -q, h, t, h, -q};
        fb.analysis_hi = {0.0, h, -2.0 * h, h, 0.0, 0.0};
        fb.synthesis_lo = {0.0, h, 2.0 * h, h, 0.0, 0.0};
        fb.synthesis_hi = {0.0, -q, -h, t, -h, -q};
        return fb;
    }
    return make_error<FilterBank>(MathError::not_implemented);
}

// ---------------------------------------------------------------------------
// CONTINUOUS wavelets psi(t) — NUMERICAL — and the CWT.
// ---------------------------------------------------------------------------

// Complex Morlet: psi(t) = pi^{-1/4} (e^{i w0 t} - e^{-w0^2/2}) e^{-t^2/2}. The
// second term enforces the (near-)zero-mean admissibility condition.
[[nodiscard]] auto morlet(double t, double omega0 = 5.0) -> std::complex<double> {
    const double c = std::pow(detail::pi_v, -0.25);
    const double gauss = std::exp(-0.5 * t * t);
    const double correction = std::exp(-0.5 * omega0 * omega0);
    const std::complex<double> osc{std::cos(omega0 * t), std::sin(omega0 * t)};
    return c * (osc - correction) * gauss;
}

// Real Morlet: the real part of the complex Morlet.
[[nodiscard]] auto morlet_real(double t, double omega0 = 5.0) -> double {
    return morlet(t, omega0).real();
}

// Ricker / Mexican-hat (the negative second derivative of a Gaussian):
//   psi(t) = 2/(sqrt(3 sigma) pi^{1/4}) (1 - (t/sigma)^2) e^{-t^2/(2 sigma^2)}.
[[nodiscard]] auto ricker(double t, double sigma = 1.0) -> double {
    const double norm = 2.0 / (std::sqrt(3.0 * sigma) * std::pow(detail::pi_v, 0.25));
    const double x = t / sigma;
    return norm * (1.0 - x * x) * std::exp(-0.5 * x * x);
}

// Shannon (sinc / band-pass) wavelet: psi(t) = (sin(2 pi t) - sin(pi t)) / (pi t),
// with the removable singularity psi(0) = 1.
[[nodiscard]] auto shannon(double t) -> double {
    if (std::abs(t) < 1e-12) {
        return 1.0;
    }
    return (std::sin(2.0 * detail::pi_v * t) - std::sin(detail::pi_v * t)) / (detail::pi_v * t);
}

// Meyer frequency-domain magnitude window |hat_psi(w)| (the nu-smoothed band on
// [2pi/3, 8pi/3]); the e^{iw/2} phase is dropped. Exposed for spectral inspection.
[[nodiscard]] auto meyer_hat(double omega) -> double {
    return detail::meyer_window(omega);
}

// Meyer wavelet psi(t), obtained by a bounded numerical quadrature of the inverse
// Fourier transform of the (band-limited, real-valued after phase folding) window:
//   psi(t) = (1/pi) integral_0^{8pi/3} M(w) cos(w (t + 1/2)) dw.
// NUMERICAL: the value is a quadrature approximation, not a closed form.
[[nodiscard]] auto meyer(double t) -> double {
    constexpr int steps = 4000;
    const double upper = 8.0 * detail::pi_v / 3.0;
    const double dw = upper / static_cast<double>(steps);
    double acc = 0.0;
    for (int i = 0; i <= steps; ++i) {
        const double w = static_cast<double>(i) * dw;
        const double weight = (i == 0 || i == steps) ? 0.5 : 1.0;  // trapezoidal
        acc += weight * detail::meyer_window(w) * std::cos(w * (t + 0.5));
    }
    return acc * dw / detail::pi_v;
}

// First-order complex Gaussian wavelet cgau1: the (normalised) first derivative of
// the modulated Gaussian g(t) = e^{-t^2} e^{-i t}, i.e. (-i - 2t) e^{-t^2} e^{-i t}.
[[nodiscard]] auto complex_gaussian(double t) -> std::complex<double> {
    const double c = std::sqrt(2.0) * std::pow(detail::pi_v, -0.25);
    const std::complex<double> phase{std::cos(t), -std::sin(t)};  // e^{-i t}
    const std::complex<double> factor{-2.0 * t, -1.0};            // (-i - 2t)
    return c * factor * std::exp(-t * t) * phase;
}

// Continuous wavelet transform over a scale grid:
//   W(a, b) = (1/sqrt(a)) Sum_t x[t] conj(psi((t - b)/a))
// evaluated at integer translations b = 0 .. n-1. Row i corresponds to scales[i];
// each row has n columns. NUMERICAL. A non-positive scale -> domain_error.
[[nodiscard]] auto cwt(std::span<const double> signal, std::span<const double> scales,
                       const std::function<std::complex<double>(double)>& psi)
    -> Result<std::vector<std::vector<std::complex<double>>>> {
    const std::size_t n = signal.size();
    if (n == 0 || scales.empty()) {
        return make_error<std::vector<std::vector<std::complex<double>>>>(MathError::domain_error);
    }
    std::vector<std::vector<std::complex<double>>> out(scales.size());
    for (std::size_t si = 0; si < scales.size(); ++si) {
        const double a = scales[si];
        if (!(a > 0.0)) {
            return make_error<std::vector<std::vector<std::complex<double>>>>(
                MathError::domain_error);
        }
        const double inv_sqrt_a = 1.0 / std::sqrt(a);
        std::vector<std::complex<double>> row(n);
        for (std::size_t b = 0; b < n; ++b) {
            std::complex<double> acc{0.0, 0.0};
            for (std::size_t t = 0; t < n; ++t) {
                const double arg = (static_cast<double>(t) - static_cast<double>(b)) / a;
                acc += signal[t] * std::conj(psi(arg));
            }
            row[b] = inv_sqrt_a * acc;
        }
        out[si] = std::move(row);
    }
    return out;
}

// ---------------------------------------------------------------------------
// DIRECTIONAL multiscale systems — HONEST not_implemented (no faking).
// ---------------------------------------------------------------------------
// Curvelets and shearlets require a genuine 2D framing (parabolic scaling, shear /
// rotation, and a wedge tiling of the frequency plane) beyond this first 1D-centric
// module. They are declared here so the API is discoverable, and they return
// not_implemented rather than a fabricated result. `data` is a row-major rows x cols
// image; the intended output would be the directional coefficient subbands.
[[nodiscard]] auto curvelet_transform_2d([[maybe_unused]] std::span<const double> data,
                                         [[maybe_unused]] std::size_t rows,
                                         [[maybe_unused]] std::size_t cols)
    -> Result<std::vector<std::vector<double>>> {
    return make_error<std::vector<std::vector<double>>>(MathError::not_implemented);
}

[[nodiscard]] auto shearlet_transform_2d([[maybe_unused]] std::span<const double> data,
                                         [[maybe_unused]] std::size_t rows,
                                         [[maybe_unused]] std::size_t cols)
    -> Result<std::vector<std::vector<double>>> {
    return make_error<std::vector<std::vector<double>>>(MathError::not_implemented);
}

}  // namespace nimblecas::wavelets
