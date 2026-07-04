// NimbleCAS freestanding WebAssembly compute kernel.
// @author Olumuyiwa Oluwasanmi
//
// Compiled with clang's native wasm32 target (NO Emscripten, NO libc / libc++):
//   scripts/build_wasm.sh   ->   web/kernel.wasm
//
// It exports a tiny, dependency-free numeric ABI the browser front-end loads to
// sample functions live in-page. This is the FEASIBLE slice of the "WASM" story:
// a freestanding compute kernel. Compiling the FULL modular CAS (which needs
// `import std` + libc++ + oneTBB) to WASM is a separate infrastructure task that
// requires Emscripten plus a wasm-targeted libc++/std module and a TBB-free build.
//
// ABI (all C linkage; f64 == double, i32 == the wasm linear-memory offset):
//   int     coeff_capacity();                             // == kCapacity (256)
//   double* coeff_buffer();                               // offset of a 256-slot
//                                                         // scratch array; JS writes
//                                                         // polynomial coefficients here
//   double  poly_eval(const double* c, int n, double x);  // Horner: sum_{i<n} c[i]*x^i
//   double  poly_eval_buffer(int n, double x);            // poly_eval(coeff_buffer(), n, x)
//
// The linear memory is exported as "memory", so JS can construct a Float64Array
// view over it at coeff_buffer() to load coefficients before calling poly_eval.

extern "C" {

enum : int { kCapacity = 256 };

// BSS scratch buffer (zero-initialised by wasm-ld). Holds coefficients written
// from JS. Low, ascending index = low polynomial degree (c[0] is the constant term).
static double g_coeffs[kCapacity];

int coeff_capacity() { return kCapacity; }

double* coeff_buffer() { return g_coeffs; }

// Horner evaluation of the degree-(n-1) polynomial c[0] + c[1]x + ... + c[n-1]x^{n-1}.
// n is clamped to [0, kCapacity] defensively so a bad length can never read past
// the buffer when called through poly_eval_buffer.
double poly_eval(const double* c, int n, double x) {
    if (n < 0) {
        n = 0;
    }
    double acc = 0.0;
    for (int i = n - 1; i >= 0; --i) {
        acc = acc * x + c[i];
    }
    return acc;
}

// Convenience: evaluate the polynomial currently loaded in the exported scratch
// buffer. n is clamped to the buffer capacity so JS cannot induce an OOB read.
double poly_eval_buffer(int n, double x) {
    if (n > kCapacity) {
        n = kCapacity;
    }
    return poly_eval(g_coeffs, n, x);
}

}  // extern "C"
