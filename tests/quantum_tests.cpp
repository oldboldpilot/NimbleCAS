// Tests for nimblecas.quantum: exact symbolic non-commutative operator algebra.
// @author Olumuyiwa Oluwasanmi
//
// Covers the Lie-algebra identities (bilinearity, antisymmetry, [A,A]=0, Leibniz, Jacobi)
// that must fall out of normal_form on commutators, a canonical-commutation example, the
// dagger/adjoint rules (order reversal, conjugation of i, involution, self-adjoint,
// bra<->ket), non-commutativity, and normal-form collection of like monomials with EXACT
// Gaussian-rational coefficients.

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.complex;
import nimblecas.quantum;
import nimblecas.testing;

using nimblecas::Complex;
using nimblecas::Op;
using nimblecas::Rational;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace nq = nimblecas;

namespace {

[[nodiscard]] auto ci(std::int64_t re, std::int64_t im) -> Complex {
    return Complex::make(Rational::from_int(re), Rational::from_int(im));
}
[[nodiscard]] auto rat(std::int64_t n, std::int64_t d) -> Complex {
    return Complex::from_real(Rational::make(n, d).value());
}

// The canonical test: two operators are equal iff their normal forms are structurally equal.
[[nodiscard]] auto nf_eq(const Op& a, const Op& b) -> bool {
    auto na = nq::normal_form(a);
    auto nb = nq::normal_form(b);
    return na.has_value() && nb.has_value() && na->is_equivalent_to(*nb);
}

// The zero operator (empty normal form).
[[nodiscard]] auto is_zero_op(const Op& a) -> bool {
    return nf_eq(a, nq::op_scalar(Complex{}));
}

}  // namespace

auto main() -> int {
    const Op A = nq::op_symbol("A");
    const Op B = nq::op_symbol("B");
    const Op C = nq::op_symbol("C");

    return TestSuite("nimblecas.quantum")
        .test("non_commutativity",
              [&](TestContext& t) {
                  // AB and BA are distinct in general (order matters).
                  t.expect(!nf_eq(nq::multiply(A, B), nq::multiply(B, A)),
                           "AB != BA for generic non-commuting symbols");
                  // But a scalar commutes with everything: (cA) == (Ac) in the algebra.
                  const Complex c = ci(3, 0);
                  t.expect(nf_eq(nq::multiply(nq::op_scalar(c), A),
                                 nq::multiply(A, nq::op_scalar(c))),
                           "scalars commute: (c I)A == A(c I)");
                  // The identity commutes: I A == A I == A.
                  t.expect(nf_eq(nq::multiply(nq::identity(), A), A), "I A == A");
                  t.expect(nf_eq(nq::multiply(A, nq::identity()), A), "A I == A");
              })
        .test("commutator_antisymmetry_and_self",
              [&](TestContext& t) {
                  // [A,B] = -[B,A].
                  t.expect(nf_eq(nq::commutator(A, B), nq::negate(nq::commutator(B, A))),
                           "[A,B] = -[B,A]");
                  // [A,A] = 0.
                  t.expect(is_zero_op(nq::commutator(A, A)), "[A,A] = 0");
                  // [A,B] + [B,A] = 0.
                  t.expect(is_zero_op(nq::add(nq::commutator(A, B), nq::commutator(B, A))),
                           "[A,B] + [B,A] = 0");
              })
        .test("commutator_bilinearity",
              [&](TestContext& t) {
                  const Complex a = ci(2, 1);   // 2 + i
                  const Complex b = ci(0, -3);  // -3i
                  // Left-linearity: [aA + bB, C] = a[A,C] + b[B,C].
                  const Op lhs =
                      nq::commutator(nq::add(nq::scale(a, A), nq::scale(b, B)), C);
                  const Op rhs = nq::add(nq::scale(a, nq::commutator(A, C)),
                                         nq::scale(b, nq::commutator(B, C)));
                  t.expect(nf_eq(lhs, rhs), "[aA+bB, C] = a[A,C] + b[B,C]");
                  // Right-linearity: [C, aA + bB] = a[C,A] + b[C,B].
                  const Op lhs2 =
                      nq::commutator(C, nq::add(nq::scale(a, A), nq::scale(b, B)));
                  const Op rhs2 = nq::add(nq::scale(a, nq::commutator(C, A)),
                                          nq::scale(b, nq::commutator(C, B)));
                  t.expect(nf_eq(lhs2, rhs2), "[C, aA+bB] = a[C,A] + b[C,B]");
              })
        .test("leibniz_product_rule",
              [&](TestContext& t) {
                  // [A, BC] = [A,B]C + B[A,C].
                  const Op lhs = nq::commutator(A, nq::multiply(B, C));
                  const Op rhs = nq::add(nq::multiply(nq::commutator(A, B), C),
                                         nq::multiply(B, nq::commutator(A, C)));
                  t.expect(nf_eq(lhs, rhs), "[A,BC] = [A,B]C + B[A,C]");
                  // Mirror (right) Leibniz: [AB, C] = A[B,C] + [A,C]B.
                  const Op lhs2 = nq::commutator(nq::multiply(A, B), C);
                  const Op rhs2 = nq::add(nq::multiply(A, nq::commutator(B, C)),
                                          nq::multiply(nq::commutator(A, C), B));
                  t.expect(nf_eq(lhs2, rhs2), "[AB,C] = A[B,C] + [A,C]B");
              })
        .test("jacobi_identity",
              [&](TestContext& t) {
                  // [A,[B,C]] + [B,[C,A]] + [C,[A,B]] = 0.
                  const Op j = nq::add(
                      nq::add(nq::commutator(A, nq::commutator(B, C)),
                              nq::commutator(B, nq::commutator(C, A))),
                      nq::commutator(C, nq::commutator(A, B)));
                  t.expect(is_zero_op(j), "Jacobi: [A,[B,C]]+[B,[C,A]]+[C,[A,B]] = 0");
              })
        .test("canonical_commutation_exact_i",
              [&](TestContext& t) {
                  const Op x = nq::op_symbol("x");
                  const Op p = nq::op_symbol("p");
                  // [x,p] normal-forms to exactly xp - px (two ordered monomials, coeffs +/-1).
                  const Op expected =
                      nq::add(nq::multiply(x, p), nq::scale(ci(-1, 0), nq::multiply(p, x)));
                  t.expect(nf_eq(nq::commutator(x, p), expected), "[x,p] = xp - px");
                  // Antisymmetry of the canonical pair: [x,p] = -[p,x].
                  t.expect(nf_eq(nq::commutator(x, p), nq::negate(nq::commutator(p, x))),
                           "[x,p] = -[p,x]");
                  // The imaginary unit stays EXACT: i*[x,p] carries coefficients +i and -i,
                  // matching i*xp - i*px built independently (no float ever enters).
                  const Complex i = Complex::i();
                  const Op scaled = nq::scale(i, nq::commutator(x, p));
                  const Op expect_scaled =
                      nq::add(nq::scale(i, nq::multiply(x, p)),
                              nq::scale(ci(0, -1), nq::multiply(p, x)));  // i and -i exactly
                  t.expect(nf_eq(scaled, expect_scaled), "i[x,p] = i xp - i px (exact i)");
              })
        .test("dagger_rules",
              [&](TestContext& t) {
                  // (AB)† = B†A†  (ORDER REVERSED).
                  t.expect(nf_eq(nq::dagger(nq::multiply(A, B)),
                                 nq::multiply(nq::dagger(B), nq::dagger(A))),
                           "(AB)† = B†A†");
                  // (A†)† = A.
                  t.expect(nf_eq(nq::dagger(nq::dagger(A)), A), "(A†)† = A");
                  // A† is a DISTINCT symbol from A (not self-adjoint).
                  t.expect(!nf_eq(nq::dagger(A), A), "A† != A for a generic symbol");
                  // Conjugation of i: (i I)† = -i I.
                  t.expect(nf_eq(nq::dagger(nq::op_scalar(Complex::i())),
                                 nq::op_scalar(ci(0, -1))),
                           "(iI)† = -iI  (conj on i)");
                  // (aA)† = conj(a) A†, checked with a = i.
                  t.expect(nf_eq(nq::dagger(nq::scale(Complex::i(), A)),
                                 nq::scale(ci(0, -1), nq::dagger(A))),
                           "(iA)† = -i A†");
                  // (A+B)† = A†+B†.
                  t.expect(nf_eq(nq::dagger(nq::add(A, B)),
                                 nq::add(nq::dagger(A), nq::dagger(B))),
                           "(A+B)† = A†+B†");
              })
        .test("self_adjoint_and_dirac",
              [&](TestContext& t) {
                  const Op H = nq::self_adjoint("H");
                  // A declared-Hermitian symbol is a fixed point of dagger.
                  t.expect(nf_eq(nq::dagger(H), H), "self-adjoint H: H† = H");
                  // dagger(|psi>) = <psi| and dagger(<phi|) = |phi>.
                  t.expect(nf_eq(nq::dagger(nq::ket("psi")), nq::bra("psi")),
                           "(|psi>)† = <psi|");
                  t.expect(nf_eq(nq::dagger(nq::bra("phi")), nq::ket("phi")),
                           "(<phi|)† = |phi>");
                  // Order reversal with bra/ket: (|a><b|)† = |b><a|.
                  t.expect(nf_eq(nq::dagger(nq::multiply(nq::ket("a"), nq::bra("b"))),
                                 nq::multiply(nq::ket("b"), nq::bra("a"))),
                           "(|a><b|)† = |b><a|");
              })
        .test("normal_form_collection_and_exact_coeffs",
              [&](TestContext& t) {
                  // Like monomials collect: A + A = 2A.
                  t.expect(nf_eq(nq::add(A, A), nq::scale(ci(2, 0), A)), "A + A = 2A");
                  // Cancellation to zero: 2A - 3A + A = 0.
                  const Op z = nq::add(nq::add(nq::scale(ci(2, 0), A), nq::scale(ci(-3, 0), A)), A);
                  t.expect(is_zero_op(z), "2A - 3A + A = 0");
                  // Distribution over sums: A(B+C) = AB + AC.
                  t.expect(nf_eq(nq::multiply(A, nq::add(B, C)),
                                 nq::add(nq::multiply(A, B), nq::multiply(A, C))),
                           "A(B+C) = AB + AC");
                  // EXACT rational coefficients: (1/2)A + (1/2)A = A (no float rounding).
                  t.expect(nf_eq(nq::add(nq::scale(rat(1, 2), A), nq::scale(rat(1, 2), A)), A),
                           "(1/2)A + (1/2)A = A exactly");
                  // (1/3)A summed three times = A.
                  const Op third = nq::scale(rat(1, 3), A);
                  t.expect(nf_eq(nq::add(nq::add(third, third), third), A),
                           "(1/3)A * 3 = A exactly");
              })
        .run();
}
