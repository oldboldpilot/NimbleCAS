// Tests for nimblecas.modgcd: Brown's modular polynomial GCD math core.
// @author Olumuyiwa Oluwasanmi
//
// Test cases T1..T9 from M5_SPEC.md section 8.1 covering known GCDs, coprime inputs,
// content extraction, non-trivial leading coefficient multiplier gamma, unlucky-prime
// discard mechanism and reset rule, codec validation and round-trips, zero and constant
// edges, a 50-case pseudo-random sweep vs PRS, and prime-budget honesty.

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.bigint;
import nimblecas.numbertheory;
import nimblecas.modgcd;
import nimblecas.testing;

using nimblecas::BigInt;
using nimblecas::Candidate;
using nimblecas::CoprimeProven;
using nimblecas::decode_image_request;
using nimblecas::decode_image_result;
using nimblecas::decode_polynomial;
using nimblecas::encode_image_request;
using nimblecas::encode_image_result;
using nimblecas::encode_polynomial;
using nimblecas::gcd_image_mod_p;
using nimblecas::ImageRequest;
using nimblecas::ImageResult;
using nimblecas::landau_mignotte_bound;
using nimblecas::MathError;
using nimblecas::merge_images;
using nimblecas::MergeOutcome;
using nimblecas::modular_gcd;
using nimblecas::NeedMorePrimes;
using nimblecas::next_prime;
using nimblecas::Polynomial;
using nimblecas::ZpImage;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

auto poly(std::vector<std::int64_t> coeffs) -> Polynomial {
    return Polynomial{std::move(coeffs)};
}

auto expect_poly(TestContext& t, const Polynomial& actual,
                 const std::vector<std::int64_t>& expected,
                 std::string_view what) -> void {
    const auto actual_coeffs = actual.coefficients();
    t.expect(actual_coeffs.size() == expected.size(),
             std::format("{}: degree/count match (expected {}, got {})",
                         what, expected.size(), actual_coeffs.size()));
    for (std::size_t i = 0; i < std::min(actual_coeffs.size(), expected.size()); ++i) {
        t.expect(actual_coeffs[i] == expected[i],
                 std::format("{}: coefficient[{}] = {} (expected {})",
                             what, i, actual_coeffs[i], expected[i]));
    }
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.modgcd")
        // -------------------------------------------------------------------
        // T1: known gcd
        // -------------------------------------------------------------------
        .test("T1_known_gcd", [](TestContext& t) {
            // a = x^2 - x - 2 = (x + 1)(x - 2), coeffs = [-2, -1, 1]
            // b = x^2 + 4x + 3 = (x + 1)(x + 3), coeffs = [3, 4, 1]
            // Expected gcd = x + 1, coeffs = [1, 1]
            const auto a = poly({-2, -1, 1});
            const auto b = poly({3, 4, 1});
            auto res = modular_gcd(a, b);
            t.expect(res.has_value(), "modular_gcd(a, b) succeeds for T1");
            if (res) {
                expect_poly(t, *res, {1, 1}, "T1: gcd(x^2-x-2, x^2+4x+3) == x+1");
                auto ref = a.gcd(b);
                t.expect(ref.has_value(), "Polynomial::gcd succeeds");
                if (ref) {
                    t.expect(res->is_equal(*ref), "modular_gcd matches Polynomial::gcd bit-for-bit");
                }
            }
        })

        // -------------------------------------------------------------------
        // T2: coprime inputs
        // -------------------------------------------------------------------
        .test("T2_coprime", [](TestContext& t) {
            // a = x^2 + 1, coeffs = [1, 0, 1]
            // b = x^2 - 2, coeffs = [-2, 0, 1]
            // Irreducible over Q, distinct roots (±i vs ±√2) => gcd = 1, coeffs = [1]
            // Internally image has degree 0 => CoprimeProven fast path
            const auto a = poly({1, 0, 1});
            const auto b = poly({-2, 0, 1});
            auto res = modular_gcd(a, b);
            t.expect(res.has_value(), "modular_gcd(a, b) succeeds for T2");
            if (res) {
                expect_poly(t, *res, {1}, "T2: gcd(x^2+1, x^2-2) == 1");
                auto ref = a.gcd(b);
                t.expect(ref.has_value(), "Polynomial::gcd succeeds");
                if (ref) {
                    t.expect(res->is_equal(*ref), "modular_gcd matches Polynomial::gcd bit-for-bit");
                }
            }
        })

        // -------------------------------------------------------------------
        // T3: content handling
        // -------------------------------------------------------------------
        .test("T3_content_handling", [](TestContext& t) {
            // a = 6x^2 - 6 = 6(x^2 - 1), coeffs = [-6, 0, 6], content = 6, pp = x^2 - 1
            // b = 4x^2 - 8x + 4 = 4(x - 1)^2, coeffs = [4, -8, 4], content = 4, pp = x^2 - 2x + 1
            // gcd(content(a), content(b)) = gcd(6, 4) = 2
            // gcd(pp(a), pp(b)) = gcd(x^2 - 1, x^2 - 2x + 1) = x - 1
            // Expected gcd = 2*(x - 1) = 2x - 2, coeffs = [-2, 2]
            const auto a = poly({-6, 0, 6});
            const auto b = poly({4, -8, 4});
            auto res = modular_gcd(a, b);
            t.expect(res.has_value(), "modular_gcd(a, b) succeeds for T3");
            if (res) {
                expect_poly(t, *res, {-2, 2}, "T3: gcd(6x^2-6, 4x^2-8x+4) == 2x-2");
                auto ref = a.gcd(b);
                t.expect(ref.has_value(), "Polynomial::gcd succeeds");
                if (ref) {
                    t.expect(res->is_equal(*ref), "modular_gcd matches Polynomial::gcd bit-for-bit");
                }
            }
        })

        // -------------------------------------------------------------------
        // T4: nontrivial gamma, deg-3 gcd
        // -------------------------------------------------------------------
        .test("T4_nontrivial_gamma_deg3", [](TestContext& t) {
            // g = 3x^3 + 2x^2 + x + 5, lc(g) = 3
            // u = x^2 + 7, v = x^2 - x + 11
            // a = g * u = 3x^5 + 2x^4 + 22x^3 + 19x^2 + 7x + 35, coeffs = [35, 7, 19, 22, 2, 3]
            // b = g * v = 3x^5 - x^4 + 32x^3 + 26x^2 + 6x + 55, coeffs = [55, 6, 26, 32, -1, 3]
            // gamma = gcd(lc(A), lc(B)) = gcd(3, 3) = 3
            // Expected gcd = 3x^3 + 2x^2 + x + 5, coeffs = [5, 1, 2, 3]
            const auto a = poly({35, 7, 19, 22, 2, 3});
            const auto b = poly({55, 6, 26, 32, -1, 3});
            auto res = modular_gcd(a, b);
            t.expect(res.has_value(), "modular_gcd(a, b) succeeds for T4");
            if (res) {
                expect_poly(t, *res, {5, 1, 2, 3}, "T4: gcd == 3x^3 + 2x^2 + x + 5");
                auto ref = a.gcd(b);
                t.expect(ref.has_value(), "Polynomial::gcd succeeds");
                if (ref) {
                    t.expect(res->is_equal(*ref), "modular_gcd matches Polynomial::gcd bit-for-bit");
                }
            }
        })

        // -------------------------------------------------------------------
        // T5: unlucky first prime, discard verified
        // -------------------------------------------------------------------
        .test("T5_unlucky_first_prime_discard", [](TestContext& t) {
            // a = x - 1, coeffs = [-1, 1]
            // b = x - 1073741828, coeffs = [-1073741828, 1]
            // Note: q_1 = next_prime(2^30, 0) = 1073741827
            // 1073741828 = q_1 + 1 => b ≡ x - 1 (mod q_1)
            // Hence mod q_1, gcd_image_mod_p(a, b, q_1, gamma=1) produces x - 1 (degree 1, unlucky).
            // Next prime q_2 = 1073741831: b ≡ x + 3 (mod q_2), gcd_image_mod_p produces 1 (degree 0).
            const auto a = poly({-1, 1});
            const auto b = poly({-1073741828, 1});

            // 1. Assert end result via modular_gcd
            auto res = modular_gcd(a, b);
            t.expect(res.has_value(), "modular_gcd succeeds for T5");
            if (res) {
                expect_poly(t, *res, {1}, "T5: gcd(x-1, x-1073741828) == 1");
            }

            // 2. Assert the discard mechanism directly via merge_images
            const std::uint64_t q1 = 1073741827;
            const std::uint64_t q2 = 1073741831;
            auto img1 = gcd_image_mod_p(a, b, q1, 1);
            t.expect(img1.has_value(), "img1 computed");
            t.expect(img1->coeffs.size() == 2, "img1 has degree 1 (unlucky)");

            auto img2 = gcd_image_mod_p(a, b, q2, 1);
            t.expect(img2.has_value(), "img2 computed");
            t.expect(img2->coeffs.size() == 1, "img2 has degree 0");

            auto b_lm = landau_mignotte_bound(a, b);
            t.expect(b_lm.has_value(), "bound computed");

            std::vector<ZpImage> images = {*img1, *img2};
            auto outcome = merge_images(images, 1, *b_lm);
            t.expect(outcome.has_value(), "merge_images succeeds");
            if (outcome) {
                t.expect(std::holds_alternative<CoprimeProven>(*outcome),
                         "T5: merge_images resets on smaller degree and reports CoprimeProven");
            }
        })

        // -------------------------------------------------------------------
        // T6: kernel + codec units
        // -------------------------------------------------------------------
        .test("T6_kernel_and_codec_units", [](TestContext& t) {
            // 1. Image kernel unit check
            // A = x^2 - 1 = [-1, 0, 1], B = x^2 - 2x + 1 = [1, -2, 1], p = 7, gamma = 1
            // Monic gcd = x - 1 ≡ x + 6 (mod 7), coeffs = [6, 1]
            const auto a = poly({-1, 0, 1});
            const auto b = poly({1, -2, 1});
            auto img = gcd_image_mod_p(a, b, 7, 1);
            t.expect(img.has_value(), "gcd_image_mod_p mod 7 succeeds");
            if (img) {
                t.expect(img->p == 7, "image modulus is 7");
                t.expect(img->coeffs.size() == 2, "image degree is 1 (count 2)");
                if (img->coeffs.size() == 2) {
                    t.expect(img->coeffs[0] == 6, "coeff[0] == 6");
                    t.expect(img->coeffs[1] == 1, "coeff[1] == 1");
                }
            }

            // 2. Landau-Mignotte bound hand checks
            // For T1: A = x^2 - x - 2, B = x^2 + 4x + 3
            // gamma = 1, dA = 2, dB = 2, min(dA, dB) = 2
            // ||A||_inf = 2, (dA+1)*||A||_inf = 3 * 2 = 6
            // ||B||_inf = 4, (dB+1)*||B||_inf = 3 * 4 = 12
            // min_term = 6
            // B_lm = 1 * 2^2 * 6 = 24
            const auto a_t1 = poly({-2, -1, 1});
            const auto b_t1 = poly({3, 4, 1});
            auto bound = landau_mignotte_bound(a_t1, b_t1);
            t.expect(bound.has_value(), "bound computed for T1");
            if (bound) {
                t.expect(*bound == BigInt::from_u64(24), "T1 bound == 24 exactly");
            }

            // 3. Codec round-trips
            // Polynomial inner block
            const auto p_test = poly({-10, 0, 42, -999});
            auto enc_p = encode_polynomial(p_test);
            t.expect(enc_p.has_value(), "encode_polynomial succeeds");
            if (enc_p) {
                auto dec_p = decode_polynomial(*enc_p);
                t.expect(dec_p.has_value(), "decode_polynomial succeeds");
                if (dec_p) {
                    t.expect(dec_p->is_equal(p_test), "polynomial round-trip bit-identical");
                }
            }

            // ImageRequest round-trip
            const ImageRequest req{
                .p = 1073741827,
                .gamma = 3,
                .a = poly({35, 7, 19, 22, 2, 3}),
                .b = poly({55, 6, 26, 32, -1, 3}),
            };
            auto enc_req = encode_image_request(req);
            t.expect(enc_req.has_value(), "encode_image_request succeeds");
            if (enc_req) {
                auto dec_req = decode_image_request(*enc_req);
                t.expect(dec_req.has_value(), "decode_image_request succeeds");
                if (dec_req) {
                    t.expect(dec_req->p == req.p, "request p matches");
                    t.expect(dec_req->gamma == req.gamma, "request gamma matches");
                    t.expect(dec_req->a.is_equal(req.a), "request a matches");
                    t.expect(dec_req->b.is_equal(req.b), "request b matches");
                }
            }

            // ImageResult round-trip
            const ImageResult res_test{
                .p = 1073741827,
                .coeffs = {5, 1, 2, 3},
            };
            auto enc_res = encode_image_result(res_test);
            t.expect(enc_res.has_value(), "encode_image_result succeeds");
            if (enc_res) {
                auto dec_res = decode_image_result(*enc_res);
                t.expect(dec_res.has_value(), "decode_image_result succeeds");
                if (dec_res) {
                    t.expect(dec_res->p == res_test.p, "result p matches");
                    t.expect(dec_res->coeffs == res_test.coeffs, "result coeffs match");
                }
            }

            // 4. Malformed-bytes rejections
            // Short buffer
            std::vector<std::byte> short_buf(10, std::byte{0});
            t.expect(decode_image_request(short_buf).error() == MathError::syntax_error,
                     "short request buffer rejected with syntax_error");
            t.expect(decode_image_result(short_buf).error() == MathError::syntax_error,
                     "short result buffer rejected with syntax_error");

            // Bad magic in ImageRequest
            if (enc_req) {
                auto corrupt = *enc_req;
                corrupt[0] = std::byte{0xFF};
                t.expect(decode_image_request(corrupt).error() == MathError::syntax_error,
                         "bad request magic rejected with syntax_error");
            }

            // Bad magic in ImageResult
            if (enc_res) {
                auto corrupt = *enc_res;
                corrupt[0] = std::byte{0xFF};
                t.expect(decode_image_result(corrupt).error() == MathError::syntax_error,
                         "bad result magic rejected with syntax_error");
            }

            // Residue >= p in ImageResult
            if (enc_res) {
                auto corrupt = *enc_res;
                // Coeff offset is 20, set first 8 bytes of coeff to value > p
                for (int i = 0; i < 8; ++i) {
                    corrupt[20 + i] = std::byte{0xFF};
                }
                t.expect(decode_image_result(corrupt).error() == MathError::domain_error,
                         "residue >= p rejected with domain_error");
            }

            // Untrimmed top coefficient in decode_image_result
            ImageResult untrimmed_res{
                .p = 1073741827,
                .coeffs = {1, 0},
            };
            auto enc_untrimmed = encode_image_result(untrimmed_res);
            if (enc_untrimmed) {
                t.expect(decode_image_result(*enc_untrimmed).error() == MathError::domain_error,
                         "untrimmed top coefficient rejected with domain_error");
            }

            // Prime out of range in decode_image_request (e.g. p = 7)
            const ImageRequest req_bad_p{
                .p = 7,
                .gamma = 1,
                .a = poly({-1, 0, 1}),
                .b = poly({1, -2, 1}),
            };
            auto enc_bad_p = encode_image_request(req_bad_p);
            if (enc_bad_p) {
                t.expect(decode_image_request(*enc_bad_p).error() == MathError::domain_error,
                         "p out of (2^30, 2^31) range rejected with domain_error");
            }
        })

        // -------------------------------------------------------------------
        // T7: zero & constant edges
        // -------------------------------------------------------------------
        .test("T7_zero_and_constant_edges", [](TestContext& t) {
            const auto zero = poly({});
            const auto a = poly({-2, -1, 1});
            const auto neg_a = poly({2, 1, -1});  // leading coeff < 0
            const auto c6 = poly({6});
            const auto c4 = poly({4});

            // gcd(0, b) == b normalised to positive lc
            auto r1 = modular_gcd(zero, a);
            t.expect(r1.has_value(), "gcd(0, a) succeeds");
            if (r1) {
                expect_poly(t, *r1, {-2, -1, 1}, "gcd(0, a) == a");
            }

            auto r2 = modular_gcd(zero, neg_a);
            t.expect(r2.has_value(), "gcd(0, -a) succeeds");
            if (r2) {
                expect_poly(t, *r2, {-2, -1, 1}, "gcd(0, -a) normalises to positive lc");
            }

            // gcd(a, 0) == a normalised
            auto r3 = modular_gcd(a, zero);
            t.expect(r3.has_value(), "gcd(a, 0) succeeds");
            if (r3) {
                expect_poly(t, *r3, {-2, -1, 1}, "gcd(a, 0) == a");
            }

            // gcd(0, 0) == 0
            auto r4 = modular_gcd(zero, zero);
            t.expect(r4.has_value(), "gcd(0, 0) succeeds");
            if (r4) {
                t.expect(r4->is_zero(), "gcd(0, 0) == 0");
            }

            // gcd(6, 4) == 2
            auto r5 = modular_gcd(c6, c4);
            t.expect(r5.has_value(), "gcd(6, 4) succeeds");
            if (r5) {
                expect_poly(t, *r5, {2}, "gcd(6, 4) == 2");
            }

            // gcd(constant, poly) matches Polynomial::gcd
            auto r6 = modular_gcd(c6, a);
            auto ref6 = c6.gcd(a);
            t.expect(r6.has_value() && ref6.has_value(), "gcd(c, a) succeeds");
            if (r6 && ref6) {
                t.expect(r6->is_equal(*ref6), "gcd(c, a) matches Polynomial::gcd");
            }
        })

        // -------------------------------------------------------------------
        // T8: seeded sweep vs PRS (50 cases)
        // -------------------------------------------------------------------
        .test("T8_seeded_sweep_vs_prs", [](TestContext& t) {
            // Simple deterministic linear congruential generator for reproducible tests
            std::uint64_t state = 0x123456789ABCDEF0ULL;
            auto next_rand = [&]() -> std::int64_t {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                // Small coefficients in [-8, 8]
                return static_cast<std::int64_t>((state >> 60) & 0xF) - 7;
            };

            for (std::size_t trial = 0; trial < 50; ++trial) {
                // Generate factor g (degree 1 or 2)
                const std::size_t deg_g = (trial % 2) + 1;
                std::vector<std::int64_t> g_coeffs(deg_g + 1);
                for (auto& c : g_coeffs) {
                    c = next_rand();
                }
                if (g_coeffs.back() == 0) {
                    g_coeffs.back() = 1;
                }
                const auto g = poly(g_coeffs);

                // Generate factor u (degree 1 or 2)
                const std::size_t deg_u = ((trial + 1) % 2) + 1;
                std::vector<std::int64_t> u_coeffs(deg_u + 1);
                for (auto& c : u_coeffs) {
                    c = next_rand();
                }
                if (u_coeffs.back() == 0) {
                    u_coeffs.back() = 1;
                }
                const auto u = poly(u_coeffs);

                // Generate factor v (degree 1 or 2)
                const std::size_t deg_v = ((trial + 2) % 2) + 1;
                std::vector<std::int64_t> v_coeffs(deg_v + 1);
                for (auto& c : v_coeffs) {
                    c = next_rand();
                }
                if (v_coeffs.back() == 0) {
                    v_coeffs.back() = 1;
                }
                const auto v = poly(v_coeffs);

                auto a_res = g.multiply(u);
                auto b_res = g.multiply(v);
                if (!a_res || !b_res || a_res->is_zero() || b_res->is_zero()) {
                    continue;
                }

                auto mod_res = modular_gcd(*a_res, *b_res);
                auto prs_res = a_res->gcd(*b_res);

                t.expect(mod_res.has_value(), std::format("sweep trial {} modular_gcd succeeds", trial));
                t.expect(prs_res.has_value(), std::format("sweep trial {} Polynomial::gcd succeeds", trial));
                if (mod_res && prs_res) {
                    t.expect(mod_res->is_equal(*prs_res),
                             std::format("sweep trial {} modular_gcd matches Polynomial::gcd bit-for-bit", trial));
                }
            }
        })

        // -------------------------------------------------------------------
        // T9: prime budget honesty
        // -------------------------------------------------------------------
        .test("T9_budget_honesty", [](TestContext& t) {
            // T4 inputs requiring multiple primes: forcing budget to 0 primes must return not_converged
            const auto a = poly({35, 7, 19, 22, 2, 3});
            const auto b = poly({55, 6, 26, 32, -1, 3});

            auto res_0 = modular_gcd(a, b, /*max_primes=*/0);
            t.expect(!res_0.has_value(), "budget = 0 fails");
            t.expect(res_0.error() == MathError::not_converged,
                     "budget = 0 returns MathError::not_converged");

            // A budget BELOW the Landau-Mignotte target is NOT by itself a failure, and
            // asserting that it is would be wrong: the candidate is proven by exact trial
            // division, so a lift that verifies is the true gcd no matter how few primes
            // produced it (spec section 3.3). The honest invariant is therefore not "a small
            // budget fails" but "whatever comes back is either an honest not_converged or
            // the exact gcd -- never a wrong value".
            auto res_1 = modular_gcd(a, b, /*max_primes=*/1);
            if (res_1.has_value()) {
                expect_poly(t, *res_1, {5, 1, 2, 3},
                            "a verified single-prime lift is the exact gcd");
            } else {
                t.expect(res_1.error() == MathError::not_converged,
                         "an unverified budget-limited run returns not_converged");
            }

            // Genuine budget exhaustion, constructed so ONE prime provably cannot suffice:
            // gcd = x + 900000000, whose constant term exceeds the symmetric range
            // [-(p-1)/2, (p-1)/2] of a single ~2^30 prime, so the one-prime lift is wrong,
            // trial division rejects it, and the budget runs out with nothing verified.
            const auto wide_a = poly({900000000, 900000001, 1});   // (x + 9e8)(x + 1)
            const auto wide_b = poly({1800000000, 900000002, 1});  // (x + 9e8)(x + 2)

            auto tight = modular_gcd(wide_a, wide_b, /*max_primes=*/1);
            t.expect(!tight.has_value(), "one prime cannot cover a coefficient above p/2");
            t.expect(tight.error() == MathError::not_converged,
                     "exhausting the budget unverified returns not_converged, never a candidate");

            auto ample = modular_gcd(wide_a, wide_b);
            t.expect(ample.has_value(), "the same inputs succeed under the default budget");
            if (ample.has_value()) {
                expect_poly(t, *ample, {900000000, 1},
                            "and the recovered gcd is exactly x + 900000000");
            }
        })

        .run();
}
