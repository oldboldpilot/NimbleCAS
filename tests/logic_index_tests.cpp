// Tests for nimblecas.logic_index: the batched first-argument clause probe.
// @author Olumuyiwa Oluwasanmi
//
// Two things are being checked, and the second matters more than the first.
//
//  1. The probe computes the right candidate set — it must never rule out a clause that could
//     match, because a false negative silently loses an answer.
//  2. Every SIMD path agrees with the scalar path BIT FOR BIT on the same input. The scalar
//     loop is the definition; AVX2 and AVX-512 are optimisations that are only worth having if
//     they compute exactly it, so they are compared against it rather than against their own
//     expectations.

import std;
import nimblecas.core;
import nimblecas.logic;
import nimblecas.logic_index;
import nimblecas.logic_parser;
import nimblecas.testing;

using nimblecas::make_atom;
using nimblecas::make_compound;
using nimblecas::make_int;
using nimblecas::make_var;
using nimblecas::MathError;
using nimblecas::Program;
using nimblecas::logic_index::clause_index_keys;
using nimblecas::logic_index::decode_candidates;
using nimblecas::logic_index::detect_probe_isa;
using nimblecas::logic_index::first_arg_key;
using nimblecas::logic_index::index_probe_batch;
using nimblecas::logic_index::index_probe_batch_with;
using nimblecas::logic_index::probe_row_words;
using nimblecas::logic_index::ProbeIsa;
using nimblecas::logic_index::to_string_view;
using nimblecas::logic_parser::parse_program;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// A deterministic spread of keys with wildcards seeded through it, so a probe has to handle
// both the "matches everything" rows and the ordinary ones in the same pass.
[[nodiscard]] auto synthetic_keys(std::size_t n, std::size_t distinct, std::size_t wildcard_every)
    -> std::vector<std::uint64_t> {
    std::vector<std::uint64_t> keys;
    keys.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (wildcard_every != 0 && i % wildcard_every == 0) {
            keys.push_back(0U);  // a variable first argument: candidate for every goal
        } else {
            keys.push_back(static_cast<std::uint64_t>((i % distinct) + 1));
        }
    }
    return keys;
}

// The probe written the most obvious way possible. Independent of the module's own scalar loop
// so that "the paths agree" is not a claim about one implementation compared with itself.
[[nodiscard]] auto reference_probe(const std::vector<std::uint64_t>& clause_keys,
                                   const std::vector<std::uint64_t>& goal_keys)
    -> std::vector<std::uint64_t> {
    const std::size_t words = probe_row_words(clause_keys.size());
    std::vector<std::uint64_t> out(goal_keys.size() * words, 0U);
    for (std::size_t g = 0; g < goal_keys.size(); ++g) {
        for (std::size_t c = 0; c < clause_keys.size(); ++c) {
            const bool candidate =
                goal_keys[g] == 0U || clause_keys[c] == 0U || goal_keys[g] == clause_keys[c];
            if (candidate) {
                out[g * words + (c / 64U)] |= std::uint64_t{1} << (c % 64U);
            }
        }
    }
    return out;
}

[[nodiscard]] auto run_isa(ProbeIsa isa, const std::vector<std::uint64_t>& ck,
                           const std::vector<std::uint64_t>& gk)
    -> std::optional<std::vector<std::uint64_t>> {
    std::vector<std::uint64_t> out(gk.size() * probe_row_words(ck.size()), 0U);
    auto const r = index_probe_batch_with(isa, ck, gk, out);
    if (!r) {
        return std::nullopt;
    }
    return out;
}

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.logic_index")
        .test("a_variable_first_argument_has_the_wildcard_key_and_everything_else_does_not",
              [](TestContext& t) {
                  t.expect(first_arg_key(make_var("X")) == 0U,
                           "an unbound first argument discriminates nothing, so its key is 0");
                  t.expect(first_arg_key(make_atom("a")) != 0U, "an atom gets a real key");
                  t.expect(first_arg_key(make_int(1)) != 0U, "an integer gets a real key");
                  t.expect(first_arg_key(make_compound("f", {make_atom("a")})) != 0U,
                           "a compound gets a real key");
              })
        .test("keys_separate_terms_that_cannot_unify", [](TestContext& t) {
            // The integer 1 and the atom '1' render the same and must NOT share a key: confusing
            // them would let the filter drop a clause that genuinely matches.
            t.expect(first_arg_key(make_int(1)) != first_arg_key(make_atom("1")),
                     "the integer 1 and the atom '1' get different keys");
            t.expect(first_arg_key(make_atom("a")) != first_arg_key(make_atom("b")),
                     "different atoms get different keys");
            t.expect(first_arg_key(make_int(1)) != first_arg_key(make_int(2)),
                     "different integers get different keys");
            // Arity is part of the principal functor: f(a) and f(a,b) cannot unify.
            t.expect(first_arg_key(make_compound("f", {make_atom("a")})) !=
                         first_arg_key(make_compound("f", {make_atom("a"), make_atom("b")})),
                     "the same functor at different arities gets different keys");
        })
        .test("identical_terms_get_identical_keys", [](TestContext& t) {
            t.expect(first_arg_key(make_atom("abc")) == first_arg_key(make_atom("abc")),
                     "the key is a function of the term, not of the object");
            t.expect(first_arg_key(make_compound("f", {make_int(7)})) ==
                         first_arg_key(make_compound("f", {make_int(9)})),
                     "only the principal functor is keyed, so the arguments do not matter");
        })
        .test("clause_keys_follow_program_order_and_mark_wildcards", [](TestContext& t) {
            auto p = parse_program("p(a). p(X). q. p(f(1)). p(1).");
            t.expect(p.has_value(), "the program parses");
            if (!p) {
                return;
            }
            const auto keys = clause_index_keys(*p);
            t.expect(keys.size() == 5, "one key per clause");
            if (keys.size() != 5) {
                return;
            }
            t.expect(keys[0] == first_arg_key(make_atom("a")), "p(a) keys on the atom a");
            t.expect(keys[1] == 0U, "p(X) has a variable first argument, so it is a wildcard");
            t.expect(keys[2] == 0U, "the 0-arity q has no first argument to discriminate on");
            t.expect(keys[3] == first_arg_key(make_compound("f", {make_int(1)})),
                     "p(f(1)) keys on f/1");
            t.expect(keys[4] == first_arg_key(make_int(1)), "p(1) keys on the integer 1");
        })
        .test("probe_admits_exactly_the_clauses_that_could_match", [](TestContext& t) {
            //            clause:  0=a   1=wild  2=b    3=a
            const std::vector<std::uint64_t> ck = {first_arg_key(make_atom("a")), 0U,
                                                   first_arg_key(make_atom("b")),
                                                   first_arg_key(make_atom("a"))};
            const std::vector<std::uint64_t> gk = {first_arg_key(make_atom("a")),
                                                   first_arg_key(make_atom("b")), 0U};
            std::vector<std::uint64_t> out(gk.size() * probe_row_words(ck.size()), 0U);
            const auto r = index_probe_batch(ck, gk, out);
            t.expect(r.has_value(), "the probe succeeds");
            if (!r) {
                return;
            }
            const std::size_t w = probe_row_words(ck.size());
            const auto row_a = decode_candidates({out.data(), w}, ck.size());
            const auto row_b = decode_candidates({out.data() + w, w}, ck.size());
            const auto row_var = decode_candidates({out.data() + 2 * w, w}, ck.size());
            t.expect(row_a == std::vector<std::size_t>({0, 1, 3}),
                     "a goal keyed on a admits the two a-clauses and the wildcard, in order");
            t.expect(row_b == std::vector<std::size_t>({1, 2}),
                     "a goal keyed on b admits the b-clause and the wildcard");
            t.expect(row_var == std::vector<std::size_t>({0, 1, 2, 3}),
                     "an unbound goal admits every clause");
        })
        .test("candidates_come_back_in_ascending_clause_order", [](TestContext& t) {
            const auto ck = synthetic_keys(200, 7, 11);
            const std::vector<std::uint64_t> gk = {3U};
            std::vector<std::uint64_t> out(probe_row_words(ck.size()), 0U);
            const auto r = index_probe_batch(ck, gk, out);
            t.expect(r.has_value(), "the probe succeeds");
            if (!r) {
                return;
            }
            const auto got = decode_candidates(out, ck.size());
            bool ascending = true;
            for (std::size_t i = 1; i < got.size(); ++i) {
                if (got[i - 1] >= got[i]) {
                    ascending = false;
                }
            }
            t.expect(ascending, "clause order is preserved, which is what keeps answers ordered");
            t.expect(!got.empty(), "the wildcard clauses alone guarantee some candidate");
        })
        .test("bits_past_the_last_clause_are_never_set", [](TestContext& t) {
            // 70 clauses is two words with 58 unused bits in the second. A stray bit there would
            // decode as a clause index that does not exist.
            const auto ck = synthetic_keys(70, 3, 0);
            const std::vector<std::uint64_t> gk = {0U};  // the wildcard goal sets the most bits
            std::vector<std::uint64_t> out(probe_row_words(ck.size()), 0U);
            const auto r = index_probe_batch(ck, gk, out);
            t.expect(r.has_value(), "the probe succeeds");
            if (!r) {
                return;
            }
            const auto got = decode_candidates(out, ck.size());
            t.expect(got.size() == 70, "an unbound goal admits all 70 clauses and no more");
            const std::uint64_t tail = out[1] >> 6U;  // bits 70.. of the second word
            t.expect(tail == 0U, "no bit is set past the last clause");
        })
        .test("a_wrong_sized_output_span_is_refused_rather_than_truncated", [](TestContext& t) {
            const auto ck = synthetic_keys(100, 5, 0);
            const std::vector<std::uint64_t> gk = {1U, 2U};
            std::vector<std::uint64_t> too_small(1, 0U);
            const auto r = index_probe_batch(ck, gk, too_small);
            t.expect(!r.has_value(), "an undersized output is an error");
            if (!r) {
                t.expect(r.error() == MathError::domain_error, "and the error is domain_error");
            }
        })
        .test("empty_inputs_are_a_no_op_not_an_error", [](TestContext& t) {
            std::vector<std::uint64_t> out;
            const auto r1 = index_probe_batch({}, {}, out);
            t.expect(r1.has_value(), "no clauses and no goals is a successful no-op");
            const auto ck = synthetic_keys(10, 3, 0);
            std::vector<std::uint64_t> out2;
            const auto r2 = index_probe_batch(ck, {}, out2);
            t.expect(r2.has_value(), "clauses but no goals is a successful no-op");
        })
        .test("the_scalar_path_matches_an_independent_reference", [](TestContext& t) {
            const auto ck = synthetic_keys(333, 9, 13);
            const auto gk = synthetic_keys(64, 9, 7);
            const auto expected = reference_probe(ck, gk);
            const auto got = run_isa(ProbeIsa::scalar, ck, gk);
            t.expect(got.has_value(), "the scalar path runs");
            if (got) {
                t.expect(*got == expected, "the scalar path matches the obvious implementation");
            }
        })
        .test("every_simd_path_matches_the_scalar_path_bit_for_bit", [](TestContext& t) {
            // The claim the whole module rests on. Sizes are chosen to straddle the vector
            // widths: 333 clauses is not a multiple of 4, 8 or 64, so every path has to handle
            // its own tail as well as its wide body.
            const auto ck = synthetic_keys(333, 9, 13);
            const auto gk = synthetic_keys(64, 9, 7);
            const auto scalar = run_isa(ProbeIsa::scalar, ck, gk);
            t.expect(scalar.has_value(), "the scalar path always runs");
            if (!scalar) {
                return;
            }
            const ProbeIsa best = detect_probe_isa();
            t.expect(!to_string_view(best).empty(), "the detected path has a name");

            if (best == ProbeIsa::avx2 || best == ProbeIsa::avx512) {
                const auto avx2 = run_isa(ProbeIsa::avx2, ck, gk);
                t.expect(avx2.has_value(), "AVX2 runs when the host advertises it");
                if (avx2) {
                    t.expect(*avx2 == *scalar, "AVX2 matches scalar bit for bit");
                }
            }
            if (best == ProbeIsa::avx512) {
                const auto avx512 = run_isa(ProbeIsa::avx512, ck, gk);
                t.expect(avx512.has_value(), "AVX-512 runs when the host advertises it");
                if (avx512) {
                    t.expect(*avx512 == *scalar, "AVX-512 matches scalar bit for bit");
                }
            }
            // The dispatching entry point must agree with whatever it dispatched to.
            std::vector<std::uint64_t> dispatched(gk.size() * probe_row_words(ck.size()), 0U);
            const auto r = index_probe_batch(ck, gk, dispatched);
            t.expect(r.has_value(), "the dispatching probe runs");
            if (r) {
                t.expect(dispatched == *scalar, "dispatch agrees with scalar whichever it chose");
            }
        })
        .test("paths_agree_across_a_sweep_of_awkward_sizes", [](TestContext& t) {
            // One clause, exactly one word, one past a word, and a wide batch — the sizes where
            // a tail-handling mistake hides.
            bool all_match = true;
            std::size_t checked = 0;
            for (const std::size_t nc : {std::size_t{1}, std::size_t{3}, std::size_t{7},
                                         std::size_t{8}, std::size_t{63}, std::size_t{64},
                                         std::size_t{65}, std::size_t{127}, std::size_t{128}}) {
                const auto ck = synthetic_keys(nc, 5, 6);
                const auto gk = synthetic_keys(17, 5, 4);
                const auto expected = reference_probe(ck, gk);
                const auto scalar = run_isa(ProbeIsa::scalar, ck, gk);
                if (!scalar || *scalar != expected) {
                    all_match = false;
                }
                ++checked;
                const ProbeIsa best = detect_probe_isa();
                if (best != ProbeIsa::scalar) {
                    const auto wide = run_isa(best, ck, gk);
                    if (!wide || *wide != expected) {
                        all_match = false;
                    }
                }
            }
            t.expect(checked == 9, "every size in the sweep was exercised");
            t.expect(all_match, "every path matches the reference at every size");
        })
        .test("asking_for_a_path_this_host_cannot_run_is_refused_not_downgraded",
              [](TestContext& t) {
                  // Silently running scalar while the caller believes it measured AVX-512 would
                  // make every number that caller reports a lie.
                  const auto ck = synthetic_keys(32, 4, 0);
                  const std::vector<std::uint64_t> gk = {1U};
                  const ProbeIsa best = detect_probe_isa();
                  if (best == ProbeIsa::scalar) {
                      const auto r = run_isa(ProbeIsa::avx512, ck, gk);
                      t.expect(!r.has_value(), "a host without AVX-512 refuses the AVX-512 path");
                  } else {
                      const auto r = run_isa(ProbeIsa::scalar, ck, gk);
                      t.expect(r.has_value(), "the scalar path is always available");
                  }
              })
        .test("the_probe_agrees_with_what_the_solver_would_have_to_try", [](TestContext& t) {
            // Ties the kernel back to its purpose: for a real program, the probe's candidates
            // for a bound first argument must be exactly the clauses whose first argument could
            // unify with it — no more, and critically no fewer.
            auto p = parse_program("p(a, 1). p(b, 2). p(X, 3). p(a, 4). p(f(z), 5).");
            t.expect(p.has_value(), "the program parses");
            if (!p) {
                return;
            }
            const auto ck = clause_index_keys(*p);
            const std::vector<std::uint64_t> gk = {first_arg_key(make_atom("a"))};
            std::vector<std::uint64_t> out(probe_row_words(ck.size()), 0U);
            const auto r = index_probe_batch(ck, gk, out);
            t.expect(r.has_value(), "the probe succeeds");
            if (!r) {
                return;
            }
            const auto got = decode_candidates(out, ck.size());
            t.expect(got == std::vector<std::size_t>({0, 2, 3}),
                     "p(a,_) admits the two a-clauses and the variable-headed one, in order");
            t.expect(std::ranges::find(got, std::size_t{1}) == got.end(),
                     "the b-clause is ruled out");
            t.expect(std::ranges::find(got, std::size_t{4}) == got.end(),
                     "the f(z)-clause is ruled out");
        })
        .run();
}
