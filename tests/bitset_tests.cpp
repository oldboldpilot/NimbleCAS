// Tests for nimblecas.bitset: the branchless, word-parallel bit set substrate.
// @author Olumuyiwa Oluwasanmi
//
// Concrete bit patterns pin every operation: single-bit set/reset/test, the word-parallel
// AND/OR/XOR/ANDNOT combinators (both the Result and in-place forms), the popcount/any/none/all
// reductions, first_set and the ascending set_bits scan (including across word boundaries), the
// capacity edges (0, exactly one word, one past a word), and the capacity-mismatch domain_error.

import std;
import nimblecas.core;
import nimblecas.bitset;
import nimblecas.testing;

using nimblecas::Bitset;
using nimblecas::MathError;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

// Builds a bitset of the given capacity with exactly the listed indices set.
[[nodiscard]] auto make_bits(std::size_t capacity, std::initializer_list<std::size_t> idx)
    -> Bitset {
    Bitset b(capacity);
    for (const std::size_t i : idx) {
        b.set(i);
    }
    return b;
}

using Indices = std::vector<std::size_t>;

}  // namespace

auto main() -> int {
    return TestSuite("nimblecas.bitset")
        .test("set_test_reset_single_bits",
              [](TestContext& t) {
                  Bitset b(10);
                  t.expect(b.none(), "fresh bitset is empty");
                  b.set(0);
                  b.set(3);
                  b.set(9);
                  t.expect(b.test(0) && b.test(3) && b.test(9), "set bits read back true");
                  t.expect(!b.test(1) && !b.test(8), "unset bits read back false");
                  t.expect(b.count() == 3, "three bits set");
                  b.reset(3);
                  t.expect(!b.test(3), "reset clears the bit");
                  t.expect(b.count() == 2, "count drops after reset");
                  b.set(3);
                  b.set(3);  // idempotent
                  t.expect(b.count() == 3, "re-setting is idempotent");
              })
        .test("count_any_none_all",
              [](TestContext& t) {
                  Bitset b(5);
                  t.expect(b.none() && !b.any(), "empty: none and not any");
                  t.expect(!b.all(), "empty is not all (capacity 5)");
                  t.expect(b.count() == 0, "empty count is 0");
                  b.set_all();
                  t.expect(b.all(), "set_all makes all() true");
                  t.expect(b.any() && !b.none(), "set_all: any true, none false");
                  t.expect(b.count() == 5, "set_all sets exactly capacity bits");
                  b.clear();
                  t.expect(b.none() && b.count() == 0, "clear empties the set");
              })
        .test("first_set_scan",
              [](TestContext& t) {
                  Bitset empty(8);
                  t.expect(empty.first_set() == 8, "first_set of empty returns capacity");
                  Bitset b = make_bits(8, {5, 6});
                  t.expect(b.first_set() == 5, "first_set returns the lowest set index");
                  Bitset c = make_bits(8, {0});
                  t.expect(c.first_set() == 0, "first_set finds bit 0");
              })
        .test("set_bits_ascending",
              [](TestContext& t) {
                  Bitset b = make_bits(16, {1, 4, 7, 15});
                  t.expect(b.set_bits() == Indices({1, 4, 7, 15}),
                           "set_bits lists indices ascending");
                  Bitset empty(16);
                  t.expect(empty.set_bits().empty(), "set_bits of empty is empty");
              })
        .test("and_or_xor_andnot_concrete",
              [](TestContext& t) {
                  // a = {0,1,2,3} (0x0F), b = {2,3,4,5} (0x3C) over an 8-bit capacity.
                  Bitset a = make_bits(8, {0, 1, 2, 3});
                  Bitset b = make_bits(8, {2, 3, 4, 5});

                  auto a_and = a.and_with(b);
                  auto a_or = a.or_with(b);
                  auto a_xor = a.xor_with(b);
                  auto a_andnot = a.andnot_with(b);
                  t.expect(a_and && a_or && a_xor && a_andnot, "all combinators succeed");

                  t.expect(a_and.value().set_bits() == Indices({2, 3}), "AND = {2,3}");
                  t.expect(a_or.value().set_bits() == Indices({0, 1, 2, 3, 4, 5}),
                           "OR = {0,1,2,3,4,5}");
                  t.expect(a_xor.value().set_bits() == Indices({0, 1, 4, 5}), "XOR = {0,1,4,5}");
                  t.expect(a_andnot.value().set_bits() == Indices({0, 1}),
                           "ANDNOT (a & ~b) = {0,1}");

                  // Operands are left untouched by the Result forms.
                  t.expect(a.set_bits() == Indices({0, 1, 2, 3}), "AND-with leaves a unchanged");
                  t.expect(b.set_bits() == Indices({2, 3, 4, 5}), "AND-with leaves b unchanged");
              })
        .test("inplace_combinators_match",
              [](TestContext& t) {
                  Bitset a = make_bits(8, {0, 1, 2, 3});
                  Bitset b = make_bits(8, {2, 3, 4, 5});

                  Bitset x = a;
                  x.and_inplace(b);
                  t.expect(x.set_bits() == Indices({2, 3}), "and_inplace = {2,3}");

                  Bitset y = a;
                  y.or_inplace(b);
                  t.expect(y.set_bits() == Indices({0, 1, 2, 3, 4, 5}), "or_inplace = union");

                  Bitset z = a;
                  z.xor_inplace(b);
                  t.expect(z.set_bits() == Indices({0, 1, 4, 5}), "xor_inplace = symmetric diff");

                  Bitset w = a;
                  w.andnot_inplace(b);
                  t.expect(w.set_bits() == Indices({0, 1}), "andnot_inplace = a minus b");

                  // In-place agrees with the Result form.
                  t.expect(x == a.and_with(b).value(), "in-place AND == Result AND");
              })
        .test("equality",
              [](TestContext& t) {
                  Bitset a = make_bits(8, {1, 3});
                  Bitset b = make_bits(8, {1, 3});
                  Bitset c = make_bits(8, {1, 4});
                  Bitset d = make_bits(16, {1, 3});  // same bits, different capacity
                  t.expect(a == b, "same capacity and bits compare equal");
                  t.expect(!(a == c), "different bits compare unequal");
                  t.expect(!(a == d), "different capacity compares unequal");
              })
        .test("cross_word_boundary",
              [](TestContext& t) {
                  // Capacity spanning three words; bits on both sides of the 64-bit boundaries.
                  Bitset b = make_bits(130, {0, 63, 64, 65, 129});
                  t.expect(b.word_count() == 3, "130 bits occupy three 64-bit words");
                  t.expect(b.count() == 5, "five bits set across words");
                  t.expect(b.set_bits() == Indices({0, 63, 64, 65, 129}),
                           "set_bits crosses word boundaries in order");
                  t.expect(b.first_set() == 0, "first_set is 0");
                  t.expect(b.test(64) && b.test(129), "high-word bits read back");
                  t.expect(!b.test(1) && !b.test(128), "neighbouring bits stay clear");
              })
        .test("capacity_edges",
              [](TestContext& t) {
                  // Capacity 0: no words, vacuously all, never any.
                  Bitset zero(0);
                  t.expect(zero.word_count() == 0, "capacity 0 has no words");
                  t.expect(zero.none() && !zero.any(), "capacity 0 is empty");
                  t.expect(zero.all(), "capacity 0 is vacuously all()");
                  t.expect(zero.count() == 0 && zero.first_set() == 0,
                           "capacity 0 count 0, first_set == capacity");
                  t.expect(zero.set_bits().empty(), "capacity 0 has no set bits");

                  // Capacity exactly one word: set_all must fill all 64 and none above.
                  Bitset full64(64);
                  full64.set_all();
                  t.expect(full64.word_count() == 1 && full64.count() == 64,
                           "64-bit capacity fills exactly one word");
                  t.expect(full64.all(), "64-capacity set_all is all()");

                  // Capacity one past a word: set_all masks the tail so count == capacity, not 128.
                  Bitset full65(65);
                  full65.set_all();
                  t.expect(full65.word_count() == 2, "65 bits need two words");
                  t.expect(full65.count() == 65,
                           "set_all masks unused high bits (count is 65, not 128)");
                  t.expect(full65.all(), "65-capacity set_all is all()");
                  t.expect(full65.first_set() == 0, "first_set of a full set is 0");
              })
        .test("capacity_mismatch_domain_error",
              [](TestContext& t) {
                  Bitset a(8);
                  Bitset b(16);
                  auto r_and = a.and_with(b);
                  auto r_or = a.or_with(b);
                  auto r_xor = a.xor_with(b);
                  auto r_andnot = a.andnot_with(b);
                  t.expect(!r_and.has_value() && r_and.error() == MathError::domain_error,
                           "and_with mismatched capacity -> domain_error");
                  t.expect(!r_or.has_value() && r_or.error() == MathError::domain_error,
                           "or_with mismatched capacity -> domain_error");
                  t.expect(!r_xor.has_value() && r_xor.error() == MathError::domain_error,
                           "xor_with mismatched capacity -> domain_error");
                  t.expect(!r_andnot.has_value() && r_andnot.error() == MathError::domain_error,
                           "andnot_with mismatched capacity -> domain_error");

                  // Equal capacity succeeds.
                  Bitset c(8);
                  t.expect(a.and_with(c).has_value(), "equal capacity and_with succeeds");
              })
        .run();
}
