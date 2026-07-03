# `nimblecas.testing` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/testing/testing.cppm`

The internal test framework. Code Policy **Rule 39** forbids external test
frameworks (GTest / Catch2), so NimbleCAS ships a minimal, dependency-free,
fluent test runner. A `TestSuite` collects named cases; each case receives a
`TestContext` to record expectations; `run()` prints a summary and returns a
process exit code (`0` = all passed) for **ctest** integration.

```cpp
import nimblecas.testing;
```

Namespace: `nimblecas::testing`. Depends only on `std`.

## `class TestContext`

Accumulates the outcome of the expectations made inside a single test case.

| Method | Signature | Behavior |
| :--- | :--- | :--- |
| `expect` | `auto expect(bool condition, std::string_view what) -> TestContext&` | Records a check; on failure stores `what`. Chainable. |
| `expect_eq` | `template<class A, class B> auto expect_eq(const A&, const B&, std::string_view) -> TestContext&` | `expect(actual == expected, what)`. |
| `expect_ne` | `template<class A, class B> auto expect_ne(const A&, const B&, std::string_view) -> TestContext&` | `expect(!(lhs == rhs), what)`. |
| `ok` | `auto ok() const noexcept -> bool` | `true` if no expectation failed. |
| `checks` | `auto checks() const noexcept -> std::size_t` | Number of expectations recorded. |
| `failures` | `auto failures() const noexcept -> std::span<const std::string>` | The messages of failed expectations. |

## `class TestSuite`

Collects and runs a group of test cases with a fluent registration API.

```cpp
using TestFn = std::function<void(TestContext&)>;

explicit TestSuite(std::string name);
auto test(std::string name, TestFn fn) -> TestSuite&;   // register a case (chainable)
[[nodiscard]] auto run() const -> int;                  // run all; 0 = all passed
```

`run()` executes every registered case with a fresh `TestContext`, prints a
`[PASS]`/`[FAIL]` line per case (with each failure message under a failing case)
and an overall `"<suite>: <p> passed, <f> failed"` summary, then returns `0` if
all passed or `1` otherwise — suitable for returning straight from `main()`.

## ctest integration

Each test executable under `tests/` builds one or more `TestSuite`s, returns
their combined exit code from `main()`, and is registered as a ctest test in
`CMakeLists.txt`. `scripts/build.sh` runs the whole suite via `ctest`. A
non-zero exit from any suite fails the build. There is one test target per
module (`core_tests`, `symbolic_tests`, `simplify_tests`, `diff_tests`,
`cache_tests`, `parallel_tests`, `simd_tests`, `polynomial_tests`,
`polyexpr_tests`), plus the Python smoke test `tests/test_bindings.py`.

## Example

```cpp
import nimblecas.testing;
import nimblecas.symbolic;
using namespace nimblecas;
using nimblecas::testing::TestSuite;

auto main() -> int {
    TestSuite suite("symbolic");
    suite
        .test("symbol round-trips", [](auto& ctx) {
            ctx.expect_eq(Expr::symbol("x").to_string(), std::string("x"),
                          "symbol prints its name");
        })
        .test("structural equality", [](auto& ctx) {
            ctx.expect(Expr::symbol("x").is_equivalent_to(Expr::symbol("x")),
                       "equal symbols compare equivalent");
        });
    return suite.run();   // 0 = all passed
}
```

## See also

- [Sanitizer & memory-safety testing](../testing/sanitizers.md)
- [Quickstart — Build & test](../QUICKSTART.md)
- [Documentation hub](../Index.md)
