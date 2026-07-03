// NimbleCAS internal test framework.
// @author Olumuyiwa Oluwasanmi
//
// Code Policy Rule 39 forbids external test frameworks (GTest/Catch2). This module
// provides a minimal, fluent test runner: a TestSuite collects named cases, each of
// which receives a TestContext to record expectations. run() reports a summary and
// returns a process exit code (0 = all passed) for ctest integration.

export module nimblecas.testing;

import std;

export namespace nimblecas::testing {

// Accumulates the outcome of the expectations made inside a single test case.
class TestContext {
public:
    auto expect(bool condition, std::string_view what) -> TestContext& {
        ++checks_;
        if (!condition) {
            failures_.emplace_back(what);
        }
        return *this;
    }

    template <typename A, typename B>
    auto expect_eq(const A& actual, const B& expected, std::string_view what) -> TestContext& {
        return expect(actual == expected, what);
    }

    template <typename A, typename B>
    auto expect_ne(const A& lhs, const B& rhs, std::string_view what) -> TestContext& {
        return expect(!(lhs == rhs), what);
    }

    [[nodiscard]] auto ok() const noexcept -> bool { return failures_.empty(); }
    [[nodiscard]] auto checks() const noexcept -> std::size_t { return checks_; }
    [[nodiscard]] auto failures() const noexcept -> std::span<const std::string> {
        return failures_;
    }

private:
    std::size_t checks_{};
    std::vector<std::string> failures_{};
};

// Collects and runs a group of test cases with a fluent registration API.
class TestSuite {
public:
    using TestFn = std::function<void(TestContext&)>;

    explicit TestSuite(std::string name) : name_(std::move(name)) {}

    auto test(std::string name, TestFn fn) -> TestSuite& {
        cases_.emplace_back(std::move(name), std::move(fn));
        return *this;
    }

    // Runs every case, prints a per-case and overall summary, and returns an exit
    // code (0 = all passed) suitable for returning from main().
    [[nodiscard]] auto run() const -> int {
        std::size_t passed = 0;
        std::size_t failed = 0;
        for (const auto& [case_name, fn] : cases_) {
            TestContext ctx;
            fn(ctx);
            if (ctx.ok()) {
                ++passed;
                std::println("  [PASS] {} ({} checks)", case_name, ctx.checks());
            } else {
                ++failed;
                std::println("  [FAIL] {}", case_name);
                for (std::string_view failure : ctx.failures()) {
                    std::println("         - {}", failure);
                }
            }
        }
        std::println("{}: {} passed, {} failed", name_, passed, failed);
        return failed == 0 ? 0 : 1;
    }

private:
    std::string name_;
    std::vector<std::pair<std::string, TestFn>> cases_{};
};

}  // namespace nimblecas::testing
