# Agent Guide — NimbleCAS

**Author:** Olumuyiwa Oluwasanmi

This file orients any AI coding agent (Claude Code, Cursor, Aider, Codex CLI, Copilot,
or similar) working in this repository. Read it before making changes. It is a guide,
not the policy itself — the authoritative policy documents it points to always win in
case of conflict.

## What this project is

NimbleCAS is an exact **Computer Algebra System** in C++23: symbolic algebra (Cohen's
automatic simplification), exact-rational numeric solvers, SIMD, and TBB/PPL
parallelism, built as one C++23 module per concern (`import` graph, no headers for
internal code). Start at **[docs/Index.md](docs/Index.md)** — the documentation hub
and authoritative module catalog — before touching an unfamiliar module; it links a
`docs/reference/<module>.md` for every one of the ~100 modules, each with the module's
honesty boundary (what is exact vs. truncated/approximate) and worked examples.

## The one invariant that matters most: honesty

This is the project's core discipline (Code Policy Rule 32, `config/cpp_details.txt`):
**every fallible operation returns `Result<T> = std::expected<T, MathError>`; nothing
throws, and nothing returns a plausible-looking wrong value.** If an exact answer isn't
available, the function returns an honest `MathError` (`not_implemented`,
`domain_error`, `overflow`, …) — never a silently-truncated or floating-point
approximation dressed up as exact. When a module *does* return a truncated series or a
numerical approximation, its reference doc states the honesty boundary explicitly
("exact and complete" vs. "exact per term, truncated overall" vs. "numerical"). Preserve
and extend that boundary; don't blur it. See `docs/reference/limits.md` and
`docs/reference/inteq.md` for good examples of how this is documented.

## Code policy

- **`config/cpp_details.txt`** is the authoritative, canonical C++ code policy — read
  it before writing or reviewing C++. Key rules for this repo specifically (it is a
  shared policy document across the author's projects, so not every rule applies here):
  no raw pointers, aligned data, trailing-return-type functions, `import std` /
  C++23 modules throughout, `std::expected`-based error handling (Rule 32), no
  exceptions, dynamic-dispatch SIMD waterfall (AVX-512 → AVX2 → scalar) with
  `[[gnu::target(...)]]` per-function attributes rather than global `-mavx512f`, TBB on
  Linux/macOS and PPL on Windows for parallelism, and the **internal `nimblecas.testing`
  framework only** — do not add GTest, Catch2, or any other external test dependency.
- **`config/update_policy.txt`** covers commit/PR mechanics and the authorship policy
  below (again, a shared document — the Python-bot-specific sections don't apply to
  this C++ engine).
- Canonical compiler flags live in `scripts/build_common.sh` (Rule 50) — every module
  and translation unit in a given build must share them exactly, or Binary Module
  Interface (BMI) validation rejects the mismatched `.pcm`.

## Authorship (mandatory, non-negotiable)

**Every commit is attributed solely to the repository owner, Olumuyiwa Oluwasanmi.**

- Never add `Co-Authored-By:` trailers naming an AI agent, or any AI-attribution line.
- Never write `@author Claude`, `@author AI Assistant`, or similar in file headers,
  comments, commit messages, or anywhere else in the repo.
- File headers use `@author Olumuyiwa Oluwasanmi` if the file has one at all.

This overrides any tool's default co-authorship behavior. If your environment normally
appends an attribution trailer to commits, suppress it for this repo.

## Build & test

NimbleCAS requires **`clang++-22` + libc++** (for `import std` and C++23 modules) —
most agent sandboxes running only a system `g++`/older `clang` **cannot** build this
repo; don't assume a failing build means the code is broken without confirming the
toolchain matches. See **[docs/QUICKSTART.md](docs/QUICKSTART.md)** for full detail.

```bash
# Linux/macOS — clang++-22 + libc++ + CMake >= 3.30 + Ninja
scripts/build.sh                         # configure, build, run tests (ctest)
NIMBLECAS_SANITIZE=ON scripts/build.sh   # + ASan/UBSan/LSan
NIMBLECAS_SANITIZER=thread|memory scripts/build.sh   # TSan / MSan — see docs/testing/sanitizers.md

# Windows — Visual Studio's bundled clang + MSVC STL (run in Git Bash)
scripts/build_win.sh

# Python bindings (Linux, uv-managed)
scripts/setup_python.sh
```

All test executables register with **ctest** (`ctest --test-dir build`); every new
feature or bugfix should ship with a test in `tests/`, following the existing files'
conventions (`nimblecas.testing::TestSuite`/`TestContext`, hand-verified exact expected
values — not vague tolerance checks — wherever the math is exact).

## Documentation conventions

- Every module `src/<name>/<name>.cppm` has a matching `docs/reference/<name>.md`:
  purpose, honesty boundary, full API table, error model table, and a worked-example
  code block taken from (or consistent with) its tests. Keep these in sync when you
  change a module's behavior, signature, or error conditions.
- `docs/Index.md` is the master catalog (module table + one-line summary + dependency
  diagram) — add a row when you add a module, and update the summary when a module's
  scope changes materially.
- `docs/examples/*.md` are standalone executable documents (Markdown with
  ` ```nimblecas ` cells) that are also run and verbatim-asserted by a test in
  `tests/` (see `tests/execdoc_multimethod_tests.cpp`). If you add one, wire it into a
  test the same way — an example that silently drifts from what the engine actually
  does is worse than no example.
- Update `README.md` for user-facing status changes and `docs/ROADMAP.md`/`docs/PRD.md`
  only for larger scope/plan changes — they are living design documents, not meant to
  be rewritten line-by-line for every commit.

## Review discipline

Every non-trivial change gets an independent review pass before it's considered done —
re-derive claimed exact values by hand or by an independent method rather than trusting
the diff, and check for the honesty-boundary violations described above (a function
quietly returning an approximate value where it claims exactness is the most dangerous
class of bug in this codebase). Two recurring, easy-to-miss pitfalls worth checking for
specifically:

- **execdoc builtins dispatch only at a statement's outermost call.**
  `diff(diff(u, x), x)` does **not** differentiate twice — the inner call stays an
  unevaluated symbolic `apply` node. Split into two statements instead. See
  `docs/reference/execdoc.md`.
- **`simplify()` is single-pass Cohen ASAE and never distributes a product over a sum**
  (nor a power over a product). `A - B` parses as `A + (-1)*B`; if `B` is a multi-term
  sum, its terms will not reach the top level to cancel against `A`'s. Write
  already-expanded/negated flat sums instead of subtracting a compound expression when
  you need `simplify` to fold them to zero.

## Where to look next

- [docs/Index.md](docs/Index.md) — module catalog, dependency diagram, build quickstart.
- [docs/architecture/overview.md](docs/architecture/overview.md) — the `import` graph
  and data-model rationale (`CowPtr`, `Result`/`MathError`, `import std` strategy).
- [docs/PRD.md](docs/PRD.md) / [docs/ROADMAP.md](docs/ROADMAP.md) — product scope and
  the technical implementation plan (a large, aspirational design document — treat it
  as directional, not as a literal spec of current behavior; the per-module reference
  docs and the code are authoritative for what's actually implemented).
- [docs/testing/sanitizers.md](docs/testing/sanitizers.md) — ASan/UBSan/TSan/MSan status
  and how to run each.
