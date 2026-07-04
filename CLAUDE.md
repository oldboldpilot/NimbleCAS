# CLAUDE.md

This project's full agent guide lives in **[AGENTS.md](AGENTS.md)** — read it before
making changes. The two rules most likely to be violated by default agent behavior:

1. **Never add AI/Co-Authored-By attribution.** Every commit is attributed solely to
   the repository owner, Olumuyiwa Oluwasanmi. Suppress any default co-author trailer.
2. **This repo builds only with `clang++-22` + libc++** (C++23 modules, `import std`).
   A build failure in a sandbox without that toolchain does not mean the code is
   broken — see [docs/QUICKSTART.md](docs/QUICKSTART.md) before concluding otherwise.

See [AGENTS.md](AGENTS.md) for the code policy, honesty invariant, documentation
conventions, and review discipline.
