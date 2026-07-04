# `nimblecas.hmm` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/hmm/hmm.cppm`

Discrete-emission **hidden Markov models**, exact over the rationals (ROADMAP §7.8).
It complements [`stochastic.md`](stochastic.md) (the *structure* of an observed
Markov chain — stationary law, spectrum, hitting times); here the chain is
**hidden**: we never see the state, only a symbol emitted from it, and the
deliverables are the classical HMM inference algorithms.

## The model

A `HiddenMarkovModel` over `N` hidden states and `M` observation symbols is three
**exact rational** objects:

- `pi` — the initial state distribution (length `N`);
- `A` — the row-stochastic transition matrix (`N×N`, `A(i,j) = P(state j at t+1 | state i at t)`);
- `B` — the row-stochastic emission matrix (`N×M`, `B(i,k) = P(symbol k | state i)`).

Validity — correct dimensions, non-negative entries, and rows summing **exactly**
to `1` — is checked over `Q` with **no tolerance**. An observation sequence is a
list of symbol indices in `[0, M)`.

## Public API (namespace `nimblecas`)

- `forward(model, obs)` — `α_t(i) = P(o_0..o_t, X_t = i)` by the sum–product
  recursion, and the likelihood `P(O | model) = Σ_i α_{T-1}(i)`. **Exact over Q.**
- `backward(model, obs)` — `β_t(i) = P(o_{t+1}..o_{T-1} | X_t = i)`. Exact; the two
  agree: `P(O) = Σ_i α_t(i) β_t(i)` for every `t`.
- `posteriors(model, obs)` — smoothing: `γ_t(i) = P(X_t = i | O)` and
  `ξ_t(i,j) = P(X_t = i, X_{t+1} = j | O)`, exact rational ratios.
- `viterbi(model, obs)` — the single most likely state path by max–product dynamic
  programming.
- `baum_welch(model, obs, …)` — Baum–Welch (EM) parameter re-estimation, exact over `Q`.
- `forward_scaled(model, obs)` — the **numerical** underflow/overflow-safe
  alternative to `forward`: `double` `α̂` rescaled to sum `1` each step, returning
  `log P(O)` as the sum of log scaling constants.

## Honesty boundary

- The exact algorithms (`forward`, `backward`, `posteriors`, `viterbi`) are exact
  rational arithmetic — sums of products of rationals, **no rounding**.
- **int64 ceiling.** The exact numerators/denominators can grow fast: a long
  sequence, or Baum–Welch's per-iteration re-estimation (which blows up
  denominators), can exceed `int64` and returns `MathError::overflow` — honestly,
  rather than silently wrapping. `forward_scaled` is the practical route for long
  sequences (numerical, with a documented log-likelihood).
- `P(O) = 0` — an observation impossible under the model — is guarded as
  `domain_error` rather than divided by zero in the posteriors.
- A `BigRational` backing would lift the int64 ceiling for the exact path; the
  current core is `int64`-precision by design.

## See also

- [`stochastic.md`](stochastic.md) — observed Markov chains (stationary distribution, WSS, PSD).
- [`smc.md`](smc.md) — numerical sequential Monte Carlo / particle filters for continuous state.
- [`mcmc.md`](mcmc.md) — Metropolis–Hastings sampling.
