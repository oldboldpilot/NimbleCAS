# `nimblecas.modgcd_dist` — Reference

**Author:** Olumuyiwa Oluwasanmi

Source: `src/modgcd_dist/modgcd_dist.cppm`

Distributed driver and task-DAG operation registration for Brown's modular GCD algorithm
over univariate polynomials in $\mathbb{Z}[x]$ (ROADMAP §6.2). Distributes per-prime modular
image evaluations across an abstract [`Executor`](taskdag.md) seam (such as `serial_executor`,
`local_parallel_executor`, or [`SgeeDistributedExecutor`](taskdag_sgee.md)), lifts the
accumulated images coordinator-locally via symmetric CRT, and **strictly verifies the candidate
by exact trial division** before returning.

## Honesty boundary

This module is **exact and complete**:

- **Bit-identical exactness**: The distributed driver `modular_gcd_with` returns byte-identical
  results to the in-process exact GCD (`modular_gcd` and `Polynomial::gcd`) on any executor
  (Code Policy Rule 32).
- **Verified candidates only**: Every candidate $G$ must pass exact trial division
  ([`Polynomial::divide_exact`](polynomial.md)) on both inputs before acceptance. It **never**
  returns an unverified or approximate candidate.
- **Iteration budget**: If the prime budget (default 256 primes) is exhausted without producing
  a verified candidate, the function returns an honest `MathError::not_converged`.
- **Dynamic range**: If any lifted coefficient or the content-scaled product exceeds the `int64`
  range, the operation returns `MathError::overflow`.
- **Fault honesty**: A worker op returning a `MathError` is treated as data and propagated
  faithfully to the coordinator; transport-level failures abort the execution with
  `MathError::distributed_error` and are **never** silently retried into a math answer or
  masked as a math error.

```cpp
import nimblecas.modgcd_dist;
```

Depends on [`core`](core.md), [`polynomial`](polynomial.md), [`bigint`](bigint.md),
[`numbertheory`](numbertheory.md), [`taskdag`](taskdag.md), and [`modgcd`](modgcd.md).

## Exported API

| Function | Signature | Behavior |
| :--- | :--- | :--- |
| `register_modgcd_ops` | `auto register_modgcd_ops(TaskRegistry& reg) -> Result<void>` | Registers the modular GCD image evaluation op `nimblecas.modgcd.image/v1` into `reg`. |
| `modular_gcd_with` | `auto modular_gcd_with(const Polynomial& a, const Polynomial& b, Executor& exec, const TaskRegistry& reg, std::size_t max_primes = 256) -> Result<Polynomial>` | Executes Brown's modular GCD over `exec`, dispatching per-prime image tasks with bound literals and merging coordinator-locally. |

## Registered Operations

### `nimblecas.modgcd.image/v1`

- **Arguments**: Exactly 1 bound literal payload containing an encoded `ImageRequest`
  (magic `NCGQ`, version 1).
- **Returns**: Encoded `ImageResult` (magic `NCGI`, version 1) containing canonical residues in $[0, p)$.
- **Body**: Decodes `ImageRequest`, invokes [`gcd_image_mod_p`](modgcd.md), and encodes `ImageResult`.

## Error model

| Condition | Error |
| :--- | :--- |
| Malformed task argument or payload decode failure | `MathError::syntax_error` (propagated as task data) |
| Precondition violation in image evaluation ($lc \equiv 0 \pmod p$, $p$ out of range) | `MathError::domain_error` (propagated as task data) |
| Integer overflow during coefficient narrowing or content scaling | `MathError::overflow` |
| Prime budget exhausted without producing a verified candidate | `MathError::not_converged` |
| Transport failure, queue dead state, or lost result beyond recovery limit | `MathError::distributed_error` |

## Worked examples

```cpp
import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.modgcd_dist;

using namespace nimblecas;

// 1. Populate registry with modular GCD op
TaskRegistry reg;
auto reg_res = register_modgcd_ops(reg);

// 2. Set up polynomials: a = (x + 1)(x - 2), b = (x + 1)(x + 3)
Polynomial a({-2, -1, 1});
Polynomial b({3, 4, 1});

// 3. Execute over local parallel executor
auto par_exec = local_parallel_executor();
auto g_par = modular_gcd_with(a, b, *par_exec, reg).value();
// g_par == x + 1, coefficients: {1, 1}

// 4. Execute over SGEE distributed executor
FakeBrokerPort port;
InMemoryResultChannel results;
SgeeExecutorConfig cfg;
cfg.with_registry(reg).with_num_workers(2).with_poll_interval_ms(1);
SgeeDistributedExecutor dist_exec(cfg, port, results);

auto g_dist = modular_gcd_with(a, b, dist_exec, reg).value();
// g_dist == x + 1 (bit-identical to g_par and in-process modular_gcd)
```

## See also

- [`nimblecas.modgcd`](modgcd.md) — pure math core for Brown's modular GCD.
- [`nimblecas.taskdag`](taskdag.md) — local DAG execution and TaskRegistry.
- [`nimblecas.taskdag_sgee`](taskdag_sgee.md) — distributed SGEE execution backend.
- [`nimblecas.polynomial`](polynomial.md) — dense univariate polynomials.
- [Documentation hub](../Index.md)
