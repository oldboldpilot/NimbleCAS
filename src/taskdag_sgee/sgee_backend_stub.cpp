// SGEE distributed taskdag backend stub when NIMBLECAS_SGEE=OFF.
// @author Olumuyiwa Oluwasanmi

module nimblecas.taskdag_sgee;

import std;
import nimblecas.core;

namespace nimblecas {

auto sgee_distributed_executor(SgeeExecutorConfig cfg) -> Result<std::unique_ptr<Executor>> {
    // Validate the config identically to the real (ON) factory so callers get a stable error for
    // the same input across build configs; a valid config then honestly reports not_implemented.
    if (cfg.registry == nullptr || cfg.wal_dir.empty() || cfg.num_workers == 0 ||
        cfg.max_attempts == 0 || cfg.visibility_timeout_ms == 0) {
        return make_error<std::unique_ptr<Executor>>(MathError::domain_error);
    }
    return make_error<std::unique_ptr<Executor>>(MathError::not_implemented);
}

}  // namespace nimblecas
