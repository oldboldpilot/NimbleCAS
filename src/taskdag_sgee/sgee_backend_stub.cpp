// SGEE distributed taskdag backend stub when NIMBLECAS_SGEE=OFF.
// @author Olumuyiwa Oluwasanmi

module nimblecas.taskdag_sgee;

import std;
import nimblecas.core;

namespace nimblecas {

auto sgee_distributed_executor(SgeeExecutorConfig cfg) -> Result<std::unique_ptr<Executor>> {
    if (cfg.registry == nullptr || cfg.num_workers == 0 || cfg.max_attempts == 0) {
        return make_error<std::unique_ptr<Executor>>(MathError::domain_error);
    }
    return make_error<std::unique_ptr<Executor>>(MathError::not_implemented);
}

}  // namespace nimblecas
