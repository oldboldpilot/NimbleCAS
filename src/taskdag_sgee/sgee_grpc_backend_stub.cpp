// SGEE cross-process gRPC backend stub when NIMBLECAS_SGEE_GRPC=OFF.
// @author Olumuyiwa Oluwasanmi

module nimblecas.taskdag_sgee;

import std;
import nimblecas.core;

namespace nimblecas {

auto sgee_grpc_distributed_executor(SgeeExecutorConfig cfg, SgeeGrpcExecutorOptions opts)
    -> Result<std::unique_ptr<Executor>> {
    // Validate identically to the real (ON) factory so callers get a stable error for the same
    // input across build configs; a valid config then honestly reports not_implemented. wal_dir is
    // ignored and num_workers == 0 is valid here too — mirror the ON factory's checks exactly.
    if (cfg.registry == nullptr || cfg.max_attempts == 0 || cfg.visibility_timeout_ms == 0 ||
        opts.endpoint.empty()) {
        return make_error<std::unique_ptr<Executor>>(MathError::domain_error);
    }
    return make_error<std::unique_ptr<Executor>>(MathError::not_implemented);
}

}  // namespace nimblecas
