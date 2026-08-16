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
    if (cfg.registry == nullptr || cfg.max_attempts == 0 || cfg.visibility_timeout_ms == 0) {
        return make_error<std::unique_ptr<Executor>>(MathError::domain_error);
    }

    // TLS validation: all-or-nothing (ca, cert, key must all be present if any TLS field is set)
    const bool has_ca = !opts.tls.ca_cert_path.empty();
    const bool has_cert = !opts.tls.cert_path.empty();
    const bool has_key = !opts.tls.key_path.empty();
    const bool has_override = !opts.tls.target_name_override.empty();
    const bool any_tls = has_ca || has_cert || has_key || has_override;
    const bool all_tls = has_ca && has_cert && has_key;
    if (any_tls && !all_tls) {
        return make_error<std::unique_ptr<Executor>>(MathError::domain_error);
    }

    // Endpoints validation
    if (!opts.endpoints.empty()) {
        for (const auto& ep : opts.endpoints) {
            if (ep.empty()) {
                return make_error<std::unique_ptr<Executor>>(MathError::domain_error);
            }
        }
    } else if (opts.endpoint.empty()) {
        return make_error<std::unique_ptr<Executor>>(MathError::domain_error);
    }

    return make_error<std::unique_ptr<Executor>>(MathError::not_implemented);
}

}  // namespace nimblecas
