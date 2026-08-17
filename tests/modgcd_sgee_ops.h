// Shared modular GCD ops for the cross-process gRPC integration test.
// @author Olumuyiwa Oluwasanmi
//
// Included by BOTH the coordinator cluster test (modgcd_sgee_grpc_cluster_tests.cpp) and the worker
// binary (modgcd_sgee_worker.cpp) so the two processes register the IDENTICAL op set. The registry
// fingerprint is registration-order-independent (FNV-1a over sorted op-ids), so identical op-ids on
// both sides yield identical fingerprints — the invariant that lets a task envelope minted by the
// coordinator be accepted by the worker pump.
//
// The including TU MUST import nimblecas.core, nimblecas.polynomial, nimblecas.modgcd, and
// nimblecas.taskdag (with import std in scope) BEFORE including this header.

#ifndef NIMBLECAS_MODGCD_SGEE_OPS_H
#define NIMBLECAS_MODGCD_SGEE_OPS_H

namespace nimblecas::modgcd_ops {

inline constexpr std::string_view k_modgcd_image_op_id = "nimblecas.modgcd.image/v1";

// Registers the modular GCD image op "nimblecas.modgcd.image/v1" into reg.
// Decodes an ImageRequest from the bound literal, invokes gcd_image_mod_p, and encodes the resulting ZpImage.
[[nodiscard]] inline auto register_modgcd_ops(TaskRegistry& reg) -> Result<void> {
    return reg.register_op(k_modgcd_image_op_id, [](std::span<const Payload> ps) -> Result<Payload> {
        if (ps.empty()) {
            return make_error<Payload>(MathError::domain_error);
        }
        const auto req = decode_image_request(ps[0]);
        if (!req) {
            return make_error<Payload>(req.error());
        }
        const auto img = gcd_image_mod_p(req->a, req->b, req->p, req->gamma);
        if (!img) {
            return make_error<Payload>(img.error());
        }
        return encode_image_result(*img);
    });
}

// Registers a gated modular GCD image op into reg.
// When gate_file is specified:
// 1. If gate_entered_file is specified, writes "entered\n" when executing an image op.
// 2. Blocks execution until gate_file exists or a timeout occurs.
// 3. Executes the image kernel normally.
// Notice the OpId registered is still "nimblecas.modgcd.image/v1", so the registry fingerprint is identical.
[[nodiscard]] inline auto register_gated_modgcd_ops(TaskRegistry& reg,
                                                     const std::filesystem::path& gate_file = {},
                                                     const std::filesystem::path& gate_entered_file = {}) -> Result<void> {
    if (gate_file.empty()) {
        return register_modgcd_ops(reg);
    }
    return reg.register_op(k_modgcd_image_op_id,
                           [gate_file, gate_entered_file,
                            entered_flag = std::make_shared<std::atomic<bool>>(false)](
                               std::span<const Payload> ps) -> Result<Payload> {
                               if (!gate_entered_file.empty() && !entered_flag->exchange(true)) {
                                   std::ofstream f(gate_entered_file);
                                   f << "entered\n";
                               }
                               const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                               while (!std::filesystem::exists(gate_file)) {
                                   if (std::chrono::steady_clock::now() >= deadline) {
                                       return make_error<Payload>(MathError::distributed_error);
                                   }
                                   std::this_thread::sleep_for(std::chrono::milliseconds(10));
                               }
                               if (ps.empty()) {
                                   return make_error<Payload>(MathError::domain_error);
                               }
                               const auto req = decode_image_request(ps[0]);
                               if (!req) {
                                   return make_error<Payload>(req.error());
                               }
                               const auto img = gcd_image_mod_p(req->a, req->b, req->p, req->gamma);
                               if (!img) {
                                   return make_error<Payload>(img.error());
                               }
                               return encode_image_result(*img);
                           });
}

}  // namespace nimblecas::modgcd_ops

#endif  // NIMBLECAS_MODGCD_SGEE_OPS_H
