// Shared diamond-DAG ops for the cross-process gRPC integration test.
// @author Olumuyiwa Oluwasanmi
//
// Included by BOTH the coordinator test (taskdag_sgee_grpc_tests.cpp) and the worker binary
// (taskdag_sgee_grpc_worker.cpp) so the two processes register the IDENTICAL op set. The registry
// fingerprint is registration-order-independent (FNV-1a over op-ids), so identical op-ids on both
// sides yield identical fingerprints — the invariant that lets a task envelope minted by the
// coordinator be accepted by the worker.
//
// The including TU MUST `import nimblecas.core;` and `import nimblecas.taskdag;` (and have
// `import std;` in scope) BEFORE including this header — it deliberately declares no imports so it
// can be dropped into either module-consuming TU.
#ifndef NIMBLECAS_TASKDAG_SGEE_DIAMOND_OPS_H
#define NIMBLECAS_TASKDAG_SGEE_DIAMOND_OPS_H

namespace nimblecas::diamond_ops {

[[nodiscard]] inline auto encode_i64(std::int64_t v) -> Payload {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(std::int64_t)>>(v);
    return Payload(bytes.begin(), bytes.end());
}

[[nodiscard]] inline auto decode_i64(std::span<const std::byte> p) -> std::int64_t {
    std::array<std::byte, sizeof(std::int64_t)> bytes{};
    // Copy at most 8 bytes — a payload wider than an i64 would otherwise overrun the array. Inputs
    // are always encode_i64 output (exactly 8 bytes), so this guard only hardens against misuse.
    std::ranges::copy_n(p.begin(), std::min<std::size_t>(p.size(), sizeof(std::int64_t)),
                        bytes.begin());
    return std::bit_cast<std::int64_t>(bytes);
}

struct DiamondIds {
    TaskId a{}, b{}, c{}, d{}, probe{};
};

// Register the five diamond ops into reg. Same op-ids as the in-process suite's build_diamond so
// the semantics (and expected outputs) are identical: A=7, B=A*2, C=A+3, D=B+C, probe=B*1000+C.
inline auto register_diamond_ops(TaskRegistry& reg) -> void {
    (void)reg.register_op("test.const7/v1", [](auto) -> Result<Payload> { return encode_i64(7); });
    (void)reg.register_op("test.mul2/v1",
                          [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) * 2); });
    (void)reg.register_op("test.add3/v1",
                          [](auto ps) -> Result<Payload> { return encode_i64(decode_i64(ps[0]) + 3); });
    (void)reg.register_op("test.add/v1", [](auto ps) -> Result<Payload> {
        return encode_i64(decode_i64(ps[0]) + decode_i64(ps[1]));
    });
    (void)reg.register_op("test.probe/v1", [](auto ps) -> Result<Payload> {
        return encode_i64(decode_i64(ps[0]) * 1000 + decode_i64(ps[1]));
    });
}

// Register the ops AND build the diamond DAG into g. Expected: A=7, B=14, C=10, D=24, probe=14010.
[[nodiscard]] inline auto build_diamond(TaskRegistry& reg, TaskGraph& g) -> DiamondIds {
    register_diamond_ops(reg);
    DiamondIds ids;
    ids.a = g.add_named_task(reg, "test.const7/v1").value();
    ids.b = g.add_named_task(reg, "test.mul2/v1", std::vector<TaskId>{ids.a}).value();
    ids.c = g.add_named_task(reg, "test.add3/v1", std::vector<TaskId>{ids.a}).value();
    ids.d = g.add_named_task(reg, "test.add/v1", std::vector<TaskId>{ids.b, ids.c}).value();
    ids.probe = g.add_named_task(reg, "test.probe/v1", std::vector<TaskId>{ids.b, ids.c}).value();
    return ids;
}

}  // namespace nimblecas::diamond_ops

#endif  // NIMBLECAS_TASKDAG_SGEE_DIAMOND_OPS_H
