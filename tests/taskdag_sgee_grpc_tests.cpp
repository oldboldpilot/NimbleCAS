// Cross-process integration test: NimbleCAS coordinator + sgee_queue_node + gRPC workers.
// @author Olumuyiwa Oluwasanmi
//
// The end-to-end proof of the M3b gRPC backend: a real sgee_queue_node process, two separate
// worker PROCESSES leasing over gRPC, and a NimbleCAS coordinator whose diamond-DAG outputs must
// be BIT-IDENTICAL to serial_executor. Linux-only (posix_spawn + sockets; the queue node is
// Linux-only regardless). An honest CTest skip (return 77) when NIMBLECAS_SGEE_QUEUE_NODE is unset.
//
// Env contract (from SGEE's own queue_node integration tests): a single node with
// SGEE_PEERS="1=[::1]:<cport>" self-elects (quorum == 1); leadership shows as `"is_leader": true`
// on GET http://[::1]:<hport>/statusz; the queue endpoint is [::1]:<qport>.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstring>

import std;
import nimblecas.core;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.testing;

#include "taskdag_sgee_diamond_ops.h"

extern char** environ;

using nimblecas::MathError;
using nimblecas::Payload;
using nimblecas::Result;
using nimblecas::SgeeExecutorConfig;
using nimblecas::SgeeGrpcExecutorOptions;
using nimblecas::TaskGraph;
using nimblecas::TaskId;
using nimblecas::TaskRegistry;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;
namespace ops = nimblecas::diamond_ops;

namespace {

[[nodiscard]] auto results_equal(const Result<Payload>& a, const Result<Payload>& b) -> bool {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    return a.has_value() ? (*a == *b) : (a.error() == b.error());
}

// Bind `n` IPv6 loopback sockets to port 0 simultaneously, read back the kernel-assigned ports,
// then close them all. Holding all n open while reading avoids handing out the same port twice.
// A small TOCTOU window remains (another process could grab a freed port before the node binds);
// there is no retry — such a race surfaces as a leadership-gate timeout that fails the test
// loudly (with the node's captured log), never a silent pass.
[[nodiscard]] auto pick_free_ports(std::size_t n) -> std::vector<int> {
    std::vector<int> fds;
    std::vector<int> ports;
    for (std::size_t i = 0; i < n; ++i) {
        const int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
        if (fd < 0) {
            break;
        }
        int on = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
        sockaddr_in6 addr{};
        addr.sin6_family = AF_INET6;
        addr.sin6_addr = in6addr_loopback;
        addr.sin6_port = 0;
        if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            ::close(fd);
            break;
        }
        socklen_t len = sizeof(addr);
        if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
            ::close(fd);
            break;
        }
        ports.push_back(ntohs(addr.sin6_port));
        fds.push_back(fd);
    }
    for (const int fd : fds) {
        ::close(fd);
    }
    return ports;
}

// GET path from http://[::1]:port, returning the raw response (headers + body) or "" on failure.
[[nodiscard]] auto http_get(int port, std::string_view path) -> std::string {
    const int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (fd < 0) {
        return {};
    }
    sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_loopback;
    addr.sin6_port = htons(static_cast<std::uint16_t>(port));
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return {};
    }
    const std::string req =
        std::format("GET {} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n", path);
    if (::write(fd, req.data(), req.size()) < 0) {
        ::close(fd);
        return {};
    }
    std::string resp;
    std::array<char, 4096> buf{};
    for (;;) {
        const auto got = ::read(fd, buf.data(), buf.size());
        if (got <= 0) {
            break;
        }
        resp.append(buf.data(), static_cast<std::size_t>(got));
    }
    ::close(fd);
    return resp;
}

// Build an environ-style char* array: the current environment with any SGEE_*/PORT entries
// removed, then `overrides` appended. Returned strings must outlive the posix_spawn call.
struct SpawnEnv {
    std::vector<std::string> storage;
    std::vector<char*> ptrs;
};
[[nodiscard]] auto make_env(const std::vector<std::string>& overrides) -> SpawnEnv {
    SpawnEnv env;
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
        std::string_view entry{*e};
        if (entry.starts_with("SGEE_") || entry.starts_with("PORT=")) {
            continue;
        }
        env.storage.emplace_back(*e);
    }
    for (const auto& o : overrides) {
        env.storage.push_back(o);
    }
    env.ptrs.reserve(env.storage.size() + 1);
    for (auto& s : env.storage) {
        env.ptrs.push_back(s.data());
    }
    env.ptrs.push_back(nullptr);
    return env;
}

// Spawn `path argv...` with `env`, redirecting stdout+stderr to `log_path`. Returns pid or -1.
[[nodiscard]] auto spawn_logged(const std::string& path, const std::vector<std::string>& argv,
                                SpawnEnv& env, const std::string& log_path) -> pid_t {
    std::vector<std::string> args_storage{path};
    args_storage.insert(args_storage.end(), argv.begin(), argv.end());
    std::vector<char*> args;
    args.reserve(args_storage.size() + 1);
    for (auto& s : args_storage) {
        args.push_back(s.data());
    }
    args.push_back(nullptr);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, log_path.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);

    pid_t pid = -1;
    const int rc = ::posix_spawn(&pid, path.c_str(), &fa, nullptr, args.data(), env.ptrs.data());
    posix_spawn_file_actions_destroy(&fa);
    return (rc == 0) ? pid : -1;
}

// SIGTERM a pid and reap it (bounded); SIGKILL if it will not go. Returns the exit status, or -1.
auto stop_process(pid_t pid, std::chrono::milliseconds grace = std::chrono::seconds(10)) -> int {
    if (pid <= 0) {
        return -1;
    }
    ::kill(pid, SIGTERM);
    const auto deadline = std::chrono::steady_clock::now() + grace;
    int status = 0;
    for (;;) {
        const pid_t r = ::waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            return status;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, &status, 0);
            return status;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

[[nodiscard]] auto read_file(const std::string& path) -> std::string {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Poll /statusz until the node reports leadership, or the budget elapses.
[[nodiscard]] auto await_leadership(int health_port, std::chrono::milliseconds budget) -> bool {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (std::chrono::steady_clock::now() < deadline) {
        const auto body = http_get(health_port, "/statusz");
        if (body.find("\"is_leader\": true") != std::string::npos) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

// One node + workers + temp dir, torn down in reverse on destruction.
struct Cluster {
    std::string node_path;
    std::string worker_path;
    std::filesystem::path data_dir;
    pid_t node_pid{-1};
    std::vector<pid_t> workers;
    int cport{0}, qport{0}, hport{0};

    ~Cluster() {
        for (auto it = workers.rbegin(); it != workers.rend(); ++it) {
            stop_process(*it);
        }
        stop_process(node_pid);
        std::error_code ec;
        std::filesystem::remove_all(data_dir, ec);
    }

    [[nodiscard]] auto endpoint() const -> std::string { return std::format("[::1]:{}", qport); }
    [[nodiscard]] auto node_log() const -> std::string {
        return (data_dir / "node.log").string();
    }
};

}  // namespace

auto main(int /*argc*/, char** /*argv*/) -> int {
    const char* node_env = std::getenv("NIMBLECAS_SGEE_QUEUE_NODE");
    if (node_env == nullptr || *node_env == '\0') {
        std::println("SKIP: set NIMBLECAS_SGEE_QUEUE_NODE to a built sgee_queue_node to run this test");
        return 77;  // CTest SKIP_RETURN_CODE
    }
    const std::string node_path = node_env;
    const char* worker_env = std::getenv("NIMBLECAS_SGEE_GRPC_WORKER");
    if (worker_env == nullptr || *worker_env == '\0') {
        std::println("SKIP: set NIMBLECAS_SGEE_GRPC_WORKER to the built worker binary");
        return 77;
    }
    const std::string worker_path = worker_env;
    if (!std::filesystem::exists(node_path) || !std::filesystem::exists(worker_path)) {
        std::println("SKIP: node ({}) or worker ({}) binary does not exist", node_path, worker_path);
        return 77;
    }

    return TestSuite("taskdag_sgee_grpc")
        .test("cross_process_diamond_dag_bit_identical_to_serial",
              [&](TestContext& t) {
                  // --- reference: serial_executor over the same diamond -----------------------
                  TaskRegistry ref_reg;
                  TaskGraph ref_g;
                  const auto ids = ops::build_diamond(ref_reg, ref_g);
                  const auto ser = nimblecas::serial_executor();
                  const auto ser_res = ser->run(ref_g).value();

                  // --- bring up node + two workers --------------------------------------------
                  const auto ports = pick_free_ports(3);
                  t.expect(ports.size() == 3, "picked three free loopback ports");
                  if (ports.size() != 3) {
                      return;
                  }

                  Cluster cluster;
                  cluster.node_path = node_path;
                  cluster.worker_path = worker_path;
                  cluster.cport = ports[0];
                  cluster.qport = ports[1];
                  cluster.hport = ports[2];
                  cluster.data_dir =
                      std::filesystem::temp_directory_path() /
                      std::format("nc-grpc-it-{}", static_cast<long>(::getpid()));
                  std::error_code ec;
                  std::filesystem::create_directories(cluster.data_dir, ec);

                  auto node_env_vars = make_env({
                      "SGEE_NODE_ID=1",
                      std::format("SGEE_PEERS=1=[::1]:{}", cluster.cport),
                      std::format("SGEE_CONSENSUS_PORT={}", cluster.cport),
                      std::format("SGEE_QUEUE_PORT={}", cluster.qport),
                      std::format("SGEE_DATA_DIR={}", cluster.data_dir.string()),
                      "SGEE_SNAPSHOT_RETENTION_MS=3600000",
                      std::format("PORT={}", cluster.hport),
                  });
                  cluster.node_pid =
                      spawn_logged(node_path, {}, node_env_vars, cluster.node_log());
                  t.expect(cluster.node_pid > 0, "sgee_queue_node spawned");
                  if (cluster.node_pid <= 0) {
                      return;
                  }

                  const bool led = await_leadership(cluster.hport, std::chrono::seconds(20));
                  t.expect(led, "node self-elected leader within 20s");
                  if (!led) {
                      std::println("--- node.log ---\n{}", read_file(cluster.node_log()));
                      return;
                  }

                  auto worker_env0 = make_env({});
                  auto worker_env1 = make_env({});
                  cluster.workers.push_back(spawn_logged(
                      worker_path, {"--endpoint", cluster.endpoint(), "--worker-id", "1"},
                      worker_env0, (cluster.data_dir / "worker1.log").string()));
                  cluster.workers.push_back(spawn_logged(
                      worker_path, {"--endpoint", cluster.endpoint(), "--worker-id", "2"},
                      worker_env1, (cluster.data_dir / "worker2.log").string()));
                  t.expect(cluster.workers[0] > 0 && cluster.workers[1] > 0, "two workers spawned");

                  // --- coordinator: gRPC distributed run, num_workers=0 (remote workers only) --
                  TaskRegistry reg;
                  TaskGraph g;
                  const auto gids = ops::build_diamond(reg, g);
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg)
                      .with_num_workers(0)
                      .with_poll_interval_ms(3)
                      .with_max_attempts(3)
                      .with_visibility_timeout_ms(30'000)
                      .with_run_deadline_ms(30'000);

                  auto exec_res = nimblecas::sgee_grpc_distributed_executor(
                      cfg, SgeeGrpcExecutorOptions{.endpoint = cluster.endpoint()});
                  t.expect(exec_res.has_value(), "gRPC distributed executor factory succeeds");
                  if (!exec_res.has_value()) {
                      return;
                  }
                  const auto dist = (*exec_res)->run(g);
                  t.expect(dist.has_value(), "distributed run completes without a transport abort");
                  if (!dist.has_value()) {
                      std::println("--- node.log ---\n{}", read_file(cluster.node_log()));
                      std::println("--- worker1.log ---\n{}",
                                   read_file((cluster.data_dir / "worker1.log").string()));
                      return;
                  }
                  const auto& dist_res = *dist;

                  // --- THE ACCEPTANCE ASSERTION: bit-identical to serial -----------------------
                  t.expect(dist_res.outputs.size() == ser_res.outputs.size(), "outputs size match");
                  bool bit_identical = (dist_res.outputs.size() == ser_res.outputs.size());
                  for (std::size_t i = 0; bit_identical && i < dist_res.outputs.size(); ++i) {
                      bit_identical = results_equal(dist_res.outputs[i], ser_res.outputs[i]);
                  }
                  t.expect(bit_identical,
                           "THE ACCEPTANCE TEST: cross-process gRPC outputs are BIT-IDENTICAL to serial");

                  t.expect(dist_res.outputs[gids.a.value].has_value() &&
                               ops::decode_i64(dist_res.outputs[gids.a.value].value()) == 7,
                           "A == 7");
                  t.expect(dist_res.outputs[gids.b.value].has_value() &&
                               ops::decode_i64(dist_res.outputs[gids.b.value].value()) == 14,
                           "B == 14");
                  t.expect(dist_res.outputs[gids.c.value].has_value() &&
                               ops::decode_i64(dist_res.outputs[gids.c.value].value()) == 10,
                           "C == 10");
                  t.expect(dist_res.outputs[gids.d.value].has_value() &&
                               ops::decode_i64(dist_res.outputs[gids.d.value].value()) == 24,
                           "D == 24");
                  t.expect(dist_res.outputs[gids.probe.value].has_value() &&
                               ops::decode_i64(dist_res.outputs[gids.probe.value].value()) == 14010,
                           "probe == 14010");
                  t.expect(dist_res.executed == 5, "all five diamond tasks were executed");

                  // Clean-teardown assertions (spec §5.1 step 9): stop the workers and node HERE and
                  // check each exited 0, then null the pids so the Cluster dtor does not double-reap.
                  for (auto& w : cluster.workers) {
                      const int st = stop_process(w);
                      w = -1;
                      t.expect(st >= 0 && WIFEXITED(st) && WEXITSTATUS(st) == 0,
                               "worker exited cleanly (status 0)");
                  }
                  const int node_st = stop_process(cluster.node_pid);
                  cluster.node_pid = -1;
                  t.expect(node_st >= 0 && WIFEXITED(node_st) && WEXITSTATUS(node_st) == 0,
                           "sgee_queue_node exited cleanly (status 0)");
                  (void)ids;
              })
        .test("run_completes_bit_identical_with_one_worker_killed_from_the_pool",
              [&](TestContext& t) {
                  // Resilience leg: spawn two workers, SIGKILL one, then run the diamond. The
                  // surviving worker completes every task and the outputs stay bit-identical,
                  // proving the coordinator tolerates a worker vanishing from the pool.
                  //
                  // HONESTY NOTE: the victim is killed BEFORE any task is enqueued, so it never
                  // holds a lease — this leg does NOT exercise the node's lease-expiry sweep. That
                  // path (a worker dying WITH a lease, the lease expiring, another worker picking it
                  // up) is covered deterministically in-process by the FakeBrokerPort
                  // organic-lease-expiry-retry test, and is SGEE's own autonomous behavior. Making
                  // it a real cross-process sweep test would require racing the kill against an
                  // in-flight lease; titling this leg as a sweep proof would be a false claim.
                  TaskRegistry ref_reg;
                  TaskGraph ref_g;
                  (void)ops::build_diamond(ref_reg, ref_g);
                  const auto ser = nimblecas::serial_executor();
                  const auto ser_res = ser->run(ref_g).value();

                  const auto ports = pick_free_ports(3);
                  if (ports.size() != 3) {
                      t.expect(false, "picked three free loopback ports");
                      return;
                  }
                  Cluster cluster;
                  cluster.node_path = node_path;
                  cluster.worker_path = worker_path;
                  cluster.cport = ports[0];
                  cluster.qport = ports[1];
                  cluster.hport = ports[2];
                  cluster.data_dir =
                      std::filesystem::temp_directory_path() /
                      std::format("nc-grpc-it-fault-{}", static_cast<long>(::getpid()));
                  std::error_code ec;
                  std::filesystem::create_directories(cluster.data_dir, ec);

                  auto node_env_vars = make_env({
                      "SGEE_NODE_ID=1",
                      std::format("SGEE_PEERS=1=[::1]:{}", cluster.cport),
                      std::format("SGEE_CONSENSUS_PORT={}", cluster.cport),
                      std::format("SGEE_QUEUE_PORT={}", cluster.qport),
                      std::format("SGEE_DATA_DIR={}", cluster.data_dir.string()),
                      "SGEE_SNAPSHOT_RETENTION_MS=3600000",
                      std::format("PORT={}", cluster.hport),
                  });
                  cluster.node_pid = spawn_logged(node_path, {}, node_env_vars, cluster.node_log());
                  if (cluster.node_pid <= 0 ||
                      !await_leadership(cluster.hport, std::chrono::seconds(20))) {
                      t.expect(false, "node came up and self-elected");
                      std::println("--- node.log ---\n{}", read_file(cluster.node_log()));
                      return;
                  }

                  auto e0 = make_env({});
                  auto e1 = make_env({});
                  cluster.workers.push_back(spawn_logged(
                      worker_path, {"--endpoint", cluster.endpoint(), "--worker-id", "1"}, e0,
                      (cluster.data_dir / "worker1.log").string()));
                  cluster.workers.push_back(spawn_logged(
                      worker_path, {"--endpoint", cluster.endpoint(), "--worker-id", "2"}, e1,
                      (cluster.data_dir / "worker2.log").string()));

                  // SIGKILL worker 2 before enqueuing any work (a crashed worker in the pool).
                  if (cluster.workers.size() == 2 && cluster.workers[1] > 0) {
                      ::kill(cluster.workers[1], SIGKILL);
                      int st = 0;
                      ::waitpid(cluster.workers[1], &st, 0);
                      cluster.workers[1] = -1;  // already reaped; don't double-reap in dtor
                  }

                  TaskRegistry reg;
                  TaskGraph g;
                  (void)ops::build_diamond(reg, g);
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg)
                      .with_num_workers(0)
                      .with_poll_interval_ms(3)
                      .with_max_attempts(3)
                      .with_visibility_timeout_ms(30'000)
                      .with_run_deadline_ms(30'000);
                  auto exec_res = nimblecas::sgee_grpc_distributed_executor(
                      cfg, SgeeGrpcExecutorOptions{.endpoint = cluster.endpoint()});
                  if (!exec_res.has_value()) {
                      t.expect(false, "factory succeeds");
                      return;
                  }
                  const auto dist = (*exec_res)->run(g);
                  t.expect(dist.has_value(), "run completes with one worker down");
                  if (!dist.has_value()) {
                      return;
                  }
                  bool bit_identical = (dist->outputs.size() == ser_res.outputs.size());
                  for (std::size_t i = 0; bit_identical && i < dist->outputs.size(); ++i) {
                      bit_identical = results_equal(dist->outputs[i], ser_res.outputs[i]);
                  }
                  t.expect(bit_identical, "outputs bit-identical to serial with one worker killed");
              })
        .run();
}
