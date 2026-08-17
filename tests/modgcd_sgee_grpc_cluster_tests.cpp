// 3-node SGEE Raft cluster cross-process integration test for distributed modular polynomial GCD.
// @author Olumuyiwa Oluwasanmi
//
// End-to-end proof of M5 distributed modular polynomial GCD cluster capabilities:
// (a) Q1: Modular GCD computed across 3-node Raft quorum with mTLS and 2 external workers,
//     asserted BIT-IDENTICAL to in-process modular_gcd and Polynomial::gcd.
// (b) Q2: Leader failover mid-round during image computation: leader is SIGKILL'd mid-run,
//     survivor node takes leadership with higher term, and modular_gcd_with completes bit-identically
//     with failover rotations verified.
// (c) Q3: mTLS negative security leg: client presenting no client certificate is rejected at the transport
//     layer with MathError::distributed_error upon attempting RPCs.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdlib>
#include <cstring>

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.modgcd;
import nimblecas.modgcd_dist;
import nimblecas.taskdag;
import nimblecas.taskdag_sgee;
import nimblecas.testing;

#include "modgcd_sgee_ops.h"

extern char** environ;

using nimblecas::BrokerPort;
using nimblecas::GrpcBrokerPort;
using nimblecas::GrpcResultChannel;
using nimblecas::MathError;
using nimblecas::Payload;
using nimblecas::Polynomial;
using nimblecas::Result;
using nimblecas::SgeeDistributedExecutor;
using nimblecas::SgeeExecutorConfig;
using nimblecas::SgeeGrpcExecutorOptions;
using nimblecas::SgeeGrpcTlsOptions;
using nimblecas::sgee_grpc_distributed_executor;
using nimblecas::TaskGraph;
using nimblecas::TaskId;
using nimblecas::TaskRegistry;
using nimblecas::TaskRunResult;
using nimblecas::modular_gcd;
using nimblecas::modular_gcd_with;
using nimblecas::register_modgcd_ops;
using nimblecas::testing::TestContext;
using nimblecas::testing::TestSuite;

namespace {

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

struct NodeStatus {
    bool ok{false};
    bool is_leader{false};
    std::uint64_t current_term{0};
};

[[nodiscard]] auto get_node_status(int health_port) -> NodeStatus {
    const auto body = http_get(health_port, "/statusz");
    if (body.empty()) {
        return NodeStatus{.ok = false};
    }
    NodeStatus st{.ok = true};
    st.is_leader = (body.find("\"is_leader\": true") != std::string::npos);
    const auto term_pos = body.find("\"current_term\":");
    if (term_pos != std::string::npos) {
        const auto val_start = body.find_first_of("0123456789", term_pos + 15);
        if (val_start != std::string::npos) {
            const auto val_end = body.find_first_not_of("0123456789", val_start);
            const auto term_str = body.substr(
                val_start, val_end == std::string::npos ? std::string::npos : val_end - val_start);
            st.current_term = std::strtoull(term_str.c_str(), nullptr, 10);
        }
    }
    return st;
}

struct LeaderInfo {
    int node_index{-1};
    std::uint64_t term{0};
    int health_port{0};
};

[[nodiscard]] auto poll_leader(const std::vector<int>& health_ports,
                               const std::vector<pid_t>& pids) -> std::pair<int, LeaderInfo> {
    int count = 0;
    LeaderInfo info;
    for (std::size_t i = 0; i < health_ports.size(); ++i) {
        if (pids[i] <= 0) {
            continue;
        }
        auto st = get_node_status(health_ports[i]);
        if (st.ok && st.is_leader) {
            count++;
            info.node_index = static_cast<int>(i);
            info.term = st.current_term;
            info.health_port = health_ports[i];
        }
    }
    return {count, info};
}

[[nodiscard]] auto await_stable_leader(const std::vector<int>& health_ports,
                                       const std::vector<pid_t>& pids,
                                       std::chrono::milliseconds budget) -> std::optional<LeaderInfo> {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (std::chrono::steady_clock::now() < deadline) {
        auto [count, info] = poll_leader(health_ports, pids);
        if (count == 1) {
            return info;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return std::nullopt;
}

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

struct Certs {
    std::filesystem::path dir;
    std::string ca_crt;
    std::string ca_key;
    std::string node_crt;
    std::string node_key;
    std::string client_crt;
    std::string client_key;
};

[[nodiscard]] auto generate_test_certs(const std::filesystem::path& dir,
                                       const std::string& script_path) -> std::optional<Certs> {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    std::string cmd = std::format("bash \"{}\" \"{}\"", script_path, dir.string());
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        return std::nullopt;
    }

    Certs certs;
    certs.dir = dir;
    certs.ca_crt = (dir / "ca.crt").string();
    certs.ca_key = (dir / "ca.key").string();
    certs.node_crt = (dir / "node.crt").string();
    certs.node_key = (dir / "node.key").string();
    certs.client_crt = (dir / "client.crt").string();
    certs.client_key = (dir / "client.key").string();

    if (!std::filesystem::exists(certs.ca_crt) ||
        !std::filesystem::exists(certs.node_crt) ||
        !std::filesystem::exists(certs.node_key) ||
        !std::filesystem::exists(certs.client_crt) ||
        !std::filesystem::exists(certs.client_key)) {
        return std::nullopt;
    }
    return certs;
}

struct ClusterNode {
    pid_t pid{-1};
    int cport{0};
    int qport{0};
    int hport{0};
    std::filesystem::path data_dir;
};

struct Cluster3Node {
    std::string node_path;
    std::string worker_path;
    std::filesystem::path base_dir;
    std::vector<ClusterNode> nodes;
    std::vector<pid_t> workers;

    ~Cluster3Node() {
        for (auto it = workers.rbegin(); it != workers.rend(); ++it) {
            stop_process(*it);
        }
        for (auto& n : nodes) {
            stop_process(n.pid);
        }
        std::error_code ec;
        std::filesystem::remove_all(base_dir, ec);
    }

    [[nodiscard]] auto endpoint(std::size_t i) const -> std::string {
        return std::format("[::1]:{}", nodes[i].qport);
    }

    [[nodiscard]] auto endpoints() const -> std::vector<std::string> {
        std::vector<std::string> eps;
        for (const auto& n : nodes) {
            eps.push_back(std::format("[::1]:{}", n.qport));
        }
        return eps;
    }

    [[nodiscard]] auto hports() const -> std::vector<int> {
        std::vector<int> hp;
        for (const auto& n : nodes) {
            hp.push_back(n.hport);
        }
        return hp;
    }

    [[nodiscard]] auto pids() const -> std::vector<pid_t> {
        std::vector<pid_t> p;
        for (const auto& n : nodes) {
            p.push_back(n.pid);
        }
        return p;
    }

    [[nodiscard]] auto node_log(std::size_t i) const -> std::string {
        return (nodes[i].data_dir / "node.log").string();
    }

    [[nodiscard]] auto worker_log(std::size_t i) const -> std::string {
        return (base_dir / std::format("worker{}.log", i + 1)).string();
    }
};

}  // namespace

auto main(int /*argc*/, char** /*argv*/) -> int {
    const char* node_env = std::getenv("NIMBLECAS_SGEE_QUEUE_NODE");
    if (node_env == nullptr || *node_env == '\0') {
        std::println("SKIP: set NIMBLECAS_SGEE_QUEUE_NODE to a built sgee_queue_node to run this test");
        return 77;
    }
    const std::string node_path = node_env;
    const char* worker_env = std::getenv("NIMBLECAS_MODGCD_GRPC_WORKER");
    if (worker_env == nullptr || *worker_env == '\0') {
        std::println("SKIP: set NIMBLECAS_MODGCD_GRPC_WORKER to the built worker binary");
        return 77;
    }
    const std::string worker_path = worker_env;
    if (!std::filesystem::exists(node_path) || !std::filesystem::exists(worker_path)) {
        std::println("SKIP: node ({}) or worker ({}) binary does not exist", node_path, worker_path);
        return 77;
    }

    const char* cert_script_env = std::getenv("NIMBLECAS_GEN_TEST_CERTS_SH");
    std::string cert_script_path = (cert_script_env != nullptr && *cert_script_env != '\0')
                                       ? std::string(cert_script_env)
                                       : "tests/gen_test_certs.sh";
    if (!std::filesystem::exists(cert_script_path)) {
        if (std::filesystem::exists("../tests/gen_test_certs.sh")) {
            cert_script_path = "../tests/gen_test_certs.sh";
        } else {
            std::println("SKIP: gen_test_certs.sh script not found at {}", cert_script_path);
            return 77;
        }
    }

    // Verify openssl availability by generating throwaway test certs
    const auto probe_dir = std::filesystem::temp_directory_path() /
                           std::format("nc-modgcd-cluster-probe-{}", static_cast<long>(::getpid()));
    const auto probe_certs = generate_test_certs(probe_dir, cert_script_path);
    std::error_code ec;
    std::filesystem::remove_all(probe_dir, ec);
    if (!probe_certs) {
        std::println("SKIP: openssl not available or certificate generation failed");
        return 77;
    }

    return TestSuite("modgcd_sgee_grpc_cluster")
        .test("q1_modular_gcd_across_3_node_mtls_quorum_bit_identical_to_inproc",
              [&](TestContext& t) {
                  // --- (Q1) HEADLINE: 3-NODE QUORUM MODULAR GCD OVER mTLS -----------------------
                  const auto ports = pick_free_ports(9);
                  t.expect(ports.size() == 9, "picked 9 free loopback ports");
                  if (ports.size() != 9) {
                      return;
                  }

                  Cluster3Node cluster;
                  cluster.node_path = node_path;
                  cluster.worker_path = worker_path;
                  cluster.base_dir =
                      std::filesystem::temp_directory_path() /
                      std::format("nc-modgcd-cluster-q1-{}", static_cast<long>(::getpid()));
                  std::filesystem::create_directories(cluster.base_dir, ec);

                  const auto certs_opt = generate_test_certs(cluster.base_dir / "certs", cert_script_path);
                  t.expect(certs_opt.has_value(), "test certificates generated");
                  if (!certs_opt) {
                      return;
                  }
                  const auto& certs = *certs_opt;

                  for (std::size_t i = 0; i < 3; ++i) {
                      ClusterNode n;
                      n.cport = ports[i * 3 + 0];
                      n.qport = ports[i * 3 + 1];
                      n.hport = ports[i * 3 + 2];
                      n.data_dir = cluster.base_dir / std::format("node{}", i);
                      std::filesystem::create_directories(n.data_dir, ec);
                      cluster.nodes.push_back(std::move(n));
                  }

                  const std::string peers = std::format(
                      "1=[::1]:{},2=[::1]:{},3=[::1]:{}",
                      cluster.nodes[0].cport, cluster.nodes[1].cport, cluster.nodes[2].cport);

                  for (std::size_t i = 0; i < 3; ++i) {
                      auto env_vars = make_env({
                          std::format("SGEE_NODE_ID={}", i + 1),
                          std::format("SGEE_PEERS={}", peers),
                          std::format("SGEE_CONSENSUS_PORT={}", cluster.nodes[i].cport),
                          std::format("SGEE_QUEUE_PORT={}", cluster.nodes[i].qport),
                          std::format("SGEE_DATA_DIR={}", cluster.nodes[i].data_dir.string()),
                          "SGEE_SNAPSHOT_RETENTION_MS=3600000",
                          "SGEE_ELECTION_TIMEOUT_MS=500",
                          "SGEE_HEARTBEAT_MS=150",
                          "SGEE_FORCE_BLOCKING_IO=1",
                          std::format("PORT={}", cluster.nodes[i].hport),
                          std::format("SGEE_TLS_CA_CERT={}", certs.ca_crt),
                          std::format("SGEE_TLS_CERT={}", certs.node_crt),
                          std::format("SGEE_TLS_KEY={}", certs.node_key),
                      });
                      cluster.nodes[i].pid =
                          spawn_logged(node_path, {}, env_vars, cluster.node_log(i));
                      t.expect(cluster.nodes[i].pid > 0, std::format("node {} spawned", i + 1));
                      if (cluster.nodes[i].pid <= 0) {
                          return;
                      }
                  }

                  const auto leader_opt =
                      await_stable_leader(cluster.hports(), cluster.pids(), std::chrono::seconds(20));
                  t.expect(leader_opt.has_value(), "cluster elected leader within 20s");
                  if (!leader_opt) {
                      for (std::size_t i = 0; i < 3; ++i) {
                          std::println("--- node{} log ---\n{}", i, read_file(cluster.node_log(i)));
                      }
                      return;
                  }

                  // Quorum agreement verification: all 3 nodes report ok and match term
                  std::size_t q1_leaders = 0;
                  std::size_t q1_followers = 0;
                  for (std::size_t i = 0; i < 3; ++i) {
                      const auto st = get_node_status(cluster.nodes[i].hport);
                      t.expect(st.ok, std::format("node {} is healthy", i + 1));
                      if (st.ok) {
                          t.expect(st.current_term == leader_opt->term,
                                   std::format("node {} matches quorum term {}", i + 1, leader_opt->term));
                          if (st.is_leader) {
                              ++q1_leaders;
                          } else {
                              ++q1_followers;
                          }
                      }
                  }
                  t.expect(q1_leaders == 1 && q1_followers == 2,
                           "Raft quorum verified: exactly 1 leader and 2 followers across 3 nodes");

                  // Spawn 2 external modgcd_sgee_worker processes with all 3 endpoints + mTLS
                  const auto eps = cluster.endpoints();
                  std::vector<std::string> worker1_args = {
                      "--endpoint", eps[0], "--endpoint", eps[1], "--endpoint", eps[2],
                      "--worker-id", "1",
                      "--ca-cert", certs.ca_crt,
                      "--cert", certs.client_crt,
                      "--key", certs.client_key,
                  };
                  std::vector<std::string> worker2_args = {
                      "--endpoint", eps[0], "--endpoint", eps[1], "--endpoint", eps[2],
                      "--worker-id", "2",
                      "--ca-cert", certs.ca_crt,
                      "--cert", certs.client_crt,
                      "--key", certs.client_key,
                  };
                  auto w_env1 = make_env({});
                  auto w_env2 = make_env({});
                  cluster.workers.push_back(
                      spawn_logged(worker_path, worker1_args, w_env1, cluster.worker_log(0)));
                  cluster.workers.push_back(
                      spawn_logged(worker_path, worker2_args, w_env2, cluster.worker_log(1)));
                  t.expect(cluster.workers[0] > 0 && cluster.workers[1] > 0, "two modgcd workers spawned");
                  if (cluster.workers[0] <= 0 || cluster.workers[1] <= 0) {
                      return;
                  }

                  // T4 inputs:
                  // a = 3x⁵+2x⁴+22x³+19x²+7x+35 [35, 7, 19, 22, 2, 3]
                  // b = 3x⁵-x⁴+32x³+26x²+6x+55  [55, 6, 26, 32, -1, 3]
                  // expected gcd = 3x³+2x²+x+5  [5, 1, 2, 3]
                  const Polynomial a({35, 7, 19, 22, 2, 3});
                  const Polynomial b({55, 6, 26, 32, -1, 3});

                  TaskRegistry reg;
                  auto reg_res = nimblecas::register_modgcd_ops(reg);
                  t.expect(reg_res.has_value(), "register_modgcd_ops succeeds on coordinator");
                  if (!reg_res.has_value()) {
                      return;
                  }

                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg)
                      .with_num_workers(0)
                      .with_poll_interval_ms(3)
                      .with_max_attempts(3)
                      .with_visibility_timeout_ms(30'000)
                      .with_run_deadline_ms(30'000)
                      .with_max_result_recoveries(3);

                  SgeeGrpcTlsOptions tls{
                      .ca_cert_path = certs.ca_crt,
                      .cert_path = certs.client_crt,
                      .key_path = certs.client_key,
                  };
                  SgeeGrpcExecutorOptions opts{
                      .endpoints = eps,
                      .tls = tls,
                  };

                  auto exec_res = nimblecas::sgee_grpc_distributed_executor(cfg, opts);
                  t.expect(exec_res.has_value(), "gRPC distributed executor factory succeeds");
                  if (!exec_res.has_value()) {
                      return;
                  }

                  // Distributed execution across the 3-node quorum
                  const auto dist_gcd = nimblecas::modular_gcd_with(a, b, **exec_res, reg);
                  t.expect(dist_gcd.has_value(), "distributed modular_gcd_with completes without transport abort");
                  if (!dist_gcd.has_value()) {
                      for (std::size_t i = 0; i < 3; ++i) {
                          std::println("--- node{} log ---\n{}", i, read_file(cluster.node_log(i)));
                      }
                      std::println("--- worker1 log ---\n{}", read_file(cluster.worker_log(0)));
                      std::println("--- worker2 log ---\n{}", read_file(cluster.worker_log(1)));
                      return;
                  }

                  // In-process references
                  const auto inproc_gcd = nimblecas::modular_gcd(a, b);
                  t.expect(inproc_gcd.has_value(), "in-process modular_gcd succeeds");
                  if (!inproc_gcd.has_value()) {
                      return;
                  }

                  const auto poly_gcd = a.gcd(b);
                  t.expect(poly_gcd.has_value(), "Polynomial::gcd succeeds");
                  if (!poly_gcd.has_value()) {
                      return;
                  }

                  // THE HEADLINE ACCEPTANCE CLAIM: BIT-IDENTITY
                  t.expect(dist_gcd->is_equal(*inproc_gcd),
                           "THE ACCEPTANCE TEST (Q1): distributed modular GCD is BIT-IDENTICAL to in-process modular_gcd");
                  t.expect(dist_gcd->is_equal(*poly_gcd),
                           "THE ACCEPTANCE TEST (Q1): distributed modular GCD is BIT-IDENTICAL to Polynomial::gcd");

                  // Exact coefficient validation: [5, 1, 2, 3]
                  t.expect(dist_gcd->degree() == 3, "GCD degree is 3");
                  t.expect(dist_gcd->coefficient(0) == 5, "c0 == 5");
                  t.expect(dist_gcd->coefficient(1) == 1, "c1 == 1");
                  t.expect(dist_gcd->coefficient(2) == 2, "c2 == 2");
                  t.expect(dist_gcd->coefficient(3) == 3, "c3 == 3");

                  // Clean teardown and worker log checks
                  for (std::size_t i = 0; i < cluster.workers.size(); ++i) {
                      const int st = stop_process(cluster.workers[i]);
                      cluster.workers[i] = -1;
                      t.expect(st >= 0 && WIFEXITED(st) && WEXITSTATUS(st) == 0,
                               std::format("worker {} exited cleanly (status 0)", i + 1));
                      const auto w_log = read_file(cluster.worker_log(i));
                      t.expect(w_log.find(std::format("worker {}: clean stop", i + 1)) != std::string::npos,
                               std::format("worker {} completed pump and logged clean stop", i + 1));
                  }
                  for (std::size_t i = 0; i < cluster.nodes.size(); ++i) {
                      const int st = stop_process(cluster.nodes[i].pid);
                      cluster.nodes[i].pid = -1;
                      t.expect(st >= 0 && WIFEXITED(st) && WEXITSTATUS(st) == 0,
                               std::format("node {} exited cleanly (status 0)", i + 1));
                  }
              })
        .test("q2_leader_failover_during_images_retains_bit_identity_with_failover_rotations",
              [&](TestContext& t) {
                  // --- (Q2) LEADER FAILOVER MID-ROUND DURING IMAGE COMPUTATION -------------------
                  const auto ports = pick_free_ports(9);
                  t.expect(ports.size() == 9, "picked 9 free loopback ports");
                  if (ports.size() != 9) {
                      return;
                  }

                  Cluster3Node cluster;
                  cluster.node_path = node_path;
                  cluster.worker_path = worker_path;
                  cluster.base_dir =
                      std::filesystem::temp_directory_path() /
                      std::format("nc-modgcd-cluster-q2-{}", static_cast<long>(::getpid()));
                  std::filesystem::create_directories(cluster.base_dir, ec);

                  const auto gate_path = cluster.base_dir / "gate.lock";
                  const auto gate_entered_path = cluster.base_dir / "gate_entered.lock";

                  // Ensure gate files are clean
                  std::filesystem::remove(gate_path, ec);
                  std::filesystem::remove(gate_entered_path, ec);

                  const auto certs_opt = generate_test_certs(cluster.base_dir / "certs", cert_script_path);
                  t.expect(certs_opt.has_value(), "test certificates generated");
                  if (!certs_opt) {
                      return;
                  }
                  const auto& certs = *certs_opt;

                  for (std::size_t i = 0; i < 3; ++i) {
                      ClusterNode n;
                      n.cport = ports[i * 3 + 0];
                      n.qport = ports[i * 3 + 1];
                      n.hport = ports[i * 3 + 2];
                      n.data_dir = cluster.base_dir / std::format("node{}", i);
                      std::filesystem::create_directories(n.data_dir, ec);
                      cluster.nodes.push_back(std::move(n));
                  }

                  const std::string peers = std::format(
                      "1=[::1]:{},2=[::1]:{},3=[::1]:{}",
                      cluster.nodes[0].cport, cluster.nodes[1].cport, cluster.nodes[2].cport);

                  for (std::size_t i = 0; i < 3; ++i) {
                      auto env_vars = make_env({
                          std::format("SGEE_NODE_ID={}", i + 1),
                          std::format("SGEE_PEERS={}", peers),
                          std::format("SGEE_CONSENSUS_PORT={}", cluster.nodes[i].cport),
                          std::format("SGEE_QUEUE_PORT={}", cluster.nodes[i].qport),
                          std::format("SGEE_DATA_DIR={}", cluster.nodes[i].data_dir.string()),
                          "SGEE_SNAPSHOT_RETENTION_MS=3600000",
                          "SGEE_ELECTION_TIMEOUT_MS=80",
                          "SGEE_HEARTBEAT_MS=25",
                          "SGEE_FORCE_BLOCKING_IO=1",
                          std::format("PORT={}", cluster.nodes[i].hport),
                          std::format("SGEE_TLS_CA_CERT={}", certs.ca_crt),
                          std::format("SGEE_TLS_CERT={}", certs.node_crt),
                          std::format("SGEE_TLS_KEY={}", certs.node_key),
                      });
                      cluster.nodes[i].pid =
                          spawn_logged(node_path, {}, env_vars, cluster.node_log(i));
                      t.expect(cluster.nodes[i].pid > 0, std::format("node {} spawned", i + 1));
                      if (cluster.nodes[i].pid <= 0) {
                          return;
                      }
                  }

                  const auto initial_leader_opt =
                      await_stable_leader(cluster.hports(), cluster.pids(), std::chrono::seconds(20));
                  t.expect(initial_leader_opt.has_value(), "cluster formed and elected initial leader");
                  if (!initial_leader_opt) {
                      return;
                  }

                  const auto eps = cluster.endpoints();
                  // Spawn worker 1 with gate file and short lease timeout (1000ms)
                  std::vector<std::string> worker1_args = {
                      "--endpoint", eps[0], "--endpoint", eps[1], "--endpoint", eps[2],
                      "--worker-id", "1",
                      "--lease-timeout-ms", "1000",
                      "--ca-cert", certs.ca_crt,
                      "--cert", certs.client_crt,
                      "--key", certs.client_key,
                      "--gate-file", gate_path.string(),
                      "--gate-entered-file", gate_entered_path.string(),
                  };
                  // Spawn worker 2 as regular worker
                  std::vector<std::string> worker2_args = {
                      "--endpoint", eps[0], "--endpoint", eps[1], "--endpoint", eps[2],
                      "--worker-id", "2",
                      "--lease-timeout-ms", "1000",
                      "--ca-cert", certs.ca_crt,
                      "--cert", certs.client_crt,
                      "--key", certs.client_key,
                  };
                  auto w_env1 = make_env({});
                  auto w_env2 = make_env({});
                  cluster.workers.push_back(
                      spawn_logged(worker_path, worker1_args, w_env1, cluster.worker_log(0)));
                  cluster.workers.push_back(
                      spawn_logged(worker_path, worker2_args, w_env2, cluster.worker_log(1)));
                  t.expect(cluster.workers[0] > 0 && cluster.workers[1] > 0, "two modgcd workers spawned");
                  if (cluster.workers[0] <= 0 || cluster.workers[1] <= 0) {
                      return;
                  }

                  // Inputs T4
                  const Polynomial a({35, 7, 19, 22, 2, 3});
                  const Polynomial b({55, 6, 26, 32, -1, 3});

                  TaskRegistry reg;
                  auto reg_res = nimblecas::register_modgcd_ops(reg);
                  t.expect(reg_res.has_value(), "register_modgcd_ops succeeds");
                  if (!reg_res.has_value()) {
                      return;
                  }

                  SgeeGrpcTlsOptions tls{
                      .ca_cert_path = certs.ca_crt,
                      .cert_path = certs.client_crt,
                      .key_path = certs.client_key,
                  };

                  auto port_res = GrpcBrokerPort::connect(eps, /*auth_token=*/"", /*rpc_deadline_ms=*/0, tls);
                  t.expect(port_res.has_value(), "connected GrpcBrokerPort with Raft ring");
                  if (!port_res.has_value()) {
                      return;
                  }
                  auto channel = std::make_unique<GrpcResultChannel>((*port_res)->ring(), /*consume_on_get=*/false);

                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg)
                      .with_num_workers(0)
                      .with_poll_interval_ms(10)
                      .with_max_attempts(5)
                      .with_visibility_timeout_ms(1'000)
                      .with_run_deadline_ms(120'000)
                      .with_max_result_recoveries(5);

                  SgeeDistributedExecutor exec(cfg, **port_res, *channel);

                  // Run modular_gcd_with on runner thread
                  Result<Polynomial> dist_gcd;
                  std::thread runner([&]() {
                      dist_gcd = nimblecas::modular_gcd_with(a, b, exec, reg);
                  });

                  // Wait until a worker is actively executing an image task mid-round
                  bool task_leased = false;
                  const auto lease_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                  while (std::chrono::steady_clock::now() < lease_deadline) {
                      if (std::filesystem::exists(gate_entered_path)) {
                          task_leased = true;
                          break;
                      }
                      std::this_thread::sleep_for(std::chrono::milliseconds(10));
                  }
                  t.expect(task_leased, "image task was leased by worker mid-round");
                  if (!task_leased) {
                      {
                          std::ofstream f(gate_path);
                          f << "release\n";
                      }
                      runner.join();
                      return;
                  }

                  // Identify current leader
                  auto [curr_count, curr_leader] = poll_leader(cluster.hports(), cluster.pids());
                  t.expect(curr_count == 1, "exactly one leader identified before kill");
                  if (curr_count != 1) {
                      {
                          std::ofstream f(gate_path);
                          f << "release\n";
                      }
                      runner.join();
                      return;
                  }
                  const int victim_idx = curr_leader.node_index;
                  const std::uint64_t victim_term = curr_leader.term;
                  t.expect(victim_idx >= 0 && victim_idx < 3, "victim index is valid");
                  if (victim_idx < 0 || victim_idx >= 3) {
                      {
                          std::ofstream f(gate_path);
                          f << "release\n";
                      }
                      runner.join();
                      return;
                  }

                  // Snapshot failover rotations immediately before kill to prove strict increase
                  const auto rotations_before = (*port_res)->failover_rotations();

                  // SIGKILL the leader
                  const pid_t victim_pid = cluster.nodes[victim_idx].pid;
                  t.expect(victim_pid > 0, "victim node pid is valid");
                  if (victim_pid <= 0) {
                      {
                          std::ofstream f(gate_path);
                          f << "release\n";
                      }
                      runner.join();
                      return;
                  }
                  ::kill(victim_pid, SIGKILL);
                  int kill_st = 0;
                  ::waitpid(victim_pid, &kill_st, 0);
                  cluster.nodes[victim_idx].pid = -1;

                  // Wait for a survivor to report is_leader with a HIGHER term
                  bool new_leader_elected = false;
                  LeaderInfo new_leader{};
                  const auto election_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                  while (std::chrono::steady_clock::now() < election_deadline) {
                      auto [count, info] = poll_leader(cluster.hports(), cluster.pids());
                      if (count == 1 && info.node_index != victim_idx && info.term > victim_term) {
                          new_leader_elected = true;
                          new_leader = info;
                          break;
                      }
                      std::this_thread::sleep_for(std::chrono::milliseconds(50));
                  }
                  t.expect(new_leader_elected, "survivor elected leader with higher term after kill");
                  t.expect(new_leader.node_index != victim_idx, "new leader is a survivor (id != killed)");
                  t.expect(new_leader.term > victim_term, "new leader term > killed leader term");

                  // Release the gate
                  {
                      std::ofstream f(gate_path);
                      f << "release\n";
                  }

                  runner.join();

                  t.expect(dist_gcd.has_value(), "modular_gcd_with completed successfully after failover");
                  if (!dist_gcd.has_value()) {
                      std::println("FAILOVER MODGCD ERROR: {}", static_cast<int>(dist_gcd.error()));
                      for (std::size_t i = 0; i < 3; ++i) {
                          std::println("--- node{} log ---\n{}", i, read_file(cluster.node_log(i)));
                      }
                      std::println("--- worker1 log ---\n{}", read_file(cluster.worker_log(0)));
                      std::println("--- worker2 log ---\n{}", read_file(cluster.worker_log(1)));
                      return;
                  }

                  const auto poly_gcd = a.gcd(b);
                  t.expect(poly_gcd.has_value(), "Polynomial::gcd succeeds");
                  if (!poly_gcd.has_value()) {
                      return;
                  }

                  // THE ACCEPTANCE CLAIMS FOR Q2
                  t.expect(dist_gcd->is_equal(*poly_gcd),
                           "THE ACCEPTANCE TEST (Q2): failover modular GCD is BIT-IDENTICAL to Polynomial::gcd");
                  t.expect(dist_gcd->degree() == 3, "GCD degree is 3");
                  t.expect(dist_gcd->coefficient(0) == 5, "c0 == 5");
                  t.expect(dist_gcd->coefficient(1) == 1, "c1 == 1");
                  t.expect(dist_gcd->coefficient(2) == 2, "c2 == 2");
                  t.expect(dist_gcd->coefficient(3) == 3, "c3 == 3");

                  // Anti-vacuous proof of failover: strict increase in rotations in response to SIGKILL
                  const auto rotations_after = (*port_res)->failover_rotations();
                  t.expect(rotations_after > rotations_before,
                           std::format("failover caused client ring rotation in response to leader kill (rotations after [{}] > before [{}])",
                                       rotations_after, rotations_before));

                  // Clean teardown of survivor nodes and workers
                  for (auto& w : cluster.workers) {
                      const int st = stop_process(w);
                      w = -1;
                      t.expect(st >= 0 && WIFEXITED(st) && WEXITSTATUS(st) == 0,
                               "worker exited cleanly (status 0)");
                  }
                  for (std::size_t i = 0; i < cluster.nodes.size(); ++i) {
                      if (cluster.nodes[i].pid > 0) {
                          const int st = stop_process(cluster.nodes[i].pid);
                          cluster.nodes[i].pid = -1;
                          t.expect(st >= 0 && WIFEXITED(st) && WEXITSTATUS(st) == 0,
                                   std::format("survivor node {} exited cleanly (status 0)", i + 1));
                      }
                  }
              })
        .test("q3_mtls_rejection_unauthenticated_client_fails_with_distributed_error",
              [&](TestContext& t) {
                  // --- (Q3) mTLS NEGATIVE REJECTION LEG ------------------------------------------
                  const auto ports = pick_free_ports(9);
                  t.expect(ports.size() == 9, "picked 9 free loopback ports");
                  if (ports.size() != 9) {
                      return;
                  }

                  Cluster3Node cluster;
                  cluster.node_path = node_path;
                  cluster.worker_path = worker_path;
                  cluster.base_dir =
                      std::filesystem::temp_directory_path() /
                      std::format("nc-modgcd-cluster-q3-{}", static_cast<long>(::getpid()));
                  std::filesystem::create_directories(cluster.base_dir, ec);

                  const auto certs_opt = generate_test_certs(cluster.base_dir / "certs", cert_script_path);
                  t.expect(certs_opt.has_value(), "test certificates generated");
                  if (!certs_opt) {
                      return;
                  }
                  const auto& certs = *certs_opt;

                  for (std::size_t i = 0; i < 3; ++i) {
                      ClusterNode n;
                      n.cport = ports[i * 3 + 0];
                      n.qport = ports[i * 3 + 1];
                      n.hport = ports[i * 3 + 2];
                      n.data_dir = cluster.base_dir / std::format("node{}", i);
                      std::filesystem::create_directories(n.data_dir, ec);
                      cluster.nodes.push_back(std::move(n));
                  }

                  const std::string peers = std::format(
                      "1=[::1]:{},2=[::1]:{},3=[::1]:{}",
                      cluster.nodes[0].cport, cluster.nodes[1].cport, cluster.nodes[2].cport);

                  for (std::size_t i = 0; i < 3; ++i) {
                      auto env_vars = make_env({
                          std::format("SGEE_NODE_ID={}", i + 1),
                          std::format("SGEE_PEERS={}", peers),
                          std::format("SGEE_CONSENSUS_PORT={}", cluster.nodes[i].cport),
                          std::format("SGEE_QUEUE_PORT={}", cluster.nodes[i].qport),
                          std::format("SGEE_DATA_DIR={}", cluster.nodes[i].data_dir.string()),
                          "SGEE_SNAPSHOT_RETENTION_MS=3600000",
                          "SGEE_ELECTION_TIMEOUT_MS=500",
                          "SGEE_HEARTBEAT_MS=150",
                          "SGEE_FORCE_BLOCKING_IO=1",
                          std::format("PORT={}", cluster.nodes[i].hport),
                          std::format("SGEE_TLS_CA_CERT={}", certs.ca_crt),
                          std::format("SGEE_TLS_CERT={}", certs.node_crt),
                          std::format("SGEE_TLS_KEY={}", certs.node_key),
                      });
                      cluster.nodes[i].pid =
                          spawn_logged(node_path, {}, env_vars, cluster.node_log(i));
                      t.expect(cluster.nodes[i].pid > 0, std::format("node {} spawned", i + 1));
                      if (cluster.nodes[i].pid <= 0) {
                          return;
                      }
                  }

                  const auto leader_opt =
                      await_stable_leader(cluster.hports(), cluster.pids(), std::chrono::seconds(20));
                  t.expect(leader_opt.has_value(), "cluster elected leader within 20s");
                  if (!leader_opt) {
                      return;
                  }

                  const auto eps = cluster.endpoints();
                  TaskRegistry reg;
                  auto reg_res = nimblecas::register_modgcd_ops(reg);
                  t.expect(reg_res.has_value(), "register_modgcd_ops succeeds");
                  if (!reg_res.has_value()) {
                      return;
                  }

                  // 1. Direct BrokerPort RPC test: lazy connect succeeds, but RPC (enqueue) fails with distributed_error
                  auto unauth_port = GrpcBrokerPort::connect(eps, /*auth_token=*/"", /*rpc_deadline_ms=*/1000, SgeeGrpcTlsOptions{});
                  t.expect(unauth_port.has_value(), "lazy GrpcBrokerPort::connect succeeds without TLS credentials");
                  if (unauth_port.has_value()) {
                      auto enqueue_res = (*unauth_port)->enqueue("nimblecas.modgcd.image/v1", {1, 2, 3});
                      t.expect(!enqueue_res.has_value(),
                               "RPC (enqueue) rejected by mTLS server when client presents no credentials");
                      if (!enqueue_res.has_value()) {
                          t.expect(enqueue_res.error() == MathError::distributed_error,
                                   "rejection is transport-class MathError::distributed_error");
                      }
                  }

                  // 2. Distributed modular_gcd_with executor test: run fails with distributed_error
                  SgeeExecutorConfig cfg;
                  cfg.with_registry(reg)
                      .with_num_workers(0)
                      .with_poll_interval_ms(5)
                      .with_max_attempts(2)
                      .with_visibility_timeout_ms(1'000)
                      .with_run_deadline_ms(2'000)
                      .with_max_result_recoveries(0);

                  SgeeGrpcExecutorOptions unauth_opts{
                      .endpoints = eps,
                      .tls = SgeeGrpcTlsOptions{},  // Plaintext / no client certificate
                  };

                  auto unauth_exec = nimblecas::sgee_grpc_distributed_executor(cfg, unauth_opts);
                  t.expect(unauth_exec.has_value(), "unauthenticated executor constructed (lazy connect)");
                  if (unauth_exec.has_value()) {
                      const Polynomial a({35, 7, 19, 22, 2, 3});
                      const Polynomial b({55, 6, 26, 32, -1, 3});
                      const auto bad_dist_gcd = nimblecas::modular_gcd_with(a, b, **unauth_exec, reg);
                      t.expect(!bad_dist_gcd.has_value(),
                               "modular_gcd_with rejected when client presents no certificate");
                      if (!bad_dist_gcd.has_value()) {
                          t.expect(bad_dist_gcd.error() == MathError::distributed_error,
                                   "modular_gcd_with error is transport-class MathError::distributed_error");
                      }
                  }

                  // Clean teardown
                  for (std::size_t i = 0; i < cluster.nodes.size(); ++i) {
                      if (cluster.nodes[i].pid > 0) {
                          const int st = stop_process(cluster.nodes[i].pid);
                          cluster.nodes[i].pid = -1;
                          t.expect(st >= 0 && WIFEXITED(st) && WEXITSTATUS(st) == 0,
                                   std::format("node {} exited cleanly (status 0)", i + 1));
                      }
                  }
              })
        .run();
}
