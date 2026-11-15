// tilefish.hpp from C++17, with no GPU: the handles, tilefish::Error and its fields, loading the committed manifests and plans through the
// wrapper, and the generated launchers and plan runner of the committed fixtures, which must compile against the wrapper (and are checked
// for what can be checked without a device).
//
//     tilefish_cpp_wrapper <manifest dir> <manifest pin> <plan dir> <plan pin> <plan set dir> <plan set pin>

#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "tilefish.hpp"
#include "tilefish.hpp"  // twice: the include guard holds

#ifdef TILEFISH_HAVE_GENERATED
#include "block_runner.hpp"
#include "generated.hpp"
#endif

namespace {

int failures = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                                   \
        }                                                                                 \
    } while (0)

/// Run `f`, which must throw a tilefish::Error of `status`; returns it.
template <class F>
tilefish::Error expect_error(F f, tilefish_status status, const char* what) {
    try {
        f();
    } catch (const tilefish::Error& e) {
        if (e.status() != status) {
            std::fprintf(stderr, "%s: threw %s, want %s: %s\n", what, e.status_name(), tilefish_status_name(status), e.what());
            failures++;
        }
        return e;
    }
    std::fprintf(stderr, "%s: did not throw\n", what);
    failures++;
    return tilefish::Error(TILEFISH_OK, TILEFISH_PRECONDITION_NONE, 0, "", "", "");
}

// what the wrapper promises of its types
static_assert(!std::is_copy_constructible<tilefish::Manifest>::value, "a handle is move-only");
static_assert(std::is_nothrow_move_constructible<tilefish::Manifest>::value, "a handle moves");
static_assert(std::is_nothrow_move_constructible<tilefish::CapturedPlan>::value, "a captured plan moves");
static_assert(std::is_base_of<std::runtime_error, tilefish::Error>::value, "tilefish::Error is a std::runtime_error");
static_assert(std::is_convertible<tilefish::TensorRef<tilefish::Bf16>, tilefish::Tensor>::value, "a typed tensor is a tensor");
static_assert(!std::is_convertible<tilefish::TensorRef<tilefish::Bf16>, tilefish::TensorRef<tilefish::F32>>::value,
              "a bf16 tensor is not an f32 one");

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s <manifest dir> <manifest pin> <plan dir> <plan pin> <plan set dir> <plan set pin>\n", argv[0]);
        return 2;
    }
    const std::string manifest_dir = argv[1], manifest_pin = argv[2], plan_dir = argv[3], plan_pin = argv[4], set_dir = argv[5],
                      set_pin = argv[6];
    const std::string wrong(64, '0');

    auto manifest = tilefish::Manifest::load(manifest_dir, manifest_pin);
    CHECK(manifest.digest() == manifest_pin);
    auto moved = std::move(manifest);
    CHECK(moved.get() != nullptr && manifest.get() == nullptr);
    CHECK(moved.with_sm_partition(16).digest() == manifest_pin);
    moved.require_digest(manifest_pin.c_str());
    auto e = expect_error([&] { moved.require_digest(wrong.c_str()); }, TILEFISH_ERROR_PIN_MISMATCH, "require_digest");
    CHECK(std::string(e.what()).find(manifest_pin) != std::string::npos);
    e = expect_error([&] { tilefish::Manifest::load(manifest_dir, wrong); }, TILEFISH_ERROR_PIN_MISMATCH, "a wrong pin");
    CHECK(std::string(e.what()).find(wrong) != std::string::npos && e.precondition() == TILEFISH_PRECONDITION_NONE);
    expect_error([&] { tilefish::Manifest::load(manifest_dir, "abc"); }, TILEFISH_ERROR_BAD_PIN, "a bad pin");
    expect_error([&] { tilefish::Manifest::load("/no/such/tilefish/directory", manifest_pin); }, TILEFISH_ERROR_IO, "no directory");
    CHECK(tilefish::Manifest::load_trusting_directory(manifest_dir).digest() == manifest_pin);

    auto plan = tilefish::LaunchPlan::load(plan_dir, plan_pin);
    CHECK(plan.graph() == "block" && plan.arch() == "sm_103");
    CHECK(plan.outputs() == std::vector<std::string>{"out0"});
    auto dims = plan.dims();
    CHECK(dims.size() == 1 && dims[0].first == "M" && dims[0].second == (std::vector<uint64_t>{32, 64, 128}));
    int weights = 0;
    for (const auto& b : plan.buffers()) weights += b.role == TILEFISH_ROLE_WEIGHT;
    CHECK(weights == 7);
    CHECK(plan.manifest().digest() == plan.manifest_digest());
    expect_error([&] { tilefish::LaunchPlan::load(plan_dir, wrong); }, TILEFISH_ERROR_PLAN_PIN_MISMATCH, "a plan with a wrong pin");

    auto set = tilefish::PlanSet::load(set_dir, set_pin);
    auto names = set.names();
    CHECK(names.size() > 1);
    std::vector<const char*> graphs;
    for (const auto& n : names) graphs.push_back(n.c_str());
    auto plans = set.plans(graphs);
    CHECK(plans.size() == names.size() && plans[0].graph() == names[0]);
    CHECK(set.shared_arena_bytes(graphs) > 0);
    expect_error([&] { set.plan("no_such_graph"); }, TILEFISH_ERROR_PLAN, "an unknown graph");

    if (!tilefish::driver_available()) {
        expect_error([] { tilefish::Context::primary(0); }, TILEFISH_ERROR_NO_DRIVER, "a context without a driver");
    }

#ifdef TILEFISH_HAVE_GENERATED
    // the generated launchers of the committed fixture: their constants, and their types
    CHECK(std::strcmp(tilefish_launchers::MANIFEST_DIGEST, manifest_pin.c_str()) == 0);
    CHECK(std::strcmp(tilefish_launchers::GeluErf::STEM, "gelu_erf") == 0);
    CHECK(tilefish_launchers::GeluErf::BUCKETS.size() == 0);
    CHECK(std::strcmp(tilefish_plans::Block::PLAN_DIGEST, plan_pin.c_str()) == 0);
    CHECK(tilefish_plans::Block::ARENA_BYTES == plan.arena_bytes());
#endif

    if (failures) {
        std::fprintf(stderr, "%d checks failed\n", failures);
        return 1;
    }
    std::printf("tilefish C++ wrapper: every check passed\n");
    return 0;
}
