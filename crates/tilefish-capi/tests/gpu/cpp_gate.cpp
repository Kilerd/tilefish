// The C++ half of the byte-identity gate, on a real GPU: the cases python/tests/rust_gate.py prepared (the same as the Rust gate's), run through
// the C ABI and tilefish.hpp (the generated launchers where this file has a typed arm, the untyped handles otherwise), with the memory, the
// streams and the graphs of the host's own (driver.hpp), and every tensor written for `rust_gate.py compare --host cpp` to hold to Python's
// replay byte for byte.
//
//     python rust_gate.py prepare $WORK                                   # writes generated.hpp, generated_embedded.cpp and block_runner.hpp too
//     cmake -S crates/tilefish-capi -B build -DTILEFISH_BUILD_GPU_GATE=ON -DTILEFISH_GATE_DIR=$WORK && cmake --build build
//     build/tilefish_cpp_gate $WORK                                       # every test, or name some after the directory
//     python rust_gate.py compare $WORK --host cpp
//
// The tests: `cases` (every case of cases.json with the manifest directory, written to cpp/), `embedded` (the same with the manifest and cubins the
// generated source embeds, cpp_embedded/), `workspace` (a GEMM whose pinned algorithm needs a workspace: the entry's, exactly what it needs and
// dirty, one byte less refused, and in the host's own CUDA graph), `refusals` (a precondition, a read-only tensor for a written parameter, a
// tensor larger than its region, a wrong pin, a tampered cubin, another architecture: each refused before anything is enqueued),
// `single_launch_graph` (a kernel and a GEMM captured in the host's own CUDA graph, replayed), `context` (a borrowed primary context, and the
// thread's current context left as it was), `plans` (every plan case launched, cpp_plans/, and captured and replayed twice, cpp_plans_graph/),
// `plan_set` (a set loaded with one pin), `shared` (the stages of a set on one shared arena, launched and replayed one after the other,
// cpp_shared/ and cpp_shared_graph/), `carried` (carried buffers handed from plan to plan on a shared arena, launched and replayed, cpp_carried/
// and cpp_carried_graph/, and the refusals of a fresh arena), `weighted` (weights bound once), `runner` (the generated runner of the plan
// `block`) and `capturing` (a plan launch on a stream the host is capturing is refused).

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "driver.hpp"
#include "generated.hpp"
#include "json.hpp"
#include "tilefish.hpp"
#if __has_include("block_runner.hpp")
#include "block_runner.hpp"
#define TILEFISH_GATE_HAVE_RUNNER 1
#endif

namespace fs = std::filesystem;
namespace L = tilefish_launchers;
using tilefish::Bf16;
using tilefish::F16;
using tilefish::F32;
using tilefish::I32;
using tilefish::I64;
using tilefish::Named;
using tilefish::Region;
using tilefish::Tensor;
using tilefish::TensorMut;
using tilefish::TensorRef;
using tilefish::U8;

namespace {

fs::path gate;
int failures = 0;
int checks = 0;

#define CHECK(cond, ...)                                                                \
    do {                                                                                \
        ++checks;                                                                       \
        if (!(cond)) {                                                                  \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s: ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                                          \
            std::fprintf(stderr, "\n");                                                 \
            ++failures;                                                                 \
        }                                                                               \
    } while (0)

std::vector<uint8_t> read_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_bytes(const fs::path& path, const std::vector<uint8_t>& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

size_t differing(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t n = std::max(a.size(), b.size()) - std::min(a.size(), b.size());
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) n += a[i] != b[i];
    return n;
}

void settle() { driver::check(driver::api().cuCtxSynchronize(), "cuCtxSynchronize"); }

int32_t dtype_code(const std::string& name) {
    static const std::map<std::string, int32_t> codes = {
        {"bf16", TILEFISH_DTYPE_BF16}, {"f16", TILEFISH_DTYPE_F16}, {"f32", TILEFISH_DTYPE_F32}, {"i32", TILEFISH_DTYPE_I32},
        {"i64", TILEFISH_DTYPE_I64},   {"u8", TILEFISH_DTYPE_U8},   {"e4m3", TILEFISH_DTYPE_E4M3}, {"e5m2", TILEFISH_DTYPE_E5M2}};
    auto it = codes.find(name);
    if (it == codes.end()) throw std::runtime_error("the gate does not know the dtype " + name);
    return it->second;
}

tilefish::Dims dims_of(const json::Value& v) {
    std::vector<uint64_t> out;
    for (const auto& d : v.array()) out.push_back(d.u64());
    return tilefish::Dims(out);
}

/// The host: a context on device 0 (the primary context, which the gate's own driver calls use too) and a stream of its own.
struct Gpu {
    tilefish::Context ctx = tilefish::Context::primary(0);
    driver::Stream stream;
    Gpu() { driver::check(driver::api().cuCtxSetCurrent(ctx.handle()), "cuCtxSetCurrent"); }
    CUstream s() const { return stream.get(); }
};

Gpu& gpu() {
    static Gpu g;
    return g;
}

// -- the cases of cases.json -------------------------------------------------------------------------------------------------------------------

/// A tensor of a case: its memory (a view's whole storage), its shape and, for a view, its strides and offset.
struct Buf {
    std::string name, role, dtype;
    driver::Buffer mem;
    tilefish::Dims shape;
    bool view = false;
    tilefish::Dims strides;
    uint64_t offset = 0;

    Region region() const { return Region{mem.address(), mem.bytes(), 0}; }
    Tensor arg() const {
        int32_t access = role == "in" ? TILEFISH_ACCESS_IN : TILEFISH_ACCESS_INOUT;
        return view ? Tensor(region(), dtype_code(dtype), access, shape, strides, offset) : Tensor(region(), dtype_code(dtype), access, shape);
    }
};

template <class T>
TensorRef<T> r(const Buf& b) {
    return b.view ? TensorRef<T>(b.region(), b.shape, b.strides, b.offset) : TensorRef<T>(b.region(), b.shape);
}

template <class T>
TensorMut<T> m(const Buf& b) {
    return b.view ? TensorMut<T>(b.region(), b.shape, b.strides, b.offset) : TensorMut<T>(b.region(), b.shape);
}

std::vector<Buf> tensors_of(const json::Value& c) {
    std::vector<Buf> out;
    for (const auto& t : c["tensors"].array()) {
        Buf b;
        b.name = t["name"].str();
        b.role = t["role"].str();
        b.dtype = t["dtype"].str();
        b.shape = dims_of(t["shape"]);
        size_t nbytes = t["nbytes"].u64();
        b.mem = b.role == "out" ? driver::Buffer(nbytes) : driver::Buffer(read_bytes(gate / t["input"].str()));
        if (b.mem.bytes() != nbytes) throw std::runtime_error(b.name + ": the file is not nbytes long");
        if (t.has("strides")) {
            b.view = true;
            b.strides = dims_of(t["strides"]);
            b.offset = t["offset"].u64();
        }
        out.push_back(std::move(b));
    }
    settle();  // the uploads are done before a kernel on another stream reads them
    return out;
}

/// What a launch allocated for itself (a workspace, a global scratch): freed after the stream is synchronized.
using Keep = std::vector<driver::Buffer>;

/// The launch through `tilefish_kernel_load` / `_gemm_load` / `_conv_load` and untyped tensors.
void launch_generic(const tilefish::Manifest& manifest, const json::Value& c, const std::vector<Buf>& bufs, Keep& keep) {
    Gpu& g = gpu();
    const char* stem = c["stem"].str().c_str();
    std::vector<Tensor> args;
    for (const auto& b : bufs) args.push_back(b.arg());
    if (c["kind"] == "gemm") {
        auto gemm = manifest.gemm(g.ctx, stem);
        keep.emplace_back(gemm.workspace_bytes());
        args.push_back(TensorMut<U8>(Region{keep.back().address(), keep.back().bytes(), 0}, {keep.back().bytes()}));
        gemm.launch(g.s(), args);
        return;
    }
    if (c["kind"] == "conv") {
        auto conv = manifest.conv(g.ctx, stem);
        keep.emplace_back(conv.workspace_bytes());
        args.push_back(TensorMut<U8>(Region{keep.back().address(), keep.back().bytes(), 0}, {keep.back().bytes()}));
        conv.launch(g.s(), args);
        return;
    }
    std::vector<tilefish_given> given;
    if (c.has("given")) {
        for (const auto& kv : c["given"].object()) given.push_back(tilefish_given{kv.first.c_str(), kv.second.i64()});
    }
    if (c.has("global_scratch")) {
        keep.emplace_back(c["global_scratch"].u64());
        args.push_back(TensorMut<U8>(Region{keep.back().address(), keep.back().bytes(), 0}, {keep.back().bytes()}));
    }
    manifest.kernel(g.ctx, stem).launch(g.s(), args, given);
}

/// The launcher a case names, through the generated header where this file has a typed arm (a launcher whose signature changed does not
/// compile), the untyped handles otherwise.
void launch_case(const tilefish::Manifest& mf, const json::Value& c, const std::vector<Buf>& b, Keep& keep) {
    Gpu& g = gpu();
    const std::string& launcher = c["launcher"].str();
    const CUstream s = g.s();
    if (c["kind"] == "gemm" || c["kind"] == "conv") return launch_generic(mf, c, b, keep);
    if (launcher == "Softmax") return L::Softmax::load(mf, g.ctx).launch(s, r<F32>(b[0]), m<F32>(b[1]));
    if (launcher == "Layernorm") return L::Layernorm::load(mf, g.ctx).launch(s, r<Bf16>(b[0]), r<Bf16>(b[1]), r<Bf16>(b[2]), m<Bf16>(b[3]));
    if (launcher == "ResidualAddRmsnorm")
        return L::ResidualAddRmsnorm::load(mf, g.ctx).launch(s, r<Bf16>(b[0]), m<F32>(b[1]), r<Bf16>(b[2]), m<Bf16>(b[3]));
    if (launcher == "GeluExact") return L::GeluExact::load(mf, g.ctx).launch(s, r<Bf16>(b[0]), m<Bf16>(b[1]));
    if (launcher == "Swiglu") return L::Swiglu::load(mf, g.ctx).launch(s, r<Bf16>(b[0]), r<Bf16>(b[1]), m<Bf16>(b[2]));
    if (launcher == "BucketedRmsDb96") return L::BucketedRmsDb96::load(mf, g.ctx).launch(s, r<Bf16>(b[0]), r<Bf16>(b[1]), m<Bf16>(b[2]));
    if (launcher == "BucketedRmsDb512") return L::BucketedRmsDb512::load(mf, g.ctx).launch(s, r<Bf16>(b[0]), r<Bf16>(b[1]), m<Bf16>(b[2]));
    if (launcher == "BucketedRmsDb1000")
        return L::BucketedRmsDb1000::load(mf, g.ctx).launch(s, r<Bf16>(b[0]), r<Bf16>(b[1]), m<Bf16>(b[2]));
    if (launcher == "Rmsnorm") return L::Rmsnorm::load(mf, g.ctx).launch(s, r<Bf16>(b[0]), r<Bf16>(b[1]), m<Bf16>(b[2]));
    if (launcher == "GeluErf") return L::GeluErf::load(mf, g.ctx).launch(s, r<Bf16>(b[0]), m<Bf16>(b[1]));
    if (launcher == "Axpy") return L::Axpy::load(mf, g.ctx).launch(s, r<F32>(b[0]), m<F32>(b[1]));
    if (launcher == "Mm") return L::Mm::load(mf, g.ctx).launch(s, r<F16>(b[0]), r<F16>(b[1]), m<F32>(b[2]));
    launch_generic(mf, c, b, keep);
}

size_t run_cases(const tilefish::Manifest& manifest, const std::string& subdir) {
    auto doc = json::read_file((gate / "cases.json").string());
    fs::remove_all(gate / subdir);
    size_t ran = 0;
    for (const auto& c : doc["cases"].array()) {
        const std::string& id = c["id"].str();
        try {
            auto bufs = tensors_of(c);
            Keep keep;
            launch_case(manifest, c, bufs, keep);
            gpu().stream.sync();
            for (const auto& b : bufs) write_bytes(gate / subdir / id / (b.name + ".bin"), b.mem.download());
            ++ran;
        } catch (const std::exception& e) {
            CHECK(false, "%s: %s", id.c_str(), e.what());
        }
    }
    CHECK(ran == doc["cases"].size(), "%zu of %zu cases ran", ran, doc["cases"].size());
    return ran;
}

tilefish::Manifest gate_manifest() {
    auto doc = json::read_file((gate / "cases.json").string());
    fs::path dir = doc["manifest"].str();
    if (dir.is_relative()) dir = gate / dir;
    // what production does: pin the digest of the manifest, which the generated launchers carry
    return tilefish::Manifest::load(dir.string(), L::MANIFEST_DIGEST);
}

void test_cases() {
    size_t n = run_cases(gate_manifest(), "cpp");
    std::printf("cpp gate: launched %zu cases through the C ABI (typed launchers where the gate has an arm)\n", n);
}

void test_embedded() {
    auto manifest = L::load_embedded();
    CHECK(manifest.digest() == L::MANIFEST_DIGEST, "the embedded manifest is pinned");
    CHECK(manifest.label() == "<embedded>", "%s", manifest.label().c_str());
    size_t n = run_cases(manifest, "cpp_embedded");
    std::printf("cpp gate: launched %zu cases with the embedded manifest\n", n);
}

// -- a GEMM whose algorithm needs a workspace ----------------------------------------------------------------------------------------------

/// Capture `launch` in the host's own CUDA graph on the gate's stream, check that the capture ran nothing, replay it and compare `out`.
void capture_and_run(const std::function<void(CUstream)>& launch, driver::Buffer& out, const std::vector<uint8_t>& expect, const char* what) {
    Gpu& g = gpu();
    out.zero();
    settle();
    driver::check(driver::api().cuStreamBeginCapture(g.s(), driver::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "cuStreamBeginCapture");
    try {
        launch(g.s());
    } catch (...) {
        driver::CUgraph broken = nullptr;
        driver::api().cuStreamEndCapture(g.s(), &broken);
        if (broken) driver::api().cuGraphDestroy(broken);
        throw;
    }
    driver::CUgraph graph = nullptr;
    driver::check(driver::api().cuStreamEndCapture(g.s(), &graph), "cuStreamEndCapture");
    driver::Graph exec(graph);
    g.stream.sync();
    auto before = out.download();
    CHECK(std::all_of(before.begin(), before.end(), [](uint8_t x) { return x == 0; }), "%s: a capture ran the launch", what);
    for (int round = 0; round < 2; ++round) {
        exec.launch(g.s());
        g.stream.sync();
        auto got = out.download();
        CHECK(got == expect, "%s: replay %d differs from Python in %zu bytes", what, round, differing(got, expect));
    }
}

void test_workspace() {
    fs::path dir = gate / "workspace_case";
    auto c = json::read_file((dir / "case.json").string());
    Gpu& g = gpu();
    auto manifest = tilefish::Manifest::load_trusting_directory((dir / "manifest").string());
    auto gemm = manifest.gemm(g.ctx, c["stem"].str().c_str());
    uint64_t mm = c["m"].u64(), n = c["n"].u64(), k = c["k"].u64();
    uint64_t gm = 0, gn = 0, gk = 0;
    bool bias = false;
    tilefish::detail::check(tilefish_gemm_sizes(gemm.get(), &gm, &gn, &gk, &bias));
    CHECK(gm == mm && gn == n && gk == k && bias, "the GEMM's sizes");
    uint64_t required = gemm.workspace_required();
    CHECK(required > 0 && required <= gemm.workspace_bytes() && required == c["workspace_required"].u64(), "the workspace: %llu",
          static_cast<unsigned long long>(required));
    driver::Buffer a(read_bytes(dir / "a.bin")), b(read_bytes(dir / "b.bin")), bb(read_bytes(dir / "bias.bin"));
    auto expect = read_bytes(dir / "expect.bin");
    auto reg = [](const driver::Buffer& x) { return Region{x.address(), x.bytes(), 0}; };
    auto run = [&](driver::Buffer& out, driver::Buffer& ws, CUstream s) {
        gemm.launch(s, {TensorRef<Bf16>(reg(a), {mm, k}), TensorRef<Bf16>(reg(b), {n, k}), TensorRef<Bf16>(reg(bb), {n}),
                        TensorMut<Bf16>(reg(out), {mm, n}), TensorMut<U8>(reg(ws), {ws.bytes()})});
    };
    settle();
    for (int which = 0; which < 2; ++which) {
        driver::Buffer ws(which == 0 ? gemm.workspace_bytes() : required);
        if (which == 1) ws.fill(0xFF);  // dirty
        driver::Buffer out(mm * n * 2);
        settle();
        run(out, ws, g.s());
        g.stream.sync();
        CHECK(out.download() == expect, "%s: the result is Python's", which == 0 ? "the entry's workspace" : "exactly the workspace, dirty");
    }
    driver::Buffer out(mm * n * 2), shorter(required - 1);
    settle();
    try {
        run(out, shorter, g.s());
        CHECK(false, "a workspace one byte short was not refused");
    } catch (const tilefish::Error& e) {
        CHECK(e.status() == TILEFISH_ERROR_PRECONDITION && e.precondition() == TILEFISH_PRECONDITION_SHAPE &&
                  std::string(e.what()).find("workspace") != std::string::npos,
              "%s", e.what());
    }
    g.stream.sync();
    auto zeros = out.download();
    CHECK(std::all_of(zeros.begin(), zeros.end(), [](uint8_t x) { return x == 0; }), "a refused launch wrote");
    driver::Buffer ws(gemm.workspace_bytes());
    capture_and_run([&](CUstream s) { run(out, ws, s); }, out, expect, "the workspace GEMM in the host's graph");
    std::printf("cpp gate: the GEMM with a workspace of %llu bytes: Python's bytes with the entry's and with exactly that (dirty), in a graph; "
                "one byte less refused\n",
                static_cast<unsigned long long>(required));
}

// -- refusals --------------------------------------------------------------------------------------------------------------------------------

template <class F>
void expect_refused(F f, tilefish_status status, int32_t precondition, const char* what) {
    try {
        f();
        CHECK(false, "%s: not refused", what);
    } catch (const tilefish::Error& e) {
        CHECK(e.status() == status && (precondition < 0 || e.precondition() == precondition), "%s: %s %d: %s", what, e.status_name(),
              e.precondition(), e.what());
    }
}

const json::Value* find_case(const json::Value& doc, const std::function<bool(const json::Value&)>& keep) {
    for (const auto& c : doc["cases"].array()) {
        if (keep(c)) return &c;
    }
    return nullptr;
}

void test_refusals() {
    Gpu& g = gpu();
    auto manifest = gate_manifest();
    auto doc = json::read_file((gate / "cases.json").string());
    const json::Value* gelu = find_case(doc, [](const json::Value& c) { return c["launcher"] == "GeluErf"; });
    CHECK(gelu != nullptr, "the gate has a GeluErf case");
    if (!gelu) return;
    auto bufs = tensors_of(*gelu);
    auto kernel = L::GeluErf::load(manifest, g.ctx);
    uint64_t count = bufs[0].shape[0];
    // a shape the binary was not built for: Count must agree between x and y
    expect_refused([&] { kernel.launch(g.s(), r<Bf16>(bufs[0]), TensorMut<Bf16>(bufs[1].region(), {count / 2})); },
                   TILEFISH_ERROR_PRECONDITION, TILEFISH_PRECONDITION_SHAPE, "two sizes of one dimension");
    // a read-only tensor for a parameter that is written
    expect_refused([&] { kernel.kernel().launch(g.s(), {r<Bf16>(bufs[0]), r<Bf16>(bufs[1])}); }, TILEFISH_ERROR_PRECONDITION,
                   TILEFISH_PRECONDITION_ACCESS, "a read-only output");
    // a tensor larger than its region (the crate refuses to make it: a launch error)
    expect_refused([&] { kernel.launch(g.s(), r<Bf16>(bufs[0]), TensorMut<Bf16>(Region{bufs[1].mem.address(), 2, 0}, {count})); },
                   TILEFISH_ERROR_LAUNCH, -1, "a tensor past its region");
    // the wrong element type does not compile with the typed launcher; untyped, it is refused
    expect_refused([&] { kernel.kernel().launch(g.s(), {TensorRef<F16>(bufs[0].region(), {count}), m<Bf16>(bufs[1])}); },
                   TILEFISH_ERROR_PRECONDITION, TILEFISH_PRECONDITION_DTYPE, "another dtype");
    g.stream.sync();
    auto y = bufs[1].mem.download();
    CHECK(std::all_of(y.begin(), y.end(), [](uint8_t x) { return x == 0; }), "a refused launch wrote");
    // a wrong pin, a tampered cubin, another architecture
    fs::path dir = json::read_file((gate / "cases.json").string())["manifest"].str();
    if (dir.is_relative()) dir = gate / dir;
    expect_refused([&] { tilefish::Manifest::load(dir.string(), std::string(64, '0')); }, TILEFISH_ERROR_PIN_MISMATCH, -1, "a wrong pin");
    fs::path copy = fs::temp_directory_path() / "tilefish-cpp-gate-tampered";
    fs::remove_all(copy);
    fs::copy(dir, copy, fs::copy_options::recursive);
    fs::path cubin;
    for (const auto& e : fs::directory_iterator(copy)) {
        if (e.path().extension() == ".cubin" && e.path().filename().string().rfind("gelu_erf.", 0) == 0) cubin = e.path();
    }
    auto bytes = read_bytes(cubin);
    bytes.back() ^= 1;
    write_bytes(cubin, bytes);
    expect_refused([&] { tilefish::Manifest::load(copy.string(), L::MANIFEST_DIGEST); }, TILEFISH_ERROR_MANIFEST, -1, "a tampered cubin");
    fs::remove_all(copy);
    uint32_t arch = g.ctx.arch();
    const char* other = arch == 120 ? "sm_103" : "sm_120";
    expect_refused([&] { manifest.kernel(g.ctx, "gelu_erf", {}, other); }, TILEFISH_ERROR_NO_BINARY, -1, "another architecture");
    expect_refused([&] { manifest.kernel(g.ctx, "no_such_stem"); }, TILEFISH_ERROR_NO_BINARY, -1, "an unknown stem");
    std::printf("cpp gate: refusals (shape, access, region, dtype, pin, tampered cubin, architecture, stem) launch nothing\n");
}

// -- single launches in the host's own CUDA graph ---------------------------------------------------------------------------------------

void test_single_launch_graph() {
    auto manifest = gate_manifest();
    auto doc = json::read_file((gate / "cases.json").string());
    int graphs = 0;
    for (const char* kind : {"kernel", "gemm"}) {
        const json::Value* c = find_case(doc, [&](const json::Value& x) {
            return x["kind"] == kind && !x.has("global_scratch") && !x.has("given") && x["tensors"].size() >= 2;
        });
        if (!c) continue;
        auto bufs = tensors_of(*c);
        size_t out = 0;
        for (size_t i = 0; i < bufs.size(); ++i) {
            if ((*c)["tensors"][i]["role"] == "out") out = i;
        }
        auto expect = read_bytes(gate / (*c)["tensors"][out]["expect"].str());
        Keep keep;
        // the handles are loaded outside the capture (a load allocates and synchronizes)
        std::optional<tilefish::Kernel> kernel;
        std::optional<tilefish::Gemm> gemm;
        if (std::string(kind) == "gemm") {
            gemm.emplace(manifest.gemm(gpu().ctx, (*c)["stem"].str().c_str()));
            keep.emplace_back(gemm->workspace_bytes());
        } else {
            kernel.emplace(manifest.kernel(gpu().ctx, (*c)["stem"].str().c_str()));
        }
        capture_and_run(
            [&](CUstream s) {
                std::vector<Tensor> args;
                for (const auto& b : bufs) args.push_back(b.arg());
                if (gemm) {
                    args.push_back(TensorMut<U8>(Region{keep.back().address(), keep.back().bytes(), 0}, {keep.back().bytes()}));
                    gemm->launch(s, args);
                } else {
                    kernel->launch(s, args);
                }
            },
            bufs[out].mem, expect, (*c)["id"].str().c_str());
        ++graphs;
    }
    CHECK(graphs == 2, "a kernel case and a GEMM case: %d", graphs);
    std::printf("cpp gate: a kernel and a GEMM captured in the host's own CUDA graph replay to Python's bytes\n");
}

// -- contexts --------------------------------------------------------------------------------------------------------------------------------

void test_context() {
    Gpu& g = gpu();
    const auto& api = driver::api();
    driver::CUdevice dev = 0;
    driver::check(api.cuDeviceGet(&dev, 0), "cuDeviceGet");
    CUcontext primary = nullptr;
    driver::check(api.cuDevicePrimaryCtxRetain(&primary, dev), "cuDevicePrimaryCtxRetain");
    CHECK(primary == g.ctx.handle(), "the primary context");
    {
        auto borrowed = tilefish::Context::borrow(primary);
        CHECK(borrowed.handle() == primary && borrowed.device() == 0 && borrowed.arch() == g.ctx.arch(), "a borrowed primary context");
        // a launch through a handle made in the borrowed context, with no current context on the thread: none after it either
        auto manifest = gate_manifest();
        auto doc = json::read_file((gate / "cases.json").string());
        const json::Value* c = find_case(doc, [](const json::Value& x) { return x["launcher"] == "GeluErf"; });
        auto bufs = tensors_of(*c);
        auto kernel = L::GeluErf::load(manifest, borrowed);
        driver::check(api.cuCtxSetCurrent(nullptr), "cuCtxSetCurrent");
        kernel.launch(g.s(), r<Bf16>(bufs[0]), m<Bf16>(bufs[1]));
        CUcontext now = reinterpret_cast<CUcontext>(1);
        driver::check(api.cuCtxGetCurrent(&now), "cuCtxGetCurrent");
        CHECK(now == nullptr, "a launch left a current context behind");
        driver::check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent");
        g.stream.sync();
        CHECK(bufs[1].mem.download() == read_bytes(gate / (*c)["tensors"][1]["expect"].str()), "the borrowed context's launch");
    }
    api.cuDevicePrimaryCtxRelease(dev);
    std::printf("cpp gate: a borrowed primary context launches, and the thread's current context is left as it was\n");
}

// -- plans -------------------------------------------------------------------------------------------------------------------------------------

/// One external of a plan case on the device.
struct Ext {
    std::string name, role, dtype;
    tilefish::Dims shape;
    driver::Buffer mem;
    std::optional<std::vector<uint8_t>> input, expect;

    Named named() const {
        Region reg{mem.address(), mem.bytes(), 0};
        int32_t access = (role == "output" || role == "inout") ? TILEFISH_ACCESS_INOUT : TILEFISH_ACCESS_IN;
        return Named{name.c_str(), Tensor(reg, dtype_code(dtype), access, shape)};
    }
};

std::vector<Ext> upload(const json::Value& c) {
    std::vector<Ext> out;
    for (const auto& t : c["externals"].array()) {
        Ext e;
        e.name = t["name"].str();
        e.role = t["role"].str();
        e.dtype = t["dtype"].str();
        e.shape = dims_of(t["shape"]);
        uint64_t elements = 1;
        for (size_t i = 0; i < e.shape.size(); ++i) elements *= e.shape[i];
        static const std::map<std::string, uint64_t> sizes = {{"bf16", 2}, {"f16", 2}, {"f32", 4}, {"i32", 4}, {"i64", 8}, {"u8", 1}};
        uint64_t bytes = elements * sizes.at(e.dtype);
        if (t.has("input")) e.input = read_bytes(gate / t["input"].str());
        if (t.has("expect")) e.expect = read_bytes(gate / t["expect"].str());
        e.mem = e.input ? driver::Buffer(*e.input) : driver::Buffer(bytes);
        out.push_back(std::move(e));
    }
    settle();
    return out;
}

std::vector<Named> named(const std::vector<Ext>& exts, const std::function<bool(const Ext&)>& keep = nullptr) {
    std::vector<Named> out;
    for (const auto& e : exts) {
        if (!keep || keep(e)) out.push_back(e.named());
    }
    return out;
}

std::vector<tilefish::DevScalar> devscalars(const json::Value& c) {
    std::vector<tilefish::DevScalar> out;
    if (c.has("devscalars")) {
        for (const auto& kv : c["devscalars"].object()) out.emplace_back(kv.first.c_str(), kv.second.i64());
    }
    return out;
}

/// The outputs (and in-place tensors) of a case: written to `subdir` and held to Python's bytes.
void check_outputs(const std::vector<Ext>& exts, const std::string& subdir, const std::string& id, const char* what, int* tensors = nullptr) {
    for (const auto& e : exts) {
        if (!e.expect) continue;
        auto got = e.mem.download();
        if (!subdir.empty()) write_bytes(gate / subdir / id / (e.name + ".bin"), got);
        CHECK(got == *e.expect, "%s: %s: '%s' differs from Python in %zu bytes", id.c_str(), what, e.name.c_str(), differing(got, *e.expect));
        if (tensors) ++*tensors;
    }
}

void reset(std::vector<Ext>& exts) {
    for (auto& e : exts) {
        if (e.role == "inout") e.mem.upload(*e.input);
        if (e.role == "output") e.mem.zero();
    }
    settle();
}

json::Value plans_doc() { return json::read_file((gate / "plans.json").string()); }

tilefish::LaunchPlan load_plan(const json::Value& p) { return tilefish::LaunchPlan::load((gate / p["dir"].str()).string(), p["digest"].str()); }

void test_plans() {
    Gpu& g = gpu();
    int cases = 0, tensors = 0;
    auto doc = plans_doc();
    for (const auto& p : doc["plans"].array()) {
        auto plan = load_plan(p);
        CHECK(plan.digest() == p["digest"].str() && plan.arena_bytes() == p["arena_bytes"].u64(), "%s", p["name"].str().c_str());
        auto run = plan.instantiate(g.ctx, g.s());
        for (const auto& c : p["cases"].array()) {
            const std::string& id = c["id"].str();
            try {
                auto exts = upload(c);
                auto values = devscalars(c);
                run.launch(g.s(), named(exts), values);
                g.stream.sync();
                check_outputs(exts, "cpp_plans", id, "launch", &tensors);
                reset(exts);
                {
                    auto graph = run.capture(g.s(), named(exts), values);
                    g.stream.sync();
                    for (const auto& e : exts) {
                        if (e.role != "output") continue;
                        auto now = e.mem.download();
                        CHECK(std::all_of(now.begin(), now.end(), [](uint8_t x) { return x == 0; }), "%s: a capture ran", id.c_str());
                    }
                    for (int round = 0; round < 2; ++round) {
                        if (round == 1) reset(exts);  // a graph holds addresses: new contents, the same memory
                        graph.replay(g.s());
                        g.stream.sync();
                        check_outputs(exts, round == 0 ? "cpp_plans_graph" : "", id, round == 0 ? "replay 0" : "replay 1");
                    }
                }
                ++cases;
            } catch (const std::exception& e) {
                CHECK(false, "%s: %s", id.c_str(), e.what());
            }
        }
    }
    std::printf("cpp gate: %d plan cases, each launched and replayed twice from a captured plan: Python's bytes (%d tensors)\n", cases, tensors);
}

void test_plan_set() {
    Gpu& g = gpu();
    auto doc = plans_doc();
    std::vector<const json::Value*> members;
    for (const auto& p : doc["plans"].array()) {
        if (p.has("set") && p["set"] == "plans/set") members.push_back(&p);
    }
    CHECK(members.size() == 3, "the gate's plan set has three plans");
    if (members.empty()) return;
    auto set = tilefish::PlanSet::load((gate / (*members[0])["set"].str()).string(), (*members[0])["set_digest"].str());
    CHECK(set.names() == (std::vector<std::string>{"tiles1", "tiles2", "tiles3"}), "the set's names");
    CHECK(set.manifest().digest().size() == 64, "the set's manifest");
    int cases = 0;
    for (const json::Value* member : members) {
        auto plan = set.plan((*member)["graph"].str().c_str());
        CHECK(plan.digest() == (*member)["digest"].str(), "a plan of the set");
        auto run = plan.instantiate(g.ctx, g.s());
        for (const auto& c : (*member)["cases"].array()) {
            auto exts = upload(c);
            run.launch(g.s(), named(exts), devscalars(c));
            g.stream.sync();
            check_outputs(exts, "", c["id"].str(), "a plan of the set");
            ++cases;
        }
    }
    expect_refused([&] { tilefish::PlanSet::load((gate / (*members[0])["set"].str()).string(), std::string(64, '0')); },
                   TILEFISH_ERROR_PLAN_PIN_MISMATCH, -1, "a set with a wrong pin");
    std::printf("cpp gate: a plan set loaded with one pin: %d cases launched, Python's bytes\n", cases);
}

// -- a shared arena: the three stages of the block, one after the other ---------------------------------------------------------------------

const char* const STAGES[3] = {"stage_attn", "stage_mix", "stage_ffn"};

struct Pipeline {
    std::string label;
    int64_t n = 0;
    std::vector<Ext> bufs;
    std::vector<const json::Value*> cases;

    Ext& buf(const std::string& name) {
        for (auto& b : bufs) {
            if (b.name == name) return b;
        }
        throw std::runtime_error("no buffer " + name);
    }
    std::vector<Named> args(size_t k) {
        std::vector<Named> out;
        for (const auto& t : (*cases[k])["externals"].array()) {
            Ext& b = buf(t["name"].str());
            Region reg{b.mem.address(), b.mem.bytes(), 0};
            int32_t access = t["role"] == "output" ? TILEFISH_ACCESS_INOUT : TILEFISH_ACCESS_IN;
            out.push_back(Named{b.name.c_str(), Tensor(reg, dtype_code(b.dtype), access, b.shape)});
        }
        return out;
    }
    void check(size_t k, const std::string& subdir, const char* what) {
        const std::string& id = (*cases[k])["id"].str();
        for (const auto& t : (*cases[k])["externals"].array()) {
            if (t["role"] != "output") continue;
            Ext& b = buf(t["name"].str());
            auto got = b.mem.download();
            write_bytes(gate / subdir / id / (b.name + ".bin"), got);
            CHECK(got == *b.expect, "%s: %s: '%s' differs in %zu bytes", id.c_str(), what, b.name.c_str(), differing(got, *b.expect));
        }
    }
};

std::vector<Pipeline> pipelines(const std::vector<const json::Value*>& members) {
    std::vector<Pipeline> out;
    for (const auto& first : (*members[0])["cases"].array()) {
        Pipeline p;
        p.label = first["id"].str().substr(std::string("shared-stage_attn-").size());
        for (size_t k = 0; k < 3; ++k) {
            std::string id = std::string("shared-") + STAGES[k] + "-" + p.label;
            for (const auto& c : (*members[k])["cases"].array()) {
                if (c["id"] == id) p.cases.push_back(&c);
            }
        }
        if (p.cases.size() != 3) throw std::runtime_error("the stages of " + p.label);
        for (const json::Value* c : p.cases) {
            for (auto& e : upload(*c)) {
                bool known = false;
                for (auto& b : p.bufs) known |= b.name == e.name;  // an input of two stages, or a stage's input the stage before returned
                if (!known) p.bufs.push_back(std::move(e));
            }
        }
        p.n = (*p.cases[0])["devscalars"]["n"].i64();
        out.push_back(std::move(p));
    }
    return out;
}

void test_shared() {
    Gpu& g = gpu();
    auto doc = plans_doc();
    std::vector<const json::Value*> members;
    for (const auto& p : doc["plans"].array()) {
        if (p.has("shared")) members.push_back(&p);
    }
    CHECK(members.size() == 3, "the gate's shared arena has three stages");
    if (members.size() != 3) return;
    auto set = tilefish::PlanSet::load((gate / (*members[0])["set"].str()).string(), (*members[0])["set_digest"].str());
    std::vector<const char*> graphs(STAGES, STAGES + 3);
    auto runs = set.instantiate(g.ctx, g.s(), graphs);
    uint64_t largest = 0, sum = 0;
    for (const json::Value* p : members) {
        largest = std::max(largest, (*p)["arena_bytes"].u64());
        sum += (*p)["arena_bytes"].u64();
    }
    CHECK(runs.arena().bytes() == largest && set.shared_arena_bytes(graphs) == largest && largest < sum, "one arena of the largest stage's");
    std::vector<tilefish::PlanRun> stage;
    for (const char* name : STAGES) stage.push_back(runs.run(name));
    auto ps = pipelines(members);
    for (auto& p : ps) {
        for (size_t k = 0; k < 3; ++k) {
            std::vector<tilefish::DevScalar> values;
            if (k == 0) values.emplace_back("n", p.n);
            stage[k].launch(g.s(), p.args(k), values);
        }
        g.stream.sync();
        for (size_t k = 0; k < 3; ++k) p.check(k, "cpp_shared", "launched on the shared arena");
    }
    for (auto& p : ps) {
        for (auto& b : p.bufs) {
            if (b.role == "output") b.mem.zero();
        }
        settle();
        {
            std::vector<tilefish::CapturedPlan> captured;
            for (size_t k = 0; k < 3; ++k) {
                std::vector<tilefish::DevScalar> values;
                if (k == 0) values.emplace_back("n", p.n);
                captured.push_back(stage[k].capture(g.s(), p.args(k), values));
            }
            for (int round = 0; round < 2; ++round) {
                for (auto& graph : captured) graph.replay(g.s());
            }
            g.stream.sync();
        }
        for (size_t k = 0; k < 3; ++k) p.check(k, "cpp_shared_graph", "replayed on the shared arena");
    }
    std::printf("cpp gate: a shared arena of %llu bytes for 3 stages (%llu as 3 arenas): %zu requests launched and replayed one stage after "
                "the other: Python's bytes\n",
                static_cast<unsigned long long>(largest), static_cast<unsigned long long>(sum), ps.size());
}

// -- carried buffers -------------------------------------------------------------------------------------------------------------------------

void test_carried() {
    Gpu& g = gpu();
    auto doc = plans_doc()["carried"];
    if (doc.is_null()) {
        CHECK(false, "plans.json has no carried section");
        return;
    }
    auto set = tilefish::PlanSet::load((gate / doc["dir"].str()).string(), doc["set_digest"].str());
    std::vector<std::string> graph_names;
    for (const auto& gname : doc["graphs"].array()) graph_names.push_back(gname.str());
    std::vector<const char*> graphs;
    for (const auto& n : graph_names) graphs.push_back(n.c_str());
    const auto& calls = doc["calls"].array();
    for (int captured = 0; captured < 2; ++captured) {
        const std::string subdir = captured ? "cpp_carried_graph" : "cpp_carried";
        auto runs = set.instantiate(g.ctx, g.s(), graphs);
        CHECK(runs.arena().bytes() == set.shared_arena_bytes(graphs), "the carried arena");
        for (const auto& call : calls) {
            const std::string& id = call["id"].str();
            auto run = runs.run(call["graph"].str().c_str());
            auto exts = upload(call);
            try {
                if (captured) {
                    auto graph = run.capture(g.s(), named(exts));
                    graph.replay(g.s());
                    g.stream.sync();
                } else {
                    run.launch(g.s(), named(exts));
                    g.stream.sync();
                }
                check_outputs(exts, subdir, id, captured ? "replayed" : "launched");
            } catch (const std::exception& e) {
                CHECK(false, "%s: %s: %s", subdir.c_str(), id.c_str(), e.what());
            }
        }
    }
    // a fresh arena: stage_b reads h, which nothing wrote (no dimension); stage_update reads it before it is written; both refused
    auto fresh = set.instantiate(g.ctx, g.s(), graphs);
    auto call_of = [&](const char* graph) -> const json::Value& {
        for (const auto& c : calls) {
            if (c["graph"] == graph) return c;
        }
        throw std::runtime_error(std::string("no call of ") + graph);
    };
    {
        auto exts = upload(call_of("stage_b"));
        expect_refused([&] { fresh.run("stage_b").launch(g.s(), named(exts)); }, TILEFISH_ERROR_PRECONDITION, TILEFISH_PRECONDITION_BUCKET,
                       "a carried buffer nothing wrote binds no dimension");
    }
    {
        auto exts = upload(call_of("stage_update"));
        try {
            fresh.run("stage_update").launch(g.s(), named(exts));
            CHECK(false, "stage_update on a fresh arena was not refused");
        } catch (const tilefish::Error& e) {
            CHECK(std::string(e.what()).find("no execution on this arena has written it") != std::string::npos, "%s", e.what());
        }
        g.stream.sync();
        for (const auto& e : exts) {
            if (e.role != "output") continue;
            auto now = e.mem.download();
            CHECK(std::all_of(now.begin(), now.end(), [](uint8_t x) { return x == 0; }), "a refused call wrote '%s'", e.name.c_str());
        }
    }
    std::printf("cpp gate: %zu calls handing carried buffers from plan to plan on one shared arena, launched and replayed: Python's bytes; "
                "a fresh arena refuses what nothing wrote\n",
                calls.size());
}

// -- weights bound once, the generated runner, a capturing stream ----------------------------------------------------------------------

const json::Value* plan_named(const json::Value& doc, const char* name) {
    for (const auto& p : doc["plans"].array()) {
        if (p["name"] == name) return &p;
    }
    return nullptr;
}

void test_weighted() {
    Gpu& g = gpu();
    auto doc = plans_doc();
    const json::Value* p = plan_named(doc, "weighted");
    CHECK(p != nullptr, "the gate has the plan 'weighted'");
    if (!p) return;
    auto plan = load_plan(*p);
    auto run = plan.instantiate(g.ctx, g.s());
    const auto& cases = (*p)["cases"].array();
    auto weights = upload(cases[0]);
    auto bound = run.bind_weights(named(weights, [](const Ext& e) { return e.role == "weight"; }));
    for (const auto& c : cases) {
        auto exts = upload(c);
        bound.launch(g.s(), named(exts, [](const Ext& e) { return e.role != "weight"; }));
        g.stream.sync();
        check_outputs(exts, "", c["id"].str(), "weights bound once");
    }
    // one capture through the bound run, replayed with every case's input written into its memory
    auto exts = upload(cases[0]);
    {
        auto graph = bound.capture(g.s(), named(exts, [](const Ext& e) { return e.role != "weight"; }));
        for (auto it = cases.rbegin(); it != cases.rend(); ++it) {
            auto inputs = upload(*it);
            for (auto& e : exts) {
                for (auto& i : inputs) {
                    if (i.name == e.name && e.role == "input") e.mem.upload(*i.input);
                    if (i.name == e.name) e.expect = i.expect;
                }
            }
            settle();
            graph.replay(g.s());
            g.stream.sync();
            check_outputs(exts, "", (*it)["id"].str(), "a capture of the bound run");
        }
    }
    auto again = upload(cases[0]);
    try {
        bound.launch(g.s(), named(again));
        CHECK(false, "a weight given again was not refused");
    } catch (const tilefish::Error& e) {
        CHECK(std::string(e.what()).find("is a weight of the plan, bound once") != std::string::npos, "%s", e.what());
    }
    std::printf("cpp gate: weights bound once: %zu cases launched and replayed through one capture: Python's bytes\n", cases.size());
}

void test_runner() {
#ifdef TILEFISH_GATE_HAVE_RUNNER
    Gpu& g = gpu();
    auto doc = plans_doc();
    const json::Value* p = plan_named(doc, "block");
    CHECK(p != nullptr, "the gate has the plan 'block'");
    if (!p) return;
    CHECK(std::string(tilefish_plans::Block::PLAN_DIGEST) == (*p)["digest"].str(), "the runner's pin");
    auto block = tilefish_plans::Block::load((gate / (*p)["dir"].str()).string(), g.ctx, g.s());
    int cases = 0;
    for (const auto& c : (*p)["cases"].array()) {
        auto exts = upload(c);
        std::map<std::string, const Ext*> by;
        for (const auto& e : exts) by[e.name] = &e;
        auto reg = [&](const char* n) { return Region{by.at(n)->mem.address(), by.at(n)->mem.bytes(), 0}; };
        auto shp = [&](const char* n) { return by.at(n)->shape; };
        tilefish_plans::Block::Externals e{
            TensorRef<Bf16>(reg("x"), shp("x")),         TensorRef<Bf16>(reg("w_ln1"), shp("w_ln1")),
            TensorRef<Bf16>(reg("w_qkv"), shp("w_qkv")), TensorRef<Bf16>(reg("b_qkv"), shp("b_qkv")),
            TensorRef<Bf16>(reg("w_o"), shp("w_o")),     TensorRef<Bf16>(reg("w_ln2"), shp("w_ln2")),
            TensorRef<Bf16>(reg("w_up"), shp("w_up")),   TensorRef<Bf16>(reg("w_down"), shp("w_down")),
            TensorMut<Bf16>(reg("out0"), shp("out0"))};
        tilefish_plans::Block::Scalars values{};
        values.n = static_cast<int32_t>(c["devscalars"]["n"].i64());
        block.launch(g.s(), e, values);
        g.stream.sync();
        check_outputs(exts, "", c["id"].str(), "the generated runner");
        ++cases;
    }
    std::printf("cpp gate: the generated runner of the plan 'block': %d cases, Python's bytes\n", cases);
#else
    CHECK(false, "block_runner.hpp is not in the gate directory (rust_gate.py prepare writes it)");
#endif
}

void test_capturing() {
    Gpu& g = gpu();
    auto doc = plans_doc();
    const json::Value& p = doc["plans"][0];
    auto plan = load_plan(p);
    auto run = plan.instantiate(g.ctx, g.s());
    auto exts = upload(p["cases"][0]);
    driver::Stream other;
    driver::check(driver::api().cuStreamBeginCapture(other.get(), driver::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "cuStreamBeginCapture");
    try {
        run.launch(other.get(), named(exts), devscalars(p["cases"][0]));
        CHECK(false, "a plan launch on a capturing stream was not refused");
    } catch (const tilefish::Error& e) {
        CHECK(e.status() == TILEFISH_ERROR_LAUNCH, "%s: %s", e.status_name(), e.what());
    }
    driver::CUgraph graph = nullptr;
    driver::api().cuStreamEndCapture(other.get(), &graph);
    if (graph) driver::api().cuGraphDestroy(graph);
    std::printf("cpp gate: a plan launch on a stream the host captures is refused (capture with tilefish_plan_run_capture)\n");
}

}  // namespace

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : std::getenv("TILEFISH_GATE_DIR");
    if (!dir) {
        std::fprintf(stderr, "usage: %s <the directory rust_gate.py prepare filled> [test ...]\n", argv[0]);
        return 2;
    }
    gate = fs::absolute(dir);
    const std::vector<std::pair<const char*, void (*)()>> tests = {
        {"cases", test_cases},       {"embedded", test_embedded}, {"workspace", test_workspace},
        {"refusals", test_refusals}, {"single_launch_graph", test_single_launch_graph},
        {"context", test_context},   {"plans", test_plans},       {"plan_set", test_plan_set},
        {"shared", test_shared},     {"carried", test_carried},   {"weighted", test_weighted},
        {"runner", test_runner},     {"capturing", test_capturing},
    };
    std::set<std::string> wanted(argv + std::min(argc, 2), argv + argc);
    int ran = 0;
    for (const auto& t : tests) {
        if (!wanted.empty() && !wanted.count(t.first)) continue;
        int before = failures;
        try {
            t.second();
        } catch (const std::exception& e) {
            CHECK(false, "%s: %s", t.first, e.what());
        }
        std::printf("%s %s\n", failures == before ? "ok  " : "FAIL", t.first);
        ++ran;
    }
    std::printf("cpp gate: %d tests, %d checks, %d failures\n", ran, checks, failures);
    return failures ? 1 : 0;
}
