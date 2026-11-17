// The few CUDA driver functions the C++ gate uses for its own memory, streams and graphs (the host's side of the C ABI), opened at run time
// from libcuda.so.1 so that the gate builds without the CUDA toolkit. Test code only.
#ifndef TILEFISH_GATE_DRIVER_HPP
#define TILEFISH_GATE_DRIVER_HPP

#include <dlfcn.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "tilefish.h"

namespace driver {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUgraph = struct CUgraph_st*;
using CUgraphExec = struct CUgraphExec_st*;

constexpr unsigned CU_STREAM_NON_BLOCKING = 1;
constexpr int CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;

struct Api {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuCtxGetCurrent)(CUcontext*);
    CUresult (*cuCtxSynchronize)();
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
    CUresult (*cuMemsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult (*cuStreamCreate)(CUstream*, unsigned);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuStreamQuery)(CUstream);
    CUresult (*cuStreamBeginCapture)(CUstream, int);
    CUresult (*cuStreamEndCapture)(CUstream, CUgraph*);
    CUresult (*cuGraphInstantiateWithFlags)(CUgraphExec*, CUgraph, unsigned long long);
    CUresult (*cuGraphLaunch)(CUgraphExec, CUstream);
    CUresult (*cuGraphExecDestroy)(CUgraphExec);
    CUresult (*cuGraphDestroy)(CUgraph);
};

template <class F>
void bind(void* lib, F& f, const char* name) {
    void* p = dlsym(lib, name);
    if (!p) throw std::runtime_error(std::string("libcuda has no ") + name);
    f = reinterpret_cast<F>(p);
}

inline const Api& api() {
    static const Api a = [] {
        void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!lib) throw std::runtime_error("cannot open libcuda.so.1");
        Api x{};
        bind(lib, x.cuInit, "cuInit");
        bind(lib, x.cuDeviceGet, "cuDeviceGet");
        bind(lib, x.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
        bind(lib, x.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
        bind(lib, x.cuCtxSetCurrent, "cuCtxSetCurrent");
        bind(lib, x.cuCtxGetCurrent, "cuCtxGetCurrent");
        bind(lib, x.cuCtxSynchronize, "cuCtxSynchronize");
        bind(lib, x.cuMemAlloc, "cuMemAlloc_v2");
        bind(lib, x.cuMemFree, "cuMemFree_v2");
        bind(lib, x.cuMemcpyHtoD, "cuMemcpyHtoD_v2");
        bind(lib, x.cuMemcpyDtoH, "cuMemcpyDtoH_v2");
        bind(lib, x.cuMemsetD8, "cuMemsetD8_v2");
        bind(lib, x.cuStreamCreate, "cuStreamCreate");
        bind(lib, x.cuStreamDestroy, "cuStreamDestroy_v2");
        bind(lib, x.cuStreamSynchronize, "cuStreamSynchronize");
        bind(lib, x.cuStreamQuery, "cuStreamQuery");
        bind(lib, x.cuStreamBeginCapture, "cuStreamBeginCapture_v2");
        bind(lib, x.cuStreamEndCapture, "cuStreamEndCapture");
        bind(lib, x.cuGraphInstantiateWithFlags, "cuGraphInstantiateWithFlags");
        bind(lib, x.cuGraphLaunch, "cuGraphLaunch");
        bind(lib, x.cuGraphExecDestroy, "cuGraphExecDestroy");
        bind(lib, x.cuGraphDestroy, "cuGraphDestroy");
        if (x.cuInit(0) != 0) throw std::runtime_error("cuInit failed");
        return x;
    }();
    return a;
}

inline void check(CUresult r, const char* what) {
    if (r != 0) throw std::runtime_error(std::string(what) + " failed: CUresult " + std::to_string(r));
}

/// Wait for the legacy default stream (the NULL stream), where `cuMemsetD8` and `cuMemcpyHtoD` run: their work is finished when they return to
/// the host only in part (a memset is asynchronous, a copy from pageable memory may still be in flight), and the gate's kernels run on a
/// non-blocking stream, which does not wait for the legacy stream.
inline void finish_legacy() { check(api().cuStreamSynchronize(nullptr), "cuStreamSynchronize(NULL)"); }

/// Device memory the gate owns (freed with it). Every write of it (the zeros it is made with, `upload`, `zero`, `fill`) is finished when the
/// call returns, so a kernel on any stream may read it next: a workspace whose zeros landed while a cuDNN convolution used it gave all-zero
/// outputs, now and then (the C++ gate's `conv_k7.T646`, and others).
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(size_t bytes) : bytes_(bytes) {
        check(api().cuMemAlloc(&ptr_, bytes ? bytes : 1), "cuMemAlloc");
        check(api().cuMemsetD8(ptr_, 0, bytes ? bytes : 1), "cuMemsetD8");
        finish_legacy();
    }
    Buffer(const std::vector<uint8_t>& data) : Buffer(data.size()) { upload(data); }  // NOLINT
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) { o.ptr_ = 0; o.bytes_ = 0; }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            bytes_ = o.bytes_;
            o.ptr_ = 0;
            o.bytes_ = 0;
        }
        return *this;
    }
    ~Buffer() { release(); }
    void upload(const std::vector<uint8_t>& data) {
        if (data.size() != bytes_) throw std::runtime_error("an upload of another size");
        if (!data.empty()) check(api().cuMemcpyHtoD(ptr_, data.data(), data.size()), "cuMemcpyHtoD");
        finish_legacy();
    }
    std::vector<uint8_t> download() const {
        std::vector<uint8_t> out(bytes_);
        if (bytes_) check(api().cuMemcpyDtoH(out.data(), ptr_, bytes_), "cuMemcpyDtoH");
        return out;
    }
    void zero() { fill(0); }
    void fill(unsigned char v) {
        check(api().cuMemsetD8(ptr_, v, bytes_ ? bytes_ : 1), "cuMemsetD8");
        finish_legacy();
    }
    uint64_t address() const { return ptr_; }
    size_t bytes() const { return bytes_; }

private:
    void release() {
        if (ptr_) api().cuMemFree(ptr_);
        ptr_ = 0;
    }
    CUdeviceptr ptr_ = 0;
    size_t bytes_ = 0;
};

/// A non-blocking stream the gate owns.
class Stream {
public:
    Stream() { check(api().cuStreamCreate(&s_, CU_STREAM_NON_BLOCKING), "cuStreamCreate"); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() { api().cuStreamDestroy(s_); }
    CUstream get() const { return s_; }
    void sync() const { check(api().cuStreamSynchronize(s_), "cuStreamSynchronize"); }

private:
    CUstream s_ = nullptr;
};

/// A CUDA graph the gate captures itself on a stream, instantiated.
class Graph {
public:
    Graph(CUgraph g) {  // NOLINT
        graph_ = g;
        check(api().cuGraphInstantiateWithFlags(&exec_, g, 0), "cuGraphInstantiateWithFlags");
    }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    ~Graph() {
        if (exec_) api().cuGraphExecDestroy(exec_);
        if (graph_) api().cuGraphDestroy(graph_);
    }
    void launch(CUstream s) const { check(api().cuGraphLaunch(exec_, s), "cuGraphLaunch"); }

private:
    CUgraph graph_ = nullptr;
    CUgraphExec exec_ = nullptr;
};

}  // namespace driver

#endif  // TILEFISH_GATE_DRIVER_HPP
