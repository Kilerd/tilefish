// tilefish.hpp: a header-only C++17 wrapper of tilefish.h (the C ABI of the tilefish host, libtilefish_capi).
//
// Licensed under either of Apache License, Version 2.0 or MIT license at your option.
//
// RAII handles that own one C handle each (move-only; share one with std::shared_ptr), typed tensors, and tilefish::Error, which every call
// that fails throws. A program built without exceptions uses tilefish.h. See docs/rfcs/0002-cpp-host.md and docs/guide/cpp.md.
//
//     auto ctx = tilefish::Context::primary(0);
//     auto manifest = tilefish::Manifest::load("build/", "<the manifest's digest>");
//     auto kernel = manifest.kernel(ctx, "gelu_erf");
//     tilefish::Region xr{x_ptr, x_bytes, 0}, yr{y_ptr, y_bytes, 0};   // device memory the host owns (cudaMalloc, cuMemAlloc, ...)
//     kernel.launch(stream, {tilefish::TensorRef<tilefish::Bf16>(xr, {4096}), tilefish::TensorMut<tilefish::Bf16>(yr, {4096})});
//
// Every precondition of the binary is checked before anything is enqueued (dtype, rank, shapes, buckets, alignment, aliasing, the region's
// length); the memory, its lifetime and its ordering against the host's other streams are the host's (see tilefish_tensor in tilefish.h).

#ifndef TILEFISH_HPP
#define TILEFISH_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tilefish.h"

namespace tilefish {

/// What every failing call throws: the status of the C call and the details of tilefish_last_error.
class Error : public std::runtime_error {
public:
    Error(tilefish_status status, int32_t precondition, int32_t cu_result, const std::string& message, std::string argument,
          std::string call)
        : std::runtime_error(message),
          status_(status),
          precondition_(precondition),
          cu_result_(cu_result),
          argument_(std::move(argument)),
          call_(std::move(call)) {}

    /// The status (TILEFISH_ERROR_*).
    tilefish_status status() const noexcept { return status_; }
    /// Its name ("TILEFISH_ERROR_PRECONDITION").
    const char* status_name() const noexcept { return tilefish_status_name(status_); }
    /// For TILEFISH_ERROR_PRECONDITION, which one (TILEFISH_PRECONDITION_*); otherwise TILEFISH_PRECONDITION_NONE.
    int32_t precondition() const noexcept { return precondition_; }
    /// For TILEFISH_ERROR_DRIVER, the CUresult; otherwise 0.
    int32_t cu_result() const noexcept { return cu_result_; }
    /// The argument or dimension a precondition is about ("" when none).
    const std::string& argument() const noexcept { return argument_; }
    /// The driver, cuBLASLt or cuDNN function that failed ("" when none).
    const std::string& call() const noexcept { return call_; }

private:
    tilefish_status status_;
    int32_t precondition_;
    int32_t cu_result_;
    std::string argument_;
    std::string call_;
};

namespace detail {

[[noreturn]] inline void throw_last(tilefish_status status) {
    tilefish_error_info info{};
    tilefish_last_error(&info);
    throw Error(status, info.precondition, info.cu_result, info.message ? info.message : "", info.argument ? info.argument : "",
                info.call ? info.call : "");
}

inline void check(tilefish_status status) {
    if (status != TILEFISH_OK) throw_last(status);
}

inline std::string str(const char* s) { return s ? std::string(s) : std::string(); }

/// The library loaded implements the ABI of this header (checked once, when the first handle is made).
inline void require_abi() {
    static const bool compatible = tilefish_abi_compatible();
    if (!compatible) {
        uint32_t v = tilefish_abi_version();
        throw Error(TILEFISH_ERROR_INVALID_ARGUMENT, TILEFISH_PRECONDITION_NONE, 0,
                    "libtilefish_capi implements ABI " + std::to_string(v >> 16) + "." + std::to_string(v & 0xffffu) +
                        ", but tilefish.h describes " + std::to_string(TILEFISH_ABI_MAJOR) + "." + std::to_string(TILEFISH_ABI_MINOR) +
                        ": build against the headers of the library you link",
                    "", "");
    }
}

/// One C handle, destroyed with its owner.
template <class T, void (*Destroy)(T*)>
class Handle {
public:
    Handle() = default;
    explicit Handle(T* p) : p_(p) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) {
            reset();
            p_ = std::exchange(o.p_, nullptr);
        }
        return *this;
    }
    ~Handle() { reset(); }
    void reset() noexcept {
        if (p_) Destroy(std::exchange(p_, nullptr));
    }
    T* get() const noexcept { return p_; }
    explicit operator bool() const noexcept { return p_ != nullptr; }

private:
    T* p_ = nullptr;
};

/// A contiguous range of `T` (a std::vector, an initializer list, or a pointer and a count): what the calls take.
template <class T>
class Span {
public:
    Span() = default;
    Span(const T* data, size_t size) : data_(data), size_(size) {}
    // A Span is a parameter type only: the array of a braced list a call is given lives until the call returns, which is all a Span needs.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-list-lifetime"
#endif
    Span(std::initializer_list<T> list) : data_(list.begin()), size_(list.size()) {}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    Span(const std::vector<T>& v) : data_(v.data()), size_(v.size()) {}  // NOLINT(google-explicit-constructor)
    const T* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    const T* begin() const noexcept { return data_; }
    const T* end() const noexcept { return data_ + size_; }

private:
    const T* data_ = nullptr;
    size_t size_ = 0;
};

/// Up to `N` items in place, more on the heap: the C arrays of a call, built without allocating in the common case.
template <class T, size_t N = 16>
class SmallVec {
public:
    void push_back(const T& v) {
        if (heap_.empty() && size_ < N) {
            inline_[size_++] = v;
            return;
        }
        if (heap_.empty()) heap_.assign(inline_.begin(), inline_.begin() + size_);
        heap_.push_back(v);
        ++size_;
    }
    const T* data() const noexcept { return heap_.empty() ? inline_.data() : heap_.data(); }
    size_t size() const noexcept { return size_; }

private:
    std::array<T, N> inline_{};
    std::vector<T> heap_;
    size_t size_ = 0;
};

}  // namespace detail

/// Sizes or strides of a tensor: up to 8 in place, more on the heap.
class Dims {
public:
    Dims() = default;
    Dims(std::initializer_list<uint64_t> list) { assign(list.begin(), list.size()); }
    Dims(const std::vector<uint64_t>& v) { assign(v.data(), v.size()); }  // NOLINT(google-explicit-constructor)
    Dims(const uint64_t* data, size_t size) { assign(data, size); }
    const uint64_t* data() const noexcept { return heap_.empty() ? inline_.data() : heap_.data(); }
    size_t size() const noexcept { return size_; }
    uint64_t operator[](size_t i) const noexcept { return data()[i]; }

private:
    void assign(const uint64_t* data, size_t size) {
        size_ = size;
        if (size <= inline_.size()) {
            for (size_t i = 0; i < size; ++i) inline_[i] = data[i];
        } else {
            heap_.assign(data, data + size);
        }
    }
    std::array<uint64_t, 8> inline_{};
    std::vector<uint64_t> heap_;
    size_t size_ = 0;
};

/// The element types, as tags of TensorRef and TensorMut.
struct Bf16 { static constexpr int32_t dtype = TILEFISH_DTYPE_BF16; static constexpr size_t size = 2; };
struct F16 { static constexpr int32_t dtype = TILEFISH_DTYPE_F16; static constexpr size_t size = 2; };
struct F32 { static constexpr int32_t dtype = TILEFISH_DTYPE_F32; static constexpr size_t size = 4; };
struct I32 { static constexpr int32_t dtype = TILEFISH_DTYPE_I32; static constexpr size_t size = 4; };
struct I64 { static constexpr int32_t dtype = TILEFISH_DTYPE_I64; static constexpr size_t size = 8; };
struct U8 { static constexpr int32_t dtype = TILEFISH_DTYPE_U8; static constexpr size_t size = 1; };
struct E4m3 { static constexpr int32_t dtype = TILEFISH_DTYPE_E4M3; static constexpr size_t size = 1; };
struct E5m2 { static constexpr int32_t dtype = TILEFISH_DTYPE_E5M2; static constexpr size_t size = 1; };

/// A region of device memory the host vouches for: its address (a CUdeviceptr), its length in bytes and its device (see tilefish_tensor).
struct Region {
    uint64_t address = 0;
    uint64_t bytes = 0;
    int32_t device = 0;
};

/// A tensor of any element type: a region and the tensor in it (dense, or a view with element strides and an offset).
class Tensor {
public:
    /// A dense row-major tensor at the start of `region`.
    Tensor(Region region, int32_t dtype, int32_t access, Dims shape)
        : region_(region), dtype_(dtype), access_(access), shape_(std::move(shape)) {}
    /// A view: element `strides`, its first element `offset` elements into `region`.
    Tensor(Region region, int32_t dtype, int32_t access, Dims shape, Dims strides, uint64_t offset)
        : region_(region),
          dtype_(dtype),
          access_(access),
          shape_(std::move(shape)),
          strides_(std::move(strides)),
          view_(true),
          offset_(offset) {}

    /// The C tensor (its arrays point into this object, which must outlive the use).
    tilefish_tensor raw() const noexcept {
        tilefish_tensor t{};
        t.address = region_.address;
        t.bytes = region_.bytes;
        t.device = region_.device;
        t.dtype = dtype_;
        t.access = access_;
        t.rank = static_cast<uint32_t>(shape_.size());
        t.shape = shape_.data();
        t.strides = view_ ? strides_.data() : nullptr;
        t.offset = offset_;
        return t;
    }
    const Region& region() const noexcept { return region_; }
    const Dims& shape() const noexcept { return shape_; }

private:
    Region region_;
    int32_t dtype_;
    int32_t access_;
    Dims shape_;
    Dims strides_;
    bool view_ = false;
    uint64_t offset_ = 0;
};

/// A tensor the kernel only reads, of element type `T`.
template <class T>
class TensorRef : public Tensor {
public:
    TensorRef(Region region, Dims shape) : Tensor(region, T::dtype, TILEFISH_ACCESS_IN, std::move(shape)) {}
    TensorRef(Region region, Dims shape, Dims strides, uint64_t offset)
        : Tensor(region, T::dtype, TILEFISH_ACCESS_IN, std::move(shape), std::move(strides), offset) {}
};

/// A tensor the kernel may write (and read), of element type `T`.
template <class T>
class TensorMut : public Tensor {
public:
    TensorMut(Region region, Dims shape) : Tensor(region, T::dtype, TILEFISH_ACCESS_INOUT, std::move(shape)) {}
    TensorMut(Region region, Dims shape, Dims strides, uint64_t offset)
        : Tensor(region, T::dtype, TILEFISH_ACCESS_INOUT, std::move(shape), std::move(strides), offset) {}
};

/// A tensor by name: an external, a weight of a plan.
struct Named {
    const char* name;
    Tensor tensor;
};

/// The value of a device scalar of a plan, by name.
struct DevScalar {
    tilefish_devscalar value{};
    DevScalar(const char* name, int64_t v) { set(name, TILEFISH_DEVSCALAR_INT); value.int_value = v; }
    DevScalar(const char* name, int32_t v) : DevScalar(name, static_cast<int64_t>(v)) {}
    DevScalar(const char* name, float v) { set(name, TILEFISH_DEVSCALAR_F32); value.f32_value = v; }
    /// The bits of one element of the scalar's dtype, in the low bytes.
    static DevScalar bits(const char* name, uint64_t v) {
        DevScalar d(name, int64_t{0});
        d.value.kind = TILEFISH_DEVSCALAR_BITS;
        d.value.bits = v;
        return d;
    }

private:
    void set(const char* name, int32_t kind) {
        value.name = name;
        value.kind = kind;
    }
};

/// A debug tap: the plan's buffer `buffer` copied into `into` after step `after` (negative: after the last step that writes it).
struct Tap {
    const char* buffer;
    Tensor into;
    int64_t after = -1;
};

namespace detail {

inline SmallVec<tilefish_tensor> raw(Span<Tensor> tensors) {
    SmallVec<tilefish_tensor> out;
    for (const auto& t : tensors) out.push_back(t.raw());
    return out;
}

inline SmallVec<tilefish_named_tensor> raw(Span<Named> named) {
    SmallVec<tilefish_named_tensor> out;
    for (const auto& n : named) out.push_back(tilefish_named_tensor{n.name, n.tensor.raw()});
    return out;
}

inline SmallVec<tilefish_devscalar, 8> raw(Span<DevScalar> values) {
    SmallVec<tilefish_devscalar, 8> out;
    for (const auto& v : values) out.push_back(v.value);
    return out;
}

inline SmallVec<tilefish_tap, 4> raw(Span<Tap> taps) {
    SmallVec<tilefish_tap, 4> out;
    for (const auto& t : taps) out.push_back(tilefish_tap{t.buffer, t.after, t.into.raw()});
    return out;
}

}  // namespace detail

/// A CUDA context: the primary context of a device (the CUDA runtime API's), or one the host lends.
class Context {
public:
    /// The primary context of device `device`, retained (and released with the last handle made in it).
    static Context primary(int device) {
        detail::require_abi();
        tilefish_context* p = nullptr;
        detail::check(tilefish_context_primary(device, &p));
        return Context(p);
    }
    /// A context the host made (or a primary context), which tilefish never destroys: keep it alive while handles made in it live.
    static Context borrow(CUcontext context) {
        detail::require_abi();
        tilefish_context* p = nullptr;
        detail::check(tilefish_context_borrow(context, &p));
        return Context(p);
    }
    CUcontext handle() const noexcept { return tilefish_context_handle(h_.get()); }
    int device() const noexcept { return tilefish_context_device(h_.get()); }
    /// The architecture of the device (103 for sm_103).
    uint32_t arch() const {
        uint32_t a = 0;
        detail::check(tilefish_context_arch(h_.get(), &a));
        return a;
    }
    const tilefish_context* get() const noexcept { return h_.get(); }

private:
    explicit Context(tilefish_context* p) : h_(p) {}
    detail::Handle<tilefish_context, tilefish_context_destroy> h_;
};

/// The cuDNN handle of a context, made early and kept while this lives.
class CudnnHandle {
public:
    explicit CudnnHandle(const Context& ctx) {
        tilefish_cudnn_handle* p = nullptr;
        detail::check(tilefish_cudnn_handle_create(ctx.get(), &p));
        h_ = detail::Handle<tilefish_cudnn_handle, tilefish_cudnn_handle_destroy>(p);
    }

private:
    detail::Handle<tilefish_cudnn_handle, tilefish_cudnn_handle_destroy> h_;
};

/// A kernel loaded in a context.
class Kernel {
public:
    explicit Kernel(tilefish_kernel* p) : h_(p) {}
    /// Check the call and launch on `stream` (a stream of the kernel's context, or nullptr for the legacy default stream).
    void launch(CUstream stream, detail::Span<Tensor> tensors, detail::Span<tilefish_given> given = {}) const {
        auto t = detail::raw(tensors);
        detail::check(tilefish_kernel_launch(h_.get(), stream, t.data(), t.size(), given.data(), given.size()));
    }
    /// Check the call as launch does, and launch nothing.
    void check(detail::Span<Tensor> tensors, detail::Span<tilefish_given> given = {}) const {
        auto t = detail::raw(tensors);
        detail::check(tilefish_kernel_check(h_.get(), t.data(), t.size(), given.data(), given.size()));
    }
    std::string stem() const { return detail::str(tilefish_kernel_stem(h_.get())); }
    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        for (size_t i = 0; i < tilefish_kernel_tensor_count(h_.get()); ++i) names.push_back(tilefish_kernel_tensor_name(h_.get(), i));
        return names;
    }
    const tilefish_kernel* get() const noexcept { return h_.get(); }

private:
    detail::Handle<tilefish_kernel, tilefish_kernel_destroy> h_;
};

/// A pinned cuBLASLt GEMM loaded in a context: launch with a, b, bias (when it has one), c (when it has one: `has_c`), out and a workspace.
class Gemm {
public:
    explicit Gemm(tilefish_gemm* p) : h_(p) {}
    void launch(CUstream stream, detail::Span<Tensor> tensors) const {
        auto t = detail::raw(tensors);
        detail::check(tilefish_gemm_launch(h_.get(), stream, t.data(), t.size()));
    }
    void check(detail::Span<Tensor> tensors) const {
        auto t = detail::raw(tensors);
        detail::check(tilefish_gemm_check(h_.get(), t.data(), t.size()));
    }
    /// The epilogue, one of `TILEFISH_EPILOGUE_*` (ABI 1.1).
    int32_t epilogue() const {
        int32_t value = TILEFISH_EPILOGUE_NONE;
        detail::check(tilefish_gemm_epilogue(h_.get(), &value, nullptr));
        return value;
    }
    /// Whether a launch takes `c`, the tensor before `out` (a `cublaslt.matmul` with `beta` 1: the epilogue acts on `acc + c`; ABI 1.1).
    bool has_c() const {
        bool value = false;
        detail::check(tilefish_gemm_epilogue(h_.get(), nullptr, &value));
        return value;
    }
    /// Whether a launch takes `bias`.
    bool has_bias() const {
        bool value = false;
        detail::check(tilefish_gemm_sizes(h_.get(), nullptr, nullptr, nullptr, &value));
        return value;
    }
    uint64_t workspace_bytes() const noexcept { return tilefish_gemm_workspace_bytes(h_.get()); }
    uint64_t workspace_required() const noexcept { return tilefish_gemm_workspace_required(h_.get()); }
    std::string stem() const { return detail::str(tilefish_gemm_stem(h_.get())); }
    const tilefish_gemm* get() const noexcept { return h_.get(); }

private:
    detail::Handle<tilefish_gemm, tilefish_gemm_destroy> h_;
};

/// A pinned cuDNN forward convolution loaded in a context: launch with x, w, y and a workspace.
class Conv {
public:
    explicit Conv(tilefish_conv* p) : h_(p) {}
    void launch(CUstream stream, detail::Span<Tensor> tensors) const {
        auto t = detail::raw(tensors);
        detail::check(tilefish_conv_launch(h_.get(), stream, t.data(), t.size()));
    }
    void check(detail::Span<Tensor> tensors) const {
        auto t = detail::raw(tensors);
        detail::check(tilefish_conv_check(h_.get(), t.data(), t.size()));
    }
    void check_canary() const { detail::check(tilefish_conv_check_canary(h_.get())); }
    uint64_t workspace_bytes() const noexcept { return tilefish_conv_workspace_bytes(h_.get()); }
    uint64_t workspace_required() const noexcept { return tilefish_conv_workspace_required(h_.get()); }
    std::string stem() const { return detail::str(tilefish_conv_stem(h_.get())); }
    const tilefish_conv* get() const noexcept { return h_.get(); }

private:
    detail::Handle<tilefish_conv, tilefish_conv_destroy> h_;
};

/// A manifest, loaded and verified (needs no driver).
class Manifest {
public:
    /// Load and verify the manifest directory, pinned to its digest.
    static Manifest load(const std::string& directory, const std::string& pin) {
        detail::require_abi();
        tilefish_manifest* p = nullptr;
        detail::check(tilefish_manifest_load(directory.c_str(), pin.c_str(), &p));
        return Manifest(p);
    }
    /// Load and verify a manifest directory trusting its own digests (development).
    static Manifest load_trusting_directory(const std::string& directory) {
        detail::require_abi();
        tilefish_manifest* p = nullptr;
        detail::check(tilefish_manifest_load_trusting_directory(directory.c_str(), &p));
        return Manifest(p);
    }
    /// A manifest whose files are in memory, verified like a directory and pinned.
    static Manifest from_embedded(const uint8_t* manifest, size_t len, detail::Span<tilefish_file> files, const std::string& pin) {
        detail::require_abi();
        tilefish_manifest* p = nullptr;
        detail::check(tilefish_manifest_from_embedded(manifest, len, files.data(), files.size(), pin.c_str(), &p));
        return Manifest(p);
    }
    /// A copy for a host that runs on a partition of `sms` SMs (0: the whole device).
    Manifest with_sm_partition(uint64_t sms) const {
        tilefish_manifest* p = nullptr;
        detail::check(tilefish_manifest_with_sm_partition(h_.get(), sms, &p));
        return Manifest(p);
    }
    void check_cuda_math(const std::string& cuda) const { detail::check(tilefish_manifest_check_cuda_math(h_.get(), cuda.c_str())); }
    std::string digest() const { return detail::str(tilefish_manifest_digest(h_.get())); }
    std::string label() const { return detail::str(tilefish_manifest_label(h_.get())); }
    /// Refuse a manifest whose digest is not `pinned` (what a generated launcher does at load).
    void require_digest(const char* pinned) const {
        std::string digest = this->digest();
        if (digest != pinned) {
            throw Error(TILEFISH_ERROR_PIN_MISMATCH, TILEFISH_PRECONDITION_NONE, 0,
                        "the manifest from " + label() + " has digest " + digest + ", but the launcher was generated from " + pinned +
                            ": generate it again from this manifest",
                        "", "");
        }
    }
    /// Load a kernel for the device of `ctx` (`arch` nullptr: the device's architecture).
    Kernel kernel(const Context& ctx, const char* stem, detail::Span<tilefish_bucket> buckets = {}, const char* arch = nullptr) const {
        tilefish_kernel* p = nullptr;
        detail::check(tilefish_kernel_load(h_.get(), ctx.get(), stem, buckets.data(), buckets.size(), arch, &p));
        return Kernel(p);
    }
    Gemm gemm(const Context& ctx, const char* stem, detail::Span<tilefish_bucket> buckets = {}, const char* arch = nullptr) const {
        tilefish_gemm* p = nullptr;
        detail::check(tilefish_gemm_load(h_.get(), ctx.get(), stem, buckets.data(), buckets.size(), arch, &p));
        return Gemm(p);
    }
    Conv conv(const Context& ctx, const char* stem, detail::Span<tilefish_bucket> buckets = {}, const char* arch = nullptr) const {
        tilefish_conv* p = nullptr;
        detail::check(tilefish_conv_load(h_.get(), ctx.get(), stem, buckets.data(), buckets.size(), arch, &p));
        return Conv(p);
    }
    const tilefish_manifest* get() const noexcept { return h_.get(); }
    explicit Manifest(tilefish_manifest* p) : h_(p) {}

private:
    detail::Handle<tilefish_manifest, tilefish_manifest_destroy> h_;
};

class CapturedPlan;
class WeightedRun;

/// A plan captured in a CUDA graph (one bucket). One thread at a time.
class CapturedPlan {
public:
    explicit CapturedPlan(tilefish_captured_plan* p) : h_(p) {}
    void replay(CUstream stream) { detail::check(tilefish_captured_plan_replay(h_.get(), stream)); }
    void set_devscalar(const DevScalar& value) { detail::check(tilefish_captured_plan_set_devscalar(h_.get(), &value.value)); }
    tilefish_captured_plan* get() const noexcept { return h_.get(); }

private:
    detail::Handle<tilefish_captured_plan, tilefish_captured_plan_destroy> h_;
};

/// A plan run with its weights bound once.
class WeightedRun {
public:
    explicit WeightedRun(tilefish_weighted_run* p) : h_(p) {}
    void check(detail::Span<Named> externals, detail::Span<DevScalar> devscalars = {}) const {
        auto e = detail::raw(externals);
        auto d = detail::raw(devscalars);
        detail::check(tilefish_weighted_run_check(h_.get(), e.data(), e.size(), d.data(), d.size()));
    }
    void launch(CUstream stream, detail::Span<Named> externals, detail::Span<DevScalar> devscalars = {},
                detail::Span<Tap> taps = {}) const {
        auto e = detail::raw(externals);
        auto d = detail::raw(devscalars);
        auto t = detail::raw(taps);
        detail::check(tilefish_weighted_run_launch(h_.get(), stream, e.data(), e.size(), d.data(), d.size(), t.data(), t.size()));
    }
    CapturedPlan capture(CUstream stream, detail::Span<Named> externals, detail::Span<DevScalar> devscalars = {},
                         detail::Span<Tap> taps = {}) const {
        auto e = detail::raw(externals);
        auto d = detail::raw(devscalars);
        auto t = detail::raw(taps);
        tilefish_captured_plan* p = nullptr;
        detail::check(tilefish_weighted_run_capture(h_.get(), stream, e.data(), e.size(), d.data(), d.size(), t.data(), t.size(), &p));
        return CapturedPlan(p);
    }

private:
    detail::Handle<tilefish_weighted_run, tilefish_weighted_run_destroy> h_;
};

/// A plan instantiated in a context: its binaries loaded, its arena allocated (or shared).
class PlanRun {
public:
    explicit PlanRun(tilefish_plan_run* p) : h_(p) {}
    void check(detail::Span<Named> externals, detail::Span<DevScalar> devscalars = {}) const {
        auto e = detail::raw(externals);
        auto d = detail::raw(devscalars);
        detail::check(tilefish_plan_run_check(h_.get(), e.data(), e.size(), d.data(), d.size()));
    }
    /// Run the plan on `stream`: every step checked before the first launch, nothing allocated.
    void launch(CUstream stream, detail::Span<Named> externals, detail::Span<DevScalar> devscalars = {},
                detail::Span<Tap> taps = {}) const {
        auto e = detail::raw(externals);
        auto d = detail::raw(devscalars);
        auto t = detail::raw(taps);
        detail::check(tilefish_plan_run_launch_tapped(h_.get(), stream, e.data(), e.size(), d.data(), d.size(), t.data(), t.size()));
    }
    /// Capture one execution in a CUDA graph on `stream`; the externals' memory stays allocated while the captured plan lives.
    CapturedPlan capture(CUstream stream, detail::Span<Named> externals, detail::Span<DevScalar> devscalars = {},
                         detail::Span<Tap> taps = {}) const {
        auto e = detail::raw(externals);
        auto d = detail::raw(devscalars);
        auto t = detail::raw(taps);
        tilefish_captured_plan* p = nullptr;
        detail::check(
            tilefish_plan_run_capture_tapped(h_.get(), stream, e.data(), e.size(), d.data(), d.size(), t.data(), t.size(), &p));
        return CapturedPlan(p);
    }
    /// Bind the weights once; their memory stays allocated while the weighted run (and its captures) live.
    WeightedRun bind_weights(detail::Span<Named> weights) const {
        auto w = detail::raw(weights);
        tilefish_weighted_run* p = nullptr;
        detail::check(tilefish_plan_run_bind_weights(h_.get(), w.data(), w.size(), &p));
        return WeightedRun(p);
    }
    uint64_t arena_bytes() const noexcept { return tilefish_plan_run_arena_bytes(h_.get()); }
    const tilefish_plan_run* get() const noexcept { return h_.get(); }

private:
    detail::Handle<tilefish_plan_run, tilefish_plan_run_destroy> h_;
};

/// An arena shared by plans that run one at a time.
class SharedArena {
public:
    explicit SharedArena(tilefish_shared_arena* p) : h_(p) {}
    uint64_t bytes() const noexcept { return tilefish_shared_arena_bytes(h_.get()); }
    uint64_t address() const noexcept { return tilefish_shared_arena_address(h_.get()); }
    const tilefish_shared_arena* get() const noexcept { return h_.get(); }

private:
    detail::Handle<tilefish_shared_arena, tilefish_shared_arena_destroy> h_;
};

/// A buffer of a plan.
struct BufferInfo {
    std::string name;
    int32_t role;  // TILEFISH_ROLE_*
    int32_t dtype;  // TILEFISH_DTYPE_*
    uint32_t rank;
    uint64_t bytes;
};

/// A launch plan, verified (needs no driver).
class LaunchPlan {
public:
    static LaunchPlan load(const std::string& path, const std::string& pin) {
        detail::require_abi();
        tilefish_plan* p = nullptr;
        detail::check(tilefish_plan_load(path.c_str(), pin.c_str(), &p));
        return LaunchPlan(p);
    }
    static LaunchPlan load_trusting_directory(const std::string& path) {
        detail::require_abi();
        tilefish_plan* p = nullptr;
        detail::check(tilefish_plan_load_trusting_directory(path.c_str(), &p));
        return LaunchPlan(p);
    }
    static LaunchPlan load_with_manifest(const std::string& path, const Manifest& manifest, const std::string& pin) {
        tilefish_plan* p = nullptr;
        detail::check(tilefish_plan_load_with_manifest(path.c_str(), manifest.get(), pin.c_str(), &p));
        return LaunchPlan(p);
    }
    static LaunchPlan from_embedded(const uint8_t* plan, size_t len, const Manifest& manifest, const std::string& pin) {
        tilefish_plan* p = nullptr;
        detail::check(tilefish_plan_from_embedded(plan, len, manifest.get(), pin.c_str(), &p));
        return LaunchPlan(p);
    }
    std::string digest() const { return detail::str(tilefish_plan_digest(h_.get())); }
    std::string manifest_digest() const { return detail::str(tilefish_plan_manifest_digest(h_.get())); }
    std::string graph() const { return detail::str(tilefish_plan_graph(h_.get())); }
    std::string arch() const { return detail::str(tilefish_plan_arch(h_.get())); }
    std::string label() const { return detail::str(tilefish_plan_label(h_.get())); }
    uint64_t arena_bytes() const noexcept { return tilefish_plan_arena_bytes(h_.get()); }
    std::vector<BufferInfo> buffers() const {
        std::vector<BufferInfo> out;
        for (size_t i = 0; i < tilefish_plan_buffer_count(h_.get()); ++i) {
            tilefish_buffer_info b{};
            detail::check(tilefish_plan_buffer(h_.get(), i, &b));
            out.push_back(BufferInfo{b.name, b.role, b.dtype, b.rank, b.bytes});
        }
        return out;
    }
    std::vector<std::string> outputs() const {
        std::vector<std::string> out;
        for (size_t i = 0; i < tilefish_plan_output_count(h_.get()); ++i) out.emplace_back(tilefish_plan_output(h_.get(), i));
        return out;
    }
    std::vector<std::pair<std::string, std::vector<uint64_t>>> dims() const {
        std::vector<std::pair<std::string, std::vector<uint64_t>>> out;
        for (size_t i = 0; i < tilefish_plan_dim_count(h_.get()); ++i) {
            const char* name = nullptr;
            const uint64_t* values = nullptr;
            size_t n = 0;
            detail::check(tilefish_plan_dim(h_.get(), i, &name, &values, &n));
            out.emplace_back(name, std::vector<uint64_t>(values, values + n));
        }
        return out;
    }
    Manifest manifest() const {
        tilefish_manifest* p = nullptr;
        detail::check(tilefish_plan_manifest(h_.get(), &p));
        return Manifest(p);
    }
    /// Instantiate in `ctx`, the arena allocated on `stream` (valid until the arena is freed); no buckets: every bucket.
    PlanRun instantiate(const Context& ctx, CUstream stream, detail::Span<tilefish_bucket> buckets = {}) const {
        tilefish_plan_run* p = nullptr;
        detail::check(tilefish_plan_instantiate(h_.get(), ctx.get(), stream, buckets.data(), buckets.size(), &p));
        return PlanRun(p);
    }
    PlanRun instantiate_in(const SharedArena& arena, detail::Span<tilefish_bucket> buckets = {}) const {
        tilefish_plan_run* p = nullptr;
        detail::check(tilefish_plan_instantiate_in(h_.get(), arena.get(), buckets.data(), buckets.size(), &p));
        return PlanRun(p);
    }
    const tilefish_plan* get() const noexcept { return h_.get(); }
    explicit LaunchPlan(tilefish_plan* p) : h_(p) {}

private:
    detail::Handle<tilefish_plan, tilefish_plan_destroy> h_;
};

/// Allocate one arena for `plans` in `ctx` on `stream`.
inline SharedArena make_shared_arena(const Context& ctx, CUstream stream, detail::Span<const LaunchPlan*> plans) {
    detail::SmallVec<const tilefish_plan*, 8> raw;
    for (const LaunchPlan* p : plans) raw.push_back(p->get());
    tilefish_shared_arena* p = nullptr;
    detail::check(tilefish_shared_arena_create(ctx.get(), stream, raw.data(), raw.size(), &p));
    return SharedArena(p);
}

/// Plans of a set instantiated on one shared arena.
class SetRun {
public:
    explicit SetRun(tilefish_set_run* p) : h_(p) {}
    PlanRun run(const char* graph) const {
        tilefish_plan_run* p = nullptr;
        detail::check(tilefish_set_run_plan(h_.get(), graph, &p));
        return PlanRun(p);
    }
    SharedArena arena() const {
        tilefish_shared_arena* p = nullptr;
        detail::check(tilefish_set_run_arena(h_.get(), &p));
        return SharedArena(p);
    }

private:
    detail::Handle<tilefish_set_run, tilefish_set_run_destroy> h_;
};

/// Plans exported together over one manifest, loaded with one pin.
class PlanSet {
public:
    static PlanSet load(const std::string& directory, const std::string& pin) {
        detail::require_abi();
        tilefish_plan_set* p = nullptr;
        detail::check(tilefish_plan_set_load(directory.c_str(), pin.c_str(), &p));
        return PlanSet(p);
    }
    static PlanSet load_trusting_directory(const std::string& directory) {
        detail::require_abi();
        tilefish_plan_set* p = nullptr;
        detail::check(tilefish_plan_set_load_trusting_directory(directory.c_str(), &p));
        return PlanSet(p);
    }
    std::string digest() const { return detail::str(tilefish_plan_set_digest(h_.get())); }
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (size_t i = 0; i < tilefish_plan_set_name_count(h_.get()); ++i) out.emplace_back(tilefish_plan_set_name(h_.get(), i));
        return out;
    }
    LaunchPlan plan(const char* graph) const {
        tilefish_plan* p = nullptr;
        detail::check(tilefish_plan_set_plan(h_.get(), graph, &p));
        return LaunchPlan(p);
    }
    /// The plans of `graphs`, loaded on several threads.
    std::vector<LaunchPlan> plans(detail::Span<const char*> graphs) const {
        std::vector<tilefish_plan*> raw(graphs.size(), nullptr);
        detail::check(tilefish_plan_set_plans(h_.get(), graphs.data(), graphs.size(), raw.data()));
        std::vector<LaunchPlan> out;
        for (tilefish_plan* p : raw) out.emplace_back(p);
        return out;
    }
    Manifest manifest() const {
        tilefish_manifest* p = nullptr;
        detail::check(tilefish_plan_set_manifest(h_.get(), &p));
        return Manifest(p);
    }
    uint64_t shared_arena_bytes(detail::Span<const char*> graphs) const {
        uint64_t bytes = 0;
        detail::check(tilefish_plan_set_shared_arena_bytes(h_.get(), graphs.data(), graphs.size(), &bytes));
        return bytes;
    }
    /// Instantiate the plans `graphs` in `ctx` on one shared arena allocated on `stream` (valid until the arena is freed).
    SetRun instantiate(const Context& ctx, CUstream stream, detail::Span<const char*> graphs) const {
        tilefish_set_run* p = nullptr;
        detail::check(tilefish_plan_set_instantiate(h_.get(), ctx.get(), stream, graphs.data(), graphs.size(), &p));
        return SetRun(p);
    }

private:
    explicit PlanSet(tilefish_plan_set* p) : h_(p) {}
    detail::Handle<tilefish_plan_set, tilefish_plan_set_destroy> h_;
};

/// Whether the CUDA driver can be loaded (manifests and plans load without one).
inline bool driver_available() { return tilefish_driver_available(); }

}  // namespace tilefish

#endif  // TILEFISH_HPP
