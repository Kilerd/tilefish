//! The calls that enqueue a launch, and the second place where the host takes on an obligation.
//!
//! The CUDA driver is cudarc's safe layer: contexts, modules, functions, attributes and streams are cudarc's, and so is the launch
//! builder that [`Kernel::launch`] goes through; [`Kernel::launch_raw`] uses cudarc's `result::launch_kernel` for a stream that
//! cudarc does not own. A kernel of thread-block clusters (manifest schema 3, section 4.5) is launched with `cuLaunchKernelEx` and the cluster
//! attribute ([`launch_prepared`]), which cudarc has no wrapper of. cuBLASLt is cudarc's `cublaslt::result` and `cublaslt::sys`, called from the `cublaslt` submodule. The calls to
//! `cudarc::driver::sys` in this crate are the probe of [`driver_available`], the version of the driver (`cuDriverGetVersion`), the query of the
//! parameter sizes of a loaded function (`cuFuncGetParamInfo`) and the eager load of a function of a plan (`cuFuncLoad`, in the `plan`
//! submodule), which cudarc has no wrappers for; `result::graph::launch` launches a captured plan on another stream than its own. The `unsafe` code of the crate is in the items of this module and its
//! `cublaslt` submodule (the launch of a kernel, [`Kernel::launch_raw`], [`Gemm::launch_raw`], the restore of the current context, the
//! probes, the views of the bytes of a slice, and every call of cuBLASLt) and one in `region` ([`crate::DevicePtr::from_raw`]), each with
//! the reason it is sound beside it.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceSlice as _, LaunchConfig,
    PushKernelArg, SyncOnDrop, result, sys,
};

use crate::conv::{Conv, ConvCall};
use crate::error::{Error, Result};
use crate::gemm::{Call, Gemm};
use crate::kernel::Kernel;
use crate::memory::sealed;
use crate::plan::{MAX_PARAMS, NOT_A_TENSOR, Prepared};
use crate::tensor::{Backing, TensorArg};

pub(crate) mod cublaslt;
pub(crate) mod cudnn;
pub(crate) mod plan;

#[cfg(target_endian = "big")]
compile_error!("tilefish packs a 32-bit kernel argument into the low bytes of a 64-bit slot: little-endian hosts only");

/// Whether the CUDA driver library (`libcuda`) can be loaded on this machine.
///
/// cudarc loads the driver when it is first called, and **`CudaContext::new` panics inside cudarc when there is no driver**: this
/// crate builds, and its tests that need no GPU run, on a machine without one, so a program that may run on such a machine asks
/// first.
///
/// The probe opens the library (`dlopen`) and keeps it open when it is there. Opening a library runs its initializers, so on a
/// machine that has one this has the effect of loading the driver into the process, as the first call of cudarc would; it calls no
/// function of the driver.
#[allow(unsafe_code)]
pub fn driver_available() -> bool {
    // SAFETY: `is_culib_present` only tries to open the library by its usual names (and keeps it open when it is there): loading
    // the driver, which runs its initializers as any use of it does, and calling no function in it.
    unsafe { sys::is_culib_present() }
}

/// The oldest CUDA driver this crate launches with, as `cuDriverGetVersion` writes it (major * 1000 + minor * 10): 12.4, which has
/// `cuFuncGetParamInfo` (the check of a loaded function's parameters against the ABI of the manifest) and `cuFuncLoad`.
const OLDEST_DRIVER: i32 = 12040;

/// The version of the CUDA driver (`cuDriverGetVersion`: `13020` for 13.2), which cudarc has no wrapper of.
#[allow(unsafe_code)]
pub(crate) fn driver_version() -> Result<i32> {
    let mut version = 0;
    // SAFETY: the out-pointer is a live local; the function only writes the version (it is in every driver).
    let status = unsafe { sys::cuDriverGetVersion(&mut version) };
    status.result().map_err(Error::driver("cuDriverGetVersion"))?;
    Ok(version)
}

/// Refuse a driver older than [`OLDEST_DRIVER`]: cudarc loads a function of the driver when it is first called, and **panics** when the driver
/// does not have it, so the functions that came with 12.4 are not called on an older one.
fn require_driver(version: i32) -> Result<()> {
    if version < OLDEST_DRIVER {
        return Err(Error::Unsupported(format!(
            "the CUDA driver of this machine is {}.{}, but this host needs 12.4 or newer (cuFuncGetParamInfo, with which a loaded binary's \
             parameters are checked against its ABI): update the driver",
            version / 1000,
            (version % 1000) / 10
        )));
    }
    Ok(())
}

/// The sizes in bytes of the parameters of a loaded function, as the driver reports them (`cuFuncGetParamInfo`, one query per parameter
/// until it has no more): what the ABI of the manifest is checked against, as Python's `_check_params` does at load. Refused (and nothing
/// asked) when the driver is older than 12.4, which does not have the function.
pub(crate) fn param_sizes(function: &CudaFunction) -> Result<Vec<usize>> {
    require_driver(driver_version()?)?;
    Ok(sizes_of(function))
}

#[allow(unsafe_code)]
fn sizes_of(function: &CudaFunction) -> Vec<usize> {
    let mut sizes = Vec::new();
    loop {
        let (mut offset, mut size) = (0usize, 0usize);
        // SAFETY: `function` is a live function of a loaded module; the two out-pointers are live locals. An index past the last
        // parameter is answered with an error, not undefined behavior (the end of the loop).
        let status = unsafe { sys::cuFuncGetParamInfo(function.cu_function(), sizes.len(), &mut offset, &mut size) };
        if status != sys::CUresult::CUDA_SUCCESS {
            return sizes;
        }
        sizes.push(size);
    }
}

/// A view of the bytes of a cudarc slice from byte `from` on, for a tensor whose first element is not at the start of the slice. `None`
/// when `from` is past the end.
#[allow(unsafe_code)]
pub(crate) fn bytes_of_slice<E>(slice: &CudaSlice<E>, from: u64) -> Option<CudaView<'_, u8>> {
    // SAFETY: a `u8` has no invalid bit pattern and no alignment, and the view covers exactly the bytes of the slice (`num_bytes`), so
    // reading it as bytes is what any kernel does with the memory; cudarc's own check keeps it within the slice.
    let all = unsafe { slice.transmute::<u8>(slice.num_bytes()) }?;
    all.try_slice(usize::try_from(from).ok()?..)
}

/// As [`bytes_of_slice`], for a view of cudarc's.
#[allow(unsafe_code)]
pub(crate) fn bytes_of_view<'a, E>(view: &CudaView<'a, E>, from: u64) -> Option<CudaView<'a, u8>> {
    // SAFETY: as in `bytes_of_slice`.
    let all = unsafe { view.transmute::<u8>(view.num_bytes()) }?;
    all.try_slice(usize::try_from(from).ok()?..)
}

/// As [`bytes_of_slice`], for a slice that is written.
///
/// The view it returns has length 1: it is a pointer to the first element and the events of the slice, which is all a launch needs of it
/// (cudarc's `chunks_exact_mut` is the call that gives a view of the original lifetime from one made by `transmute_mut`; its first chunk
/// is the view of the byte `from`).
#[allow(unsafe_code)]
pub(crate) fn bytes_of_slice_mut<E>(slice: &mut CudaSlice<E>, from: u64) -> Option<CudaViewMut<'_, u8>> {
    let bytes = slice.num_bytes();
    // SAFETY: as in `bytes_of_slice` (the memory is only handed to a kernel, which writes bytes).
    let all = unsafe { slice.transmute_mut::<u8>(bytes) }?;
    all.chunks_exact_mut(1).nth(usize::try_from(from).ok()?)
}

/// As [`bytes_of_slice_mut`], for a mutable view of cudarc's.
#[allow(unsafe_code)]
pub(crate) fn bytes_of_view_mut<'a, E>(view: &'a mut CudaViewMut<'_, E>, from: u64) -> Option<CudaViewMut<'a, u8>> {
    let bytes = view.num_bytes();
    // SAFETY: as in `bytes_of_slice_mut`.
    let all = unsafe { view.transmute_mut::<u8>(bytes) }?;
    all.chunks_exact_mut(1).nth(usize::try_from(from).ok()?)
}

/// All the bytes of a cudarc slice that is written, as a view of its whole length (sharing its events).
#[allow(unsafe_code)]
pub(crate) fn all_bytes_of_slice_mut<E>(slice: &mut CudaSlice<E>) -> Option<CudaViewMut<'_, u8>> {
    let bytes = slice.num_bytes();
    // SAFETY: as in `bytes_of_slice_mut`: a `u8` has no invalid bit pattern and no alignment, and the view covers exactly the bytes of the
    // slice.
    unsafe { slice.transmute_mut::<u8>(bytes) }
}

/// All the bytes of a mutable view of cudarc's, as a view that is read.
#[allow(unsafe_code)]
pub(crate) fn all_bytes_of_view_mut<'a, E>(view: &'a CudaViewMut<'_, E>) -> Option<CudaView<'a, u8>> {
    // SAFETY: as in `bytes_of_slice`.
    unsafe { view.transmute::<u8>(view.num_bytes()) }
}

/// All the bytes of a mutable view of cudarc's, as a view that may be written.
#[allow(unsafe_code)]
pub(crate) fn all_bytes_of_view_mut_mut<'a, E>(view: &'a mut CudaViewMut<'_, E>) -> Option<CudaViewMut<'a, u8>> {
    let bytes = view.num_bytes();
    // SAFETY: as in `bytes_of_slice_mut`.
    unsafe { view.transmute_mut::<u8>(bytes) }
}

/// The 128 bytes of an encoded `CUtensorMap`, 64-byte aligned as its kernel parameter is (`.param .align 64 .b8 [128]`): what a launch passes
/// by value for a TMA descriptor.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(64))]
pub(crate) struct TensorMapBytes(pub(crate) [u64; 16]);

impl std::fmt::Debug for TensorMapBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorMapBytes(")?;
        for w in &self.0 {
            write!(f, "{w:016x}")?;
        }
        write!(f, ")")
    }
}

// SAFETY: a plain array of 16 `u64`, `repr(C)`, with no padding and no pointer the kernel follows: the driver copies its 128 bytes into the
// parameter of the kernel, which reads them as the `CUtensorMap` that `cuTensorMapEncodeTiled` wrote into them (`encode_tensor_maps`).
#[allow(unsafe_code)]
unsafe impl cudarc::driver::DeviceRepr for TensorMapBytes {}

/// Encode the TMA descriptors of a checked call (`cuTensorMapEncodeTiled`, which cudarc has no wrapper of): each with the arguments
/// `Plan::prepare` computed from its tensor (the address, shape and strides it was checked with) and the map of its parameter. A call the driver
/// refuses is an `Error::Driver`; the checks of `Plan::prepare` refuse, with a kind, everything the driver is documented to refuse of a tensor.
/// The map is a function of those numbers alone: a launch encodes its own, and a graph that captured a launch holds the bytes of that one.
#[allow(unsafe_code)]
pub(crate) fn encode_tensor_maps(prepared: &mut Prepared) -> Result<()> {
    for map in &mut prepared.maps {
        let e = &map.encode;
        let rank = e.rank as usize;
        if !(1..=5).contains(&rank)
            || [e.global_dim.len(), e.global_strides.len(), e.box_dim.len(), e.element_strides.len()] != [rank; 4]
        {
            return Err(Error::Launch(format!("a tensor map of rank {rank} has arguments of other lengths: {e:?}")));
        }
        let data_type = match e.data_type {
            0 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            3 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
            5 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
            6 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            7 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            9 => sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            other => return Err(Error::Launch(format!("a tensor map of data type {other} is not supported"))),
        };
        let swizzle = match e.swizzle {
            0 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            1 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
            2 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
            _ => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        };
        let l2 = match e.l2_promotion {
            0 => sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            1 => sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
            2 => sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            _ => sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        };
        let fill = if e.oob_fill == 1 {
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
        } else {
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        };
        let mut out = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
        // SAFETY: `out` is a live, aligned `CUtensorMap` (cudarc's type has the alignment the driver requires) that the call writes; the four
        // arrays are live locals of `rank` entries each (checked above: the driver reads `rank` dimensions, box sizes and element strides and
        // `rank - 1` strides of them); the address is only recorded in the map (the driver does not dereference it), and the enumerators are
        // valid values of their types. The function exists in every driver this crate runs on (CUDA 12.0 and newer).
        let status = unsafe {
            sys::cuTensorMapEncodeTiled(
                out.as_mut_ptr(),
                data_type,
                e.rank,
                e.address as *mut c_void,
                e.global_dim.as_ptr(),
                e.global_strides.as_ptr(),
                e.box_dim.as_ptr(),
                e.element_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle,
                l2,
                fill,
            )
        };
        status.result().map_err(Error::driver("cuTensorMapEncodeTiled"))?;
        // SAFETY: the call succeeded, so it wrote the whole map (and it was zeroed before).
        let encoded = unsafe { out.assume_init() };
        map.bytes = Some(TensorMapBytes(encoded.opaque));
    }
    Ok(())
}

/// What a launch pushes for one tensor: the value in the slot (a region), the slice itself, or a view of it from the first element on; or,
/// for the tensor of a TMA descriptor (whose slot is the map), the guard of the slice's tracking for the time of the launch.
enum Holder<'t> {
    Value,
    Read(&'t dyn sealed::Read),
    Write(&'t mut dyn sealed::Write),
    ReadView(CudaView<'t, u8>),
    WriteView(CudaViewMut<'t, u8>),
    Guard(#[allow(dead_code)] SyncOnDrop<'t>),
}

/// The `cuLaunchKernelEx` attributes of a launch of a kernel of clusters: the cluster's dimensions, and the spread scheduling policy Triton's own
/// launcher asks for (`CU_CLUSTER_SCHEDULING_POLICY_SPREAD`).
fn cluster_attributes(cluster: [u32; 3]) -> [sys::CUlaunchAttribute; 2] {
    let mut dimension = sys::CUlaunchAttributeValue { pad: [0; 64] };
    dimension.clusterDim =
        sys::CUlaunchAttributeValue_union__bindgen_ty_1 { x: cluster[0], y: cluster[1], z: cluster[2] };
    let mut policy = sys::CUlaunchAttributeValue { pad: [0; 64] };
    policy.clusterSchedulingPolicyPreference = sys::CUclusterSchedulingPolicy::CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    [
        sys::CUlaunchAttribute {
            id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
            pad: [0; 4],
            value: dimension,
        },
        sys::CUlaunchAttribute {
            id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
            pad: [0; 4],
            value: policy,
        },
    ]
}

/// Launch `function` on the raw `stream` with the `kernelParams` of `prepared` (each slot pointing at its value, the slot of a TMA descriptor at
/// the 128 bytes of its encoded map): `cuLaunchKernel` (cudarc's `result::launch_kernel`), or for a kernel of clusters `cuLaunchKernelEx` with the
/// cluster attribute (cudarc has no wrapper of it); the grid is `prepared.grid`, the blocks of the launch.
///
/// # Safety
///
/// `function` is a function of a loaded module whose function was checked to take the parameter sizes of the ABI `prepared` was made from (by
/// `Plan::prepare`, for that very entry: its slots, its grid against the device, its addresses against their memory); `stream` is a valid stream
/// of the function's context, which is current on this thread; and the memory the addresses name stays allocated, and is ordered against other
/// streams, until the kernel has run (the caller's: `ordered`, a plan's arena, or the contract of `launch_raw`).
#[allow(unsafe_code)]
pub(crate) unsafe fn launch_prepared(
    function: sys::CUfunction,
    prepared: &Prepared,
    block: [u32; 3],
    shared: u32,
    stream: sys::CUstream,
) -> Result<()> {
    let count = prepared.count;
    let mut values = prepared.values;
    let mut maps = prepared
        .maps
        .iter()
        .map(|m| m.bytes.ok_or_else(|| Error::Launch("a tensor map was not encoded".to_owned())))
        .collect::<Result<Vec<_>>>()?;
    let mut params: [*mut c_void; MAX_PARAMS] = [std::ptr::null_mut(); MAX_PARAMS];
    for (slot, value) in params.iter_mut().zip(values.iter_mut()).take(count) {
        *slot = std::ptr::from_mut(value).cast::<c_void>();
    }
    for (map, bytes) in prepared.maps.iter().zip(maps.iter_mut()) {
        params[map.slot] = std::ptr::from_mut(bytes).cast::<c_void>(); // the slot of a TMA descriptor: its 128 bytes, by value
    }
    let grid = prepared.grid;
    match prepared.cluster {
        // SAFETY: `params[..count]` point at the live slots copied from `prepared` (and the maps' bytes), which the driver reads before it
        // returns, in the order of the manifest's ABI; the rest is the contract of this function.
        None => unsafe {
            result::launch_kernel(
                function,
                (grid[0], grid[1], grid[2]),
                (block[0], block[1], block[2]),
                shared,
                stream,
                &mut params[..count],
            )
        }
        .map_err(Error::driver("cuLaunchKernel")),
        Some(cluster) => {
            let mut attributes = cluster_attributes(cluster);
            let config = sys::CUlaunchConfig {
                gridDimX: grid[0],
                gridDimY: grid[1],
                gridDimZ: grid[2],
                blockDimX: block[0],
                blockDimY: block[1],
                blockDimZ: block[2],
                sharedMemBytes: shared,
                hStream: stream,
                attrs: attributes.as_mut_ptr(),
                numAttrs: attributes.len() as u32,
            };
            // SAFETY: as above, and `config` and its two attributes are live locals the driver reads before it returns (the grid is a whole number
            // of clusters: `Plan::prepare` made it the programs times the cluster); `cuLaunchKernelEx` is in every driver this crate runs on (12.0
            // and newer).
            unsafe { sys::cuLaunchKernelEx(&config, function, params.as_mut_ptr(), std::ptr::null_mut()) }
                .result()
                .map_err(Error::driver("cuLaunchKernelEx"))
        }
    }
}

/// Enqueue `function` on `stream` with the arguments of `prepared` in the order of the ABI of the manifest: a tensor's pointer, a scalar or
/// a fixed argument in its slot, then the scratch pointers a Triton kernel has (null, but the global scratch of a kernel that has one). A kernel of
/// clusters is launched with `cuLaunchKernelEx` ([`launch_prepared`]), its tensors ordered through cudarc's guards as a GEMM's are.
///
/// This is cudarc's launch builder (`CudaStream::launch_builder`, `.arg`, `.launch`): it binds the stream's context to the thread
/// and calls `cuLaunchKernel`, which copies the arguments before it returns. A tensor that is cudarc's own memory is given to the
/// builder as the slice or view itself (or, for a view of a tensor that does not start at the first byte of its slice, as the bytes of
/// the slice from its first element on), which makes cudarc order the launch on `stream` after the slice's earlier writes (and reads,
/// for a write) and record the launch on the slice's `read` or `write` event. A tensor that is a region the host vouched for is given
/// as its address, and nothing tracks it. A scalar is read through the first four (or eight) bytes of its slot (little-endian, which the
/// crate insists on).
///
/// What [`ordered`] adds to cudarc's tracking, which keeps one `read` and one `write` event per slice, is the same for a kernel and a GEMM.
#[allow(unsafe_code)]
pub(crate) fn enqueue(
    function: &CudaFunction,
    stream: &CudaStream,
    prepared: &Prepared,
    block: [u32; 3],
    shared: u32,
    tensors: &mut [TensorArg<'_>],
) -> Result<()> {
    let mapped: Vec<usize> = prepared.maps.iter().map(|m| usize::from(prepared.tensor_of[m.slot])).collect();
    if prepared.maps.iter().any(|m| m.bytes.is_none()) {
        return Err(Error::Launch("a tensor map of the launch was not encoded".to_owned()));
    }
    if prepared.cluster.is_some() {
        return ordered(stream, tensors, |tensors, want_end| {
            // cudarc's tracking of every slice through its guards (the wait for its earlier uses now, the record of this use when the guard is
            // dropped, after the launch), as `enqueue_gemm` does: cudarc's launch builder has no cluster attribute
            let mut guards = Vec::with_capacity(tensors.len());
            for tensor in tensors.iter_mut() {
                match &mut tensor.backing {
                    Backing::Raw => {}
                    Backing::Read(memory) => guards.push(memory.sync_read(stream)),
                    Backing::Write(memory) => guards.push(memory.sync_write(stream)),
                }
            }
            stream.context().bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
            // SAFETY: the function is the loaded kernel `prepared` was checked for (`Kernel::check_with`), the stream a cudarc stream of its
            // context made current above, and the memory the tensors name is borrowed for the call and handed to cudarc's tracking on the stream
            // (the guards), or vouched for by the host (a region): what `enqueue` argues for a launch through the builder.
            unsafe { launch_prepared(function.cu_function(), prepared, block, shared, stream.cu_stream()) }?;
            let end = if want_end {
                Some(
                    stream
                        .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                        .map_err(Error::driver("cuEventRecord"))?,
                )
            } else {
                None
            };
            drop(guards);
            Ok(end)
        });
    }
    ordered(stream, tensors, |tensors, want_end| {
        // what the builder is given for each tensor: made before the builder, which holds pointers into it until the launch
        let mut holders: Vec<Holder<'_>> = Vec::with_capacity(tensors.len());
        for (t, tensor) in tensors.iter_mut().enumerate() {
            let offset = tensor.byte_offset;
            let missing = || Error::Launch("a view starts past the end of its slice".to_owned());
            if mapped.contains(&t) {
                // the tensor of a TMA descriptor: its slot is the map, and its slice is tracked by a guard that lives until after the launch
                // (cudarc's wait for the slice's earlier uses on `stream` now, the record of this use when the guard is dropped)
                holders.push(match &mut tensor.backing {
                    Backing::Raw => Holder::Value,
                    Backing::Read(memory) => Holder::Guard(memory.sync_read(stream)),
                    Backing::Write(memory) => Holder::Guard(memory.sync_write(stream)),
                });
                continue;
            }
            holders.push(match &mut tensor.backing {
                Backing::Raw => Holder::Value,
                Backing::Read(memory) => {
                    if offset == 0 {
                        Holder::Read(*memory)
                    } else {
                        Holder::ReadView(memory.bytes_from(offset).ok_or_else(missing)?)
                    }
                }
                Backing::Write(memory) => {
                    if offset == 0 {
                        Holder::Write(&mut **memory)
                    } else {
                        Holder::WriteView(memory.bytes_from_mut(offset).ok_or_else(missing)?)
                    }
                }
            });
        }
        let mut launch = stream.launch_builder(function);
        if want_end {
            launch.record_kernel_launch(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
        }
        let mut pushed = holders.iter_mut();
        for (slot, value) in prepared.params().iter().enumerate() {
            if prepared.tensor_of[slot] == NOT_A_TENSOR {
                launch.arg(value);
                continue;
            }
            // the tensors are in the order of the slots: `Plan::new` numbers them as it meets them
            let Some(holder) = pushed.next() else {
                return Err(Error::Launch("fewer tensors than the ABI has".to_owned()));
            };
            if let Some(map) = prepared.map_at(slot) {
                // the slot of a TMA descriptor: the 128 bytes of its map, by value (its tensor is held by the guard above)
                let bytes = map.bytes.as_ref().expect("encoded: checked above");
                launch.arg(bytes);
                continue;
            }
            match holder {
                Holder::Value | Holder::Guard(_) => {
                    launch.arg(value);
                }
                Holder::Read(memory) => {
                    let memory: &dyn sealed::Read = *memory;
                    memory.push(&mut launch);
                }
                Holder::Write(memory) => {
                    let memory: &mut dyn sealed::Write = &mut **memory;
                    memory.push(&mut launch);
                }
                Holder::ReadView(view) => {
                    launch.arg(&*view);
                }
                Holder::WriteView(view) => {
                    launch.arg(&mut *view);
                }
            }
        }
        let grid = prepared.grid;
        let config = LaunchConfig {
            grid_dim: (grid[0], grid[1], grid[2]),
            block_dim: (block[0], block[1], block[2]),
            shared_mem_bytes: shared,
        };
        // SAFETY: cudarc asks the caller to vouch for the arguments and for what the kernel does with them. The arguments are the slots of
        // a `Prepared`, made by `Plan::prepare` from the manifest's ABI (a pointer, a scalar or a fixed value in the width of its type,
        // and the null scratch pointers a Triton kernel has, in the order of its parameters), and the cubin is the one whose digests the
        // manifest pinned, whose loaded function was checked to take the parameter sizes of the ABI. The memory the pointers name is
        // what the tensors borrow: a cudarc slice that the tensor holds the borrow of, for the whole call, and that this launch hands to
        // cudarc on `stream` (so it is ordered and recorded there, and the slice's own stream is made to wait for the kernel below), or
        // a region whose lifetime and contents the host vouched for (`DevicePtr::from_raw`); the tensors were checked to fit their memory
        // (a view: its extent from its first element), not to overlap where one is written, to be in this kernel's context and
        // device, to be writable where the kernel writes, and to be aligned as the binary was compiled; the grid was checked against
        // the device's limits and the shared memory against its opt-in maximum when the kernel was loaded.
        let launched = unsafe { launch.launch(config) }.map_err(Error::driver("cuLaunchKernel"));
        launched.map(|events| events.map(|(_, end)| end))
    })
}

/// Order a use of `tensors` on `stream` as the crate promises, around `work`, which enqueues the work itself (a kernel, a GEMM) and returns
/// the event it recorded right after it when it is asked to (`want_end`: some slice's owning stream is not `stream`).
///
/// * a launch of slices is refused while `stream` (or a stream a slice is freed on) is capturing a CUDA graph, whatever mode cudarc's
///   context is in: a graph would hold the slices' addresses with nothing to keep them alive ([`refuse_capture`]);
/// * before the work, for a slice that is written and whose owning stream (the one it is freed on) is not `stream`, `stream` waits for
///   everything enqueued on the owning stream so far. The owning stream has waited for every earlier launch that used the slice from
///   another stream (next item), so the writer is ordered after all earlier readers and writers, not only the last one that cudarc
///   recorded;
/// * after the work, for every slice whose owning stream is not `stream`, the owning stream waits for the event recorded right after
///   it (timing disabled): whatever is enqueued on the owning stream afterwards (the free, a use) runs after the work, however many
///   other launches read the slice. Nothing is waited for when every slice is on `stream` (stream order already says it);
/// * an error after the work may have been enqueued does not leave a slice's owner unordered ([`settle`]).
pub(crate) fn ordered(
    stream: &CudaStream,
    tensors: &mut [TensorArg<'_>],
    work: impl FnOnce(&mut [TensorArg<'_>], bool) -> Result<Option<CudaEvent>>,
) -> Result<()> {
    refuse_per_thread(stream, tensors.iter())?;
    refuse_capture(stream, tensors)?;
    let across = |t: &TensorArg<'_>| t.owner.as_ref().is_some_and(|owner| owner.cu_stream() != stream.cu_stream());
    let owners = distinct_owners(tensors.iter().filter(|t| across(t)));
    // before the work: a writer waits for what is on the owning stream (see above); nothing is enqueued for it yet, so an error here
    // is a launch that did not happen
    for owner in distinct_owners(tensors.iter().filter(|t| across(t) && matches!(t.backing, Backing::Write(_)))) {
        let behind = owner.record_event(None).map_err(Error::driver("cuEventRecord"))?;
        stream.wait(&behind).map_err(Error::driver("cuStreamWaitEvent"))?;
    }
    let launched = work(tensors, !owners.is_empty());
    settle(
        launched,
        &owners,
        |owner, end| owner.wait(end).map_err(Error::driver("cuStreamWaitEvent")),
        || {
            // best effort: if this fails too, there is nothing more to do
            let _ = stream.synchronize();
        },
    )
}

/// The owners of `tensors` (the streams their memory is freed on), each once.
fn distinct_owners<'t, 'a: 't>(tensors: impl Iterator<Item = &'t TensorArg<'a>>) -> Vec<Arc<CudaStream>> {
    let mut owners: Vec<Arc<CudaStream>> = Vec::new();
    for owner in tensors.filter_map(|t| t.owner.as_ref()) {
        if !owners.iter().any(|seen| seen.cu_stream() == owner.cu_stream()) {
            owners.push(Arc::clone(owner));
        }
    }
    owners
}

/// What follows the work of a launch, whose result is `launched` (the event recorded after it, when it was asked for): each of
/// `owners`, the streams that slices of the launch are freed on and that are not the launch stream, waits for that event.
///
/// A call of cudarc's `launch` that returns an error may have enqueued the kernel already (the error can be that of the event
/// recorded after it, or of the records on the slices), and then there is no event to wait for; and the wait of one owner may fail
/// while another's does not. In either case the kernel may be running with nothing ordering the free of a slice behind it, so every
/// owner is still asked to wait, and then `synchronize` blocks the host until the launch stream has finished everything it has,
/// the kernel included, before the first error is returned: whatever the host enqueues on an owner after that is ordered after it.
/// With no `owners` (every slice is on the launch stream) stream order says it and the result is `launched`.
///
/// An `Err` from a launch is therefore not a promise that nothing was launched: a precondition that fails (an `Error::Launch` or a
/// precondition error) enqueues nothing, a driver error may have.
fn settle<Owner, Event>(
    launched: Result<Option<Event>>,
    owners: &[Owner],
    mut wait: impl FnMut(&Owner, &Event) -> Result<()>,
    synchronize: impl FnOnce(),
) -> Result<()> {
    if owners.is_empty() {
        return launched.map(drop);
    }
    let mut first_error = None;
    match launched {
        Ok(Some(end)) => {
            for owner in owners {
                if let Err(error) = wait(owner, &end) {
                    first_error.get_or_insert(error);
                }
            }
        }
        Ok(None) => {
            first_error = Some(Error::Launch("the driver gave no event for the end of the launch".to_owned()));
        }
        Err(error) => first_error = Some(error),
    }
    match first_error {
        None => Ok(()),
        Some(error) => {
            synchronize();
            Err(error)
        }
    }
}

/// The handle of the per-thread default stream (`CU_STREAM_PER_THREAD`, what cudarc's `CudaContext::per_thread_stream` wraps).
const PER_THREAD_STREAM: usize = 0x2;

/// Whether a stream handle is the per-thread default stream's: the same handle in every thread, naming another stream in each.
pub(crate) fn is_per_thread(handle: sys::CUstream) -> bool {
    handle as usize == PER_THREAD_STREAM
}

/// Refuse the per-thread default stream (`ctx.per_thread_stream()`) as the stream of a safe launch, and as the stream a slice of one of
/// `tensors` is owned by (allocated and freed on).
///
/// Its handle, `CU_STREAM_PER_THREAD`, is the same in every thread and names that thread's own stream: two of them compare equal, so a slice
/// owned by one thread's per-thread stream and launched on another thread's would count as being on the launch stream, and neither the wait
/// of the owning stream for the end of the kernel nor the wait of a writer for the owning stream would be made (the slice could be freed, and
/// its memory reused, under the kernel); and a wait enqueued on the handle from another thread than the owner's waits on that other thread's
/// stream. The owning stream of a slice cannot be named from another thread, so the ordering this crate promises cannot be made: the launch
/// is refused. The legacy default stream (`ctx.default_stream()`, one stream per context) and the streams of `ctx.new_stream()` are fine.
/// (`unsafe` `launch_raw` takes any handle, under its contract.)
pub(crate) fn refuse_per_thread<'t, 'a: 't>(
    stream: &CudaStream,
    tensors: impl Iterator<Item = &'t TensorArg<'a>>,
) -> Result<()> {
    if is_per_thread(stream.cu_stream()) {
        return Err(Error::Launch(format!(
            "the launch stream is {PER_THREAD}, which is refused: {PER_THREAD_WHY}. Launch on a stream of `ctx.new_stream()`, or on \
             `ctx.default_stream()`"
        )));
    }
    for t in tensors {
        if t.owner.as_ref().is_some_and(|owner| is_per_thread(owner.cu_stream())) {
            return Err(Error::Launch(format!(
                "a tensor is made of a slice allocated on {PER_THREAD}, which is refused: {PER_THREAD_WHY}. Allocate it on a stream of \
                 `ctx.new_stream()`, or on `ctx.default_stream()`"
            )));
        }
    }
    Ok(())
}

const PER_THREAD: &str = "the per-thread default stream (`ctx.per_thread_stream()`, the handle `CU_STREAM_PER_THREAD`)";
const PER_THREAD_WHY: &str = "its handle is the same in every thread and names another stream in each, so the launch cannot be \
    ordered against the uses (and the free) of a slice from other threads";

/// Refuse a launch of tensors made of cudarc slices while `stream`, or a stream a slice is freed on, is capturing a CUDA graph.
///
/// A graph holds the addresses of the memory its kernels use, and nothing that keeps that memory alive: `CudaGraph::launch` is safe
/// and runs the kernel again with those addresses wherever the graph is launched, after the slices are freed and their memory is
/// used by something else. Whether a launch of slices is refused during a capture is not something cudarc's tracking decides: it
/// waits on events when its context has more than one stream and tracks events, and then the driver refuses the launch on a capturing
/// stream (`CUDA_ERROR_STREAM_CAPTURE_ISOLATION`), but a context that never made a stream of its own (it uses
/// `default_stream`, or a stream of another library) has no such events, and the kernel would be captured. So the capture is looked for here, in every mode. Regions
/// the host vouched for (`DevicePtr::from_raw`) are not looked at: capture is what they are for, and their contract says that the
/// graph is not replayed after the memory is gone.
///
/// The check is made when the call is: a capture begun on `stream` by another thread while this call runs is not excluded by this
/// crate (CUDA does not define the order of a capture and a launch on one stream from two threads either).
fn refuse_capture(stream: &CudaStream, tensors: &[TensorArg<'_>]) -> Result<()> {
    if tensors.iter().all(|t| matches!(t.backing, Backing::Raw)) {
        return Ok(());
    }
    let owners = distinct_owners(tensors.iter());
    for candidate in std::iter::once(stream).chain(owners.iter().map(|owner| &**owner)) {
        let status = candidate.capture_status().map_err(Error::driver("cuStreamIsCapturing"))?;
        if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(Error::Launch(CAPTURE_REFUSED.to_owned()));
        }
    }
    Ok(())
}

/// The message of a launch of slices that is refused because a stream is capturing.
const CAPTURE_REFUSED: &str = "a tensor is made of a cudarc slice and the stream is capturing a CUDA graph: a graph would hold the \
    slice's address and nothing that keeps the slice alive; capture tensors made of regions (`unsafe DevicePtr::from_raw`), \
    with `launch_raw`";

/// A context made current on this thread for as long as the guard lives, with what was current before put back when it is
/// dropped (on an error as well).
struct Current {
    changed: bool,
    previous: Option<sys::CUcontext>,
}

impl Current {
    fn make(context: &CudaContext) -> Result<Self> {
        let previous = result::ctx::get_current().map_err(Error::driver("cuCtxGetCurrent"))?;
        if previous == Some(context.cu_ctx()) {
            return Ok(Self { changed: false, previous });
        }
        context.bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
        Ok(Self { changed: true, previous })
    }
}

impl Drop for Current {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if self.changed {
            // SAFETY: `previous` was this thread's current context when the guard was made (a null pointer: there was none), and
            // setting it again is what the thread had. A failure is ignored: there is nothing to do about it in a destructor.
            let _ = unsafe { result::ctx::set_current(self.previous.unwrap_or(std::ptr::null_mut())) };
        }
    }
}

/// Refuse a launch of tensors made of cudarc slices on the raw stream `stream` while it is capturing.
#[allow(unsafe_code)]
fn refuse_capture_raw(stream: sys::CUstream, tensors: &[TensorArg<'_>]) -> Result<()> {
    if tensors.iter().any(|t| !matches!(t.backing, Backing::Raw)) {
        // SAFETY: `stream` is a valid `CUstream`, as the contract of the `launch_raw` that calls this says; the call only reads its
        // capture status.
        let status = unsafe { result::stream::is_capturing(stream) }.map_err(Error::driver("cuStreamIsCapturing"))?;
        if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(Error::Launch(CAPTURE_REFUSED.to_owned()));
        }
    }
    Ok(())
}

impl Kernel {
    /// Check the call and launch the kernel on a raw `CUstream`.
    ///
    /// What [`Kernel::launch`] does for a stream that cudarc owns, for a handle that nothing owns: another library's stream, or the
    /// stream that a CUDA graph is being captured on (the call is captured like any kernel node). The kernel's context is made
    /// current on this thread for the call, and what was current is put back afterwards (also when the launch fails).
    ///
    /// **This is how a CUDA graph is captured**, with tensors made of regions ([`DevicePtr::from_raw`](crate::DevicePtr::from_raw)):
    /// a graph holds the addresses it captured and nothing that keeps the memory alive, so it is the host's (see the contract of
    /// `from_raw`) that the graph is not replayed after the memory is gone.
    ///
    /// A tensor made of a cudarc slice is used by its address here: cudarc's event tracking is for its own streams, so nothing
    /// orders this launch against other streams' use of the memory, or a later free of it against this launch. That is the caller's,
    /// as it is for a region. A launch with such a tensor on a stream that is capturing is refused (an `Error::Launch`), as in
    /// [`Kernel::launch`]: a captured launch of a slice would outlive the borrow that keeps the slice alive.
    ///
    /// # Safety
    ///
    /// * `stream` must be a valid `CUstream` of the CUDA context the kernel was loaded in ([`Kernel::context`]): a handle from another
    ///   context, a destroyed stream or a pointer that is not a stream is undefined behavior in the driver, and it must stay valid
    ///   until this returns;
    /// * the tensors' memory must stay allocated until the launch has completed on `stream`, which this call does not wait for, and
    ///   it must be ordered against the other streams that use it (nothing here is).
    #[allow(unsafe_code)]
    pub unsafe fn launch_raw(&self, stream: sys::CUstream, tensors: &[TensorArg<'_>]) -> Result<()> {
        // SAFETY: the caller's contract is this function's.
        unsafe { self.launch_raw_with(stream, tensors, &[]) }
    }

    /// [`Kernel::launch_raw`] of a call that gives the values of the binary's fixed arguments per call (see [`Kernel::launch_with`]).
    ///
    /// # Safety
    ///
    /// As for [`Kernel::launch_raw`].
    #[allow(unsafe_code)]
    pub unsafe fn launch_raw_with(
        &self,
        stream: sys::CUstream,
        tensors: &[TensorArg<'_>],
        given: &[(&str, i64)],
    ) -> Result<()> {
        let prepared = self.check_with(tensors, given)?;
        let _current = Current::make(self.context())?;
        refuse_capture_raw(stream, tensors)?;
        // SAFETY: the kernel's function and the `Prepared` it checked (the slots, at the slot of a TMA descriptor the 128 bytes of its encoded map,
        // the grid of blocks and, for a kernel of clusters, its cluster); its context is current (`_current`); the stream and the memory are the
        // caller's, as the contract above says.
        unsafe {
            launch_prepared(self.function().cu_function(), &prepared, self.plan().block(), self.plan().shared(), stream)
        }
    }
}

/// Enqueue the pinned GEMM of a checked call on `stream`, ordered as a kernel launch is ([`ordered`]): the same wait for the earlier writes
/// of a slice (and reads, for a write) on `stream` and record of the use on it that cudarc's launch builder does for a kernel, here through
/// the guards cudarc's `device_ptr` and `device_ptr_mut` give, which stay alive until the work is enqueued and the end event is recorded.
#[allow(unsafe_code)]
pub(crate) fn enqueue_gemm(gemm: &Gemm, stream: &CudaStream, call: &Call, tensors: &mut [TensorArg<'_>]) -> Result<()> {
    ordered(stream, tensors, |tensors, want_end| {
        let mut guards = Vec::with_capacity(tensors.len());
        for tensor in tensors.iter_mut() {
            match &mut tensor.backing {
                Backing::Raw => {}
                Backing::Read(memory) => guards.push(memory.sync_read(stream)),
                Backing::Write(memory) => guards.push(memory.sync_write(stream)),
            }
        }
        stream.context().bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
        // SAFETY: the stream is the cudarc stream of this GEMM's context, which is current on this thread; the addresses in `call` are those of
        // tensors that `Gemm::check_call` checked to fit their memory and the shapes of the problem, to be bf16 (the workspace bytes) and
        // aligned, not to overlap where one is written, and to be in this context; the slices among them are borrowed by the tensors for the
        // whole call and handed to cudarc's tracking (the guards) on this stream, and a region's memory is the host's, as its contract says.
        unsafe { gemm.run(stream.cu_stream(), call) }?;
        let end = if want_end {
            Some(
                stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                    .map_err(Error::driver("cuEventRecord"))?,
            )
        } else {
            None
        };
        drop(guards);
        Ok(end)
    })
}

/// Enqueue the pinned convolution of a checked call on `stream`, ordered as a GEMM is ([`enqueue_gemm`]).
#[allow(unsafe_code)]
pub(crate) fn enqueue_conv(
    conv: &Conv,
    stream: &CudaStream,
    call: &ConvCall,
    tensors: &mut [TensorArg<'_>],
) -> Result<()> {
    ordered(stream, tensors, |tensors, want_end| {
        let mut guards = Vec::with_capacity(tensors.len());
        for tensor in tensors.iter_mut() {
            match &mut tensor.backing {
                Backing::Raw => {}
                Backing::Read(memory) => guards.push(memory.sync_read(stream)),
                Backing::Write(memory) => guards.push(memory.sync_write(stream)),
            }
        }
        stream.context().bind_to_thread().map_err(Error::driver("cuCtxSetCurrent"))?;
        // SAFETY: as `enqueue_gemm`: the stream is the cudarc stream of this convolution's context, current on this thread; the addresses in
        // `call` are those of tensors `Conv::check_call` checked (shapes of the problem, dtype, alignment, the workspace, no overlap where one is
        // written, this context), borrowed for the whole call and handed to cudarc's tracking on this stream.
        unsafe { conv.run(stream.cu_stream(), call) }?;
        let end = if want_end {
            Some(
                stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                    .map_err(Error::driver("cuEventRecord"))?,
            )
        } else {
            None
        };
        drop(guards);
        Ok(end)
    })
}

impl Conv {
    /// Check the call and run the convolution on a raw `CUstream`: [`Conv::launch`] for a handle that nothing owns, or the stream a CUDA graph is
    /// being captured on (the call is captured like any other; the workspace is the caller's, so nothing is allocated).
    ///
    /// **This is how a convolution is captured in a CUDA graph**, with tensors made of regions: the contract is that of [`Gemm::launch_raw`].
    ///
    /// # Safety
    ///
    /// As [`Gemm::launch_raw`]: `stream` must be a valid `CUstream` of the CUDA context the convolution was loaded in ([`Conv::context`]) until
    /// this returns, and the tensors' memory (the workspace included) must stay allocated until the work has completed on `stream`, and be ordered
    /// against the other streams that use it.
    #[allow(unsafe_code)]
    pub unsafe fn launch_raw(&self, stream: sys::CUstream, tensors: &[TensorArg<'_>]) -> Result<()> {
        let call = self.check_call(tensors)?;
        let _current = Current::make(self.context())?;
        refuse_capture_raw(stream, tensors)?;
        // SAFETY: `stream` is the caller's valid stream of this convolution's context, as the contract above says; the pointers are the addresses
        // of tensors that were checked to fit their memory and the shapes of the problem, which the caller keeps allocated.
        unsafe { self.run(stream, &call) }
    }
}

impl Gemm {
    /// Run the pinned algorithm of a checked call on `stream`: the descriptors of the call are made here and destroyed at its end.
    ///
    /// # Safety
    ///
    /// `stream` must be a valid `CUstream` of this GEMM's context, which is current on this thread, and the addresses of `call` must be
    /// device memory of that context holding what the problem says, allocated until the work has completed on `stream`: what
    /// `Gemm::check_call` checks of the tensors they came from.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn run(&self, stream: sys::CUstream, call: &Call) -> Result<()> {
        let engine = self.engine();
        let [m, n, k] = engine.sizes;
        let problem =
            cublaslt::Problem::new(m, n, k, engine.epilogue, call.bias, &engine.form, engine.sm_count_target)?;
        let addresses =
            cublaslt::Addresses { a: call.a, b: call.b, c: call.c, out: call.out, workspace: call.workspace };
        // SAFETY: the contract of this function.
        unsafe { cublaslt::matmul(engine.handle, &problem, engine.algo, &addresses, engine.required, stream) }
    }

    /// Check the call and run the GEMM on a raw `CUstream`: [`Gemm::launch`] for a handle that nothing owns, or the stream a CUDA graph is
    /// being captured on (the call is captured like any other; the workspace is the caller's, so nothing is allocated).
    ///
    /// `tensors` are `a`, `b`, `bias` (when it has one), `c` (when it has one), `out` and the workspace, as for [`Gemm::launch`]. The GEMM's context is made
    /// current on this thread for the call, and what was current is put back afterwards (also when it fails).
    ///
    /// **This is how a GEMM is captured in a CUDA graph**, with tensors made of regions ([`DevicePtr::from_raw`](crate::DevicePtr::from_raw)):
    /// the contract is that of [`Kernel::launch_raw`], and the workspace must stay allocated, and be used by nothing else while the graph
    /// runs, for as long as the graph can be replayed. A tensor made of a cudarc slice is used by its address, and refused on a stream
    /// that is capturing, as there.
    ///
    /// # Safety
    ///
    /// As [`Kernel::launch_raw`]: `stream` must be a valid `CUstream` of the CUDA context the GEMM was loaded in ([`Gemm::context`]) until
    /// this returns, and the tensors' memory (the workspace included) must stay allocated until the work has completed on `stream`, and
    /// be ordered against the other streams that use it.
    #[allow(unsafe_code)]
    pub unsafe fn launch_raw(&self, stream: sys::CUstream, tensors: &[TensorArg<'_>]) -> Result<()> {
        let addresses = self.check_call(tensors)?;
        let _current = Current::make(self.context())?;
        refuse_capture_raw(stream, tensors)?;
        // SAFETY: `stream` is the caller's valid stream of this GEMM's context, as the contract above says; the pointers are the addresses
        // of tensors that were checked to fit their memory and the shapes of the problem, which the caller keeps allocated.
        unsafe { self.run(stream, &addresses) }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;

    fn boom(what: &str) -> Error {
        Error::Launch(what.to_owned())
    }

    /// `settle` with owners numbered 1, 2, 3, of which `failing` fail to wait: what they waited for, how often the host was
    /// synchronized, and the result.
    fn run(
        launched: Result<Option<&'static str>>,
        owners: &[u32],
        failing: &[u32],
    ) -> (Vec<(u32, &'static str)>, u32, Result<()>) {
        let waited = RefCell::new(Vec::new());
        let synchronized = Cell::new(0);
        let result = settle(
            launched,
            owners,
            |owner, end| {
                waited.borrow_mut().push((*owner, *end));
                if failing.contains(owner) { Err(boom(&format!("wait {owner}"))) } else { Ok(()) }
            },
            || synchronized.set(synchronized.get() + 1),
        );
        (waited.into_inner(), synchronized.get(), result)
    }

    #[test]
    fn every_owner_waits_for_the_end_of_the_kernel_and_nothing_else_is_done() {
        let (waited, synchronized, result) = run(Ok(Some("end")), &[1, 2, 3], &[]);
        assert_eq!(waited, [(1, "end"), (2, "end"), (3, "end")]);
        assert_eq!(synchronized, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn a_failing_owner_does_not_stop_the_others_and_the_host_waits_for_the_stream() {
        let (waited, synchronized, result) = run(Ok(Some("end")), &[1, 2, 3], &[1, 3]);
        assert_eq!(waited, [(1, "end"), (2, "end"), (3, "end")], "every owner was asked, although the first failed");
        assert_eq!(synchronized, 1, "the launch stream is synchronized once, so that no free is left unordered");
        assert_eq!(result.unwrap_err().to_string(), boom("wait 1").to_string(), "the first error is the one returned");
    }

    #[test]
    fn a_launch_that_failed_after_the_kernel_may_have_been_enqueued_has_no_event_and_synchronizes() {
        let (waited, synchronized, result) = run(Err(boom("end event")), &[1, 2], &[]);
        assert!(waited.is_empty(), "there is no event to wait for");
        assert_eq!(synchronized, 1);
        assert_eq!(result.unwrap_err().to_string(), boom("end event").to_string());
    }

    #[test]
    fn a_launch_that_gave_no_event_is_an_error_and_synchronizes() {
        let (waited, synchronized, result) = run(Ok(None), &[1], &[]);
        assert!(waited.is_empty());
        assert_eq!(synchronized, 1);
        assert!(result.is_err());
    }

    #[test]
    fn with_every_slice_on_the_launch_stream_the_result_is_the_launch_and_nothing_is_synchronized() {
        let (waited, synchronized, result) = run(Ok(None), &[], &[]);
        assert!(waited.is_empty() && result.is_ok());
        assert_eq!(synchronized, 0);
        let (_, synchronized, result) = run(Err(boom("launch")), &[], &[]);
        assert_eq!(synchronized, 0, "stream order orders the free after the kernel, enqueued or not");
        assert_eq!(result.unwrap_err().to_string(), boom("launch").to_string());
    }

    #[test]
    fn a_driver_older_than_12_4_is_refused_before_a_function_of_12_4_is_called() {
        for old in [11080, 12000, 12030] {
            let e = require_driver(old).unwrap_err();
            assert!(matches!(&e, Error::Unsupported(m) if m.contains("needs 12.4 or newer")), "{e}");
        }
        assert!(require_driver(12030).unwrap_err().to_string().contains("is 12.3"));
        for new in [12040, 12080, 13000, 13020] {
            assert!(require_driver(new).is_ok());
        }
    }

    #[test]
    fn only_the_per_thread_default_stream_is_the_per_thread_handle() {
        // cudarc's `per_thread_stream` is the handle 0x2 in every thread; the legacy default stream is null (and `CU_STREAM_LEGACY`, 0x1, is
        // one stream per context too); a stream of `new_stream` is a pointer the driver made
        assert!(is_per_thread(0x2 as sys::CUstream));
        assert!(!is_per_thread(std::ptr::null_mut()));
        assert!(!is_per_thread(0x1 as sys::CUstream));
        assert!(!is_per_thread(0x7f00_dead_beef_usize as sys::CUstream));
    }
}
