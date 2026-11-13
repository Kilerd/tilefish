//! Synthetic manifests: entries of every kind of schema 3, around a small ELF file, for the tests that need no GPU and no fixture.
#![allow(dead_code)]

use serde_json::Value;
pub use serde_json::json;
use tilefish::{Error, Manifest, Pin};

/// A small ELF64 file shaped like a cubin: a null section, `sections` (`(name, type, flags, contents)`) and a `.shstrtab`. `osabi` and `flags`
/// say the architecture the way ptxas writes it (`osabi` 0x41 and the SM in the second byte of `flags` from sm_100 on).
pub fn elf(sections: &[(&str, u32, u64, &[u8])], osabi: u8, flags: u32) -> Vec<u8> {
    let mut names = vec![0u8];
    let mut name_at = Vec::new();
    for (name, ..) in sections {
        name_at.push(names.len() as u32);
        names.extend_from_slice(name.as_bytes());
        names.push(0);
    }
    let shstr_at = names.len() as u32;
    names.extend_from_slice(b".shstrtab\0");
    let mut body = Vec::new();
    let mut offsets = Vec::new();
    for (.., data) in sections {
        offsets.push(64 + body.len() as u64);
        body.extend_from_slice(data);
    }
    let names_offset = 64 + body.len() as u64;
    body.extend_from_slice(&names);
    let shoff = 64 + body.len() as u64;
    let mut headers = vec![0u8; 64]; // the null section
    let mut header = |name: u32, ty: u32, flags: u64, offset: u64, size: u64, align: u64| {
        headers.extend_from_slice(&name.to_le_bytes());
        headers.extend_from_slice(&ty.to_le_bytes());
        headers.extend_from_slice(&flags.to_le_bytes());
        headers.extend_from_slice(&0u64.to_le_bytes());
        headers.extend_from_slice(&offset.to_le_bytes());
        headers.extend_from_slice(&size.to_le_bytes());
        headers.extend_from_slice(&0u32.to_le_bytes());
        headers.extend_from_slice(&0u32.to_le_bytes());
        headers.extend_from_slice(&align.to_le_bytes());
        headers.extend_from_slice(&0u64.to_le_bytes());
    };
    for (i, (_, ty, flags, data)) in sections.iter().enumerate() {
        header(name_at[i], *ty, *flags, offsets[i], data.len() as u64, 1);
    }
    header(shstr_at, 3, 0, names_offset, names.len() as u64, 1);
    let mut out = Vec::new();
    out.extend_from_slice(&[0x7F, b'E', b'L', b'F', 2, 1, 1, osabi, 8]);
    out.extend_from_slice(&[0u8; 7]);
    out.extend_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    out.extend_from_slice(&190u16.to_le_bytes()); // EM_CUDA
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&shoff.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&64u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&64u16.to_le_bytes());
    out.extend_from_slice(&(sections.len() as u16 + 2).to_le_bytes());
    out.extend_from_slice(&(sections.len() as u16 + 1).to_le_bytes());
    out.extend_from_slice(&body);
    out.extend_from_slice(&headers);
    out
}

/// A cubin for `sm` (103 for sm_103) with the code section of `symbol`.
pub fn cubin(symbol: &str, sm: u32) -> Vec<u8> {
    let text = format!(".text.{symbol}");
    elf(&[(&text, 1, 6, &[1, 2, 3, 4, 5, 6, 7, 8])], 0x41, (sm << 8) | 0x0A)
}

/// A pointer parameter of the ABI. `view` is `(strides, offset)` of a strided pointer.
pub fn pointer(slot: usize, name: &str, dtype: &str, access: &str, shape: Value, view: Option<(Value, &str)>) -> Value {
    let ty = match dtype {
        "bf16" => "*bf16",
        "f32" => "*fp32",
        "i32" => "*i32",
        "u8" => "*u8",
        other => panic!("no pointer type for {other}"),
    };
    let mut p = json!({"slot": slot, "index": slot, "name": name, "kind": "pointer", "type": ty, "ptx_type": "u64", "size": 8,
                       "dtype": dtype, "access": access, "shape": shape});
    if let Some((strides, offset)) = view {
        p["strides"] = strides;
        p["offset"] = offset.into();
    }
    p
}

/// A runtime scalar parameter (`ty` is `i32` or `i64`); `extra` is merged into it (`count_of`, `stride_of`).
pub fn scalar(slot: usize, name: &str, ty: &str, dim: &str, extra: Value) -> Value {
    let (ptx, size) = if ty == "i64" { ("u64", 8) } else { ("u32", 4) };
    let mut p = json!({"slot": slot, "index": slot, "name": name, "kind": "scalar", "type": ty, "ptx_type": ptx, "size": size, "dim": dim});
    for (k, v) in extra.as_object().into_iter().flatten() {
        p[k] = v.clone();
    }
    p
}

/// The scalar that carries one stride of a strided pointer.
pub fn stride_scalar(slot: usize, name: &str, ty: &str, arg: &str, axis: usize) -> Value {
    scalar(slot, name, ty, name, json!({"stride_of": {"arg": arg, "axis": axis}}))
}

/// A fixed argument of a prebuilt binary.
pub fn fixed(slot: usize, name: &str, ty: &str, value: Value) -> Value {
    let (ptx, size) = match ty {
        "i64" => ("u64", 8),
        "f32" => ("f32", 4),
        _ => ("u32", 4),
    };
    json!({"slot": slot, "index": slot, "name": name, "kind": "fixed", "type": ty, "ptx_type": ptx, "size": size, "value": value})
}

/// A fixed argument per call of a prebuilt binary: no value, every launch gives it.
#[allow(dead_code)]
pub fn given(slot: usize, name: &str, ty: &str) -> Value {
    let (ptx, size) = if ty == "i64" { ("u64", 8) } else { ("u32", 4) };
    json!({"slot": slot, "index": slot, "name": name, "kind": "given", "type": ty, "ptx_type": ptx, "size": size})
}

pub fn alignment(index: usize, arg: &str, bytes: u64) -> Value {
    json!({"kind": "pointer_alignment", "arg": arg, "index": index, "bytes": bytes})
}

pub fn max_extent(index: usize, arg: &str) -> Value {
    json!({"kind": "max_extent", "arg": arg, "index": index, "value": 2147483647})
}

/// One kernel entry of a synthetic manifest and the cubin it names. `kind` is `"triton"` or `"binary"`.
pub struct Kernel {
    pub entry: Value,
    pub cubin: Vec<u8>,
}

impl Kernel {
    /// The entry of `stem` for `sm_<sm>` with these ABI parameters, preconditions and table of dimensions; `grid` is the grid expression.
    pub fn new(
        kind: &str,
        stem: &str,
        sm: u32,
        params: Vec<Value>,
        preconditions: Vec<Value>,
        dims: Value,
        grid: &str,
    ) -> Self {
        let symbol = format!("tf_{}", stem.replace('.', "_"));
        let cubin = cubin(&symbol, sm);
        let sha = tilefish::elf::sha256_hex(&cubin);
        let code = tilefish::elf::Elf::parse(&cubin).unwrap().code_digest();
        let triton = kind == "triton";
        let n = params.len();
        let scratch = if triton {
            json!([{"slot": n, "name": "global_scratch", "type": "u64", "ptx_type": "u64", "size": 8, "value": 0},
                   {"slot": n + 1, "name": "profile_scratch", "type": "u64", "ptx_type": "u64", "size": 8, "value": 0}])
        } else {
            json!([])
        };
        let mut entry = json!({
            "stem": stem, "op": stem, "kind": kind, "arch": format!("sm_{sm}"), "ptx_target": if triton { format!("sm_{sm}a") } else { String::new() },
            "symbol": symbol, "cubin": format!("{stem}.sm_{sm}.cubin"),
            "abi": {"params": params, "scratch": scratch, "scratch_slots": if triton { 2 } else { 0 }},
            "grid": grid, "block": if triton { json!([128, 1, 1]) } else { json!([64, 1, 1]) }, "shared": 0,
            "launch": {"global_scratch_size": 0, "profile_scratch_size": 0, "launch_pdl": false, "cluster_dims": [1, 1, 1], "num_ctas": 1, "cooperative": false},
            "constexprs": [], "buckets": {}, "dims": dims, "preconditions": preconditions,
            "declaration_digest": "d".repeat(64), "sha256": sha, "sha256_code": code,
        });
        if triton {
            entry["num_warps"] = 4.into();
            entry["num_stages"] = 3.into();
            entry["options"] = json!({"num_warps": 4, "num_stages": 3});
            entry["toolchain"] = json!({"triton": "3.5.1", "ptxas": "ptxas 12.9"});
            entry["source_sha256"] = "5".repeat(64).into();
            entry["source_digest"] = "6".repeat(64).into();
            entry["source_digest_kind"] = "tilefish.triton-ir.v1".into();
        } else {
            entry["binary"] =
                json!({"source": "ptx", "ptxas_flags": ["-lineinfo", "-v", format!("--gpu-name=sm_{sm}a")]});
            entry["toolchain"] = json!({"ptxas": "ptxas 12.9"});
            entry["source_digest"] = "7".repeat(64).into();
            entry["source_digest_kind"] = "tilefish.binary.v1".into();
        }
        Self { entry, cubin }
    }

    /// The same entry with its cubin replaced (the digests of the entry follow it).
    pub fn with_cubin(mut self, cubin: Vec<u8>) -> Self {
        self.entry["sha256"] = tilefish::elf::sha256_hex(&cubin).into();
        self.entry["sha256_code"] = tilefish::elf::Elf::parse(&cubin).unwrap().code_digest().into();
        self.cubin = cubin;
        self
    }
}

/// The text of a manifest of these kernels and library entries.
pub fn manifest_text(kernels: &[Kernel], library: &[Value]) -> String {
    let doc = json!({"schema": 3, "tilefish": "0.0.3", "kernels": kernels.iter().map(|k| k.entry.clone()).collect::<Vec<_>>(), "library": library});
    serde_json::to_string_pretty(&doc).unwrap()
}

/// Read a manifest of these kernels (and library entries) from memory, as `from_embedded` does, trusting it.
pub fn read(kernels: &[Kernel], library: &[Value]) -> Result<Manifest, Error> {
    let text = manifest_text(kernels, library);
    let files: Vec<(String, &[u8])> =
        kernels.iter().map(|k| (k.entry["cubin"].as_str().unwrap().to_owned(), k.cubin.as_slice())).collect();
    let files: Vec<(&str, &[u8])> = files.iter().map(|(n, b)| (n.as_str(), *b)).collect();
    Manifest::from_embedded(text.as_bytes(), &files, Pin::TrustDirectory)
}

/// The message of a manifest that is refused.
pub fn refusal(kernels: &[Kernel], library: &[Value]) -> String {
    match read(kernels, library) {
        Err(Error::Manifest(message)) => message,
        Err(other) => panic!("expected a manifest error, got {other:?}"),
        Ok(_) => panic!("the manifest was accepted"),
    }
}
