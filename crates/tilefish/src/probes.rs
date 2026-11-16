//! The records of an **instrumented** binary (`docs/instrument.md`): what its probes are, the probe buffer a launch gives it, and the records read
//! back from that buffer, as `tilefish.probe-trace.v1` that `tilefish-explain DIR --probes FILE` renders.
//!
//! An instrumented export (Python's `tilefish.runtime.export(..., instrument=True)`) is a diagnostic variant: its Triton binaries record, at every
//! `tilefish.probe("name")` of their kernels and at three probes of the build's own (`@begin`, `@calib` twice, `@end`), a record of 16 bytes into a
//! **probe buffer** the host gives each launch in the kernel's *profile scratch* parameter (the second of a Triton kernel's two scratch slots, null
//! in every production launch, so the ABI is the production one). Its entries have `instrument` ([`Instrument`]: the record format, the records each
//! warp keeps, the kernel's probes and where each is written), and [`Manifest::load`](crate::Manifest::load) refuses it: it is loaded with
//! [`Manifest::load_instrumented`](crate::Manifest::load_instrumented) only.
//!
//! **The buffer of a launch** has a slot per warp of every CTA of the CUDA grid launched (a cluster's CTAs counted, a persistent grid as launched):
//! the slot of warp `w` of the CTA of linear index `c` (`x + y * nx + z * nx * ny`) is at `(c * warps + w) * (16 + 16 * capacity)`, where `warps`
//! is the block's (`ceil(block.x / 32)`). A slot is a header of 16 bytes (`u32` magic [`PROBE_MAGIC`], the CTA, the warp, the number of records the
//! warp attempted), then `capacity` records of 16 bytes: `u64` `%globaltimer` (nanoseconds, one clock for the GPU), `u32` `%clock` (the SM's cycle
//! counter, low 32 bits), `u16` the probe's id (0 `@begin`, 1 `@end`, 2 `@calib`, then the kernel's probes), `u16` `%smid`. A warp keeps its first
//! `capacity` records and counts the rest. A slot whose header is not intact (another magic, another CTA or warp) is listed as missing.
//!
//! [`ProbeBuffer`] and [`Kernel::launch_probed`](crate::Kernel::launch_probed) record one launch; [`LaunchPlan::instantiate_probed`](crate::LaunchPlan::instantiate_probed)
//! gives every instrumented step of a plan a region of a probe buffer of the instantiation, read with [`PlanRun::collect_probes`](crate::PlanRun::collect_probes).
//! [`ProbeTrace::save`] writes what was collected for Python.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use serde_json::{Value, json};

use crate::error::{Error, Result};

/// The record format, the buffer layout and the probes the build adds: the `format` of an entry's `instrument`. Python's `_probe.FORMAT`.
pub const PROBE_FORMAT: &str = "tilefish.probes.v1";
/// The schema of a probe trace file (what [`ProbeTrace::save`] writes and Python's `ProbeTrace.load` reads).
pub const PROBE_TRACE_SCHEMA: &str = "tilefish.probe-trace.v1";
/// The first word of a warp's header: the bytes `"tfp1"`.
pub const PROBE_MAGIC: u32 = 0x3170_6674;
/// The bytes of a warp's header: magic, CTA, warp, count (`u32` each).
pub const PROBE_HEADER_BYTES: u64 = 16;
/// The bytes of a record: globaltimer (`u64`), clock (`u32`), id (`u16`), SM (`u16`).
pub const PROBE_RECORD_BYTES: u64 = 16;
/// The most records a warp keeps (an entry's `capacity` is from 1 to this).
pub const PROBE_MAX_CAPACITY: u64 = 1 << 16;
/// The id of a kernel's first probe: 0 is `@begin`, 1 `@end`, 2 `@calib`.
pub const PROBE_FIRST_ID: u64 = 3;
/// The largest id of a probe.
pub const PROBE_MAX_ID: u64 = 0xFFFF;
/// The names of the probes every instrumented binary has, by id.
pub const IMPLICIT_PROBES: [&str; 3] = ["@begin", "@end", "@calib"];
/// What a probe buffer's address must be a multiple of (the records are written 16 bytes at a time).
pub const PROBE_ALIGNMENT: u64 = 16;

/// Where a probe of a kernel is written (`instrument.probes[].sites[]` of its entry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeSite {
    /// The module-relative file and line (`kernels/mm.py:40`; `""` when the build had no location for it).
    pub loc: String,
    /// The partition of a warp-specialised kernel the probe runs in (`"default"`, `"worker 0"`, ...), `""` for a kernel that is not partitioned.
    pub role: String,
    /// The jit function the probe is written in (`kernels.mm._issue`).
    pub function: String,
}

/// One probe of a kernel: its id in the records, its name, and where it is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    /// The id its records carry (from [`PROBE_FIRST_ID`]).
    pub id: u16,
    /// The name the kernel gave it (`tilefish.probe("load.wait")`).
    pub name: String,
    /// Where it is written: one site, or several for a name written in several places.
    pub sites: Vec<ProbeSite>,
}

/// The `instrument` of an instrumented binary's entry (manifest schema 3, section 4.8): the records each warp keeps per launch and the kernel's
/// probes. The format is [`PROBE_FORMAT`] (the only one this host reads).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instrument {
    /// The records each warp keeps per launch (1 to [`PROBE_MAX_CAPACITY`]); the rest are counted, not kept.
    pub capacity: u32,
    /// The kernel's probes, by id (the build's own `@begin`, `@end` and `@calib` are not listed).
    pub probes: Vec<Probe>,
}

impl Instrument {
    /// The bytes of one warp's slot of the probe buffer.
    pub fn slot_bytes(&self) -> u64 {
        PROBE_HEADER_BYTES + PROBE_RECORD_BYTES * u64::from(self.capacity)
    }

    /// The name of the probe `id`: one of the build's own, or the kernel's.
    pub fn name_of(&self, id: u16) -> Option<&str> {
        IMPLICIT_PROBES
            .get(usize::from(id))
            .copied()
            .or_else(|| self.probes.iter().find(|p| p.id == id).map(|p| p.name.as_str()))
    }

    /// The table of probes as the manifest writes it (`[{"id", "name", "sites": [{"loc", "role", "function"}]}]`).
    pub fn probes_json(&self) -> Value {
        Value::Array(
            self.probes
                .iter()
                .map(|p| {
                    json!({
                        "id": p.id,
                        "name": p.name,
                        "sites": p.sites.iter().map(|s| json!({"loc": s.loc, "role": s.role, "function": s.function})).collect::<Vec<_>>(),
                    })
                })
                .collect(),
        )
    }
}

/// Whether `name` is the name of a probe: 1 to 64 characters, a letter or `_` first, then letters, digits and `_ . : / -`. Python's
/// `_probe.check_name`.
pub fn is_probe_name(name: &str) -> bool {
    let b = name.as_bytes();
    !b.is_empty()
        && b.len() <= 64
        && (b[0].is_ascii_alphabetic() || b[0] == b'_')
        && b[1..].iter().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b':' | b'/' | b'-'))
}

/// The bytes of the probe buffer of a launch of `ctas` CTAs of `warps` warps whose binary keeps `capacity` records per warp (`None` when it does not
/// fit 64 bits). Python's `_probe.buffer_bytes`.
pub fn probe_buffer_bytes(capacity: u32, warps: u32, ctas: u64) -> Option<u64> {
    ctas.checked_mul(u64::from(warps))?.checked_mul(PROBE_HEADER_BYTES + PROBE_RECORD_BYTES * u64::from(capacity))
}

/// The warps of a block of `block` threads (what the slots of a CTA are).
pub fn warps_of(block: [u32; 3]) -> u32 {
    block[0].div_ceil(32).max(1)
}

/// One record: the probe's id, the SM, `%globaltimer` (ns) and `%clock` (cycles, low 32 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeRecord {
    /// The probe (0 `@begin`, 1 `@end`, 2 `@calib`, then the kernel's).
    pub id: u16,
    /// `%smid` when the record was made.
    pub sm: u16,
    /// `%globaltimer`, nanoseconds.
    pub globaltimer: u64,
    /// `%clock`, the low 32 bits of the SM's cycle counter.
    pub clock: u32,
}

/// What one warp of a launch recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarpRecords {
    /// The linear index of its CTA over the CUDA grid.
    pub cta: u32,
    /// The warp in its CTA (`threadIdx.x / 32`).
    pub warp: u32,
    /// The records it attempted (more than the capacity: the rest were dropped).
    pub count: u32,
    /// The records kept, in the order it made them.
    pub records: Vec<ProbeRecord>,
}

impl WarpRecords {
    /// How many of its records did not fit the capacity.
    pub fn dropped(&self) -> u32 {
        self.count.saturating_sub(self.records.len() as u32)
    }
}

fn u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().expect("4 bytes"))
}

/// What [`decode_probes`] reads of a probe buffer: the warps whose header is intact, and the `(cta, warp)` slots that are not.
pub type DecodedProbes = (Vec<WarpRecords>, Vec<(u32, u32)>);

/// The records of a launch's probe buffer (its first [`probe_buffer_bytes`] bytes, read back from the device): every warp whose header is intact
/// (the magic, its own CTA and warp), and the `(cta, warp)` slots that are not. Python's `_probe.decode`.
pub fn decode_probes(data: &[u8], capacity: u32, warps: u32, ctas: u64) -> Result<DecodedProbes> {
    let need = probe_buffer_bytes(capacity, warps, ctas)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::Launch("the probe buffer does not fit the address space".to_owned()))?;
    if data.len() < need {
        return Err(Error::Launch(format!(
            "the probe buffer has {} bytes, but {ctas} CTAs of {warps} warps of {capacity} records need {need}",
            data.len()
        )));
    }
    let slot = (PROBE_HEADER_BYTES + PROBE_RECORD_BYTES * u64::from(capacity)) as usize;
    let mut out = Vec::new();
    let mut missing = Vec::new();
    for c in 0..ctas {
        for w in 0..warps {
            let at = (c as usize * warps as usize + w as usize) * slot;
            let (magic, cta, warp, count) =
                (u32_at(data, at), u32_at(data, at + 4), u32_at(data, at + 8), u32_at(data, at + 12));
            if magic != PROBE_MAGIC || u64::from(cta) != c || warp != w {
                missing.push((c as u32, w));
                continue;
            }
            let kept = count.min(capacity) as usize;
            let records = (0..kept)
                .map(|i| {
                    let r = at + PROBE_HEADER_BYTES as usize + i * PROBE_RECORD_BYTES as usize;
                    let word = u32_at(data, r + 12);
                    ProbeRecord {
                        id: (word & 0xFFFF) as u16,
                        sm: (word >> 16) as u16,
                        globaltimer: u64::from_le_bytes(data[r..r + 8].try_into().expect("8 bytes")),
                        clock: u32_at(data, r + 8),
                    }
                })
                .collect();
            out.push(WarpRecords { cta, warp, count, records });
        }
    }
    Ok((out, missing))
}

/// The records of one launch of an instrumented binary, with what explains them: which binary, its CUDA grid and block, the capacity and the
/// table of its probes, and, for a step of a plan, the graph, the step and the bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeLaunch {
    /// The stem of the binary.
    pub stem: String,
    /// Its architecture.
    pub arch: String,
    /// Its operator.
    pub op: String,
    /// The graph of the plan the launch was a step of (`None`: a launch of its own).
    pub graph: Option<String>,
    /// The step of the plan.
    pub step: Option<usize>,
    /// The bucket the plan ran at, by dimension.
    pub bucket: Option<Vec<(String, u64)>>,
    /// The digest of the manifest of the binary, when the host knows it.
    pub manifest_digest: Option<String>,
    /// The CUDA grid launched.
    pub grid: [u32; 3],
    /// The block launched.
    pub block: [u32; 3],
    /// The binary's probes.
    pub instrument: Instrument,
    /// What each warp recorded.
    pub warps: Vec<WarpRecords>,
    /// The `(cta, warp)` slots without an intact header.
    pub missing: Vec<(u32, u32)>,
}

impl ProbeLaunch {
    /// The launch as `tilefish.probe-trace.v1` writes it (Python's `Launch.to_json`): the times from `t0`, the launch's earliest globaltimer,
    /// which is a decimal string (a globaltimer does not fit a JSON number exactly).
    pub fn to_json(&self) -> Value {
        let t0 = self.warps.iter().flat_map(|w| w.records.iter().map(|r| r.globaltimer)).min().unwrap_or(0);
        let warps: Vec<Value> = self
            .warps
            .iter()
            .map(|w| {
                let recs: Vec<Value> =
                    w.records.iter().map(|r| json!([r.id, r.sm, r.globaltimer - t0, r.clock])).collect();
                json!([w.cta, w.warp, w.count, recs])
            })
            .collect();
        let bucket = self.bucket.as_ref().map(|b| {
            Value::Object(
                b.iter().map(|(k, v)| (k.clone(), Value::from(*v))).collect::<serde_json::Map<String, Value>>(),
            )
        });
        json!({
            "stem": self.stem,
            "arch": self.arch,
            "op": self.op,
            "graph": self.graph,
            "step": self.step,
            "bucket": bucket,
            "manifest_digest": self.manifest_digest,
            "grid": self.grid,
            "block": self.block,
            "capacity": self.instrument.capacity,
            "probes": self.instrument.probes_json(),
            "t0": t0.to_string(),
            "warps": warps,
            "missing": self.missing.iter().map(|(c, w)| json!([c, w])).collect::<Vec<_>>(),
        })
    }
}

/// The records of one or more instrumented launches, for `tilefish-explain DIR --probes FILE` (and Python's `tilefish.ProbeTrace.load`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProbeTrace {
    /// The launches, in the order they were collected.
    pub launches: Vec<ProbeLaunch>,
}

impl ProbeTrace {
    /// The trace as `tilefish.probe-trace.v1` JSON.
    pub fn to_json(&self) -> Value {
        json!({"schema": PROBE_TRACE_SCHEMA, "launches": self.launches.iter().map(ProbeLaunch::to_json).collect::<Vec<_>>()})
    }

    /// Write [`ProbeTrace::to_json`] to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut text = serde_json::to_string(&self.to_json())
            .map_err(|e| Error::Launch(format!("the probe trace cannot be written: {e}")))?;
        text.push('\n');
        std::fs::write(path, text).map_err(|source| Error::Io { path: path.display().to_string(), source })
    }
}

/// What a probe buffer knows of the launch it was given to, to decode its records.
#[derive(Debug, Clone)]
pub(crate) struct ProbeMeta {
    pub(crate) stem: String,
    pub(crate) arch: String,
    pub(crate) op: String,
    pub(crate) grid: [u32; 3],
    pub(crate) block: [u32; 3],
    pub(crate) instrument: Arc<Instrument>,
}

impl ProbeMeta {
    /// The bytes of the launch's records.
    pub(crate) fn bytes(&self) -> Option<u64> {
        let ctas = self.grid.iter().map(|g| u64::from(*g)).product::<u64>();
        probe_buffer_bytes(self.instrument.capacity, warps_of(self.block), ctas)
    }

    /// The records of `data` (the launch's bytes read back) as a [`ProbeLaunch`].
    pub(crate) fn decode(&self, data: &[u8]) -> Result<ProbeLaunch> {
        let ctas = self.grid.iter().map(|g| u64::from(*g)).product::<u64>();
        let (warps, missing) = decode_probes(data, self.instrument.capacity, warps_of(self.block), ctas)?;
        Ok(ProbeLaunch {
            stem: self.stem.clone(),
            arch: self.arch.clone(),
            op: self.op.clone(),
            graph: None,
            step: None,
            bucket: None,
            manifest_digest: None,
            grid: self.grid,
            block: self.block,
            instrument: (*self.instrument).clone(),
            warps,
            missing,
        })
    }
}

/// The device memory an instrumented launch records into ([`Kernel::launch_probed`](crate::Kernel::launch_probed)), and what it read back.
///
/// It owns a cudarc slice allocated on a stream (the slice's own stream): a launch hands it to cudarc's tracking as memory the kernel writes, which
/// makes its own stream wait for the kernel (as for any tensor a launch writes), so [`ProbeBuffer::collect`], which copies it back on that stream,
/// reads what the last launch wrote. Every launch writes the slots of every warp it runs (their headers first), so a buffer is reused from launch to
/// launch without clearing: collect each launch before the next one, which writes over it.
#[derive(Debug)]
pub struct ProbeBuffer {
    pub(crate) slice: CudaSlice<u8>,
    stream: Arc<CudaStream>,
    pub(crate) pending: Option<ProbeMeta>,
}

impl ProbeBuffer {
    /// A probe buffer of `bytes` bytes on `stream` (allocated there, freed there; [`Kernel::probe_bytes`](crate::Kernel::probe_bytes) says how many
    /// a launch needs).
    pub fn new(stream: &Arc<CudaStream>, bytes: usize) -> Result<Self> {
        let slice = stream.alloc_zeros::<u8>(bytes.max(1)).map_err(Error::driver("cuMemAllocAsync"))?;
        Ok(Self { slice, stream: Arc::clone(stream), pending: None })
    }

    /// Its bytes.
    pub fn bytes(&self) -> usize {
        self.slice.len()
    }

    /// The stream it was allocated on (its own stream).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The records of the last launch it was given to: copied back on its own stream (which waits for that launch), then decoded. Refused when no
    /// launch was given it since the last collect.
    pub fn collect(&mut self) -> Result<ProbeLaunch> {
        let Some(meta) = self.pending.take() else {
            return Err(Error::Launch(
                "the probe buffer was given to no instrumented launch since it was last collected (Kernel::launch_probed)".to_owned(),
            ));
        };
        let need = meta
            .bytes()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::Launch("the probe buffer does not fit the address space".to_owned()))?;
        let view = self.slice.slice(0..need);
        let data = self.stream.clone_dtoh(&view).map_err(Error::driver("cuMemcpyDtoHAsync"))?;
        self.stream.synchronize().map_err(Error::driver("cuStreamSynchronize"))?;
        meta.decode(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(cta: u32, warp: u32, count: u32, records: &[(u16, u16, u64, u32)], capacity: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for v in [PROBE_MAGIC, cta, warp, count] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for (id, sm, t, clk) in records {
            out.extend_from_slice(&t.to_le_bytes());
            out.extend_from_slice(&clk.to_le_bytes());
            out.extend_from_slice(&((u32::from(*sm) << 16) | u32::from(*id)).to_le_bytes());
        }
        out.resize(out.len() + 16 * (capacity as usize - records.len()), 0xAB);
        out
    }

    /// As Python's `test_decode_reads_every_warp_and_counts_what_was_dropped`: every intact warp, the slots without a header, the dropped count.
    #[test]
    fn decode_reads_every_warp_and_counts_what_was_dropped() {
        let cap = 4;
        let recs = [(0, 7, 1000, 10), (2, 7, 1010, 30), (2, 7, 1020, 70), (3, 7, 1500, 1000)];
        let mut data = slot(0, 0, 9, &recs, cap);
        data.extend(vec![0u8; 16 + 16 * cap as usize]);
        data.extend(slot(1, 0, 2, &recs[..2], cap));
        data.extend(slot(1, 1, 0, &[], cap));
        let (warps, missing) = decode_probes(&data, cap, 2, 2).unwrap();
        assert_eq!(missing, vec![(0, 1)]);
        let summary: Vec<_> = warps.iter().map(|w| (w.cta, w.warp, w.count, w.records.len(), w.dropped())).collect();
        assert_eq!(summary, vec![(0, 0, 9, 4, 5), (1, 0, 2, 2, 0), (1, 1, 0, 0, 0)]);
        assert_eq!(warps[0].records[3], ProbeRecord { id: 3, sm: 7, globaltimer: 1500, clock: 1000 });
        let mut wrong = slot(1, 0, 1, &recs[..1], cap);
        wrong.extend(slot(0, 0, 1, &recs[..1], cap));
        assert_eq!(decode_probes(&wrong, cap, 2, 1).unwrap().1, vec![(0, 0), (0, 1)]);
        assert!(decode_probes(&data[..data.len() - 1], cap, 2, 2).is_err());
        assert_eq!(probe_buffer_bytes(cap, 2, 2), Some(data.len() as u64));
        assert_eq!(probe_buffer_bytes(u32::MAX, u32::MAX, u64::MAX), None);
    }

    #[test]
    fn the_names_of_probes() {
        for good in ["load", "load.wait", "mma/issue", "_x", "tile:3", "a-b", &"x".repeat(64)] {
            assert!(is_probe_name(good), "{good}");
        }
        for bad in ["", "3load", "has space", &"x".repeat(65), "@begin", "q\"", "é"] {
            assert!(!is_probe_name(bad), "{bad}");
        }
        assert_eq!(warps_of([96, 1, 1]), 3);
        assert_eq!(warps_of([100, 1, 1]), 4);
    }

    /// The launch as Python's `Launch.to_json` writes it: times from `t0` (a decimal string), the table of probes verbatim.
    #[test]
    fn a_launch_is_written_as_the_probe_trace_schema() {
        let instrument = Instrument {
            capacity: 8,
            probes: vec![Probe {
                id: 3,
                name: "load.wait".to_owned(),
                sites: vec![ProbeSite {
                    loc: "kernels/mm.py:30".to_owned(),
                    role: "worker 0".to_owned(),
                    function: "kernels.mm._load".to_owned(),
                }],
            }],
        };
        assert_eq!(instrument.name_of(1), Some("@end"));
        assert_eq!(instrument.name_of(3), Some("load.wait"));
        assert_eq!(instrument.name_of(4), None);
        let t0 = 1_794_824_135_279_526_464u64; // more than 2**53: not a JSON number that a reader keeps exact
        let launch = ProbeLaunch {
            stem: "mm.M256".to_owned(),
            arch: "sm_103".to_owned(),
            op: "mm".to_owned(),
            graph: Some("model".to_owned()),
            step: Some(2),
            bucket: Some(vec![("M".to_owned(), 256)]),
            manifest_digest: Some("ab".repeat(32)),
            grid: [2, 1, 1],
            block: [96, 1, 1],
            instrument,
            warps: vec![WarpRecords {
                cta: 0,
                warp: 1,
                count: 3,
                records: vec![
                    ProbeRecord { id: 0, sm: 5, globaltimer: t0 + 7, clock: 10 },
                    ProbeRecord { id: 3, sm: 5, globaltimer: t0, clock: 4_000_000_000 },
                ],
            }],
            missing: vec![(1, 2)],
        };
        let doc = ProbeTrace { launches: vec![launch] }.to_json();
        assert_eq!(doc["schema"], PROBE_TRACE_SCHEMA);
        let l = &doc["launches"][0];
        assert_eq!(l["t0"], t0.to_string());
        assert_eq!(l["warps"], json!([[0, 1, 3, [[0, 5, 7, 10], [3, 5, 0, 4_000_000_000u32]]]]));
        assert_eq!(l["bucket"], json!({"M": 256}));
        assert_eq!(l["missing"], json!([[1, 2]]));
        assert_eq!(l["capacity"], 8);
        assert_eq!(
            l["probes"],
            json!([{"id": 3, "name": "load.wait", "sites": [{"loc": "kernels/mm.py:30", "role": "worker 0", "function": "kernels.mm._load"}]}])
        );
        let keys: Vec<&str> = l.as_object().unwrap().keys().map(String::as_str).collect();
        let mut want = vec![
            "arch",
            "block",
            "bucket",
            "capacity",
            "grid",
            "graph",
            "manifest_digest",
            "missing",
            "op",
            "probes",
            "stem",
            "step",
            "t0",
            "warps",
        ];
        want.sort_unstable();
        let mut got = keys.clone();
        got.sort_unstable();
        assert_eq!(got, want);
    }
}
