//! A plan set: several launch plans exported to one directory with **one manifest** (`docs/plan-schema-1.md`, section 15).
//!
//! Python's `tilefish.runtime.export([g1, g2, ...], directory)` writes one `manifest.json` for all the plans (each binary once, whichever plans launch
//! it), `<graph>.plan.json` per graph and the index `plans.json`, which lists the digest of every plan and has a digest of its own (`set_digest`). A
//! port of `read_plan_set`, `validate_set` and `set_digest` of `python/src/tilefish/_plan.py`. [`PlanSet::load`] reads the index with its pin, loads
//! and verifies the manifest once (pinned to the index's `manifest_digest`), and [`PlanSet::plan`] loads one plan with the digest the index lists as
//! its pin and the set's manifest: one pin covers every plan, their manifest and every binary, and the plans share the manifest's loaded modules.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::elf::sha256_hex;
use crate::error::{Error, Result};
use crate::json::{canonical, parse_strict, type_name};
use crate::launch_plan::LaunchPlan;
use crate::manifest::{MANIFEST_NAME, Manifest, Pin, is_arch, is_digest, is_ident};

/// The `schema` of a plan set's index.
pub const PLAN_SET_SCHEMA: &str = "tilefish.plan-set.v1";
/// The index of a plan set, next to its `manifest.json`.
pub const PLAN_SET_NAME: &str = "plans.json";
const SET_DIGEST_TAG: &[u8] = b"tilefish.plan-set-digest.v1\0";
const SET_KEYS: [&str; 6] = ["schema", "tilefish", "arch", "manifest_digest", "plans", "set_digest"];
const MAX_PLANS: usize = 1 << 16;
const MAX_SET_BYTES: u64 = 1 << 28;

/// The file of the plan of graph `graph` in the directory of a plan set: `<graph>.plan.json`.
pub fn plan_file_name(graph: &str) -> String {
    format!("{graph}.plan.json")
}

/// Python's `set_digest`: `sha256(b"tilefish.plan-set-digest.v1\0" + canonical_json(doc without "set_digest" and "tilefish"))`, lowercase hex.
pub fn plan_set_digest(doc: &Value) -> String {
    let body = match doc {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(k, _)| k.as_str() != "set_digest" && k.as_str() != "tilefish")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        other => other.clone(),
    };
    let mut bytes = SET_DIGEST_TAG.to_vec();
    bytes.extend_from_slice(canonical(&body).as_bytes());
    sha256_hex(&bytes)
}

fn bad(message: impl Into<String>) -> Error {
    let message = message.into();
    let text: String = message.chars().take(1000).collect();
    Error::Plan(format!("plans.json: {text}"))
}

/// The plans of one directory over one manifest, verified against the pin of the set. Cheap to clone the plans it loads (they share its manifest).
#[derive(Debug, Clone)]
pub struct PlanSet {
    directory: PathBuf,
    digest: String,
    version: String,
    arch: String,
    plans: Vec<(String, String)>,
    manifest: Manifest,
}

impl PlanSet {
    /// Read `<directory>/plans.json` and the manifest next to it (`docs/plan-schema-1.md`, section 15).
    ///
    /// `pin` is the trust anchor: [`Pin::Digest`] with the set's digest (what Python's export returned, `PlanSet.digest`), and an index with any
    /// other content is refused; or [`Pin::TrustDirectory`]. Either way the index's own `set_digest` must be its digest, every rule of the index must
    /// hold, and the manifest is loaded with the index's `manifest_digest` as its pin (every cubin verified). The plans are read by [`PlanSet::plan`].
    pub fn load(directory: impl AsRef<Path>, pin: Pin<'_>) -> Result<Self> {
        let directory = directory.as_ref().to_owned();
        let file = directory.join(PLAN_SET_NAME);
        let label = file.display().to_string();
        if let Pin::Digest(d) = pin
            && !is_digest(d)
        {
            return Err(Error::BadPin(d.to_owned()));
        }
        let io = |source| Error::Io { path: label.clone(), source };
        let meta = std::fs::symlink_metadata(&file).map_err(io)?;
        if meta.file_type().is_symlink() || !meta.is_file() {
            return Err(Error::Plan(format!(
                "{label:?} is not a regular file (or is a symbolic link); refusing to read it"
            )));
        }
        if meta.len() > MAX_SET_BYTES {
            return Err(Error::Plan(format!(
                "the plan set {label:?} is {} bytes, more than the {MAX_SET_BYTES} it may have",
                meta.len()
            )));
        }
        let bytes = std::fs::read(&file).map_err(|source| Error::Io { path: label.clone(), source })?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| Error::Plan(format!("cannot read the plan set {label:?}: it is not UTF-8 text: {e}")))?;
        let doc =
            parse_strict(text).map_err(|e| Error::Plan(format!("the plan set {label:?} is not valid JSON: {e}")))?;
        if doc.get("schema").and_then(Value::as_str) != Some(PLAN_SET_SCHEMA) {
            return Err(Error::Plan(format!("{label:?} is not a tilefish plan set of schema {PLAN_SET_SCHEMA:?}")));
        }
        let found = plan_set_digest(&doc);
        if let Pin::Digest(pinned) = pin
            && found != pinned
        {
            return Err(Error::PlanPinMismatch { path: label, found, pinned: pinned.to_owned() });
        }
        if doc.get("set_digest").and_then(Value::as_str) != Some(found.as_str()) {
            return Err(Error::Plan(format!(
                "the plan set in {label:?} was edited: its set_digest does not say its content's digest {found}"
            )));
        }
        let Value::Object(map) = &doc else {
            return Err(bad(format!("the document must be an object, got {}", type_name(&doc))));
        };
        let missing: Vec<&str> = SET_KEYS.iter().copied().filter(|k| !map.contains_key(*k)).collect();
        let mut unknown: Vec<&str> = map.keys().map(String::as_str).filter(|k| !SET_KEYS.contains(k)).collect();
        unknown.sort_unstable();
        if !missing.is_empty() || !unknown.is_empty() {
            return Err(bad(format!(
                "the document must have exactly the fields {SET_KEYS:?}; missing {missing:?}; unknown {unknown:?}"
            )));
        }
        let Some(version) = map["tilefish"].as_str() else { return Err(bad("tilefish must be a string")) };
        let arch = match map["arch"].as_str() {
            Some(a) if is_arch(a) => a.to_owned(),
            _ => return Err(bad(format!("arch must look like 'sm_103', got {}", map["arch"]))),
        };
        let manifest_digest = match map["manifest_digest"].as_str() {
            Some(d) if is_digest(d) => d.to_owned(),
            _ => return Err(bad("manifest_digest must be 64 lowercase hex digits (a sha256)")),
        };
        let plans = match &map["plans"] {
            Value::Object(p) if !p.is_empty() && p.len() <= MAX_PLANS => p,
            _ => {
                return Err(bad(format!(
                    "plans must be a non-empty object of at most {MAX_PLANS} graph names to plan digests"
                )));
            }
        };
        let mut listed = Vec::with_capacity(plans.len());
        for (name, digest) in plans {
            match digest.as_str() {
                Some(d) if is_ident(name) && is_digest(d) => listed.push((name.clone(), d.to_owned())),
                _ => {
                    return Err(bad(format!(
                        "plans maps graph names (identifiers) to plan digests (64 lowercase hex digits), got {name:?}: {digest}"
                    )));
                }
            }
        }
        let manifest = match Manifest::load(directory.join(MANIFEST_NAME), Pin::Digest(&manifest_digest)) {
            Ok(m) => m,
            Err(Error::PinMismatch { directory, found, pinned }) => {
                return Err(bad(format!(
                    "the plan set was built against the manifest with digest {pinned}, but the manifest in {directory:?} has digest {found}"
                )));
            }
            Err(other) => return Err(other),
        };
        Ok(Self { directory, digest: found, version: version.to_owned(), arch, plans: listed, manifest })
    }

    /// Load the plan of graph `name` (`<directory>/<name>.plan.json`) with the digest the index lists as its pin and the set's manifest
    /// ([`LaunchPlan::load_with_manifest`]): refused when the index lists no such graph, or the file is not the plan it lists.
    pub fn plan(&self, name: &str) -> Result<LaunchPlan> {
        let Some(digest) = self.plan_digest(name) else {
            let names: Vec<&str> = self.names().collect();
            return Err(Error::Plan(format!("the plan set lists no plan of graph '{name}'; it has {names:?}")));
        };
        let plan = LaunchPlan::load_with_manifest(
            self.directory.join(plan_file_name(name)),
            &self.manifest,
            Pin::Digest(digest),
        )?;
        if plan.graph() != name {
            return Err(Error::Plan(format!("the file of graph '{name}' is the plan of graph '{}'", plan.graph())));
        }
        Ok(plan)
    }

    /// The digest of the set (`set_digest`): what a pin is compared with.
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// The version of tilefish that wrote the index (informative: not digested).
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The architecture the plans are for.
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// The names of the graphs of the set, in code point order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.plans.iter().map(|(n, _)| n.as_str())
    }

    /// The digest the index lists for the plan of graph `name`.
    pub fn plan_digest(&self, name: &str) -> Option<&str> {
        self.plans.iter().find(|(n, _)| n == name).map(|(_, d)| d.as_str())
    }

    /// The manifest of every plan of the set, loaded and verified once.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
}
