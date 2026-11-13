//! A strict ELF64 section walk and the digest of the machine code of a cubin (`sha256_code`).
//!
//! This is a port of `python/src/tilefish/_elf.py`, and [`Elf::code_digest`] must return the same 64 hex digits as Python's
//! `code_digest` for every cubin. A cubin is an ELF file whose debug sections hold the path of the source it was
//! compiled from, so two builds of one kernel from different directories differ byte for byte while running the same
//! code. The digest covers the identity fields of the ELF header and every section that is not debug information, as one
//! canonical byte stream (see [`Elf::code_digest`]), and skips `.debug_*` and `.nv_debug_*` with their wrapped variants.
//!
//! The walk is strict: anything that is not a well-formed little-endian ELF64 file whose section header table lies
//! inside the file is an error, and nothing is guessed. All offsets are checked with 128-bit arithmetic, so no file can
//! make a bound wrap around.

use sha2::{Digest, Sha256};

/// Changes with the canonical byte stream of [`Elf::code_digest`]: every digest changes with it. Python's `CODE_DIGEST_FORMAT`.
pub const CODE_DIGEST_FORMAT: u32 = 1;

const SHT_NULL: u32 = 0;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_NOBITS: u32 = 8;
const SHT_DYNSYM: u32 = 11;
const SHF_INFO_LINK: u64 = 0x40;
const SHN_XINDEX: u16 = 0xFFFF;
const EHDR_SIZE: usize = 64;
const SHDR_SIZE: usize = 64;

// prefixes that wrap a section name: the mercury (sm_100 and later) copy of it, and relocations of it
const WRAPPERS: [&str; 3] = [".nv.merc.", ".rela.", ".rel."];
const DEBUG_PREFIXES: [&str; 4] = [".debug_", ".zdebug_", ".nv_debug_", ".nv.debug_"];

/// Whether a section holds debug information: `.debug_*`, `.nv_debug_*` or a wrapped variant of one.
///
/// `.rela.debug_line`, `.nv.merc.debug_line` and `.nv.merc.rela.nv_debug_line_sass` are debug sections;
/// `.rela.text.k`, `.nv.merc.nv.info` and `.text.debug_k` are not.
pub fn is_debug_section(name: &str) -> bool {
    let mut base = name.to_owned();
    let mut stripped = true;
    while stripped {
        stripped = false;
        for prefix in WRAPPERS {
            if base.starts_with(prefix) {
                base = format!(".{}", &base[prefix.len()..]);
                stripped = true;
            }
        }
    }
    DEBUG_PREFIXES.iter().any(|p| base.starts_with(p))
}

/// One section header of an ELF file.
#[derive(Debug, Clone)]
pub struct Section {
    /// Position in the section header table.
    pub index: usize,
    /// The name, from the section name table.
    pub name: String,
    /// `sh_type`.
    pub ty: u32,
    /// `sh_flags`.
    pub flags: u64,
    /// `sh_addr`.
    pub addr: u64,
    /// `sh_offset`.
    pub offset: u64,
    /// `sh_size`.
    pub size: u64,
    /// `sh_link`.
    pub link: u32,
    /// `sh_info`.
    pub info: u32,
    /// `sh_addralign`.
    pub addralign: u64,
    /// `sh_entsize`.
    pub entsize: u64,
}

/// The parts of an ELF64 file the digest needs: the identity fields, the sections and their contents.
#[derive(Debug)]
pub struct Elf<'a> {
    ident: [u8; 16],
    ty: u16,
    machine: u16,
    version: u32,
    flags: u32,
    /// All sections, the null section first.
    pub sections: Vec<Section>,
    blob: &'a [u8],
    shstrndx: usize,
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

fn bad<T>(message: impl AsRef<str>) -> Result<T, String> {
    Err(format!("malformed ELF file: {}", message.as_ref()))
}

impl<'a> Elf<'a> {
    /// Walk the section header table of an ELF64 little-endian file, checking every bound on the way.
    pub fn parse(blob: &'a [u8]) -> Result<Self, String> {
        let size = blob.len();
        if size < EHDR_SIZE {
            return bad(format!("{size} bytes is shorter than the {EHDR_SIZE}-byte ELF header"));
        }
        let mut ident = [0u8; 16];
        ident.copy_from_slice(&blob[..16]);
        let (ty, machine, version) = (u16_at(blob, 16), u16_at(blob, 18), u32_at(blob, 20));
        let shoff = u64_at(blob, 40);
        let flags = u32_at(blob, 48);
        let (ehsize, shentsize, shnum, shstrndx) =
            (u16_at(blob, 52), u16_at(blob, 58), u16_at(blob, 60), u16_at(blob, 62));
        if &ident[..4] != b"\x7fELF" {
            return bad(format!("bad magic {:?}", &ident[..4]));
        }
        if ident[4] != 2 {
            return bad(format!("EI_CLASS is {}, not 2 (ELFCLASS64)", ident[4]));
        }
        if ident[5] != 1 {
            return bad(format!("EI_DATA is {}, not 1 (little endian)", ident[5]));
        }
        if ident[6] != 1 {
            return bad(format!("EI_VERSION is {}, not 1", ident[6]));
        }
        if usize::from(ehsize) != EHDR_SIZE {
            return bad(format!("e_ehsize is {ehsize}, not {EHDR_SIZE}"));
        }
        if usize::from(shentsize) != SHDR_SIZE {
            return bad(format!("e_shentsize is {shentsize}, not {SHDR_SIZE}"));
        }
        if shnum == 0 {
            return bad("it has no section header table (e_shnum is 0)");
        }
        if shstrndx == SHN_XINDEX {
            return bad("extended section numbering (e_shstrndx = SHN_XINDEX) is not supported");
        }
        if shstrndx >= shnum {
            return bad(format!("e_shstrndx {shstrndx} is outside the {shnum} sections"));
        }
        let table_end = u128::from(shoff) + u128::from(shnum) * SHDR_SIZE as u128;
        if (shoff as u128) < EHDR_SIZE as u128 || table_end > size as u128 {
            return bad(format!(
                "the section header table (offset {shoff}, {shnum} entries) does not fit in {size} bytes"
            ));
        }
        let shoff = shoff as usize;
        let shnum = usize::from(shnum);
        let shstrndx = usize::from(shstrndx);

        struct Raw {
            name: u32,
            ty: u32,
            flags: u64,
            addr: u64,
            offset: u64,
            size: u64,
            link: u32,
            info: u32,
            addralign: u64,
            entsize: u64,
        }
        let raw: Vec<Raw> = (0..shnum)
            .map(|i| {
                let at = shoff + i * SHDR_SIZE;
                Raw {
                    name: u32_at(blob, at),
                    ty: u32_at(blob, at + 4),
                    flags: u64_at(blob, at + 8),
                    addr: u64_at(blob, at + 16),
                    offset: u64_at(blob, at + 24),
                    size: u64_at(blob, at + 32),
                    link: u32_at(blob, at + 40),
                    info: u32_at(blob, at + 44),
                    addralign: u64_at(blob, at + 48),
                    entsize: u64_at(blob, at + 56),
                }
            })
            .collect();
        for (i, s) in raw.iter().enumerate() {
            if s.ty == SHT_NOBITS {
                continue;
            }
            if s.size != 0 && s.offset < EHDR_SIZE as u64 {
                return bad(format!("section {i} starts at offset {}, inside the ELF header", s.offset));
            }
            if u128::from(s.offset) + u128::from(s.size) > size as u128 {
                return bad(format!(
                    "section {i} (offset {}, size {}) extends past the end of the {size}-byte file",
                    s.offset, s.size
                ));
            }
        }
        if raw[0].ty != SHT_NULL {
            return bad(format!("section 0 has type {}, not 0 (SHT_NULL)", raw[0].ty));
        }
        let strtab = &raw[shstrndx];
        if strtab.ty != SHT_STRTAB {
            return bad(format!(
                "the section name table (section {shstrndx}) has type {}, not 3 (SHT_STRTAB)",
                strtab.ty
            ));
        }
        let names = &blob[strtab.offset as usize..(strtab.offset + strtab.size) as usize];

        let mut sections = Vec::with_capacity(shnum);
        for (i, s) in raw.iter().enumerate() {
            let at = s.name as usize;
            if at >= names.len() {
                return bad(format!(
                    "section {i}: the name offset {at} is outside the {}-byte name table",
                    names.len()
                ));
            }
            let Some(len) = names[at..].iter().position(|b| *b == 0) else {
                return bad(format!("section {i}: its name is not NUL terminated"));
            };
            let Ok(name) = std::str::from_utf8(&names[at..at + len]) else {
                return bad(format!("section {i}: its name is not valid UTF-8"));
            };
            sections.push(Section {
                index: i,
                name: name.to_owned(),
                ty: s.ty,
                flags: s.flags,
                addr: s.addr,
                offset: s.offset,
                size: s.size,
                link: s.link,
                info: s.info,
                addralign: s.addralign,
                entsize: s.entsize,
            });
        }
        Ok(Self { ident, ty, machine, version, flags, sections, blob, shstrndx })
    }

    /// The contents of a section (empty for a `NOBITS` section, which has none in the file).
    pub fn data(&self, s: &Section) -> &'a [u8] {
        if s.ty == SHT_NOBITS {
            return &[];
        }
        &self.blob[s.offset as usize..(s.offset + s.size) as usize]
    }

    /// The first section called `name`.
    pub fn section(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Whether the kernel `symbol` uses shared memory: it has a `.nv.shared.<symbol>` section.
    ///
    /// The ELF does not say how much dynamic shared memory the code addresses: that is the `shared` the build recorded.
    pub fn uses_shared(&self, symbol: &str) -> bool {
        self.section(&format!(".nv.shared.{symbol}")).is_some()
    }

    // a link or info field as the section it names (so that moving debug sections does not change it), else as a number
    fn reference(&self, value: u32, resolve: bool) -> Vec<u8> {
        let v = value as usize;
        if resolve && 0 < v && v < self.sections.len() {
            let mut out = b"S:".to_vec();
            out.extend_from_slice(self.sections[v].name.as_bytes());
            out
        } else {
            format!("N:{value}").into_bytes()
        }
    }

    /// The sha256 (lowercase hex) of the machine code of the file: everything but build-path debug information.
    ///
    /// The digest covers, as one canonical byte stream, the ELF identity (`e_ident`, type, machine, version and flags,
    /// which carry the target architecture) and, for every section after the null section that is not a debug section,
    /// its name, type, flags, address, size, alignment, entry size, its `sh_link` and `sh_info` (as section names where
    /// they name a section) and its contents (nothing for `NOBITS`). File offsets are not covered, and neither is the
    /// section name table itself.
    pub fn code_digest(&self) -> String {
        let mut h = Sha256::new();
        fn put(h: &mut Sha256, b: &[u8]) {
            h.update((b.len() as u64).to_le_bytes());
            h.update(b);
        }
        h.update(format!("tilefish.cubin-code.v{CODE_DIGEST_FORMAT}\0").as_bytes());
        put(&mut h, &self.ident);
        h.update(self.ty.to_le_bytes());
        h.update(self.machine.to_le_bytes());
        h.update(self.version.to_le_bytes());
        h.update(self.flags.to_le_bytes());
        for s in &self.sections[1..] {
            if is_debug_section(&s.name) || s.index == self.shstrndx {
                continue;
            }
            put(&mut h, s.name.as_bytes());
            h.update(s.ty.to_le_bytes());
            h.update(s.flags.to_le_bytes());
            h.update(s.addr.to_le_bytes());
            h.update(s.size.to_le_bytes());
            h.update(s.addralign.to_le_bytes());
            h.update(s.entsize.to_le_bytes());
            put(&mut h, &self.reference(s.link, true));
            let info_is_section = s.flags & SHF_INFO_LINK != 0 && s.ty != SHT_SYMTAB && s.ty != SHT_DYNSYM;
            put(&mut h, &self.reference(s.info, info_is_section));
            put(&mut h, self.data(s));
        }
        hex(&h.finalize())
    }
}

/// Lowercase hex of some bytes.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from(DIGITS[usize::from(b >> 4)]));
        out.push(char::from(DIGITS[usize::from(b & 15)]));
    }
    out
}

/// The sha256 of some bytes, in lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_sections_are_the_ones_python_skips() {
        for name in [
            ".debug_line",
            ".nv_debug_ptx_txt",
            ".rela.debug_line",
            ".nv.merc.debug_line",
            ".nv.merc.rela.debug_line",
            ".nv.merc.rela.nv_debug_line_sass",
            ".zdebug_info",
            ".nv.debug_frame",
        ] {
            assert!(is_debug_section(name), "{name}");
        }
        for name in [".text.k", ".rela.text.k", ".nv.merc.nv.info", ".text.debug_k", ".nv.shared.k", ".shstrtab"] {
            assert!(!is_debug_section(name), "{name}");
        }
    }

    #[test]
    fn hex_is_lowercase() {
        assert_eq!(hex(&[0x00, 0xab, 0xff]), "00abff");
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn a_file_that_is_not_an_elf_is_refused() {
        assert!(Elf::parse(b"too short").unwrap_err().contains("shorter than"));
        let mut blob = vec![0u8; 64];
        assert!(Elf::parse(&blob).unwrap_err().contains("bad magic"));
        blob[..4].copy_from_slice(b"\x7fELF");
        assert!(Elf::parse(&blob).unwrap_err().contains("EI_CLASS"));
        blob[4] = 2;
        blob[5] = 1;
        blob[6] = 1;
        assert!(Elf::parse(&blob).unwrap_err().contains("e_ehsize"));
        blob[52] = 64;
        blob[58] = 64;
        assert!(Elf::parse(&blob).unwrap_err().contains("no section header table"));
        blob[60] = 1;
        blob[40] = 1; // a section header table inside the ELF header
        assert!(Elf::parse(&blob).unwrap_err().contains("does not fit"));
    }
}
