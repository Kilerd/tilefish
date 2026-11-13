//! The restricted expression grammar of a manifest: grids, dimension expressions and shapes.
//!
//! A port of `_ExprParser` in `python/src/tilefish/_specvalid.py`. The grammar is ints, identifiers, `cdiv(a, b)`,
//! `+ - * //`, unary minus and parentheses, with the same limits (512 characters, 256 tokens, depth 24). It is parsed to
//! a tree and evaluated by this module: nothing is ever handed to an interpreter.
//!
//! A shape or a runtime scalar is an expression over *dimensions* (`M`, `3072*M`, `cdiv(n, 16)`). A launch binds the
//! dimensions from the sizes of the tensors the way Python's `solve_dims` does ([`Pattern::solve`]): an equation with
//! one unbound dimension that occurs affinely is solved (`3072*M` against 6144 gives `M = 2`, and fails when it is not a
//! whole number), a fully bound one is checked, and whatever is left over is ambiguous.

use std::collections::BTreeMap;

const MAX_LEN: usize = 512;
const MAX_TOKENS: usize = 256;
const MAX_DEPTH: usize = 24;
/// The most dimension names one kernel may use (a strided pointer has one name per axis for its strides, and one for its offset).
pub const MAX_DIMS: usize = 32;

/// The dimension names of one kernel, numbered in order of first appearance.
#[derive(Debug, Clone, Default)]
pub struct Symbols {
    names: Vec<String>,
}

impl Symbols {
    pub fn intern(&mut self, name: &str) -> Result<usize, String> {
        if let Some(i) = self.find(name) {
            return Ok(i);
        }
        if self.names.len() == MAX_DIMS {
            return Err(format!("a kernel may use at most {MAX_DIMS} dimension names"));
        }
        self.names.push(name.to_owned());
        Ok(self.names.len() - 1)
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    pub fn name(&self, i: usize) -> &str {
        &self.names[i]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }
}

/// A parsed expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(i128),
    Var(usize),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    FloorDiv(Box<Expr>, Box<Expr>),
    CDiv(Box<Expr>, Box<Expr>),
}

const OVERFLOW: &str = "a dimension expression overflows";

fn bounded(v: i128) -> Result<i128, String> {
    if v.unsigned_abs() > (1u128 << 100) { Err(OVERFLOW.to_owned()) } else { Ok(v) }
}

/// `ceil(a / b)` for a positive `b` (the floor and one more when there is a remainder, which no `i128` overflows).
fn ceil_div(a: i128, b: i128) -> i128 {
    a.div_euclid(b) + i128::from(a.rem_euclid(b) != 0)
}

/// The value of `cdiv(n, d)`, an error for a divisor below 1 (as Python's `cdiv` raises for one).
fn cdiv_value(n: i128, d: i128) -> Result<i128, String> {
    if d < 1 { Err(format!("a division by {d}: the divisor must be positive")) } else { Ok(ceil_div(n, d)) }
}

impl Expr {
    /// The value under `dims` (`None` for a dimension that is not bound), or an error for a division by a non-positive
    /// number or a value that does not fit 64 bits.
    pub fn eval(&self, dims: &[Option<i64>]) -> Result<Option<i64>, String> {
        match self.eval_by(&|i| dims.get(i).copied().flatten().map(i128::from))? {
            None => Ok(None),
            Some(v) => i64::try_from(v).map(Some).map_err(|_| format!("the value {v} does not fit 64 bits")),
        }
    }

    /// The value of the expression as it is written, each dimension's value given by `lookup` (`None`: not bound). A sum or a product
    /// beyond `2^100` is an overflow.
    fn eval_by(&self, lookup: &dyn Fn(usize) -> Option<i128>) -> Result<Option<i128>, String> {
        let two = |a: &Expr, b: &Expr| -> Result<Option<(i128, i128)>, String> {
            match (a.eval_by(lookup)?, b.eval_by(lookup)?) {
                (Some(x), Some(y)) => Ok(Some((x, y))),
                _ => Ok(None),
            }
        };
        let checked = |v: Option<i128>| v.ok_or(OVERFLOW).map_err(str::to_owned).and_then(bounded);
        Ok(match self {
            Expr::Num(n) => Some(*n),
            Expr::Var(i) => lookup(*i),
            Expr::Neg(a) => a.eval_by(lookup)?.map(|v| checked(v.checked_neg())).transpose()?,
            Expr::Add(a, b) => two(a, b)?.map(|(x, y)| checked(x.checked_add(y))).transpose()?,
            Expr::Sub(a, b) => two(a, b)?.map(|(x, y)| checked(x.checked_sub(y))).transpose()?,
            Expr::Mul(a, b) => two(a, b)?.map(|(x, y)| checked(x.checked_mul(y))).transpose()?,
            Expr::FloorDiv(a, b) | Expr::CDiv(a, b) => match two(a, b)? {
                Some((_, d)) if d < 1 => return Err(format!("a division by {d}: the divisor must be positive")),
                Some((x, d)) => Some(if matches!(self, Expr::CDiv(..)) { ceil_div(x, d) } else { x.div_euclid(d) }),
                None => None,
            },
        })
    }

    /// Every dimension the expression mentions, once each.
    pub fn vars(&self, out: &mut Vec<usize>) {
        match self {
            Expr::Num(_) => {}
            Expr::Var(i) => {
                if !out.contains(i) {
                    out.push(*i);
                }
            }
            Expr::Neg(a) => a.vars(out),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::FloorDiv(a, b) | Expr::CDiv(a, b) => {
                a.vars(out);
                b.vars(out);
            }
        }
    }

    /// The affine form of the expression: what `_ExprParser._fold` in Python folds it to with the arithmetic of `DimExpr` and
    /// `dims.cdiv`, or `None` where Python has no value for it (two symbolic factors, a `//` that is not exact, a `//` by a divisor that is
    /// not a positive constant).
    ///
    /// The fold is Python's, case by case. A `//` folds a constant dividend (floor division: `-5 // 2` is `-3`) or a dividend whose every
    /// coefficient is a multiple of the divisor. A `cdiv` by a positive constant is the dividend itself (by 1), a constant (`cdiv(-5, 2)` is
    /// `-2`), an exact quotient (`cdiv(4*n, 2)` is `2*n`) or an atom of its own; a `cdiv` by a symbolic divisor is the constant `k` when the
    /// dividend is exactly `k` times the divisor, whatever the signs (`cdiv(0, n)` and `cdiv(0, -n)` are `0`, `cdiv(2*n, -n)` is `-2`), and an
    /// atom otherwise. A `cdiv` by a constant below 1 has no form (Python's `cdiv` raises, and its parser refuses the text before). The atom of a
    /// `cdiv` that does not fold holds the *folded* dividend and divisor, as Python's `_CDiv` does: in `cdiv(M + N - N, 2)` the `N` cancels, so
    /// the atom is `cdiv(M, 2)`, it equals the atom of `cdiv(M, 2)`, and evaluating it needs no value for `N`.
    ///
    /// One divergence remains: Python's integers are unbounded, the fold here is exact in 128 bits. A coefficient or a constant that does not
    /// fit (a product of four 12-digit numbers, `10^48`) has no form here where Python has one; no manifest written by `tilefish` has one.
    pub fn affine(&self) -> Option<Affine> {
        match self {
            Expr::Num(n) => Some(Affine::constant(*n)),
            Expr::Var(i) => Some(Affine::atom(Atom::Var(*i))),
            Expr::Neg(a) => a.affine()?.scaled(-1),
            Expr::Add(a, b) => a.affine()?.plus(&b.affine()?, 1),
            Expr::Sub(a, b) => a.affine()?.plus(&b.affine()?, -1),
            Expr::Mul(a, b) => {
                let (x, y) = (a.affine()?, b.affine()?);
                match (x.as_constant(), y.as_constant()) {
                    (_, Some(k)) => x.scaled(k),
                    (Some(k), _) => y.scaled(k),
                    _ => None,
                }
            }
            Expr::FloorDiv(a, b) => {
                let (x, d) = (a.affine()?, b.affine()?.as_constant()?);
                if d < 1 {
                    return None;
                }
                match x.as_constant() {
                    Some(c) => Some(Affine::constant(c.div_euclid(d))),
                    None => x.exact_div(d),
                }
            }
            Expr::CDiv(a, b) => cdiv(a.affine()?, b.affine()?).ok(),
        }
    }

    /// The value of an expression that mentions no dimension (after affine folding, so `n - n` is `0`).
    pub fn static_value(&self) -> Option<i128> {
        self.affine()?.as_constant()
    }
}

/// A ceiling division of two folded forms, folded as Python's `dims.cdiv` folds one (see [`Expr::affine`]): an error for a constant
/// divisor below 1, as Python's raises.
fn cdiv(x: Affine, y: Affine) -> Result<Affine, String> {
    match y.as_constant() {
        Some(d) if d < 1 => Err(format!("a division by {d}: the divisor must be positive")),
        Some(1) => Ok(x),
        Some(d) => Ok(match x.as_constant() {
            Some(c) => Affine::constant(ceil_div(c, d)),
            None => match x.exact_div(d) {
                Some(q) => q,
                None => Affine::atom(Atom::CDiv(Box::new(x), Box::new(y))),
            },
        }),
        None => Ok(match x.ratio(&y) {
            Some(k) => Affine::constant(k),
            None => Affine::atom(Atom::CDiv(Box::new(x), Box::new(y))),
        }),
    }
}

/// An atom of an affine form: a dimension, or a ceiling division that does not fold (Python's `_CDiv`), of the folded dividend and divisor.
#[derive(Debug, Clone, PartialEq)]
pub enum Atom {
    Var(usize),
    CDiv(Box<Affine>, Box<Affine>),
}

impl Atom {
    /// A text that two atoms share exactly when they are the same atom (an atom of a `cdiv` is known by its folded parts).
    fn key(&self) -> String {
        match self {
            Atom::Var(i) => format!("v{i:04}"),
            Atom::CDiv(n, d) => format!("c({}/{})", n.key(), d.key()),
        }
    }
}

/// `c + sum(coefficient * atom)`, its terms in the order of their atoms' keys (so that one form has one spelling).
#[derive(Debug, Clone, PartialEq)]
pub struct Affine {
    pub c: i128,
    pub terms: Vec<(Atom, i128)>,
}

impl Affine {
    fn constant(c: i128) -> Self {
        Self { c, terms: Vec::new() }
    }

    fn atom(a: Atom) -> Self {
        Self { c: 0, terms: vec![(a, 1)] }
    }

    fn as_constant(&self) -> Option<i128> {
        self.terms.is_empty().then_some(self.c)
    }

    fn key(&self) -> String {
        let mut key = self.c.to_string();
        for (a, c) in &self.terms {
            key.push_str(&format!(";{c}*{}", a.key()));
        }
        key
    }

    fn scaled(&self, k: i128) -> Option<Self> {
        if k == 0 {
            return Some(Self::constant(0));
        }
        let terms =
            self.terms.iter().map(|(a, c)| c.checked_mul(k).map(|c| (a.clone(), c))).collect::<Option<Vec<_>>>()?;
        Some(Self { c: self.c.checked_mul(k)?, terms })
    }

    fn plus(&self, other: &Self, sign: i128) -> Option<Self> {
        let mut merged: BTreeMap<String, (Atom, i128)> = BTreeMap::new();
        for (a, c) in &self.terms {
            merged.insert(a.key(), (a.clone(), *c));
        }
        for (a, c) in &other.terms {
            let add = c.checked_mul(sign)?;
            match merged.get_mut(&a.key()) {
                Some(slot) => slot.1 = slot.1.checked_add(add)?,
                None => {
                    merged.insert(a.key(), (a.clone(), add));
                }
            }
        }
        let terms = merged.into_values().filter(|(_, c)| *c != 0).collect();
        Some(Self { c: self.c.checked_add(other.c.checked_mul(sign)?)?, terms })
    }

    /// The form divided by the constant `d` (at least 2), when every coefficient and the constant are multiples of it (Python's
    /// `_div_exact_int`).
    fn exact_div(&self, d: i128) -> Option<Self> {
        if self.c % d != 0 || self.terms.iter().any(|(_, c)| c % d != 0) {
            return None;
        }
        Some(Self { c: self.c / d, terms: self.terms.iter().map(|(a, c)| (a.clone(), c / d)).collect() })
    }

    /// The integer `k` with `self == k * other`, for an `other` that is not a constant (Python's `DimExpr.exact_div`): the constants and the
    /// coefficient of every atom of either form must all be `k` times the other's.
    fn ratio(&self, other: &Self) -> Option<i128> {
        let mut pairs = Vec::with_capacity(other.terms.len() + self.terms.len() + 1);
        if self.c != 0 || other.c != 0 {
            pairs.push((self.c, other.c));
        }
        for (a, c) in &other.terms {
            pairs.push((self.coefficient_of(a), *c));
        }
        for (a, c) in &self.terms {
            if other.coefficient_of(a) == 0 {
                pairs.push((*c, 0));
            }
        }
        let mut ratio = None;
        for (mine, theirs) in pairs {
            if theirs == 0 {
                if mine != 0 {
                    return None;
                }
                continue;
            }
            if mine.checked_rem(theirs)? != 0 {
                return None;
            }
            let k = mine.checked_div(theirs)?;
            match ratio {
                None => ratio = Some(k),
                Some(r) if r != k => return None,
                Some(_) => {}
            }
        }
        ratio
    }

    fn coefficient_of(&self, atom: &Atom) -> i128 {
        self.terms.iter().find(|(a, _)| a == atom).map_or(0, |(_, c)| *c)
    }

    /// Every dimension the form mentions, once each, those inside the atom of a `cdiv` included (Python's `free_dims`): a dimension that
    /// cancelled is not one of them.
    pub fn vars(&self, out: &mut Vec<usize>) {
        for (atom, _) in &self.terms {
            match atom {
                Atom::Var(i) => {
                    if !out.contains(i) {
                        out.push(*i);
                    }
                }
                Atom::CDiv(n, d) => {
                    n.vars(out);
                    d.vars(out);
                }
            }
        }
    }

    /// The value of the affine form under `dims`, as Python's `eval_dim` of the folded expression gives it: a dimension that cancelled
    /// (`M + N - N`, `cdiv(M + N - N, 2)`) needs no value, and a form that folds to a constant once some of its dimensions are bound has that
    /// value (`cdiv(M + 2*N, N)` is 2 when `M` is 0, whatever `N` is). `None` while it still mentions an unbound dimension; an error for a
    /// `cdiv` by a value below 1, an overflow, or a value that does not fit 64 bits.
    pub fn eval(&self, dims: &[Option<i64>]) -> Result<Option<i64>, String> {
        match self.value(&|i| dims.get(i).copied().flatten().map(i128::from))? {
            None => Ok(None),
            Some(v) => i64::try_from(v).map(Some).map_err(|_| format!("the value {v} does not fit 64 bits")),
        }
    }

    /// The value under `lookup` (the value of each bound dimension): see [`Affine::eval`].
    fn value(&self, lookup: &dyn Fn(usize) -> Option<i128>) -> Result<Option<i128>, String> {
        match self.numeric(None, lookup)? {
            Some(v) => Ok(Some(v)),
            // a dimension is not bound: the form with the bound ones substituted may still fold to a constant, as Python's `subst` does
            None => Ok(self.subst(lookup)?.as_constant()),
        }
    }

    /// The value of the form without the term of the dimension `skip`, when every dimension the rest needs is bound (`None` as soon as one is
    /// not). Nothing is allocated.
    fn numeric(&self, skip: Option<usize>, lookup: &dyn Fn(usize) -> Option<i128>) -> Result<Option<i128>, String> {
        let mut acc = self.c;
        for (atom, c) in &self.terms {
            let value = match atom {
                Atom::Var(i) if Some(*i) == skip => continue,
                Atom::Var(i) => lookup(*i),
                Atom::CDiv(n, d) => match (n.numeric(None, lookup)?, d.numeric(None, lookup)?) {
                    (Some(n), Some(d)) => Some(cdiv_value(n, d)?),
                    _ => None,
                },
            };
            let Some(value) = value else { return Ok(None) };
            acc = c.checked_mul(value).and_then(|p| acc.checked_add(p)).ok_or(OVERFLOW)?;
        }
        Ok(Some(acc))
    }

    /// The form with every bound dimension replaced by its value and folded again (Python's `subst`): an atom of a `cdiv` is folded anew
    /// from its substituted parts.
    fn subst(&self, lookup: &dyn Fn(usize) -> Option<i128>) -> Result<Self, String> {
        let mut acc = Self::constant(self.c);
        for (atom, c) in &self.terms {
            let replaced = match atom {
                Atom::Var(i) => lookup(*i).map_or_else(|| Self::atom(Atom::Var(*i)), Self::constant),
                Atom::CDiv(n, d) => cdiv(n.subst(lookup)?, d.subst(lookup)?)?,
            };
            acc = replaced.scaled(*c).and_then(|r| acc.plus(&r, 1)).ok_or(OVERFLOW)?;
        }
        Ok(acc)
    }

    /// The coefficient of the dimension `var` (0 when it does not occur as an atom of its own).
    pub fn coefficient(&self, var: usize) -> i128 {
        self.coefficient_of(&Atom::Var(var))
    }
}

// -- the parser -------------------------------------------------------------------------------------------------------

struct Parser<'a> {
    what: &'a str,
    text: &'a str,
    toks: Vec<&'a str>,
    i: usize,
    syms: &'a mut Symbols,
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

impl<'a> Parser<'a> {
    fn new(text: &'a str, what: &'a str, syms: &'a mut Symbols) -> Result<Self, String> {
        if text.chars().count() > MAX_LEN {
            return Err(format!("{what} is too long"));
        }
        let stripped = text.trim_end_matches(' ');
        let b = stripped.as_bytes();
        let mut toks = Vec::new();
        let mut pos = 0;
        while pos < b.len() {
            while b.get(pos) == Some(&b' ') {
                pos += 1;
            }
            let start = pos;
            let Some(&first) = b.get(pos) else {
                return Err(format!("{what} {text:?} has a character the dimension grammar does not allow (at {pos})"));
            };
            if first.is_ascii_digit() {
                while pos < b.len() && pos - start < 12 && b[pos].is_ascii_digit() {
                    pos += 1;
                }
            } else if is_ident_start(first) {
                while pos < b.len() && (b[pos].is_ascii_alphanumeric() || b[pos] == b'_') {
                    pos += 1;
                }
            } else if stripped[pos..].starts_with("//") {
                pos += 2;
            } else if matches!(first, b'-' | b'+' | b'*' | b'(' | b')' | b',') {
                pos += 1;
            } else {
                return Err(format!("{what} {text:?} has a character the dimension grammar does not allow (at {pos})"));
            }
            toks.push(&stripped[start..pos]);
        }
        if toks.len() > MAX_TOKENS {
            return Err(format!("{what} is too long"));
        }
        Ok(Self { what, text, toks, i: 0, syms })
    }

    fn peek(&self) -> Option<&'a str> {
        self.toks.get(self.i).copied()
    }

    fn take(&mut self, expected: Option<&str>) -> Result<&'a str, String> {
        let tok = self.peek();
        match tok {
            Some(t) if expected.is_none_or(|e| e == t) => {
                self.i += 1;
                Ok(t)
            }
            _ => Err(format!(
                "{} {:?}: expected {}, got {:?}",
                self.what,
                self.text,
                expected.unwrap_or("more input"),
                tok
            )),
        }
    }

    fn done(&self) -> Result<(), String> {
        match self.peek() {
            None => Ok(()),
            Some(t) => Err(format!("{} {:?}: unexpected {t:?}", self.what, self.text)),
        }
    }

    fn positive_divisor(&self, divisor: &Expr, what: &str) -> Result<(), String> {
        match divisor.static_value() {
            Some(v) if v < 1 => {
                Err(format!("{} {:?}: {what} by {v} (the divisor must be positive)", self.what, self.text))
            }
            _ => Ok(()),
        }
    }

    fn expr(&mut self, depth: usize) -> Result<Expr, String> {
        let mut acc = self.term(depth)?;
        while matches!(self.peek(), Some("+" | "-")) {
            let op = self.take(None)?;
            let rhs = self.term(depth)?;
            acc = if op == "+" {
                Expr::Add(Box::new(acc), Box::new(rhs))
            } else {
                Expr::Sub(Box::new(acc), Box::new(rhs))
            };
        }
        Ok(acc)
    }

    fn term(&mut self, depth: usize) -> Result<Expr, String> {
        let mut acc = self.factor(depth)?;
        while matches!(self.peek(), Some("*" | "//")) {
            let op = self.take(None)?;
            let rhs = self.factor(depth)?;
            acc = if op == "*" {
                Expr::Mul(Box::new(acc), Box::new(rhs))
            } else {
                self.positive_divisor(&rhs, "division")?;
                Expr::FloorDiv(Box::new(acc), Box::new(rhs))
            };
        }
        Ok(acc)
    }

    fn factor(&mut self, depth: usize) -> Result<Expr, String> {
        if depth > MAX_DEPTH {
            return Err(format!("{} is nested too deeply", self.what));
        }
        let tok = self.take(None)?;
        if tok == "-" {
            return Ok(Expr::Neg(Box::new(self.factor(depth + 1)?)));
        }
        if tok.as_bytes()[0].is_ascii_digit() {
            return Ok(Expr::Num(
                tok.parse().map_err(|_| format!("{} {:?}: bad number {tok:?}", self.what, self.text))?,
            ));
        }
        if tok == "(" {
            let inner = self.expr(depth + 1)?;
            self.take(Some(")"))?;
            return Ok(inner);
        }
        if is_ident_start(tok.as_bytes()[0]) {
            if tok == "cdiv" {
                self.take(Some("("))?;
                let num = self.expr(depth + 1)?;
                self.take(Some(","))?;
                let den = self.expr(depth + 1)?;
                self.take(Some(")"))?;
                self.positive_divisor(&den, "cdiv")?;
                return Ok(Expr::CDiv(Box::new(num), Box::new(den)));
            }
            if self.peek() == Some("(") {
                return Err(format!("{} {:?}: only cdiv(a, b) may be called, not {tok}(...)", self.what, self.text));
            }
            return Ok(Expr::Var(self.syms.intern(tok)?));
        }
        Err(format!("{} {:?}: unexpected {tok:?}", self.what, self.text))
    }
}

/// Parse a dimension expression such as `2*S + 1` or `cdiv(n, 128)`, interning its identifiers in `syms`.
pub fn parse_expr(text: &str, what: &str, syms: &mut Symbols) -> Result<Expr, String> {
    let mut p = Parser::new(text, what, syms)?;
    let e = p.expr(0)?;
    p.done()?;
    Ok(e)
}

/// Parse a grid such as `(M,)` or `(cdiv(S, 16), 32)`: one to three entries, a single entry with a trailing comma, no
/// static entry below 1.
pub fn parse_grid(text: &str, syms: &mut Symbols) -> Result<Vec<Expr>, String> {
    let mut p = Parser::new(text, "grid", syms)?;
    p.take(Some("("))?;
    let mut entries = Vec::new();
    let mut trailing = false;
    loop {
        entries.push(p.expr(0)?);
        if p.peek() == Some(",") {
            p.take(None)?;
            if p.peek() == Some(")") {
                trailing = true;
                break;
            }
        } else {
            break;
        }
    }
    p.take(Some(")"))?;
    p.done()?;
    if !(1..=3).contains(&entries.len()) {
        return Err(format!("grid {text:?} must have 1 to 3 entries"));
    }
    if entries.len() == 1 && !trailing {
        return Err(format!("grid {text:?}: a single entry needs a trailing comma, like (n,)"));
    }
    for (position, entry) in entries.iter().enumerate() {
        if let Some(v) = entry.static_value().filter(|v| *v < 1) {
            return Err(format!(
                "grid {text:?}: entry {position} is the static size {v}; grid sizes must be at least 1"
            ));
        }
    }
    Ok(entries)
}

// -- patterns: an expression that is solved against a size -------------------------------------------------------------

/// An expression that a launch binds, checks or evaluates: the shape of a tensor, the dimension a scalar carries, an entry of the grid.
///
/// It is used in its folded form, as Python uses the folded `DimExpr` of a declaration: its dimensions are the ones the folded form mentions
/// (Python's `free_dims`: in `cdiv(M + N - N, 2)` only `M`), and its value is the one the folded form has. An expression that is not affine
/// (`M*N`, which Python never writes, but a host may be given) is evaluated as it is written.
#[derive(Debug, Clone)]
pub struct Pattern {
    pub text: String,
    pub expr: Expr,
    pub vars: Vec<usize>,
    affine: Option<Affine>,
    pub is_static: bool,
}

/// What one pass over an equation did.
#[derive(Debug, PartialEq, Eq)]
pub enum Solved {
    /// Every dimension was bound and the equation holds.
    Checked,
    /// One dimension was bound by it.
    Bound(usize, i128),
    /// It cannot be used yet (two or more unbound dimensions, or the one is not affine in it).
    Stuck,
}

/// Why an equation cannot hold.
#[derive(Debug, PartialEq, Eq)]
pub enum Unsolvable {
    /// A fully bound pattern is another value than the size.
    Mismatch { expected: i128 },
    /// `coefficient * var = target` has no integer solution.
    NotWhole { var: usize, target: i128, coefficient: i128 },
    /// The solution is below the least value the dimension may have.
    NotPositive { var: usize, value: i128 },
    /// The expression cannot be evaluated (a division by a non-positive number, an overflow).
    Eval(String),
}

impl Pattern {
    pub fn new(text: &str, expr: Expr) -> Self {
        let mut vars = Vec::new();
        let affine = expr.affine();
        match &affine {
            Some(a) => a.vars(&mut vars),
            None => expr.vars(&mut vars),
        }
        let is_static = affine.as_ref().is_some_and(|a| a.terms.is_empty());
        Self { text: text.to_owned(), expr, vars, affine, is_static }
    }

    pub fn parse(text: &str, what: &str, syms: &mut Symbols) -> Result<Self, String> {
        Ok(Self::new(text, parse_expr(text, what, syms)?))
    }

    /// The dimension the pattern is, when its folded form is one dimension and nothing else (Python's `isinstance(p.dim, Dim)`).
    pub fn as_var(&self) -> Option<usize> {
        match self.affine.as_ref()? {
            Affine { c: 0, terms } if terms.len() == 1 => match &terms[0] {
                (Atom::Var(i), 1) => Some(*i),
                _ => None,
            },
            _ => None,
        }
    }

    /// The value under `dims` (`None` while it needs a dimension that is not bound): the folded form's (see [`Affine::eval`]), or for an
    /// expression that is not affine, the value of the expression as it is written. A dimension is an integer of 128 bits here, as
    /// wide as a size or a stride of a view can make one (Python's are unbounded): whether a value fits what a kernel takes is checked
    /// where it is given to the kernel.
    pub fn eval(&self, dims: &[Option<i128>]) -> Result<Option<i128>, String> {
        let lookup = |i: usize| dims.get(i).copied().flatten();
        match &self.affine {
            Some(a) => a.value(&lookup),
            None => self.expr.eval_by(&lookup),
        }
    }

    /// Bind or check one dimension against `actual`, as Python's `_solve_one` does: a dimension is at least 1.
    #[cfg(test)]
    pub fn solve(&self, actual: i128, dims: &[Option<i128>]) -> Result<Solved, Unsolvable> {
        self.solve_from(actual, dims, 1)
    }

    /// As [`Pattern::solve`], with the dimension it binds at least `minimum`: 1 for a size, 0 for a stride (the stride of an expanded view
    /// is 0, as Python's `_solve_one` allows for the equations of a stride).
    pub fn solve_from(&self, actual: i128, dims: &[Option<i128>], minimum: i128) -> Result<Solved, Unsolvable> {
        let mut unbound = self.vars.iter().copied().filter(|v| dims.get(*v).copied().flatten().is_none());
        let Some(var) = unbound.next() else {
            return match self.eval(dims).map_err(Unsolvable::Eval)? {
                Some(expected) if expected == actual => Ok(Solved::Checked),
                Some(expected) => Err(Unsolvable::Mismatch { expected }),
                None => Ok(Solved::Stuck),
            };
        };
        if unbound.next().is_some() {
            return Ok(Solved::Stuck);
        }
        let Some(affine) = &self.affine else { return Ok(Solved::Stuck) };
        let coefficient = affine.coefficient(var);
        if coefficient == 0 {
            return Ok(Solved::Stuck); // the dimension only occurs inside a cdiv
        }
        // the rest of the form (Python's `pattern - var * coef`), which must have a value with `var` unbound
        let lookup = |i: usize| dims.get(i).copied().flatten();
        let base = match affine.numeric(Some(var), &lookup).map_err(Unsolvable::Eval)? {
            Some(base) => base,
            None => {
                let mut rest = affine.clone();
                rest.terms.retain(|(a, _)| *a != Atom::Var(var));
                match rest.value(&lookup).map_err(Unsolvable::Eval)? {
                    Some(base) => base,
                    None => return Ok(Solved::Stuck),
                }
            }
        };
        let overflow = || Unsolvable::Eval(OVERFLOW.to_owned());
        let target = actual.checked_sub(base).ok_or_else(overflow)?;
        if target.checked_rem(coefficient).ok_or_else(overflow)? != 0 {
            return Err(Unsolvable::NotWhole { var, target, coefficient });
        }
        let value = target.checked_div(coefficient).ok_or_else(overflow)?;
        if value < minimum {
            return Err(Unsolvable::NotPositive { var, value });
        }
        Ok(Solved::Bound(var, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(text: &str, names: &[&str], values: &[Option<i64>]) -> Result<Option<i64>, String> {
        let mut syms = Symbols::default();
        for n in names {
            syms.intern(n).unwrap();
        }
        parse_expr(text, "expression", &mut syms)?.eval(values)
    }

    #[test]
    fn evaluates_like_python() {
        let m = &["M", "n"];
        assert_eq!(eval("3*M + 1", m, &[Some(4), None]), Ok(Some(13)));
        assert_eq!(eval("cdiv(n, 1024)", m, &[None, Some(1025)]), Ok(Some(2)));
        assert_eq!(eval("cdiv(n, 1024)", m, &[None, Some(1024)]), Ok(Some(1)));
        assert_eq!(eval("n // 4", m, &[None, Some(10)]), Ok(Some(2)));
        assert_eq!(eval("-M + 10", m, &[Some(4), None]), Ok(Some(6)));
        assert_eq!(eval("(M + 1) * 2 - n", m, &[Some(1), Some(1)]), Ok(Some(3)));
        assert_eq!(eval("M", m, &[None, None]), Ok(None));
        assert_eq!(eval("2 * 3 // 2", m, &[None, None]), Ok(Some(3)));
        assert!(eval("cdiv(M, n)", m, &[Some(4), Some(0)]).is_err());
    }

    #[test]
    fn refuses_what_python_refuses() {
        for text in [
            "n // 0",
            "cdiv(n, 0)",
            "cdiv(n, -2)",
            "n / 2",
            "n % 2",
            "2 ** 3",
            "foo(n)",
            "n n",
            "(n",
            "n)",
            "1 +",
            "__import__('os')",
            "n.real",
            "1234567890123",
        ] {
            let mut syms = Symbols::default();
            assert!(parse_expr(text, "expression", &mut syms).is_err(), "{text}");
        }
        let mut syms = Symbols::default();
        assert!(parse_expr(&"(".repeat(30), "expression", &mut syms).is_err());
        assert!(parse_expr(&"n + ".repeat(300), "expression", &mut syms).is_err());
    }

    #[test]
    fn grids() {
        let mut s = Symbols::default();
        assert_eq!(parse_grid("(M,)", &mut s).unwrap().len(), 1);
        assert_eq!(parse_grid("(cdiv(S, 16), 32)", &mut s).unwrap().len(), 2);
        assert_eq!(parse_grid("(3*M, 1, 1)", &mut s).unwrap().len(), 3);
        for text in ["(M)", "()", "(0,)", "(-1,)", "(n - n, 4)", "(1, 2, 3, 4)", "M", "(M,", "(M,) x"] {
            assert!(parse_grid(text, &mut s).is_err(), "{text}");
        }
    }

    fn pattern(text: &str, names: &[&str]) -> Pattern {
        let mut syms = Symbols::default();
        for n in names {
            syms.intern(n).unwrap();
        }
        Pattern::parse(text, "pattern", &mut syms).unwrap()
    }

    #[test]
    fn solves_affine_equations() {
        let p = pattern("3072*M", &["M"]);
        assert_eq!(p.solve(6144, &[None]), Ok(Solved::Bound(0, 2)));
        assert_eq!(p.solve(6145, &[None]), Err(Unsolvable::NotWhole { var: 0, target: 6145, coefficient: 3072 }));
        assert_eq!(p.solve(0, &[None]), Err(Unsolvable::NotPositive { var: 0, value: 0 }));
        assert_eq!(p.solve(6144, &[Some(2)]), Ok(Solved::Checked));
        assert_eq!(p.solve(6144, &[Some(3)]), Err(Unsolvable::Mismatch { expected: 9216 }));
        let q = pattern("2*M + N + 1", &["M", "N"]);
        assert_eq!(q.solve(10, &[None, None]), Ok(Solved::Stuck));
        assert_eq!(q.solve(10, &[None, Some(3)]), Ok(Solved::Bound(0, 3)));
        let r = pattern("cdiv(M, 2)", &["M"]);
        assert_eq!(r.solve(5, &[None]), Ok(Solved::Stuck)); // inside a cdiv only
        assert_eq!(r.solve(5, &[Some(9)]), Ok(Solved::Checked));
        let s = pattern("1000", &[]);
        assert!(s.is_static);
        assert_eq!(s.solve(1000, &[]), Ok(Solved::Checked));
        assert_eq!(s.solve(999, &[]), Err(Unsolvable::Mismatch { expected: 1000 }));
    }

    #[test]
    fn a_value_that_does_not_fit_is_an_error_and_never_wraps() {
        let m = &["M", "N"];
        let (big, huge) = (i64::MAX, 1i64 << 60);
        // 64 bits: the sum of two big dimensions, and a negative one, are exact in 128 bits and refused as values
        let too_big = |r: Result<Option<i64>, String>| r.expect_err("it does not fit").contains("does not fit 64 bits");
        assert!(too_big(eval("M + N", m, &[Some(big), Some(big)])));
        assert!(too_big(eval("-M - N", m, &[Some(big), Some(big)])));
        assert_eq!(eval("M + N", m, &[Some(big - 1), Some(1)]), Ok(Some(big)));
        assert!(too_big(eval("M + N", m, &[Some(big), Some(1)])));
        assert_eq!(eval("-M - 1", m, &[Some(big), None]), Ok(Some(i64::MIN)));
        assert!(too_big(eval("-M - 2", m, &[Some(big), None])));
        // 2^60 shapes: a product of two of them is a 120-bit number, which is bounded (2^100), and the message says so
        let overflows = |r: Result<Option<i64>, String>| r.expect_err("it overflows").contains("overflows");
        assert!(overflows(eval("M * N", m, &[Some(huge), Some(huge)])));
        assert!(overflows(eval("M * N + 1", m, &[Some(huge), Some(huge)])));
        assert!(overflows(eval("M * N - 1", m, &[Some(huge), Some(huge)])));
        // the bound is 2^100 itself: 2^50 * 2^50 is within it (and then too big for 64 bits), one more is not
        assert!(too_big(eval("M * N", m, &[Some(1 << 50), Some(1 << 50)])));
        assert!(overflows(eval("M * N", m, &[Some(1 << 50), Some((1 << 50) + 1)])));
        assert!(too_big(eval("M + N", m, &[Some(1 << 62), Some(1 << 62)])));
        // a product of products that is beyond even 128 bits is the same refusal
        assert!(overflows(eval("(M * N) * (M * N)", m, &[Some(1 << 40), Some(1 << 40)])));
        // products that stay small are right, and a bound dimension of 2^60 alone is fine
        assert_eq!(eval("M * 4 - M * 3", m, &[Some(huge), None]), Ok(Some(huge)));
        assert_eq!(eval("M // 2 + cdiv(N, 2)", m, &[Some(huge), Some(huge)]), Ok(Some(huge)));
    }

    #[test]
    fn an_equation_that_overflows_is_refused_and_one_at_the_edge_is_solved() {
        // coefficients are folded in 128 bits: 10^12 * 10^12 is 10^24, and a bound dimension of 2^62 times it overflows
        let q = pattern("999999999999*999999999999*M + N", &["M", "N"]);
        let e = q.solve(5, &[Some(1 << 62), None]).unwrap_err();
        assert_eq!(e, Unsolvable::Eval("a dimension expression overflows".to_owned()));
        // a coefficient, a target or a solution beyond 64 bits is exact, as Python's integers are (its messages show them as they are)
        let big = 999_999_999_998_000_000_000_001i128; // 999999999999 squared
        let r = pattern("999999999999*999999999999*N", &["N"]);
        assert_eq!(r.solve(5, &[None]), Err(Unsolvable::NotWhole { var: 0, target: 5, coefficient: big }));
        let up = pattern("2*N - 999999999999*999999999999", &["N"]);
        assert_eq!(up.solve(0, &[None]), Err(Unsolvable::NotWhole { var: 0, target: big, coefficient: 2 }));
        let down = pattern("N + 999999999999*999999999999", &["N"]);
        assert_eq!(down.solve(5, &[None]), Err(Unsolvable::NotPositive { var: 0, value: 5 - big }));
        // a dimension of 80 bits is bound (Python binds it too): whether it fits what a kernel takes is checked where it is given one
        let wide = pattern("N - 999999999999*999999999999", &["N"]);
        assert_eq!(wide.solve(5, &[None]), Ok(Solved::Bound(0, 5 + big)));
        // the edges of a size: 2^63 - 1 and 2^64 - 1 (one-byte elements) are dimensions
        let n = pattern("N", &["N"]);
        assert_eq!(n.solve(i64::MAX.into(), &[None]), Ok(Solved::Bound(0, i64::MAX.into())));
        assert_eq!(n.solve(u64::MAX.into(), &[None]), Ok(Solved::Bound(0, u64::MAX.into())));
        assert_eq!(n.solve(1, &[None]), Ok(Solved::Bound(0, 1)));
        assert_eq!(n.solve(0, &[None]), Err(Unsolvable::NotPositive { var: 0, value: 0 }));
        assert_eq!(n.solve(-1, &[None]), Err(Unsolvable::NotPositive { var: 0, value: -1 }));
        // an evaluation that overflows while checking a fully bound pattern is an Eval error, not a mismatch
        let both = pattern("M * N", &["M", "N"]);
        assert!(
            matches!(both.solve(7, &[Some(1 << 60), Some(1 << 60)]), Err(Unsolvable::Eval(m)) if m.contains("overflows"))
        );
        assert_eq!(both.solve(6, &[Some(2), Some(3)]), Ok(Solved::Checked));
    }

    /// What an expression folds to: a constant, a form over these dimensions (sorted), no form (not affine), or a text the parser refuses.
    #[derive(Debug, PartialEq)]
    enum Folded {
        Constant(i128),
        Over(Vec<String>),
        NotAffine,
        Refused,
    }

    fn folded(text: &str) -> Folded {
        let mut syms = Symbols::default();
        let Ok(e) = parse_expr(text, "e", &mut syms) else { return Folded::Refused };
        match e.affine() {
            None => Folded::NotAffine,
            Some(a) => match a.as_constant() {
                Some(c) => Folded::Constant(c),
                None => {
                    let mut vars = Vec::new();
                    a.vars(&mut vars);
                    let mut names: Vec<String> = vars.iter().map(|v| syms.name(*v).to_owned()).collect();
                    names.sort();
                    Folded::Over(names)
                }
            },
        }
    }

    #[test]
    fn cdiv_and_floor_division_fold_as_python_folds_them() {
        use Folded::{Constant, NotAffine, Over, Refused};
        let over = |names: &[&str]| Over(names.iter().map(|n| (*n).to_owned()).collect());
        // what `_ExprParser(text).expr()` gives in Python (its value, or its `free_dims`), with `_fold`'s `DimExpr` arithmetic and `cdiv`
        let python = [
            // a dividend that is a multiple of a symbolic divisor is that multiple, of either sign: `cdiv(0, x)` is 0
            ("cdiv(0, n)", Constant(0)),
            ("cdiv(0, -n)", Constant(0)),
            ("cdiv(0*n, m)", Constant(0)),
            ("cdiv(m - m, n)", Constant(0)),
            ("cdiv(0, n) + 1", Constant(1)),
            ("cdiv(2*n, n)", Constant(2)),
            ("cdiv(2*n, -n)", Constant(-2)),
            ("cdiv(-2*n, n)", Constant(-2)),
            ("cdiv(n, -n)", Constant(-1)),
            ("cdiv(n+m, n+m)", Constant(1)),
            ("cdiv(2*n+2, n+1)", Constant(2)),
            ("cdiv(3*n, 2*n)", over(&["n"])),
            ("cdiv(n+1, 2*n+2)", over(&["n"])),
            ("cdiv(-4, n)", over(&["n"])),
            ("cdiv(-5, n)", over(&["n"])),
            // a constant divisor: the dividend (by 1), a constant (rounded up, negative dividends included), an exact quotient, or an atom
            ("cdiv(n, 1)", over(&["n"])),
            ("cdiv(n, n - n + 1)", over(&["n"])),
            ("cdiv(0, 5)", Constant(0)),
            ("cdiv(-5, 2)", Constant(-2)),
            ("cdiv(-6, 3)", Constant(-2)),
            ("cdiv(n - n - 4, 3)", Constant(-1)),
            ("cdiv(4*n, 2)", over(&["n"])),
            ("cdiv(-n, 2)", over(&["n"])),
            // a divisor that folds to a constant below 1 is refused by the parser, a `//` by a symbolic divisor has no form
            ("cdiv(n, 2 - 4)", Refused),
            ("cdiv(n, cdiv(0, m))", Refused),
            ("n // -n", NotAffine),
            ("-5 // 2", Constant(-3)),
            ("0 // 5", Constant(0)),
            ("-0 // 5", Constant(0)),
            ("(n - n) // 3", Constant(0)),
            // the atom of a cdiv holds its folded parts: a name that cancels in it is not one of its dimensions, and two spellings are one atom
            ("cdiv(M + N - N, 2)", over(&["M"])),
            ("cdiv(M + N - N, 2) - cdiv(M, 2)", Constant(0)),
            ("cdiv(M, 2) + cdiv(M+0, 2)", over(&["M"])),
        ];
        for (text, want) in python {
            assert_eq!(folded(text), want, "{text}");
        }
        let mut s = Symbols::default();
        let twice = parse_expr("cdiv(M, 2) + cdiv(M+0, 2)", "e", &mut s).unwrap().affine().unwrap();
        assert_eq!(twice.terms.len(), 1);
        assert_eq!(twice.terms[0].1, 2, "2*cdiv(M, 2)");
        // and the grids Python's `parse_grid` refuses for a static entry below 1 once folded
        for (grid, ok) in [
            ("(cdiv(0, n),)", false),
            ("(cdiv(0, -n) + 1,)", true),
            ("(cdiv(n, -n) + 1,)", false),
            ("(cdiv(n, -n) + 2,)", true),
            ("(cdiv(M + N - N, 2) - cdiv(M, 2),)", false),
            ("(cdiv(2*n, -n) + 3,)", true),
        ] {
            assert_eq!(parse_grid(grid, &mut Symbols::default()).is_ok(), ok, "{grid}");
        }
    }

    #[test]
    fn a_folded_form_is_evaluated_as_python_substitutes_into_it() {
        let names = &["M", "N"];
        let value = |text: &str, values: &[Option<i128>]| pattern(text, names).eval(values);
        // a name that cancels inside a cdiv needs no value (Python's `eval_dim` of `cdiv(M, 2)`)
        assert_eq!(value("cdiv(M + N - N, 2)", &[Some(5), None]), Ok(Some(3)));
        assert_eq!(pattern("cdiv(M + N - N, 2)", names).vars, [0]);
        // a form that folds to a constant once some dimensions are bound has that value: `cdiv(M + 2*N, N)` is 2 when M is 0
        assert_eq!(value("cdiv(M + 2*N, N)", &[Some(0), None]), Ok(Some(2)));
        assert_eq!(value("cdiv(M + 2*N, N)", &[Some(1), None]), Ok(None));
        assert_eq!(value("cdiv(M, N)", &[Some(0), None]), Ok(Some(0)));
        // a cdiv that folded to a constant has its value whatever the sign of the divisor would be; one that did not is refused for it
        assert_eq!(value("cdiv(0, -N)", &[None, Some(3)]), Ok(Some(0)));
        assert_eq!(value("cdiv(2*N, -N)", &[None, Some(3)]), Ok(Some(-2)));
        assert!(value("cdiv(M, -N)", &[Some(4), Some(2)]).unwrap_err().contains("must be positive"));
        assert!(value("cdiv(M, N)", &[Some(4), Some(0)]).unwrap_err().contains("must be positive"));
        // solving: Python binds M from the first equation and checks `cdiv(M, 2)` against the second, N never needed
        let p = pattern("cdiv(M + N - N, 2)", names);
        assert_eq!(p.solve(3, &[Some(5), None]), Ok(Solved::Checked));
        assert_eq!(p.solve(4, &[Some(5), None]), Err(Unsolvable::Mismatch { expected: 3 }));
        assert_eq!(p.solve(3, &[None, None]), Ok(Solved::Stuck), "M only occurs inside the cdiv");
        let q = pattern("M + cdiv(N + M - M, 2)", names);
        assert_eq!(q.solve(10, &[None, Some(7)]), Ok(Solved::Bound(0, 6)));
        // a cdiv of a negative constant by a dimension rounds up, as Python's does
        assert_eq!(value("cdiv(-5, N)", &[None, Some(2)]), Ok(Some(-2)));
        assert_eq!(value("cdiv(-5, N)", &[None, Some(5)]), Ok(Some(-1)));
    }

    #[test]
    fn folds_to_affine_forms() {
        let mut s = Symbols::default();
        let e = parse_expr("n - n + 3", "e", &mut s).unwrap();
        assert_eq!(e.static_value(), Some(3));
        assert_eq!(parse_expr("cdiv(4*n, 2)", "e", &mut s).unwrap().affine().unwrap().coefficient(0), 2);
        assert!(parse_expr("n * n", "e", &mut s).unwrap().affine().is_none());
        assert!(parse_expr("n // 2", "e", &mut s).unwrap().affine().is_none());
        assert_eq!(parse_expr("(2*n) // 2", "e", &mut s).unwrap().affine().unwrap().coefficient(0), 1);
    }
}
