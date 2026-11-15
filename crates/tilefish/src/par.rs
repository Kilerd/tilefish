//! The start-up work of a plan set done on several threads: loading its plans, loading the binaries and GEMMs they launch, instantiating them.
//!
//! [`map`] applies a function to every item of a slice on up to [`threads`] scoped threads and returns the results **in the order of the items**;
//! the error it returns is the one of the **first item, in that order**, that failed, so a refusal is the one a loop over the items would have
//! given (the other items may have been tried too). Few items run on the calling thread alone.

use std::num::NonZeroUsize;

use crate::error::Result;

/// The most threads a start-up step uses.
const MAX_THREADS: usize = 8;
/// Fewer items than this run on the calling thread.
const MIN_ITEMS: usize = 4;

/// The threads a start-up step uses: the machine's parallelism, at most [`MAX_THREADS`].
pub(crate) fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get).clamp(1, MAX_THREADS)
}

/// `f` of every item, on up to [`threads`] threads, in the order of the items; the first error in that order.
pub(crate) fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> Result<R> + Sync) -> Result<Vec<R>> {
    let n = threads().min(items.len());
    if n <= 1 || items.len() < MIN_ITEMS {
        return items.iter().map(&f).collect();
    }
    let f = &f;
    // thread k takes the items k, k + n, k + 2n, ...: neighbours (often the cheap and the costly ones of a set) go to different threads
    let parts: Vec<Vec<Result<R>>> = std::thread::scope(|scope| {
        let handles: Vec<_> =
            (0..n).map(|k| scope.spawn(move || items.iter().skip(k).step_by(n).map(f).collect::<Vec<_>>())).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))).collect()
    });
    let mut parts: Vec<std::vec::IntoIter<Result<R>>> = parts.into_iter().map(Vec::into_iter).collect();
    (0..items.len()).map(|i| parts[i % n].next().expect("one result per item")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn the_results_are_in_the_order_of_the_items() {
        let items: Vec<u32> = (0..1000).collect();
        assert_eq!(map(&items, |i| Ok(i * 2)).unwrap(), items.iter().map(|i| i * 2).collect::<Vec<_>>());
        assert_eq!(map(&items[..2], |i| Ok(*i)).unwrap(), vec![0, 1]);
        assert!(map(&items[..0], |i| Ok(*i)).unwrap().is_empty());
    }

    #[test]
    fn the_error_is_the_first_in_the_order_of_the_items() {
        let items: Vec<u32> = (0..100).collect();
        let e = map(&items, |i| if i % 7 == 3 { Err(Error::Plan(format!("item {i}"))) } else { Ok(*i) }).unwrap_err();
        assert_eq!(e.to_string(), Error::Plan("item 3".to_owned()).to_string());
    }
}
