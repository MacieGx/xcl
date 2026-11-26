//! Diagnostic (`XCL_CLAMP_RUNS=1`): how many two-pass mesh ExecuteIndirects follow each other on a
//! list with nothing of the title's between them that could write their records or counts. Each
//! of them has a clamp dispatch of its own today (`indirect_clamp`); one clamp could bound a whole
//! run. Two measures: a run ended only by a write of the title on the list (a dispatch, any
//! other ExecuteIndirect, a memory copy / fill / write), and one also ended by any barrier
//! call of the title (the console's "ordered after" mark, whatever it names).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_CLAMP_RUNS").is_ok_and(|v| v.trim() == "1"))
}

/// run lengths 1, 2-3, 4-7, 8-15, 16-63, 64+
const BUCKETS: usize = 6;

fn bucket(len: u32) -> usize {
    match len {
        0..=1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        8..=15 => 3,
        16..=63 => 4,
        _ => 5,
    }
}

/// [by writes only, by writes and barriers]: runs per bucket, then the calls in all runs
static RUNS: [[AtomicU64; BUCKETS + 1]; 2] = [const { [const { AtomicU64::new(0) }; BUCKETS + 1] }; 2];

/// The list a thread records two-pass calls on and the length of its open run, per measure.
#[derive(Default)]
struct Open {
    list: usize,
    len: [u32; 2],
}

impl Open {
    /// A two-pass call on `list`: the runs that end with it (a call on another list ends both).
    fn call(&mut self, list: usize) -> [u32; 2] {
        let ended = if self.list != list { std::mem::take(&mut self.len) } else { [0; 2] };
        self.list = list;
        self.len[0] += 1;
        self.len[1] += 1;
        ended
    }

    /// Something on `list` ends its open run of measure 1 (`barrier`) or of both.
    fn end(&mut self, list: usize, barrier: bool) -> [u32; 2] {
        if self.list != list {
            return [0; 2];
        }
        let mut ended = [0; 2];
        ended[1] = std::mem::take(&mut self.len[1]);
        if !barrier {
            ended[0] = std::mem::take(&mut self.len[0]);
        }
        ended
    }
}

thread_local! {
    static OPEN: RefCell<Open> = RefCell::new(Open::default());
}

fn count(ended: [u32; 2]) {
    for (k, &len) in ended.iter().enumerate() {
        if len > 0 {
            RUNS[k][bucket(len)].fetch_add(1, Ordering::Relaxed);
            RUNS[k][BUCKETS].fetch_add(u64::from(len), Ordering::Relaxed);
        }
    }
}

/// A two-pass mesh ExecuteIndirect is recorded on `list`.
pub(super) fn call(list: usize) {
    if !on() {
        return;
    }
    count(OPEN.with(|o| o.borrow_mut().call(list)));
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if n % 16384 == 0 {
        let line = |k: usize| {
            let b: Vec<u64> = (0..BUCKETS).map(|i| RUNS[k][i].load(Ordering::Relaxed)).collect();
            let runs: u64 = b.iter().sum();
            format!("{runs} runs, {:.1} calls a run, runs of 1 / 2-3 / 4-7 / 8-15 / 16-63 / 64+ calls: {b:?}", RUNS[k][BUCKETS].load(Ordering::Relaxed) as f64 / runs.max(1) as f64)
        };
        crate::xlog_info!("D3D12.X: clamp runs, {n} two-pass ExecuteIndirects so far: ended by the title's writes: {}; by its writes and barriers: {}", line(0), line(1));
    }
}

/// The title records something on `list` that can write records or counts.
pub(super) fn write(list: usize) {
    if on() {
        count(OPEN.with(|o| o.borrow_mut().end(list, false)));
    }
}

/// The title records a barrier on `list`.
pub(super) fn barrier(list: usize) {
    if on() {
        count(OPEN.with(|o| o.borrow_mut().end(list, true)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_ends_at_a_write_and_the_strict_one_at_a_barrier_too() {
        let mut o = Open::default();
        assert_eq!(o.call(1), [0, 0]);
        assert_eq!(o.call(1), [0, 0]);
        // a barrier ends the strict run only
        assert_eq!(o.end(1, true), [0, 2]);
        assert_eq!(o.call(1), [0, 0]);
        // a write ends both: 3 calls since the last write, 1 since the barrier
        assert_eq!(o.end(1, false), [3, 1]);
        // nothing open: nothing ends
        assert_eq!(o.end(1, false), [0, 0]);
    }

    #[test]
    fn another_list_ends_the_runs_and_its_writes_end_none() {
        let mut o = Open::default();
        o.call(1);
        o.call(1);
        assert_eq!(o.end(2, false), [0, 0]);
        assert_eq!(o.call(2), [2, 2]);
        assert_eq!(o.end(2, false), [1, 1]);
    }

    #[test]
    fn lengths_fall_into_their_buckets() {
        assert_eq!([1, 2, 3, 4, 7, 8, 15, 16, 63, 64, 5000].map(bucket), [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    }
}
