//! Diagnostic: where the time of a title's submit (ExecuteCommandLists) goes, as averages over the
//! last 1024 of them (logged; a few relaxed adds a submit).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Cumulative microseconds by stage, and the submits counted.
static SUM: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
const LOCK: usize = 0;
const REFRESH: usize = 1;
const DRIVER: usize = 2;
const AFTER: usize = 3;
const TOTAL: usize = 4;
const N: usize = 5;
const FILTER: usize = 6;
const TEXTURES: usize = 7;
const LOOKS: usize = 8;

pub(super) fn add(stage: usize, d: Duration) {
    SUM[stage].fetch_add(d.as_micros() as u64, Ordering::Relaxed);
}

/// the wait for the refresh lock, and the refresh itself
pub(super) fn lock_wait(d: Duration) {
    add(LOCK, d);
}

pub(super) fn refresh(d: Duration) {
    add(REFRESH, d);
}

/// the PC ExecuteCommandLists of a chunk of the title's lists, and everything XCL does after it
pub(super) fn driver(d: Duration) {
    add(DRIVER, d);
}

pub(super) fn after(d: Duration) {
    add(AFTER, d);
}

/// parts of the refresh beside those `sparse` counts: finding the textures on the written pages
/// (and the pending ones), and their compares and uploads
pub(super) fn filter(d: Duration) {
    add(FILTER, d);
}

pub(super) fn textures(d: Duration) {
    add(TEXTURES, d);
}

/// A submit asked the write watch (not one a look before it covered, `look_join`).
pub(super) fn looked() {
    SUM[LOOKS].fetch_add(1, Ordering::Relaxed);
}

/// A whole submit (from the hook's entry) is done.
pub(super) fn done(t0: Instant) {
    add(TOTAL, t0.elapsed());
    let n = SUM[N].fetch_add(1, Ordering::Relaxed) + 1;
    if n % 1024 != 0 {
        return;
    }
    struct Last {
        sum: [u64; 9],
        parts: [u64; 4],
        at: Instant,
    }
    static LAST: Mutex<Option<Last>> = Mutex::new(None);
    let mut g = LAST.lock().unwrap();
    let now: [u64; 9] = std::array::from_fn(|i| SUM[i].load(Ordering::Relaxed));
    // [look, STATE lock wait, chunk scan, upload] µs inside the refreshes
    let parts = super::sparse::timing_totals();
    if let Some(l) = g.as_ref() {
        let d = |i: usize| now[i].saturating_sub(l.sum[i]) as f64 / 1024.0 / 1000.0;
        let p = |i: usize| parts[i].saturating_sub(l.parts[i]) as f64 / 1024.0 / 1000.0;
        let refresh = d(REFRESH);
        crate::xlog_info!(
            "submit cost, average of 1024 ExecuteCommandLists: {:.2} ms in all = {:.2} refresh lock wait + {:.2} refresh (look {:.2} in {} looks, sparse lock {:.2}, scan {:.2}, upload {:.2}, texture filter {:.2}, textures {:.2}, rest {:.2}) + {:.2} driver + {:.2} after ({:.0} submits/s)",
            d(TOTAL),
            d(LOCK),
            refresh,
            p(0),
            now[LOOKS].saturating_sub(l.sum[LOOKS]),
            p(1),
            p(2),
            p(3),
            d(FILTER),
            d(TEXTURES),
            refresh - p(0) - p(1) - p(2) - p(3) - d(FILTER) - d(TEXTURES),
            d(DRIVER),
            d(AFTER),
            1024.0 / l.at.elapsed().as_secs_f64().max(1e-3)
        );
    }
    *g = Some(Last { sum: now, parts, at: Instant::now() });
}
