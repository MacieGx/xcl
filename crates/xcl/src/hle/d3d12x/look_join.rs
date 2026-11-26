//! One look for CPU writes for several of the title's submits (`XCL_LOOK_JOIN`).
//!
//! Every ExecuteCommandLists of the title is behind a look at the write watch of all the memory
//! the GPU reads (`title_textures::refresh`): its time goes with the memory watched, not with what
//! was written, and a title that submits its lists one by one pays it for each (a retail title's
//! world: ~43 submits a frame, 6.5 ms a look over ~6.5 GiB of tiles). What a list's GPU work reads
//! was written by the time the title closed the list, so a look that began after every list of a
//! submit was closed has taken it, and that submit can go without a look of its own. The rule is
//! `XCL_LOOK_SKIP`'s of the maintainer's nvidia-dev branch; here it is also bounded in time: a look
//! covers the submits of the next `XCL_LOOK_JOIN_MS` milliseconds only (4 unless given, 0: no
//! bound), so what a title writes for a list it has closed already is at most that late.
//!
//! `XCL_LOOK_JOIN=1`: such submits go without a look. `XCL_LOOK_JOIN=0`: every submit looks, and
//! the log says how many could have gone without and how many written pages their looks found
//! (what joining would have left to the next look). Unset: nothing is counted.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Off,
    Measure,
    Join,
}

fn mode_of(v: Option<&str>) -> Mode {
    match v.map(str::trim) {
        Some("1") => Mode::Join,
        Some("0") => Mode::Measure,
        _ => Mode::Off,
    }
}

fn mode() -> Mode {
    static M: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *M.get_or_init(|| mode_of(std::env::var("XCL_LOOK_JOIN").ok().as_deref()))
}

/// how long after a look ended a submit may still go without its own (µs; 0: no bound)
fn window_us() -> u64 {
    static W: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("XCL_LOOK_JOIN_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4) * 1000)
}

/// microseconds since the first call, never 0
fn now_us() -> u64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64 + 1
}

/// title list -> when the title closed it (`now_us`)
static CLOSED: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);
/// when the last look began and ended (`now_us`; 0: none yet, or one is under way)
static LOOK: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// when the hot-block walk last found a written block (`XCL_WATCH_HOT`): no look before took it
static WALK_FOUND: AtomicU64 = AtomicU64::new(0);

/// Whether anything is counted at all (the switch is set).
pub(super) fn on() -> bool {
    mode() != Mode::Off
}

/// The title closed `list` (its own pointer): what the list's GPU work reads was written by now.
pub(super) fn note_list_closed(list: usize) {
    if on() {
        CLOSED.lock().unwrap().get_or_insert_with(Default::default).insert(list, now_us());
    }
}

/// The walk of the watched memory found a block written that no look takes yet.
pub(super) fn note_walk_found() {
    if on() {
        WALK_FOUND.store(now_us(), Ordering::Release);
    }
}

/// A look begins (before the watched ranges are asked for: memory new after that is not in it).
pub(super) fn look_begins() {
    if on() {
        LOOK[1].store(0, Ordering::Release);
        LOOK[0].store(now_us(), Ordering::Release);
    }
}

/// Whether a look that began at `began` and ended at `ended` took all that submits of lists
/// closed at `closed` read (None: a list not seen closing), asked at `now`.
fn covers(began: u64, ended: u64, walk_found: u64, now: u64, window: u64, closed: impl IntoIterator<Item = Option<u64>>) -> bool {
    if began == 0 || ended < began {
        return false;
    }
    // a block the walk found written since: the look did not take it
    if walk_found >= began {
        return false;
    }
    if window != 0 && now.saturating_sub(ended) > window {
        return false;
    }
    let mut any = false;
    for t in closed {
        any = true;
        if !t.is_some_and(|t| t < began) {
            return false;
        }
    }
    any
}

/// Whether the last look took what the title's submit of `lists` (its own pointers) reads. Asked
/// with the refresh lock held: the look before is done and its uploads are submitted.
pub(super) fn covered(lists: &[*mut std::ffi::c_void]) -> bool {
    if !on() || lists.is_empty() {
        return false;
    }
    let g = CLOSED.lock().unwrap();
    let Some(m) = g.as_ref() else { return false };
    covers(
        LOOK[0].load(Ordering::Acquire),
        LOOK[1].load(Ordering::Acquire),
        WALK_FOUND.load(Ordering::Acquire),
        now_us(),
        window_us(),
        lists.iter().map(|&l| m.get(&(l as usize)).copied()),
    )
}

/// Whether a submit the last look covers goes without a look of its own.
pub(super) fn joins() -> bool {
    mode() == Mode::Join
}

/// submits, covered ones, looks, µs in looks, pages looks found, pages the looks of covered
/// submits found, µs in the looks of covered submits
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// A submit's refresh is done: `covered` by the look before it, with a look of its own that
/// took `look` (time, written pages) or without one.
pub(super) fn submit_done(covered: bool, look: Option<(std::time::Duration, usize)>) {
    if !on() {
        return;
    }
    if let Some((t, pages)) = look {
        LOOK[1].store(now_us(), Ordering::Release);
        STATS[2].fetch_add(1, Ordering::Relaxed);
        STATS[3].fetch_add(t.as_micros() as u64, Ordering::Relaxed);
        STATS[4].fetch_add(pages as u64, Ordering::Relaxed);
        if covered {
            STATS[5].fetch_add(pages as u64, Ordering::Relaxed);
            STATS[6].fetch_add(t.as_micros() as u64, Ordering::Relaxed);
        }
    }
    let c = STATS[1].fetch_add(covered as u64, Ordering::Relaxed) + covered as u64;
    let n = STATS[0].fetch_add(1, Ordering::Relaxed) + 1;
    if n % 1024 != 0 {
        return;
    }
    static LAST: Mutex<[u64; 7]> = Mutex::new([0; 7]);
    let mut last = LAST.lock().unwrap();
    let now: [u64; 7] = std::array::from_fn(|i| STATS[i].load(Ordering::Relaxed));
    let d = |i: usize| now[i].saturating_sub(last[i]);
    let covered_now = c.saturating_sub(last[1]);
    let window = match window_us() {
        0 => "no time bound".to_string(),
        w => format!("within {} ms of it", w / 1000),
    };
    if joins() {
        crate::xlog_info!(
            "D3D12.X: look join: {covered_now} of the last 1024 submits went without a look of their own (lists closed before the last look began, {window}); {} looks, {:.2} ms and {} written pages a look",
            d(2),
            d(3) as f64 / d(2).max(1) as f64 / 1000.0,
            d(4) / d(2).max(1)
        );
    } else {
        crate::xlog_info!(
            "D3D12.X: look join (XCL_LOOK_JOIN=0, counted only): {covered_now} of the last 1024 submits could go without a look (lists closed before the last look began, {window}); their looks took {:.2} ms and found {} written pages each on average ({:.2} ms and {} pages over all {} looks)",
            d(6) as f64 / covered_now.max(1) as f64 / 1000.0,
            d(5) / covered_now.max(1),
            d(3) as f64 / d(2).max(1) as f64 / 1000.0,
            d(4) / d(2).max(1),
            d(2)
        );
    }
    *last = now;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_off_unless_set_and_zero_only_counts() {
        assert_eq!(mode_of(None), Mode::Off);
        assert_eq!(mode_of(Some("")), Mode::Off);
        assert_eq!(mode_of(Some("yes")), Mode::Off);
        assert_eq!(mode_of(Some(" 0 ")), Mode::Measure);
        assert_eq!(mode_of(Some("1")), Mode::Join);
    }

    #[test]
    fn a_look_covers_only_lists_closed_before_it_began() {
        // look from 100 to 150, asked at 160, no time bound
        let c = |closed: &[Option<u64>]| covers(100, 150, 0, 160, 0, closed.iter().copied());
        assert!(c(&[Some(99)]));
        assert!(c(&[Some(5), Some(99)]));
        // closed while the look ran, or after it: its pages may have been passed already
        assert!(!c(&[Some(100)]));
        assert!(!c(&[Some(99), Some(120)]));
        assert!(!c(&[Some(155)]));
        // a list never seen closing (a bundle, a list of XCL's own)
        assert!(!c(&[Some(99), None]));
        // a submit without lists has a look of its own (KickoffX's path)
        assert!(!c(&[]));
    }

    #[test]
    fn no_look_yet_or_a_look_under_way_covers_nothing() {
        assert!(!covers(0, 0, 0, 10, 0, [Some(1)]));
        // `look_begins` clears the end: a look that began at 100 and has not ended
        assert!(!covers(100, 0, 0, 110, 0, [Some(1)]));
        // the end of the look before is not this look's
        assert!(!covers(100, 90, 0, 110, 0, [Some(1)]));
    }

    #[test]
    fn a_look_covers_the_submits_of_its_window_only() {
        // 4000 µs after the look ended at 150
        assert!(covers(100, 150, 0, 4150, 4000, [Some(99)]));
        assert!(!covers(100, 150, 0, 4151, 4000, [Some(99)]));
        // no bound
        assert!(covers(100, 150, 0, 10_000_000, 0, [Some(99)]));
    }

    #[test]
    fn a_block_the_walk_found_since_is_not_covered() {
        assert!(covers(100, 150, 99, 160, 0, [Some(50)]));
        assert!(!covers(100, 150, 100, 160, 0, [Some(50)]));
        assert!(!covers(100, 150, 155, 160, 0, [Some(50)]));
    }
}
