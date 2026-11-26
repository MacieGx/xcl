//! lab: write guards for the hot-block looks (`XCL_WATCH_GUARD=1`, with `XCL_WATCH_HOT=1`).
//!
//! A hot look (`title_textures::take_hot`) takes the written pages of the 2 MiB blocks written in
//! the last few seconds only, and a thread walks the rest. The first write to a block that went
//! cold reaches the GPU unseen when a submit comes before the walk finds it: a community run of
//! the retail title saw ~2 black frames a second on its loading screens with hot looks on, none
//! with them off (claudetools-gh, #505). Here a block that goes cold is made read-only (its
//! committed, read-write parts in the watched ranges): its first write faults, the handler gives
//! the block its protection back and marks it written (`take_woken`: hot again at the next look),
//! and the write goes on – the write watch records it as any other. Kernel calls do not fault in
//! the title, they fail: the buffers of the file calls XCL routes and DirectStorage's
//! destinations are given back first (`unguard`, next to `kernelx::lazy::ensure`).
//!
//! The idea and the measurements (0 writes missed, ~85 µs a protect) are claudetools-gh's; this
//! code is ours.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_READONLY, PAGE_READWRITE};

/// the hot looks' block (`title_textures::HOT_BLOCK`)
pub(super) const BLOCK: usize = 2 << 20;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// guarded block -> the regions made read-only in it. Held while protections change, so a write
/// that faults meanwhile waits and then finds the block either guarded or writable again.
static GUARDED: Mutex<BTreeMap<usize, Vec<(usize, usize)>>> = Mutex::new(BTreeMap::new());
/// blocks a write (or `unguard`) gave back since the last look took them
static WOKEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// blocks guarded, blocks woken by a fault, by `unguard`, by the walk
static STATS: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// `XCL_WATCH_GUARD=1`: cold blocks are guarded (only with hot looks, which the caller checks).
pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_WATCH_GUARD").is_ok_and(|v| v.trim() == "1"))
}

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    // (a write: ExceptionInformation[0] = 1)
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 || r.ExceptionInformation[0] != 1 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let at = r.ExceptionInformation[1];
    if wake_as(at & !(BLOCK - 1), 1) || writable(at) {
        // (writable: another thread gave the block back while this one waited for the lock)
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

unsafe fn writable(at: usize) -> bool {
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0 && m.State == MEM_COMMIT && m.Protect == PAGE_READWRITE
}

/// The parts of `[a, a + n)` that are committed and plainly read-write (what a guard protects).
unsafe fn plain_regions(a: usize, n: usize) -> Vec<(usize, usize)> {
    let (mut at, end, mut out) = (a, a + n, Vec::new());
    while at < end {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
            break;
        }
        let stop = (m.BaseAddress as usize + m.RegionSize).min(end);
        if stop <= at {
            break;
        }
        if m.State == MEM_COMMIT && m.Protect == PAGE_READWRITE {
            out.push((at, stop - at));
        }
        at = stop;
    }
    out
}

/// Guard `block` (its watched parts `runs`): read-only until its next write. A block guarded
/// already is left as it is. False: nothing in it could be guarded.
pub(super) unsafe fn guard(block: usize, runs: &[(usize, usize)]) -> bool {
    install();
    let mut g = GUARDED.lock().unwrap_or_else(|e| e.into_inner());
    if g.contains_key(&block) {
        return true;
    }
    let mut done = Vec::new();
    for &(a, n) in runs {
        for (ra, rn) in plain_regions(a, n) {
            let mut old = 0u32;
            if VirtualProtect(ra as *const c_void, rn, PAGE_READONLY, &mut old) != 0 {
                done.push((ra, rn));
            }
        }
    }
    if done.is_empty() {
        return false;
    }
    g.insert(block, done);
    STATS[0].fetch_add(1, Ordering::Relaxed);
    true
}

/// Give `block` its protection back if it is guarded (true: it was), counted as `why` (1: a
/// fault, 2: `unguard`, 3: the walk), and mark it written for the next look.
unsafe fn wake_as(block: usize, why: usize) -> bool {
    let mut g = GUARDED.lock().unwrap_or_else(|e| e.into_inner());
    let Some(regions) = g.remove(&block) else { return false };
    for (a, n) in regions {
        let mut old = 0u32;
        VirtualProtect(a as *const c_void, n, PAGE_READWRITE, &mut old);
    }
    drop(g);
    WOKEN.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    STATS[why].fetch_add(1, Ordering::Relaxed);
    super::look_join::note_walk_found();
    true
}

/// The walk found a written page in `block`: it is given back if guarded (it was written in a
/// way that did not fault, e.g. by the title's own `VirtualProtect` or a new commit).
pub(super) unsafe fn walk_found(block: usize) {
    wake_as(block, 3);
}

/// The blocks given back since the last call: hot from now on.
pub(super) fn take_woken() -> Vec<usize> {
    std::mem::take(&mut *WOKEN.lock().unwrap_or_else(|e| e.into_inner()))
}

/// `[va, va + n)` is about to be written by a kernel call (a file read, DirectStorage) or by XCL
/// after a protection check: its guarded blocks are given back first.
pub(crate) fn unguard(va: usize, n: usize) {
    if n == 0 || !on() {
        return;
    }
    let blocks: Vec<usize> = {
        let g = GUARDED.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            return;
        }
        g.range(va & !(BLOCK - 1)..va.saturating_add(n)).map(|(&b, _)| b).collect()
    };
    for b in blocks {
        // SAFETY: only blocks this module made read-only get their protection back
        unsafe { wake_as(b, 2) };
    }
}

/// One line for the log: blocks guarded now and the counts.
pub(super) fn stats() -> String {
    let now = GUARDED.lock().unwrap_or_else(|e| e.into_inner()).len();
    format!(
        "write guards: {now} blocks guarded now; {} guarded, {} given back by a write, {} before a kernel call, {} by the walk",
        STATS[0].load(Ordering::Relaxed),
        STATS[1].load(Ordering::Relaxed),
        STATS[2].load(Ordering::Relaxed),
        STATS[3].load(Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE};
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    extern "system" {
        fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
    }

    unsafe fn written(base: usize, size: usize) -> Vec<usize> {
        let mut buf = vec![std::ptr::null_mut(); 1024];
        let (mut n, mut page) = (buf.len(), 0u32);
        assert_eq!(GetWriteWatch(1, base as *const c_void, size, buf.as_mut_ptr(), &mut n, &mut page), 0);
        buf[..n].iter().map(|&p| p as usize).collect()
    }

    /// A guarded block's first write faults, gives the block back (marked for the next look) and
    /// goes on; the write watch records it. A kernel call's buffer is given back before it, and
    /// an uncommitted part is not guarded.
    #[test]
    fn a_guarded_blocks_first_write_goes_on_and_is_seen() {
        unsafe {
            // two blocks, write-watched as the title's sparse windows are, the second half of the
            // second block not committed
            let size = 2 * BLOCK;
            let r = VirtualAlloc(std::ptr::null(), size + BLOCK, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(r, 0);
            let base = (r + BLOCK - 1) & !(BLOCK - 1);
            assert!(!VirtualAlloc(base as *const c_void, BLOCK + BLOCK / 2, MEM_COMMIT, PAGE_READWRITE).is_null());
            let _ = written(base, size);
            let (b0, b1) = (base, base + BLOCK);
            assert!(guard(b0, &[(b0, BLOCK)]));
            assert!(guard(b1, &[(b1, BLOCK)]), "its committed half");
            assert_eq!(GUARDED.lock().unwrap()[&b1], vec![(b1, BLOCK / 2)]);
            assert!(!writable(b0 + 0x1000));
            take_woken();
            // the write faults, is let through and is recorded
            ((b0 + 0x3008) as *mut u64).write_volatile(0x5EED);
            assert_eq!(((b0 + 0x3008) as *const u64).read_volatile(), 0x5EED);
            assert!(writable(b0), "the block is writable again");
            assert!(take_woken().contains(&b0), "hot at the next look");
            assert!(written(base, size).contains(&(b0 + 0x3000)));
            // a second write to it does not fault (nothing woken)
            ((b0 + 0x5000) as *mut u8).write_volatile(1);
            assert!(!take_woken().contains(&b0));
            // a kernel call's buffer in the other block: given back first
            unguard(b1 + 0x10, 0x100);
            if on() {
                assert!(writable(b1) && take_woken().contains(&b1));
            } else {
                // (`unguard` does nothing without the switch: the walk's way back)
                walk_found(b1);
                assert!(writable(b1) && take_woken().contains(&b1));
            }
            assert!(!guard(base + size, &[]), "nothing to guard");
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
