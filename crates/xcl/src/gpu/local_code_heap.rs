//! LOCAL TEST, not for a PR: raise the NVIDIA user-mode driver's shader-code heap reservation
//! (`XCL_LOCAL_CODE_HEAP_MB`), after vinhed's report on xcl-dev/xcl#2403 (driver 621.24 only).
//!
//! In `nvwgf2umx.dll` 32.0.16.2124 the heap grow routine at RVA 0x5dc0d0 takes the heap object in
//! rcx; on the heap's first allocation (`[heap+0x30]` null) it reads the reservation as
//! `[heap+0x10] >> 12` pages (RVA 0x5dc1da). A detour on that routine raises `[heap+0x10]` before the
//! first allocation. Every byte the hook relies on is checked first; on any mismatch nothing is done.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const GROW_RVA: usize = 0x5dc0d0;
/// The routine's first 16 bytes (prologue), the `[rcx+0x30]` read and the `[rsi+0x10] >> 12` read.
const CHECKS: [(usize, &[u8]); 3] = [
    (0x5dc0d0, &[0x48, 0x89, 0x74, 0x24, 0x18, 0x48, 0x89, 0x7C, 0x24, 0x20, 0x55, 0x41, 0x54, 0x41, 0x55, 0x41]),
    (0x5dc103, &[0x48, 0x8B, 0x41, 0x30]),
    (0x5dc1da, &[0x48, 0x8B, 0x46, 0x10, 0x48, 0xC1, 0xE8, 0x0C]),
];

static ORIG: AtomicUsize = AtomicUsize::new(0);
static WANT: AtomicU64 = AtomicU64::new(0);
static RAISED: AtomicU64 = AtomicU64::new(0);

unsafe extern "system" fn grow(heap: *mut u8) -> u64 {
    if !heap.is_null() && (heap.add(0x30) as *const usize).read_unaligned() == 0 {
        let size = heap.add(0x10) as *mut u64;
        let (old, want) = (size.read_unaligned(), WANT.load(Ordering::Relaxed));
        if old < want {
            size.write_unaligned(want);
            let n = RAISED.fetch_add(1, Ordering::Relaxed) + 1;
            crate::xlog_info!("LOCAL code heap: heap {:p} reservation {} MiB -> {} MiB (heap #{n})", heap, old >> 20, want >> 20);
        }
    }
    let orig: unsafe extern "system" fn(*mut u8) -> u64 = std::mem::transmute(ORIG.load(Ordering::Acquire));
    orig(heap)
}

/// The driver DLL: already loaded, or the one copy in the driver store.
fn driver_module() -> Result<usize, String> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, LoadLibraryW};
    let h = unsafe { GetModuleHandleW(crate::util::wide("nvwgf2umx.dll").as_ptr()) };
    if !h.is_null() {
        return Ok(h as usize);
    }
    let store = std::path::Path::new(r"C:\Windows\System32\DriverStore\FileRepository");
    let found: Vec<std::path::PathBuf> = std::fs::read_dir(store)
        .map_err(|e| format!("driver store: {e}"))?
        .filter_map(|d| d.ok())
        .filter(|d| d.file_name().to_string_lossy().starts_with("nv_dispi.inf_amd64_"))
        .map(|d| d.path().join("nvwgf2umx.dll"))
        .filter(|p| p.exists())
        .collect();
    let [path] = found.as_slice() else { return Err(format!("{} copies of nvwgf2umx.dll in the driver store", found.len())) };
    let h = unsafe { LoadLibraryW(crate::util::wide(&path.to_string_lossy()).as_ptr()) };
    if h.is_null() {
        return Err(format!("LoadLibrary {} failed", path.display()));
    }
    crate::xlog_info!("LOCAL code heap: loaded {}", path.display());
    Ok(h as usize)
}

/// Install the hook once, before the first device is made (`XCL_LOCAL_CODE_HEAP_MB` = new size).
pub fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(mb) = std::env::var("XCL_LOCAL_CODE_HEAP_MB").ok().and_then(|v| v.parse::<u64>().ok()) else { return };
        match try_install(mb) {
            Ok(()) => crate::xlog_info!("LOCAL code heap: hook on nvwgf2umx+0x{GROW_RVA:x} active, reservation {mb} MiB"),
            Err(e) => crate::xlog_warn!("LOCAL code heap: not installed: {e}"),
        }
    });
}

fn try_install(mb: u64) -> Result<(), String> {
    let base = driver_module()?;
    for (rva, bytes) in CHECKS {
        let have = unsafe { std::slice::from_raw_parts((base + rva) as *const u8, bytes.len()) };
        if have != bytes {
            return Err(format!("bytes at +0x{rva:x} differ ({have:02x?}): not driver 621.24"));
        }
    }
    WANT.store(mb << 20, Ordering::Relaxed);
    let d = unsafe { retour::RawDetour::new((base + GROW_RVA) as *const (), grow as *const ()) }.map_err(|e| format!("detour: {e}"))?;
    ORIG.store(d.trampoline() as *const () as usize, Ordering::Release);
    unsafe { d.enable() }.map_err(|e| format!("enable: {e}"))?;
    Box::leak(Box::new(d));
    Ok(())
}
