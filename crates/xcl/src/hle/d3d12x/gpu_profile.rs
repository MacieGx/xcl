//! lab: `XCL_GPU_PROFILE=<from seconds>[,<every>]` – where the GPU time of a frame goes, for
//! everything XCL records on the title's direct and compute lists.
//!
//! Based on #1766 (`XCL_GPU_PROFILE=1`, wondercrash: a timestamp pair around each draw /
//! dispatch / ExecuteIndirect, resolved right behind it), changed in two ways:
//! - **Marks, not pairs.** A list gets one timestamp at each *mark*, and the GPU time between
//!   two marks of a list goes to the label of the second: `mark(list, Other)` before an op and
//!   `mark(list, <op>)` after it. So the time of the work XCL adds around a title op (the
//!   indirect clamp's dispatch and its transitions, the two passes of the two-pass mesh path,
//!   the pieces of a dispatch chain, the title's copies and barriers) is attributed on its own,
//!   and the segments of a list add up to the list's GPU time from its first mark to its last
//!   ("Other" is what lies between the marked ops: state, clears, resolves, …).
//! - **Resolved in batches.** The queries of a recording are taken in chunks of a ring and
//!   resolved once, at the list's Close (#1766's resolve after each op made the queue wait for
//!   the op). A reporter thread takes the recordings whose stamps the GPU has written.
//!
//! Frames: a recording counts for the frame (present) in which it began. With `<every>` N > 1
//! only the lists of every Nth frame are stamped (the others pay one check a mark), so the frames
//! in between show the speed without the profile in the same run. Every 5 s the log has, per
//! profiled frame: the GPU time in lists, the time at least one list was running on the GPU (the
//! union of the segments, all queues and per queue type) against the wall time those frames
//! spanned, the time by kind of work, and the 30 programs / passes with the most time.
//!
//! What the numbers mean: a timestamp is written once the work before it has finished, so a
//! segment is the time from the end of the previous marked work to the end of this one,
//! including waits on barriers in between; overlapping lists on two queues each count in full
//! (the union does not). Not covered: work on copy queues (no timestamps there on every GPU),
//! queue operations (UpdateTileMappings, waits, signals), the presenter's own list and bundles:
//! the wall time the union does not cover is those or idle.

use std::collections::{BTreeSet, HashMap};
use std::ffi::c_void;
use std::mem::{offset_of, size_of, transmute};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::{Interface, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::real;

/// timestamp queries in the ring (8 bytes of readback each)
const QUERIES: usize = 1 << 21;
/// queries a recording takes at once
const CHUNK: usize = 128;
const CHUNKS: usize = QUERIES / CHUNK;
#[cfg_attr(test, allow(dead_code))]
const POLL: Duration = Duration::from_millis(200);
const WINDOW: Duration = Duration::from_secs(5);
/// windows between two logs of the totals since the start
#[cfg_attr(test, allow(dead_code))]
const SINCE_START_EVERY: u32 = 12;
const TOP: usize = 30;
/// a closed recording whose stamps were not all written by then is dropped (never submitted)
const GIVE_UP: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Config {
    from: f64,
    every: u64,
}

fn parse(v: &str) -> Option<Config> {
    let mut f = v.trim().split(',');
    let from = f.next()?.trim().parse::<f64>().ok()?;
    let every = match f.next() {
        Some(e) => e.trim().parse::<u64>().ok()?,
        None => 1,
    };
    Some(Config { from, every: every.max(1) })
}

#[cfg(test)]
thread_local! {
    /// the profile of a test's own thread (other tests' lists are not stamped)
    static TEST_CONFIG: std::cell::Cell<Option<Config>> = const { std::cell::Cell::new(None) };
}

fn config() -> Option<Config> {
    #[cfg(test)]
    if let Some(c) = TEST_CONFIG.with(|c| c.get()) {
        return Some(c);
    }
    static C: OnceLock<Option<Config>> = OnceLock::new();
    *C.get_or_init(|| std::env::var("XCL_GPU_PROFILE").ok().and_then(|v| parse(&v)))
}

pub(crate) fn on() -> bool {
    config().is_some()
}

/// Whether the lists of frame `frame` are stamped.
fn profiled(frame: u64, every: u64) -> bool {
    frame % every.max(1) == 0
}

/// What the GPU did between two marks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub(crate) enum Kind {
    /// between marked ops: the title's state changes, clears, resolves, queries, unmarked work
    Other,
    /// a draw of a translated pipeline (VS, PS)
    Draw,
    /// an ExecuteIndirect of draws with a translated pipeline
    IndirectDraw,
    /// a draw or DispatchMesh with a PC pipeline
    PcDraw,
    /// an ExecuteIndirect with a PC pipeline
    PcIndirect,
    /// a Dispatch of a translated compute program run whole (CS)
    Dispatch,
    /// an ExecuteIndirect of dispatches with a translated pipeline
    IndirectDispatch,
    /// one piece of a compute program split into a dispatch chain (CS, piece), with its UAV barrier
    ChainPiece,
    /// the two-pass mesh path's compute pass and the UAV barrier after it (MS, PS)
    TwoPassCs,
    /// the two-pass path's scaling of the records (several mesh groups a record)
    TwoPassScale,
    /// the two-pass path's pass-through mesh shader pass and the UAV barrier after it (MS, PS)
    TwoPassMs,
    /// the indirect clamp before an ExecuteIndirect: UAV barrier, its dispatch, the transition
    Clamp,
    /// the clamp's transition back after the ExecuteIndirect
    ClampFinish,
    /// a title copy (CopyBufferRegion, CopyTextureRegion, CopyResource, …)
    Copy,
    /// a title ResourceBarrier
    Barrier,
    /// XCL's upload of pages the CPU wrote (sparse windows), a list of its own
    Upload,
    /// XCL's copies of GPU-written pages back to the CPU, a list of its own
    Readback,
    /// XCL's upload of title textures, a list of its own
    TextureUpload,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Other => "other (between marked ops)",
            Kind::Draw => "draw",
            Kind::IndirectDraw => "ExecuteIndirect of draws",
            Kind::PcDraw => "draw with a PC pipeline",
            Kind::PcIndirect => "ExecuteIndirect with a PC pipeline",
            Kind::Dispatch => "dispatch",
            Kind::IndirectDispatch => "ExecuteIndirect of dispatches",
            Kind::ChainPiece => "dispatch chain piece",
            Kind::TwoPassCs => "two-pass compute pass",
            Kind::TwoPassScale => "two-pass record scaling",
            Kind::TwoPassMs => "two-pass mesh pass",
            Kind::Clamp => "indirect clamp",
            Kind::ClampFinish => "indirect clamp transition back",
            Kind::Copy => "title copy",
            Kind::Barrier => "title barrier",
            Kind::Upload => "XCL upload (sparse pages)",
            Kind::Readback => "XCL readback copies",
            Kind::TextureUpload => "XCL texture upload",
        }
    }
}

/// A mark's label: the kind of work and what it ran ((CS, 0), (VS, PS), (MS, PS), (CS, piece);
/// (0, 0): nothing known), with its size (groups, vertices x instances, max count).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Label {
    pub kind: Kind,
    pub a: u64,
    pub b: u64,
    pub size: [u32; 3],
}

impl Label {
    pub(crate) fn of(kind: Kind) -> Label {
        Label { kind, a: 0, b: 0, size: [0; 3] }
    }

    pub(crate) fn op(kind: Kind, (a, b): (u64, u64), size: [u32; 3]) -> Label {
        Label { kind, a, b, size }
    }
}

type Key = (Kind, u64, u64);

#[derive(Default, Clone, Debug, PartialEq)]
struct Total {
    calls: u64,
    ticks: u64,
    /// ticks of the longest call, and its size
    max: u64,
    size: [u32; 3],
}

fn add(totals: &mut HashMap<Key, Total>, l: &Label, ticks: u64) {
    let t = totals.entry((l.kind, l.a, l.b)).or_default();
    t.calls += 1;
    t.ticks += ticks;
    if ticks >= t.max {
        t.max = ticks;
        t.size = l.size;
    }
}

/// The `n` entries of `totals` with the most ticks, most first.
fn top(totals: &HashMap<Key, Total>, n: usize) -> Vec<(Key, Total)> {
    let mut v: Vec<(Key, Total)> = totals.iter().map(|(k, t)| (*k, t.clone())).collect();
    v.sort_by(|a, b| b.1.ticks.cmp(&a.1.ticks).then(b.1.calls.cmp(&a.1.calls)).then(a.0.cmp(&b.0)));
    v.truncate(n);
    v
}

/// The segments of a recording from its stamps (in the order recorded): (label, start, end) of
/// each mark after the first, which only starts the first segment. None: a stamp the GPU has
/// not written yet (0).
fn segments(stamps: &[(u64, Label)]) -> Option<Vec<(Label, u64, u64)>> {
    if stamps.iter().any(|s| s.0 == 0) {
        return None;
    }
    Some(stamps.windows(2).map(|w| (w[1].1, w[0].0, w[1].0)).collect())
}

/// The time covered by at least one of `spans` (start, end).
fn union(spans: &mut [(u64, u64)]) -> u64 {
    spans.sort_unstable();
    let (mut total, mut cur): (u64, Option<(u64, u64)>) = (0, None);
    for &(a, b) in spans.iter() {
        match cur {
            Some((ca, cb)) if a <= cb => cur = Some((ca, cb.max(b))),
            Some((ca, cb)) => {
                total += cb - ca;
                cur = Some((a, b));
            }
            None => cur = Some((a, b)),
        }
    }
    total + cur.map_or(0, |(a, b)| b - a)
}

/// A list's recording.
struct Recording {
    /// the presents before it began
    frame: u64,
    /// 0 direct, 1 compute
    ty: u8,
    /// not stamped (another frame, or a list type without timestamps)
    skip: bool,
    /// (query, label of the segment that ends at it)
    stamps: Vec<(u32, Label)>,
    /// the chunks taken, and how many queries of the last one are used
    chunks: Vec<u32>,
    used: usize,
    closed: Option<Instant>,
}

impl Recording {
    /// (first query, count) of each chunk
    fn ranges(&self) -> Vec<(u32, u32)> {
        let n = self.chunks.len();
        self.chunks.iter().enumerate().map(|(i, &c)| (c * CHUNK as u32, if i + 1 == n { self.used as u32 } else { CHUNK as u32 })).filter(|r| r.1 > 0).collect()
    }
}

#[derive(Default)]
struct Window {
    totals: HashMap<Key, Total>,
    frames: BTreeSet<u64>,
    /// (start, end, queue type) of every segment
    spans: Vec<(u64, u64, u8)>,
    recordings: u64,
}

struct Profile {
    _heap: ID3D12QueryHeap,
    _buf: ID3D12Resource,
    heap: usize,
    buf: usize,
    /// the readback buffer's mapping: one u64 a query
    words: usize,
    free: Vec<u32>,
    open: HashMap<usize, Recording>,
    closed: Vec<Recording>,
    window: Window,
    since_start: Window,
    /// marks without a free chunk, segments with the end before the start, recordings dropped
    lost: u64,
    bad: u64,
    gave_up: u64,
}

// SAFETY: the readback mapping lives as long as `_buf`; D3D12 objects are free-threaded.
unsafe impl Send for Profile {}

impl Profile {
    fn word(&self, i: u32) -> &AtomicU64 {
        unsafe { &*(self.words as *const AtomicU64).add(i as usize) }
    }

    /// The chunks of `r` back to the ring, their words zeroed (only once the GPU has written them
    /// or never will).
    fn free_chunks(&mut self, r: &Recording) {
        for &c in &r.chunks {
            for q in c * CHUNK as u32..(c + 1) * CHUNK as u32 {
                self.word(q).store(0, Ordering::Relaxed);
            }
            self.free.push(c);
        }
    }
}

static STATE: Mutex<Option<Profile>> = Mutex::new(None);

/// ticks a second of the title queues' timestamps (0: no queue seen yet)
static FREQUENCY: AtomicU64 = AtomicU64::new(0);

unsafe fn init(dev: &ID3D12Device) -> Option<Profile> {
    let mut heap: Option<ID3D12QueryHeap> = None;
    let desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: QUERIES as u32, NodeMask: 0 };
    if let Err(e) = dev.CreateQueryHeap(&desc, &mut heap) {
        crate::xlog_warn!("gpu profile: no timestamp query heap ({e})");
        return None;
    }
    let heap = heap?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (QUERIES * 8) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let mut buf: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buf) {
        crate::xlog_warn!("gpu profile: no readback buffer ({e})");
        return None;
    }
    let buf = buf?;
    let mut p = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).ok()?;
    std::ptr::write_bytes(p.cast::<u8>(), 0, QUERIES * 8);
    #[cfg(not(test))]
    let _ = std::thread::Builder::new().name("xcl-gpu-profile".into()).spawn(|| {
        let (mut last, mut windows) = (Instant::now(), 0u32);
        loop {
            std::thread::sleep(POLL);
            poll();
            if last.elapsed() >= WINDOW && report(last.elapsed(), (windows + 1) % SINCE_START_EVERY == 0) {
                windows += 1;
                last = Instant::now();
            }
        }
    });
    let c = config().unwrap_or(Config { from: 0.0, every: 1 });
    crate::xlog_info!(
        "gpu profile: the title's direct / compute lists are stamped from {:.0} s, one frame in {}; a breakdown by kind and program every {} s",
        c.from,
        c.every,
        WINDOW.as_secs()
    );
    Some(Profile {
        heap: heap.as_raw() as usize,
        buf: buf.as_raw() as usize,
        _heap: heap,
        _buf: buf,
        words: p as usize,
        free: (0..CHUNKS as u32).rev().collect(),
        open: HashMap::new(),
        closed: Vec::new(),
        window: Window::default(),
        since_start: Window::default(),
        lost: 0,
        bad: 0,
        gave_up: 0,
    })
}

/// The profile, made on first use with the device of `list` (outside the lock: the device calls
/// go through XCL's hooks).
unsafe fn with_state<R>(list: *mut c_void, f: impl FnOnce(&mut Profile) -> R) -> Option<R> {
    static INIT: std::sync::Once = std::sync::Once::new();
    if !INIT.is_completed() {
        let mut dev: Option<ID3D12Device> = None;
        ID3D12GraphicsCommandList::from_raw_borrowed(&list)?.GetDevice(&mut dev).ok()?;
        let dev = dev?;
        INIT.call_once(|| {
            let s = init(&dev);
            *STATE.lock().unwrap() = s;
        });
    }
    STATE.lock().unwrap().as_mut().map(f)
}

/// The PC method in `slot` of `obj` (an adopted object's own, not XCL's console-layout entry).
unsafe fn pc_fn(obj: *mut c_void, slot: usize) -> usize {
    if real::is_adopted(obj) {
        real::orig_fn(obj, slot)
    } else {
        *(*(obj as *const *const usize)).add(slot)
    }
}

type EndQueryFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32);
type ResolveFn = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, usize, u64);

/// `queue` (a title queue) executes lists: its timestamp frequency converts the ticks.
pub(crate) unsafe fn note_queue(queue: *mut c_void) {
    if !on() || queue.is_null() || FREQUENCY.load(Ordering::Relaxed) != 0 {
        return;
    }
    const SLOT: usize = offset_of!(ID3D12CommandQueue_Vtbl, GetTimestampFrequency) / size_of::<usize>();
    let f: unsafe extern "system" fn(*mut c_void, *mut u64) -> HRESULT = transmute(pc_fn(queue, SLOT));
    let mut hz = 0u64;
    if f(queue, &mut hz).is_ok() && hz != 0 && FREQUENCY.swap(hz, Ordering::Relaxed) == 0 {
        crate::xlog_info!("gpu profile: GPU timestamps at {hz} Hz");
    }
}

/// The GPU work recorded on `list` since its previous mark is `label` (the first mark of a
/// recording only starts it). Call while `list` is open.
pub(crate) unsafe fn mark(list: *mut c_void, label: Label) {
    let Some(c) = config() else { return };
    if list.is_null() || crate::log::uptime() < c.from {
        return;
    }
    let every = c.every;
    let at = with_state(list, |s| {
        if !s.open.contains_key(&(list as usize)) {
            const GET_TYPE: usize = offset_of!(ID3D12CommandList_Vtbl, GetType) / size_of::<usize>();
            let get_type: unsafe extern "system" fn(*mut c_void) -> D3D12_COMMAND_LIST_TYPE = transmute(pc_fn(list, GET_TYPE));
            let ty = get_type(list);
            let frame = super::present::activity().0;
            let skip = !profiled(frame, every) || (ty != D3D12_COMMAND_LIST_TYPE_DIRECT && ty != D3D12_COMMAND_LIST_TYPE_COMPUTE);
            let ty = u8::from(ty == D3D12_COMMAND_LIST_TYPE_COMPUTE);
            s.open.insert(list as usize, Recording { frame, ty, skip, stamps: Vec::new(), chunks: Vec::new(), used: 0, closed: None });
        }
        let r = s.open.get_mut(&(list as usize)).unwrap();
        if r.skip {
            return None;
        }
        if r.chunks.is_empty() || r.used == CHUNK {
            let Some(c) = s.free.pop() else {
                s.lost += 1;
                return None;
            };
            r.chunks.push(c);
            r.used = 0;
        }
        let q = *r.chunks.last().unwrap() * CHUNK as u32 + r.used as u32;
        r.used += 1;
        r.stamps.push((q, label));
        Some((s.heap, q))
    })
    .flatten();
    if let Some((heap, q)) = at {
        const END_QUERY: usize = offset_of!(ID3D12GraphicsCommandList_Vtbl, EndQuery) / size_of::<usize>();
        let end_query: EndQueryFn = transmute(pc_fn(list, END_QUERY));
        end_query(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, q);
    }
}

/// Before `list` is closed: its stamps are resolved (one ResolveQueryData a chunk) and the
/// recording waits for the GPU.
pub(crate) unsafe fn list_close(list: *mut c_void) {
    if !on() || list.is_null() {
        return;
    }
    let Some((mut r, heap, buf)) = STATE.lock().unwrap().as_mut().and_then(|s| Some((s.open.remove(&(list as usize))?, s.heap, s.buf))) else { return };
    if r.skip || r.stamps.len() < 2 {
        if let Some(s) = STATE.lock().unwrap().as_mut() {
            s.free_chunks(&r);
        }
        return;
    }
    const RESOLVE: usize = offset_of!(ID3D12GraphicsCommandList_Vtbl, ResolveQueryData) / size_of::<usize>();
    let resolve: ResolveFn = transmute(pc_fn(list, RESOLVE));
    for (first, n) in r.ranges() {
        resolve(list, heap, D3D12_QUERY_TYPE_TIMESTAMP, first, n, buf, 8 * u64::from(first));
    }
    r.closed = Some(Instant::now());
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        s.closed.push(r);
    }
}

/// `list` was reset (or made): a recording it did not close is dropped.
pub(crate) fn list_reset(list: *mut c_void) {
    if !on() {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    if let Some(r) = s.open.remove(&(list as usize)) {
        s.free_chunks(&r);
    }
}

/// An XCL list of its own was reset onto its allocator: it is stamped from here (call
/// `own_list_end` before its Close).
pub(crate) unsafe fn own_list_begin(list: *mut c_void) {
    if !on() {
        return;
    }
    list_reset(list);
    mark(list, Label::of(Kind::Other));
}

/// Before the Close of an XCL list `own_list_begin` started: all of it is `kind`.
pub(crate) unsafe fn own_list_end(list: *mut c_void, kind: Kind) {
    if !on() {
        return;
    }
    mark(list, Label::of(kind));
    list_close(list);
}

/// Take the recordings whose stamps the GPU has written into the window.
fn poll() {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let mut i = 0;
    while i < s.closed.len() {
        let stamps: Vec<(u64, Label)> = s.closed[i].stamps.iter().map(|&(q, l)| (s.word(q).load(Ordering::Acquire), l)).collect();
        match segments(&stamps) {
            Some(segs) => {
                let r = s.closed.swap_remove(i);
                for (l, a, b) in segs {
                    if b < a {
                        s.bad += 1;
                        continue;
                    }
                    add(&mut s.window.totals, &l, b - a);
                    add(&mut s.since_start.totals, &l, b - a);
                    s.window.spans.push((a, b, r.ty));
                }
                s.window.frames.insert(r.frame);
                s.since_start.frames.insert(r.frame);
                s.window.recordings += 1;
                s.since_start.recordings += 1;
                s.free_chunks(&r);
            }
            None if s.closed[i].closed.is_some_and(|t| t.elapsed() > GIVE_UP) => {
                let r = s.closed.swap_remove(i);
                s.gave_up += 1;
                s.free_chunks(&r);
            }
            None => i += 1,
        }
    }
}

#[cfg_attr(test, allow(dead_code))]
fn describe(key: Key) -> String {
    let (kind, a, b) = key;
    match kind {
        Kind::Draw | Kind::IndirectDraw if a != 0 || b != 0 => format!("{} VS {a:016x} PS {b:016x}", kind.name()),
        Kind::Dispatch | Kind::IndirectDispatch if a != 0 => format!("{} CS {a:016x}", kind.name()),
        Kind::ChainPiece => format!("{} CS {a:016x} piece {b}", kind.name()),
        Kind::TwoPassCs | Kind::TwoPassMs | Kind::TwoPassScale if a != 0 || b != 0 => format!("{} MS {a:016x} PS {b:016x}", kind.name()),
        _ => kind.name().to_owned(),
    }
}

/// Log the window's breakdown (and the totals since the start). False: nothing logged (no
/// queue's timestamp frequency yet); the window goes on.
#[cfg_attr(test, allow(dead_code))]
fn report(elapsed: Duration, since_start: bool) -> bool {
    let hz = FREQUENCY.load(Ordering::Relaxed);
    if hz == 0 {
        return false;
    }
    static PRESENTS: AtomicU64 = AtomicU64::new(0);
    let presents = super::present::activity().0;
    let presented = presents.saturating_sub(PRESENTS.swap(presents, Ordering::Relaxed));
    let (window, all, open, closed, free, lost, bad, gave_up) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return false };
        let all = since_start.then(|| (s.since_start.totals.clone(), s.since_start.frames.len()));
        (std::mem::take(&mut s.window), all, s.open.len(), s.closed.len(), s.free.len(), s.lost, s.bad, s.gave_up)
    };
    let ms = |ticks: u64| ticks as f64 * 1000.0 / hz as f64;
    let frames = window.frames.len().max(1) as f64;
    let summary = |what: &str, totals: &HashMap<Key, Total>, frames: f64| {
        let total: u64 = totals.values().map(|t| t.ticks).sum();
        let mut kinds: HashMap<Kind, (u64, u64)> = HashMap::new();
        for ((k, _, _), t) in totals {
            let e = kinds.entry(*k).or_default();
            e.0 += t.ticks;
            e.1 += t.calls;
        }
        let mut kinds: Vec<_> = kinds.into_iter().collect();
        kinds.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        crate::xlog_info!(
            "gpu profile: {what}: by kind (ms a frame, share of list time, calls a frame): {}",
            kinds
                .iter()
                .map(|(k, (t, n))| format!("{} {:.2} {:.1}% {:.0}", k.name(), ms(*t) / frames, 100.0 * *t as f64 / total.max(1) as f64, *n as f64 / frames))
                .collect::<Vec<_>>()
                .join("; ")
        );
        for (i, (key, t)) in top(totals, TOP).into_iter().enumerate() {
            crate::xlog_info!(
                "gpu profile: {what}: {:>2}. {:>8.2} ms a frame {:>5.1}%  {}: {:.1} calls a frame, longest {:.2} ms ({:?})",
                i + 1,
                ms(t.ticks) / frames,
                100.0 * t.ticks as f64 / total.max(1) as f64,
                describe(key),
                t.calls as f64 / frames,
                ms(t.max),
                t.size
            );
        }
    };
    if !window.totals.is_empty() {
        let in_lists: u64 = window.totals.values().map(|t| t.ticks).sum();
        let first = window.spans.iter().map(|s| s.0).min().unwrap_or(0);
        let last = window.spans.iter().map(|s| s.1).max().unwrap_or(0);
        let busy = |ty: Option<u8>| {
            let mut v: Vec<(u64, u64)> = window.spans.iter().filter(|s| ty.is_none_or(|t| s.2 == t)).map(|s| (s.0, s.1)).collect();
            union(&mut v)
        };
        let (all_busy, direct, compute) = (busy(None), busy(Some(0)), busy(Some(1)));
        crate::xlog_info!(
            "gpu profile: last {:.1} s: {} frame(s) profiled ({presented} presents), {} lists; a profiled frame: {:.2} ms GPU time in lists, {:.2} ms with a list running (direct {:.2}, compute {:.2}); the profiled lists spanned {:.1} ms of GPU clock ({:.1} ms a frame); open {open}, waiting {closed}, free chunks {free}/{CHUNKS}, lost marks {lost}, bad {bad}, dropped {gave_up}",
            elapsed.as_secs_f64(),
            window.frames.len(),
            window.recordings,
            ms(in_lists) / frames,
            ms(all_busy) / frames,
            ms(direct) / frames,
            ms(compute) / frames,
            ms(last.saturating_sub(first)),
            ms(last.saturating_sub(first)) / frames,
        );
        summary("last window", &window.totals, frames);
    }
    if let Some((all, n)) = all {
        summary(&format!("since the start ({n} frames)"), &all, n.max(1) as f64);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_takes_a_start_time_and_a_frame_step() {
        assert_eq!(parse("1"), Some(Config { from: 1.0, every: 1 }));
        assert_eq!(parse(" 420 , 4 "), Some(Config { from: 420.0, every: 4 }));
        assert_eq!(parse("300,0"), Some(Config { from: 300.0, every: 1 }));
        assert_eq!(parse("x"), None);
        assert_eq!(parse("10,y"), None);
        assert!(profiled(8, 4) && !profiled(9, 4) && profiled(9, 1));
    }

    #[test]
    fn a_segment_goes_to_the_label_of_the_mark_that_ends_it() {
        let l = |k| Label::of(k);
        assert_eq!(segments(&[(100, l(Kind::Other)), (0, l(Kind::Clamp))]), None);
        let s = segments(&[(100, l(Kind::Draw)), (130, l(Kind::Clamp)), (180, l(Kind::TwoPassCs)), (200, l(Kind::TwoPassMs)), (201, l(Kind::ClampFinish))]).unwrap();
        assert_eq!(
            s,
            vec![(l(Kind::Clamp), 100, 130), (l(Kind::TwoPassCs), 130, 180), (l(Kind::TwoPassMs), 180, 200), (l(Kind::ClampFinish), 200, 201)]
        );
        // the segments add up to the recording's time from its first mark to its last
        assert_eq!(s.iter().map(|x| x.2 - x.1).sum::<u64>(), 101);
        assert_eq!(segments(&[(100, l(Kind::Other))]), Some(vec![]));
    }

    #[test]
    fn busy_time_counts_overlapping_segments_once() {
        assert_eq!(union(&mut []), 0);
        assert_eq!(union(&mut [(10, 20), (15, 30), (40, 50), (50, 55), (0, 5)]), 5 + 20 + 15);
        assert_eq!(union(&mut [(0, 100), (10, 20)]), 100);
    }

    #[test]
    fn programs_are_ranked_by_their_total_and_keep_their_longest_call() {
        let mut t = HashMap::new();
        let a = Label::op(Kind::Dispatch, (0xA, 0), [8, 4, 1]);
        let b = Label::op(Kind::TwoPassCs, (0xB, 0xC), [1, 0, 0]);
        add(&mut t, &Label { size: [1, 1, 1], ..a }, 10);
        add(&mut t, &a, 30);
        add(&mut t, &Label { size: [2, 1, 1], ..a }, 20);
        add(&mut t, &b, 100);
        let r = top(&t, 2);
        assert_eq!(r[0], ((Kind::TwoPassCs, 0xB, 0xC), Total { calls: 1, ticks: 100, max: 100, size: [1, 0, 0] }));
        assert_eq!(r[1], ((Kind::Dispatch, 0xA, 0), Total { calls: 3, ticks: 60, max: 30, size: [8, 4, 1] }));
        // the same program as another kind of work stays apart
        add(&mut t, &Label { kind: Kind::TwoPassMs, ..b }, 1);
        assert_eq!(top(&t, 10).len(), 3);
    }

    #[test]
    fn a_recording_resolves_each_chunk_it_took() {
        let r = Recording { frame: 0, ty: 0, skip: false, stamps: Vec::new(), chunks: vec![7, 2, 9], used: 5, closed: None };
        assert_eq!(r.ranges(), vec![(7 * CHUNK as u32, CHUNK as u32), (2 * CHUNK as u32, CHUNK as u32), (9 * CHUNK as u32, 5)]);
    }

    /// On the GPU: marks around copies on a direct list, more marks than a chunk holds, resolved
    /// at the close in one ResolveQueryData a chunk; the reporter's pass attributes the time.
    #[test]
    fn gpu_marks_on_a_list_are_resolved_at_its_close_and_attributed() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            TEST_CONFIG.with(|c| c.set(Some(Config { from: 0.0, every: 1 })));
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            note_queue(queue.as_raw());
            assert_ne!(FREQUENCY.load(Ordering::Relaxed), 0, "the queue's timestamp frequency");
            let buffer = |bytes: u64| {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            const BIG: u64 = 256 << 20;
            let (src, dst) = (buffer(BIG), buffer(BIG));
            let l = list.as_raw();
            list_reset(l);
            // a big copy, then 300 small ones (three chunks of stamps), then the big one again
            mark(l, Label::of(Kind::Other));
            list.CopyBufferRegion(&dst, 0, &src, 0, BIG);
            mark(l, Label::op(Kind::Copy, (1, 0), [0; 3]));
            for _ in 0..300 {
                list.CopyBufferRegion(&dst, 0, &src, 0, 256);
                mark(l, Label::op(Kind::Copy, (2, 0), [0; 3]));
            }
            list.CopyBufferRegion(&dst, 0, &src, 0, BIG);
            mark(l, Label::op(Kind::Copy, (1, 0), [0; 3]));
            {
                let g = STATE.lock().unwrap();
                let r = &g.as_ref().unwrap().open[&(l as usize)];
                assert_eq!(r.stamps.len(), 303);
                assert_eq!(r.chunks.len(), 303usize.div_ceil(CHUNK));
            }
            list_close(l);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let start = Instant::now();
            while fence.GetCompletedValue() < 1 && start.elapsed().as_secs() < 20 {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(fence.GetCompletedValue(), 1, "the list ran");
            poll();
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert!(s.closed.is_empty() && s.open.is_empty(), "the recording was taken");
            assert_eq!(s.free.len(), CHUNKS, "its chunks are back");
            assert_eq!((s.lost, s.bad), (0, 0));
            let big = &s.window.totals[&(Kind::Copy, 1, 0)];
            let small = &s.window.totals[&(Kind::Copy, 2, 0)];
            assert_eq!((big.calls, small.calls), (2, 300));
            let hz = FREQUENCY.load(Ordering::Relaxed) as f64;
            let ms = |t: u64| t as f64 * 1000.0 / hz;
            eprintln!("gpu profile test: two 256 MiB copies {:.3} ms, 300 copies of 256 B {:.3} ms", ms(big.ticks), ms(small.ticks));
            // 512 MiB copied: at least ~0.1 ms on any GPU; the small copies take far less
            assert!(ms(big.ticks) > 0.05, "the big copies took GPU time: {} ticks", big.ticks);
            assert!(small.ticks < big.ticks, "the small copies take less than the big ones");
            let mut spans: Vec<(u64, u64)> = s.window.spans.iter().map(|x| (x.0, x.1)).collect();
            assert_eq!(union(&mut spans), big.ticks + small.ticks, "the segments of one list do not overlap");
        }
    }
}
