//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (32 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels). The CPU never sees these textures
//! (on the console it could, but GPU-owned targets are not CPU-written).
//!
//! Arenas cost video memory for whole ranges, and a placement straddling two made a third over
//! both while the two lived on with their resources: a retail game's streaming targets, packed
//! back to back over 3.5 GB of its addresses, made 15 GB of arenas in two seconds – past the
//! card's memory and the system's commit, and the device was removed. So the first choice is
//! the console's own model (`Pages`): a 64 KiB page of video memory for each 64 KiB page of
//! the title's address a target covers, and each target a reserved (tiled) resource whose tiles
//! are mapped onto the pages of its addresses. Targets at one address share pages, a page
//! exists once however many targets cover it, and what no target covers takes nothing.
//! Arenas remain for what cannot be tiled (an address not 64 KiB aligned, a resource the
//! driver does not make reserved: D32S8, MSAA): heaps of the placement's size, shared by
//! placements inside them, let go 2 s after their last resource (256 MiB ones: 2.8 GB of D32S8).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// A heap of video memory for placements at title addresses `[base, base + size)`.
struct Arena<H> {
    base: usize,
    size: usize,
    /// the heap's alignment (a placement that needs more does not go in)
    align: usize,
    heap: H,
    /// resources placed in it
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
    id: u64,
}

/// The arenas, apart from D3D12 (`H`: the heap; numbers in the tests).
struct Arenas<H> {
    list: Vec<Arena<H>>,
    next_id: u64,
}

impl<H: Clone> Arenas<H> {
    const fn new() -> Self {
        Arenas { list: Vec::new(), next_id: 0 }
    }

    /// Arenas no resource used for `IDLE` go (their heap with the last reference).
    fn sweep(&mut self, now: std::time::Instant) {
        self.list.retain(|a| a.users > 0 || a.idle_since.is_none_or(|t| now.duration_since(t) < IDLE));
    }

    /// Where a placement of `bytes` at `va` (alignment `align`) goes: the newest arena holding
    /// it at an offset `align` allows, else a new arena of its size, `make(size, align)`.
    /// (heap, offset, arena id); None: no heap.
    fn take(&mut self, va: usize, bytes: usize, align: usize, now: std::time::Instant, make: impl FnOnce(usize, usize) -> Option<H>) -> Option<(H, usize, u64)> {
        self.sweep(now);
        let fits = |a: &Arena<H>| a.align >= align && a.base <= va && va + bytes <= a.base + a.size && (va - a.base) % align == 0;
        let i = match self.list.iter().rposition(fits) {
            Some(i) => i,
            None => {
                let size = bytes.div_ceil(align) * align;
                let heap = make(size, align)?;
                self.next_id += 1;
                self.list.push(Arena { base: va, size, align, heap, users: 0, idle_since: None, id: self.next_id });
                self.list.len() - 1
            }
        };
        let a = &mut self.list[i];
        a.users += 1;
        a.idle_since = None;
        Some((a.heap.clone(), va - a.base, a.id))
    }

    /// A resource in arena `id` released.
    fn release(&mut self, id: u64, now: std::time::Instant) {
        if let Some(a) = self.list.iter_mut().find(|a| a.id == id) {
            a.users = a.users.saturating_sub(1);
            if a.users == 0 {
                a.idle_since = Some(now);
            }
        }
    }

    fn bytes(&self) -> usize { self.list.iter().map(|a| a.size).sum() }
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arenas<ID3D12Heap> {}

static ARENAS: Mutex<Arenas<ID3D12Heap>> = Mutex::new(Arenas::new());
/// resource → its arena's id
static ARENA_OF: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
/// bytes of every arena made
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (bytes of the arenas in use, bytes of all arenas made) – `XCL_ACTIVITY`
pub fn arena_bytes() -> (usize, usize) {
    (ARENAS.lock().unwrap().bytes(), MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed))
}
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains(&resource))
}

pub(super) fn forget(resource: usize) {
    forget_small(resource);
    let placed = PLACED_TEXTURES.lock().unwrap().as_mut().is_some_and(|textures| textures.remove(&resource));
    if placed {
        forget_pages(resource);
        if let Some(id) = ARENA_OF.lock().unwrap().as_mut().and_then(|m| m.remove(&resource)) {
            ARENAS.lock().unwrap().release(id, std::time::Instant::now());
        }
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// A new arena's heap: `size` bytes of video memory for title addresses from `va`.
unsafe fn arena_heap(dev: &ID3D12Device, va: usize, size: usize, align: usize) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // 64 KiB, or 4 MiB for MSAA targets
        Alignment: align as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: video-memory arena for {va:#x}: CreateHeap: {e}");
        return None;
    }
    let made = MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: video-memory arena #{n} for title addresses {va:#x}..{:#x} ({made} bytes of arenas made so far)", va + size);
    }
    heap
}

/// bytes of a page – of the title's addresses and of a tile
const PAGE: usize = 0x1_0000;
/// pages of a video-memory pool heap (64 MiB)
const POOL_PAGES: u32 = 1024;
/// a page no target uses any more is kept this long for the next target at its address (with
/// what it holds – a title may make a target again over memory it rendered to)
const IDLE: std::time::Duration = std::time::Duration::from_secs(2);
/// a pool tile given back is not reused before this (the GPU may still be at work on what the
/// title released)
const COOL: std::time::Duration = std::time::Duration::from_secs(2);
/// Page budget (`XCL_PAGE_BUDGET_MB`): a texture the CPU fills is not evicted before this
/// long after it was placed, remapped or used in a transition.
const EVICT_MIN_AGE: std::time::Duration = std::time::Duration::from_secs(5);
/// Page budget: textures smaller than this are never evicted (UI, fonts, small lookups).
const EVICT_MIN_TILES: u32 = 4;
/// Page budget: tiles freed per eviction round once the pools are at the budget.
const EVICT_BATCH: usize = 512;
/// Page budget: after a round that gave back fewer tiles than wanted, pools grow past the
/// budget for this long instead of trying again for every tile.
const EVICT_IDLE: std::time::Duration = std::time::Duration::from_secs(1);

static LAB_EVICT_SHARED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static LAB_BIND_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static LAB_SEEN_NO_TOUCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// lab: `XCL_EVICT_SHARED=1` (textures whose pages are all shared are evicted too: a group of
/// them over the same pages was never evicted), `XCL_SEEN_TOUCH=0` (see `seen`): the variable
/// is set to "1" for the first, "0" for the second.
fn lab_flag(cell: &std::sync::OnceLock<bool>, var: &str) -> bool {
    *cell.get_or_init(|| std::env::var(var).is_ok_and(|v| v.trim() == if var == "XCL_SEEN_TOUCH" { "0" } else { "1" }))
}

/// Page budget: pool heaps the pages may take, from `XCL_PAGE_BUDGET_MB` (0: no limit).
fn pool_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let mb: usize = std::env::var("XCL_PAGE_BUDGET_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        if mb == 0 {
            return 0;
        }
        // evicting unmaps a texture from another thread: only safe while a texture's last
        // Release reaches `forget` before the object goes (`late_release` holds it until then)
        if !super::late_release::enabled() {
            crate::xlog_warn!("D3D12.X: XCL_PAGE_BUDGET_MB needs late release (XCL_LATE_RELEASE=0 is set): no page budget");
            return 0;
        }
        let pools = (mb << 20).div_ceil(POOL_PAGES as usize * PAGE).max(1);
        crate::xlog_info!("D3D12.X: page budget {mb} MB ({pools} pools of 64 MB): CPU-filled textures are evicted least recently used first beyond it");
        pools
    })
}

/// Page budget: textures whose tiles were evicted; the next upload or transition maps
/// them again and uploads them whole.
static EVICTED: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
/// the number of textures in `EVICTED` (callers skip the lookup while it is 0)
static EVICTED_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static EVICTIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REMAPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Page budget: textures with no T# in any view heap (while the heaps are looked at) may be
/// evicted this long after they were placed or last seen, small ones too.
const EVICT_UNSEEN_AGE: std::time::Duration = std::time::Duration::from_secs(2);
/// whether the console view heaps are looked at (`seen`): only then does "no T#" mean unused
static HEAPS_LOOKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A pool tile: (pool heap, tile in it).
type Tile = (u32, u32);

struct Page {
    tile: Tile,
    /// targets using the page
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
}

/// A page of `Pages::pages` (its title address) or of `Pages::extra_pages` (its key there).
#[derive(Clone, Copy)]
enum PageKey {
    Own(usize),
    Extra(usize, usize),
}

/// The pages of `idle` – those whose last user went, oldest first, each with when – that no
/// resource has used for `IDLE` go back to the pools: out of their map, their tile to `cooling`.
/// An entry whose page was used again since (in use now, or idle again from a later moment, with
/// a later entry) is dropped. Only the queue is looked at: every page of both maps, looked at at
/// every placement, grew with the square of what a title streams.
fn sweep_idle(
    idle: &mut std::collections::VecDeque<(PageKey, std::time::Instant)>,
    pages: &mut std::collections::HashMap<usize, Page>,
    extra_pages: &mut std::collections::HashMap<(usize, usize), Page>,
    cooling: &mut std::collections::VecDeque<(Tile, std::time::Instant)>,
    now: std::time::Instant,
) {
    while idle.front().is_some_and(|&(_, since)| now.duration_since(since) >= IDLE) {
        let Some((key, since)) = idle.pop_front() else { break };
        let still_idle = |p: &Page| p.idle_since == Some(since);
        let page = match key {
            PageKey::Own(a) if pages.get(&a).is_some_and(still_idle) => pages.remove(&a),
            PageKey::Extra(va, i) if extra_pages.get(&(va, i)).is_some_and(still_idle) => extra_pages.remove(&(va, i)),
            _ => None,
        };
        if let Some(p) = page {
            cooling.push_back((p.tile, now));
        }
    }
}

struct Target {
    va: usize,
    console_bytes: Option<u64>,
    total: u32,
    mapped: bool,
    used: Vec<usize>,
    used_extra: Vec<(usize, usize)>,
    /// a texture the CPU fills (its texels can be uploaded again from the title's memory)
    evictable: bool,
    /// placed, remapped, transitioned or its T# seen in a shader-visible heap last
    touched: std::time::Instant,
    /// placed as GPU-owned: its pages are keyed by the physical page under them (`page_key`)
    gpu_owned: bool,
    /// where the last look at the view heaps found its T# (`seen`)
    seen: Seen,
}

/// Where a texture's T# was at the last look at the console view heaps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Seen {
    None,
    CpuOnly,
    Visible,
}

/// Per-heap free-tile allocator for `Pages`.
///
/// Keeping free tiles partitioned by pool heap (sorted descending so `.pop()` yields ascending
/// contiguous tile indices) prevents streaming churn across 100+ 64 MiB pools from scattering a
/// single texture across dozens of heaps — which otherwise explodes `Pages::map` into dozens of
/// `UpdateTileMappings` calls per texture.
struct TilePool {
    /// Free tile indices for each pool heap (descending when `!dirty_sort[h]`).
    free_by_heap: Vec<Vec<u32>>,
    /// Whether `free_by_heap[h]` must be sorted descending before popping.
    dirty_sort: Vec<bool>,
    /// Total free tiles currently available across all heaps.
    free_total: usize,
    /// Tiles given back, reusable after `COOL`.
    cooling: std::collections::VecDeque<(Tile, std::time::Instant)>,
}

impl TilePool {
    fn new() -> Self {
        TilePool {
            free_by_heap: Vec::new(),
            dirty_sort: Vec::new(),
            free_total: 0,
            cooling: std::collections::VecDeque::new(),
        }
    }

    fn add_heap(&mut self, pool_pages: u32) -> u32 {
        let h = self.free_by_heap.len() as u32;
        self.free_by_heap.push((0..pool_pages).rev().collect());
        self.dirty_sort.push(false);
        self.free_total += pool_pages as usize;
        h
    }

    fn push_cooling(&mut self, tile: Tile, now: std::time::Instant) {
        self.cooling.push_back((tile, now));
    }

    /// Move every cooled tile (`>= COOL`) into its pool heap's free list so best-fit allocation
    /// sees all ready tiles at once.
    fn drain_cooling(&mut self, now: std::time::Instant) {
        while self.cooling.front().is_some_and(|&(_, at)| now.duration_since(at) >= COOL) {
            let Some(((h, t), _)) = self.cooling.pop_front() else { break };
            let idx = h as usize;
            if idx < self.free_by_heap.len() {
                self.free_by_heap[idx].push(t);
                self.dirty_sort[idx] = true;
                self.free_total += 1;
            }
        }
    }

    fn return_tiles(&mut self, tiles: &[Tile]) {
        for &(h, t) in tiles {
            let idx = h as usize;
            if idx < self.free_by_heap.len() {
                self.free_by_heap[idx].push(t);
                self.dirty_sort[idx] = true;
                self.free_total += 1;
            }
        }
    }

    /// Tiles free now or cooling (page budget: what a placement at the cap can have without a
    /// new pool heap).
    fn spare(&self) -> usize {
        self.free_total + self.cooling.len()
    }

    /// Page budget: up to `n` tiles still cooling are free now, oldest first (at the budget a
    /// tile is reused before it cooled).
    fn reuse_cooling(&mut self, n: usize) {
        for _ in 0..n {
            let Some(((h, t), _)) = self.cooling.pop_front() else { break };
            self.return_tiles(&[(h, t)]);
        }
    }

    /// Pick the pool heap to take up to `rem` tiles from:
    /// 1. `preferred` if it fits all `rem` tiles (keeps partial-overlap placements on 1 heap),
    /// 2. otherwise best-fit heap with `len >= rem` (fits the whole texture in 1 heap without
    ///    splitting fuller pools),
    /// 3. otherwise the heap with the most free tiles (minimizing distinct heaps touched).
    fn pick_heap(&self, rem: usize, preferred: Option<u32>) -> Option<usize> {
        if self.free_total == 0 {
            return None;
        }
        if let Some(p) = preferred.map(|h| h as usize).filter(|&h| h < self.free_by_heap.len()) {
            if self.free_by_heap[p].len() >= rem {
                return Some(p);
            }
        }
        let mut best_fit: Option<(usize, usize)> = None;
        let mut largest: Option<(usize, usize)> = None;
        for (h, tiles) in self.free_by_heap.iter().enumerate() {
            let len = tiles.len();
            if len == 0 {
                continue;
            }
            if len >= rem {
                if best_fit.is_none_or(|(_, blen)| len < blen) {
                    best_fit = Some((h, len));
                }
            } else if largest.is_none_or(|(lh, llen)| len > llen || (len == llen && preferred == Some(h as u32) && preferred != Some(lh as u32))) {
                largest = Some((h, len));
            }
        }
        best_fit.or(largest).map(|(h, _)| h)
    }

    /// Allocate `count` tiles, preferring a single heap and ascending contiguous offsets.
    /// `make_heap` allocates a new backing pool heap of `pool_pages` tiles when `free_total == 0`.
    /// On failure, any tiles already taken are returned untouched.
    fn take_batch(
        &mut self,
        count: usize,
        preferred: Option<u32>,
        now: std::time::Instant,
        pool_pages: u32,
        mut make_heap: impl FnMut() -> Option<u32>,
    ) -> Option<Vec<Tile>> {
        if count == 0 {
            return Some(Vec::new());
        }
        self.drain_cooling(now);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let rem = count - out.len();
            let h = match self.pick_heap(rem, preferred) {
                Some(h) => h,
                None => {
                    let Some(new_h) = make_heap() else {
                        self.return_tiles(&out);
                        return None;
                    };
                    let added = self.add_heap(pool_pages);
                    debug_assert_eq!(added, new_h);
                    added as usize
                }
            };
            if self.dirty_sort[h] {
                self.free_by_heap[h].sort_unstable_by(|a, b| b.cmp(a));
                self.dirty_sort[h] = false;
            }
            let free = &mut self.free_by_heap[h];
            let take = rem.min(free.len());
            for _ in 0..take {
                let t = free.pop().unwrap();
                out.push((h as u32, t));
            }
            self.free_total -= take;
        }
        Some(out)
    }
}

/// Allocate one 64 MiB default heap for `Pages`.
unsafe fn alloc_pool_heap(dev: &ID3D12Device) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        Alignment: 0,
        // every kind of texture (resource heap tier 2: the only flags for both)
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&desc, &mut heap)) {
        crate::xlog_warn!("D3D12.X: no video memory for more pages of title textures ({e})");
        return None;
    }
    heap
}

/// The title's pages of video memory for tiled textures (see the module's comment).
struct Pages {
    queue: ID3D12CommandQueue,
    /// signalled after each mapping; title queues wait for it before their next work
    fence: ID3D12Fence,
    value: u64,
    pools: Vec<ID3D12Heap>,
    pool: TilePool,
    /// title page address → its video memory
    pages: std::collections::HashMap<usize, Page>,
    /// extra / sub-64 KiB tiles keyed by `(va, tile_index)` so repeated placements at the same
    /// address share them without aliasing unrelated addresses
    extra_pages: std::collections::HashMap<(usize, usize), Page>,
    /// pages of both maps whose last user went, oldest first, with when (`sweep_idle`)
    idle: std::collections::VecDeque<(PageKey, std::time::Instant)>,
    /// resource → its mapping metadata and currently held pages / extra keys
    targets: std::collections::HashMap<usize, Target>,
    /// queue → the fence value it last waited for
    waited: std::collections::HashMap<usize, u64>,
    /// cached `reservable()` answers for `(DXGI_FORMAT, SampleCount)` on this device
    reservable_cache: std::collections::HashMap<(i32, u32), bool>,
    /// reusable scratch buffers for `map()`
    scratch_heaps: Vec<u32>,
    scratch_flags: Vec<D3D12_TILE_RANGE_FLAGS>,
    scratch_offsets: Vec<u32>,
    scratch_counts: Vec<u32>,
    /// page budget: no eviction round before this (the last one gave back fewer tiles than
    /// wanted: `EVICT_IDLE`)
    evict_idle_until: Option<std::time::Instant>,
    /// page budget: pool heaps the pages may take (`pool_cap`; 0: no limit)
    cap: usize,
    /// lab (`XCL_ZERO_TARGET_PAGES=1`): what zeroes pool tiles (`Pages::scrub`)
    scrub: Option<Scrub>,
}

/// tiles `Pages::scrub` zeroes with one copy (16 MiB)
const SCRUB_TILES: usize = 256;

/// Lab (`XCL_ZERO_TARGET_PAGES=1`): a reserved buffer mapped onto pool tiles for a moment and a
/// buffer of zeros copied over it – a tile of a pool heap cannot be written any other way before
/// a texture is mapped onto it.
struct Scrub {
    over: ID3D12Resource,
    zeros: ID3D12Resource,
    /// (allocator, list, `Pages::value` after its last copy): one is used again once the fence
    /// is there – never waited for (the mapping queue may itself be waiting for the title's
    /// work, which waits for the pages lock this runs under: runs stood still that way)
    lists: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList, u64)>,
}

/// lab: `XCL_ZERO_TARGET_PAGES=1` – pool tiles given to a render target, depth buffer or UAV
/// texture are zeroed first. A tile comes back from evicted streamed textures with their bytes,
/// and a retail title reads targets it has not written everywhere (history, layers only some of
/// its passes fill): it then read texture bytes as radiance.
fn zero_target_pages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_ZERO_TARGET_PAGES").is_ok_and(|v| v.trim() == "1"))
}

static SCRUBBED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// lab: tiles zeroed so far (`XCL_ZERO_TARGET_PAGES`)
pub fn scrubbed_tiles() -> u64 {
    SCRUBBED.load(std::sync::atomic::Ordering::Relaxed)
}

// D3D12 objects are free-threaded
unsafe impl Send for Pages {}

/// device raw pointer → its pages (None inside: tiled tier < 2 on this device)
static PAGES: Mutex<std::collections::BTreeMap<usize, Option<Pages>>> = Mutex::new(std::collections::BTreeMap::new());

impl Pages {
    unsafe fn new(dev: &ID3D12Device) -> Option<Pages> {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        // tier 2: unmapped tiles read as zero, writes to them go nowhere
        if o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            crate::xlog_info!("D3D12.X: tiled resources tier {}: GPU-owned textures in arenas", o.TiledResourcesTier.0);
            return None;
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        let _ = queue.SetName(windows::core::w!("xcl vram tile-mapping queue"));
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        crate::xlog_info!("D3D12.X: textures in title memory: tiled resources on 64 KiB pages of video memory, one per page of the title's");
        Some(Pages {
            queue,
            fence,
            value: 0,
            pools: Vec::new(),
            pool: TilePool::new(),
            pages: std::collections::HashMap::new(),
            extra_pages: std::collections::HashMap::new(),
            idle: std::collections::VecDeque::new(),
            targets: std::collections::HashMap::new(),
            waited: std::collections::HashMap::new(),
            reservable_cache: std::collections::HashMap::new(),
            scratch_heaps: Vec::new(),
            scratch_flags: Vec::new(),
            scratch_offsets: Vec::new(),
            scratch_counts: Vec::new(),
            evict_idle_until: None,
            cap: pool_cap(),
            scrub: None,
        })
    }

    /// Lab (`XCL_ZERO_TARGET_PAGES=1`): zero the pool tiles `tiles` on the mapping queue, before
    /// anything is mapped onto them (the title's queues wait for this queue's fence).
    unsafe fn scrub(&mut self, dev: &ID3D12Device, tiles: &[Tile]) {
        if self.scrub.is_none() {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: (SCRUB_TILES * PAGE) as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let made = super::title_textures::own_work(|| -> windows::core::Result<Scrub> {
                let mut over: Option<ID3D12Resource> = None;
                dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut over)?;
                let mut zeros: Option<ID3D12Resource> = None;
                // (a committed buffer is made of zeros)
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut zeros,
                )?;
                let (Some(over), Some(zeros)) = (over, zeros) else { return Err(windows::core::Error::from(windows::Win32::Foundation::E_FAIL)) };
                Ok(Scrub { over, zeros, lists: Vec::new() })
            });
            match made {
                Ok(s) => self.scrub = Some(s),
                Err(e) => {
                    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        crate::xlog_warn!("D3D12.X: lab: XCL_ZERO_TARGET_PAGES: nothing to zero tiles with ({e})");
                    }
                    return;
                }
            }
        }
        let mut sorted: Vec<Tile> = tiles.to_vec();
        sorted.sort_unstable();
        for chunk in sorted.chunks(SCRUB_TILES) {
            // a list whose last copy has run, or a new one
            let done = self.fence.GetCompletedValue();
            let Some(sc) = self.scrub.as_mut() else { return };
            let k = match sc.lists.iter().position(|l| l.2 <= done) {
                Some(k) => k,
                None => {
                    let made = super::title_textures::own_work(|| -> windows::core::Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList, u64)> {
                        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
                        list.Close()?;
                        Ok((alloc, list, 0))
                    });
                    let Ok(l) = made else { return };
                    sc.lists.push(l);
                    sc.lists.len() - 1
                }
            };
            let Some(sc) = self.scrub.as_ref() else { return };
            let (alloc, list) = (&sc.lists[k].0, &sc.lists[k].1);
            // tile i of the buffer over `chunk[i]`: one call per pool heap
            let mut at = 0usize;
            while at < chunk.len() {
                let h = chunk[at].0;
                let n = chunk[at..].iter().take_while(|t| t.0 == h).count();
                let (mut offsets, mut counts): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
                for &(_, t) in &chunk[at..at + n] {
                    match (offsets.last(), counts.last_mut()) {
                        (Some(&o), Some(c)) if o + *c == t => *c += 1,
                        _ => {
                            offsets.push(t);
                            counts.push(1);
                        }
                    }
                }
                let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; offsets.len()];
                let coord = D3D12_TILED_RESOURCE_COORDINATE { X: at as u32, ..Default::default() };
                let region = D3D12_TILE_REGION_SIZE { NumTiles: n as u32, UseBox: false.into(), ..Default::default() };
                self.queue.UpdateTileMappings(
                    &sc.over,
                    1,
                    Some(&coord),
                    Some(&region),
                    &self.pools[h as usize],
                    offsets.len() as u32,
                    Some(flags.as_ptr()),
                    Some(offsets.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
                at += n;
            }
            if alloc.Reset().is_err() || list.Reset(alloc, None).is_err() {
                return;
            }
            list.CopyBufferRegion(&sc.over, 0, &sc.zeros, 0, (chunk.len() * PAGE) as u64);
            if list.Close().is_err() {
                return;
            }
            let Ok(cl) = list.cast::<ID3D12CommandList>() else { return };
            // (XCL's own work: the queue hook must not take it for the title's and refresh)
            super::title_textures::own_work(|| self.queue.ExecuteCommandLists(&[Some(cl)]));
            self.value += 1;
            let _ = self.queue.Signal(&self.fence, self.value);
            let busy = self.value;
            if let Some(sc) = self.scrub.as_mut() {
                sc.lists[k].2 = busy;
            }
            let n = SCRUBBED.fetch_add(chunk.len() as u64, std::sync::atomic::Ordering::Relaxed) + chunk.len() as u64;
            if n == chunk.len() as u64 {
                crate::xlog_info!("D3D12.X: lab: XCL_ZERO_TARGET_PAGES: pool tiles given to render targets / UAV textures are zeroed first");
            }
        }
    }

    /// Pages no resource used for `IDLE` go back to the pools.
    fn sweep(&mut self, now: std::time::Instant) {
        sweep_idle(&mut self.idle, &mut self.pages, &mut self.extra_pages, &mut self.pool.cooling, now);
    }

    /// Map every tile of `res` (`tiles[i]`: overall tile i) – one call per pool heap.
    unsafe fn map(&mut self, res: &ID3D12Resource, tiles: &[Tile]) {
        if tiles.is_empty() {
            return;
        }
        super::real::removed_probe("vram: before map");
        self.scratch_heaps.clear();
        self.scratch_heaps.extend(tiles.iter().map(|t| t.0));
        self.scratch_heaps.sort_unstable();
        self.scratch_heaps.dedup();
        for i in 0..self.scratch_heaps.len() {
            let h = self.scratch_heaps[i];
            let ranges = heap_ranges_trimmed(h, tiles);
            if ranges.is_empty() {
                continue;
            }
            self.scratch_flags.clear();
            self.scratch_offsets.clear();
            self.scratch_counts.clear();
            let mut num_tiles = 0u32;
            for &(mapped, off, count) in &ranges {
                self.scratch_flags.push(if mapped { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP });
                self.scratch_offsets.push(off);
                self.scratch_counts.push(count);
                num_tiles += count;
            }
            let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
            let region = D3D12_TILE_REGION_SIZE { NumTiles: num_tiles, UseBox: false.into(), ..Default::default() };
            self.queue.UpdateTileMappings(
                res,
                1,
                Some(&coord),
                Some(&region),
                &self.pools[h as usize],
                ranges.len() as u32,
                Some(self.scratch_flags.as_ptr()),
                Some(self.scratch_offsets.as_ptr()),
                Some(self.scratch_counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
        super::real::removed_probe("vram: after map");
    }

    /// Unmap every tile of `res` (`D3D12_TILE_RANGE_FLAG_NULL`) so its unshared tiles can return
    /// to the pool even while the title keeps the COM object alive.
    unsafe fn unmap(&mut self, res: &ID3D12Resource, total: u32) {
        if total == 0 {
            return;
        }
        super::real::removed_probe("vram: before unmap");
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let region = D3D12_TILE_REGION_SIZE { NumTiles: total, UseBox: false.into(), ..Default::default() };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        self.queue.UpdateTileMappings(
            res,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&flag),
            None,
            Some(&total),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    fn release_tiles(&mut self, used: Vec<usize>, used_extra: Vec<(usize, usize)>, now: std::time::Instant, immediate_cool: bool) {
        for a in used {
            if let Some(p) = self.pages.get_mut(&a) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.pages.remove(&a) {
                            self.pool.push_cooling(p.tile, now);
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Own(a), now));
                    }
                }
            }
        }
        for k in used_extra {
            if let Some(p) = self.extra_pages.get_mut(&k) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.extra_pages.remove(&k) {
                            self.pool.push_cooling(p.tile, now);
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Extra(k.0, k.1), now));
                    }
                }
            }
        }
    }

    /// Page budget: unmap CPU-filled textures, least recently touched first, until `want`
    /// tiles came back (pages other mapped textures still share stay). Each goes to `EVICTED`.
    /// After a round that fell short, none for `EVICT_IDLE`: every tile taken at the budget
    /// would otherwise sort and walk the same textures again.
    unsafe fn evict_lru(&mut self, want: usize) -> usize {
        let now = std::time::Instant::now();
        if self.evict_idle_until.is_some_and(|t| now < t) {
            return 0;
        }
        let before = self.pool.spare();
        // with the view heaps looked at, a texture whose T# is in none cannot be sampled: it
        // goes first, small or not, once `EVICT_UNSEEN_AGE` old
        let heaps = HEAPS_LOOKED.load(std::sync::atomic::Ordering::Relaxed);
        let mut order: Vec<(bool, std::time::Instant, usize)> = self
            .targets
            .iter()
            .filter_map(|(&k, t)| {
                if !t.evictable || !t.mapped {
                    return None;
                }
                let unseen = heaps && t.seen == Seen::None;
                let age = now.duration_since(t.touched);
                let ok = if unseen { age >= EVICT_UNSEEN_AGE } else { t.total >= EVICT_MIN_TILES && age >= EVICT_MIN_AGE };
                ok.then_some((!unseen, t.touched, k))
            })
            .collect();
        order.sort_unstable();
        let mut evicted = Vec::new();
        let mut shared = 0u64;
        for (_, _, key) in order {
            if self.pool.spare() - before >= want {
                break;
            }
            let Some(t) = self.targets.get(&key) else { continue };
            // a texture whose pages all are other mapped textures' too gives nothing back
            let alone = t.used.iter().any(|a| self.pages.get(a).is_some_and(|p| p.users <= 1))
                || t.used_extra.iter().any(|k| self.extra_pages.get(k).is_some_and(|p| p.users <= 1));
            if !alone && !lab_flag(&LAB_EVICT_SHARED, "XCL_EVICT_SHARED") {
                shared += 1;
                continue;
            }
            let Some(t) = self.targets.get_mut(&key) else { continue };
            let total = t.total;
            let used = std::mem::take(&mut t.used);
            let used_extra = std::mem::take(&mut t.used_extra);
            t.mapped = false;
            let raw = key as *mut c_void;
            if let Some(res) = ID3D12Resource::from_raw_borrowed(&raw) {
                self.unmap(res, total);
            }
            self.release_tiles(used, used_extra, now, true);
            evicted.push(key);
        }
        let freed = self.pool.spare() - before;
        self.evict_idle_until = (freed < want).then(|| now + EVICT_IDLE);
        if !evicted.is_empty() {
            let n = EVICTIONS.fetch_add(evicted.len() as u64, std::sync::atomic::Ordering::Relaxed) + evicted.len() as u64;
            static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let step = LOGGED.load(std::sync::atomic::Ordering::Relaxed);
            if n >= step {
                LOGGED.store(n.max(1) * 2, std::sync::atomic::Ordering::Relaxed);
                crate::xlog_info!("D3D12.X: page budget: {} textures evicted ({} MB back) this round, {n} so far; {shared} passed over, all their pages shared", evicted.len(), (freed * PAGE) >> 20);
            }
            let mut g = EVICTED.lock().unwrap();
            let s = g.get_or_insert_with(HashSet::new);
            s.extend(evicted);
            EVICTED_LEN.store(s.len(), std::sync::atomic::Ordering::Relaxed);
        }
        freed
    }

    /// Acquire/share all `total` tiles for `res` at `va` and map them on the `Pages` queue
    /// (`phys[i]`: the title's physical page at owned page i, if any – see `page_key`).
    unsafe fn bind(&mut self, dev: &ID3D12Device, res: &ID3D12Resource, va: usize, console_bytes: Option<u64>, total: u32, phys: &[Option<usize>]) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        let (first, own) = (total > 0).then(|| owned_pages(va, console_bytes, total as usize)).flatten()?;
        let now = std::time::Instant::now();
        self.sweep(now);

        let count = |s: &Self| {
            let mut missing = 0usize;
            let mut preferred: Option<u32> = None;
            for i in 0..total as usize {
                let existing = if i < own {
                    s.pages.get(&page_key(first + i * PAGE, phys.get(i).copied().flatten())).map(|p| p.tile.0)
                } else {
                    s.extra_pages.get(&(va, i)).map(|p| p.tile.0)
                };
                match existing {
                    Some(h) if preferred.is_none() => preferred = Some(h),
                    Some(_) => {}
                    None => missing += 1,
                }
            }
            (missing, preferred)
        };
        let (mut missing, mut preferred) = count(self);
        // lab (`XCL_BIND_LOG=1`): how a large texture gets its pages - which of them exist already
        // (shared by address or physical page with another texture: they keep what they hold)
        // and which are new (zeroed for a target with `XCL_ZERO_TARGET_PAGES`)
        // (3-D textures of any size too, with their page keys: the colour LUTs are right after their
        // bake and hold another LUT's rows by the present - run U1)
        if lab_flag(&LAB_BIND_LOG, "XCL_BIND_LOG") && res.GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            static N3: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            if N3.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 600 {
                let d = res.GetDesc();
                let keys: Vec<String> = (0..(total as usize).min(12))
                    .map(|i| if i < own { format!("{:#x}{}", page_key(first + i * PAGE, phys.get(i).copied().flatten()), if self.pages.contains_key(&page_key(first + i * PAGE, phys.get(i).copied().flatten())) { "*" } else { "" }) } else { format!("extra{}", if self.extra_pages.contains_key(&(va, i)) { "*" } else { "" }) })
                    .collect();
                crate::xlog_info!(
                    "lab: bind 3-D {:p} at {va:#x}: {}x{}x{} format {} flags {:#x} console bytes {:?}, {total} tiles, first page {first:#x}, {own} owned, keys (* = exists already) {} at {:.1} s",
                    res.as_raw(), d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.Flags.0, console_bytes, keys.join(" "), crate::log::uptime()
                );
            }
        }
        if total >= 64 && lab_flag(&LAB_BIND_LOG, "XCL_BIND_LOG") {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let d = res.GetDesc();
            // (screen-sized textures only, from the world on: the menus' streamed textures used the lines up)
            // targets of 1280 or wider always; flag-less textures only at screen sizes (the streamed 2048 / 4096 ones used the lines up)
            let target = d.Flags.0 & 0x7 != 0;
            let screen = d.Width >= 1280 && d.Width <= 2560 && d.Height >= 720 && d.Height <= 1440 && crate::log::uptime() > 60.0;
            if (target && d.Width >= 1280 || !target && screen) && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3000 {
                let by_phys = (0..own).filter(|&i| phys.get(i).copied().flatten().is_some()).count();
                let new: Vec<usize> = (0..total as usize)
                    .filter(|&i| if i < own { !self.pages.contains_key(&page_key(first + i * PAGE, phys.get(i).copied().flatten())) } else { !self.extra_pages.contains_key(&(va, i)) })
                    .collect();
                crate::xlog_info!(
                    "lab: bind {:p} at {va:#x}: {}x{} format {} flags {:#x}, {total} tiles, {own} owned pages ({by_phys} keyed by physical page), {missing} new (first {:?}, last {:?}) at {:.1} s",
                    res.as_raw(), d.Width, d.Height, d.Format.0, d.Flags.0, new.first(), new.last(), crate::log::uptime()
                );
            }
        }

        if self.cap > 0 && self.pools.len() >= self.cap && missing > 0 {
            // page budget: rounds of `EVICT_BATCH` tiles give tiles back (to `cooling`), which this
            // and the next placements take first; a round that fell short is not repeated for
            // `EVICT_IDLE` (each one walks every target): the pools grow past the budget meanwhile
            let mut evicted = false;
            while self.pool.spare() < missing && self.evict_idle_until.is_none_or(|t| now >= t) {
                if self.evict_lru(EVICT_BATCH) == 0 {
                    self.evict_idle_until = Some(now + EVICT_IDLE);
                    break;
                }
                evicted = true;
            }
            if evicted {
                // pages this texture would have shared may have gone with the evicted ones
                (missing, preferred) = count(self);
            }
            // at the budget a tile is reused before it cooled: a draw still in flight may read
            // the newer texels there for a frame
            self.pool.drain_cooling(now);
            self.pool.reuse_cooling(missing.saturating_sub(self.pool.free_total));
        }

        let pools = &mut self.pools;
        let pool = &mut self.pool;
        let fresh = pool.take_batch(missing, preferred, now, POOL_PAGES, || {
            let heap = alloc_pool_heap(dev)?;
            pools.push(heap);
            Some(pools.len() as u32 - 1)
        })?;
        // lab: a target the GPU fills starts from zeros, not from what the tile held
        if !fresh.is_empty()
            && zero_target_pages()
            && res.GetDesc().Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) != 0
        {
            self.scrub(dev, &fresh);
        }
        let mut fresh_iter = fresh.into_iter();

        let mut tiles = Vec::with_capacity(total as usize);
        let mut used = Vec::with_capacity(own);
        let mut used_extra = Vec::new();
        for i in 0..total as usize {
            let tile = if i < own {
                let a = page_key(first + i * PAGE, phys.get(i).copied().flatten());
                let t = match self.pages.get_mut(&a) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        p.tile
                    }
                    None => {
                        let t = fresh_iter.next().unwrap();
                        self.pages.insert(a, Page { tile: t, users: 1, idle_since: None });
                        t
                    }
                };
                used.push(a);
                t
            } else {
                let k = (va, i);
                let t = match self.extra_pages.get_mut(&k) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        p.tile
                    }
                    None => {
                        let t = fresh_iter.next().unwrap();
                        self.extra_pages.insert(k, Page { tile: t, users: 1, idle_since: None });
                        t
                    }
                };
                used_extra.push(k);
                t
            };
            tiles.push(tile);
        }
        self.map(res, &tiles);
        Some((used, used_extra))
    }
}

/// The ranges of one UpdateTileMappings call for pool heap `heap` over a resource whose overall
/// tile i goes to `tiles[i]`: (mapped, tile offset in the heap, count) – runs of consecutive
/// tiles of the heap, and skipped runs for tiles of other heaps.
fn heap_ranges(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = Vec::new();
    for &(h, t) in tiles {
        let mine = h == heap;
        match out.last_mut() {
            Some(r) if mine && r.0 && r.1 + r.2 == t => r.2 += 1,
            Some(r) if !mine && !r.0 => r.2 += 1,
            _ => out.push((mine, if mine { t } else { 0 }, 1)),
        }
    }
    out
}

/// Like `heap_ranges`, with any trailing `SKIP` range trimmed so `UpdateTileMappings` only walks
/// up to the last tile belonging to `heap`.
fn heap_ranges_trimmed(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out = heap_ranges(heap, tiles);
    while out.last().is_some_and(|r| !r.0) {
        out.pop();
    }
    out
}

/// 64 KiB page boundaries `P` with `va <= P < va + bytes`, capped at `total` tiles:
/// `(first_page, count)`. Because every `P` lies strictly inside `[va, va + bytes)`, two
/// non-overlapping ranges never share a page here even when `va` is only 4 KiB aligned.
fn owned_pages(va: usize, console_bytes: Option<u64>, total: usize) -> Option<(usize, usize)> {
    match console_bytes {
        Some(b) => {
            let bytes = usize::try_from(b).ok()?;
            let first = va.checked_next_multiple_of(PAGE)?;
            let end = va.saturating_add(bytes);
            let n = if first < end { (end - first).div_ceil(PAGE) } else { 0 };
            Some((first, n.min(total)))
        }
        None => (va % PAGE == 0).then_some((va, total)),
    }
}

/// A reserved texture needs TILED in the format's support (`support2`) and, multisampled, a
/// tiled quality level at its sample count.
fn can_reserve(support2: i32, samples: u32, tiled_msaa_levels: u32) -> bool {
    support2 & D3D12_FORMAT_SUPPORT2_TILED.0 != 0 && (samples <= 1 || tiled_msaa_levels > 0)
}

/// Whether the device can make `desc` a reserved texture, asked before CreateReservedResource:
/// refused, it is an error in the debug layer (#940: R32G8X24_TYPELESS depth, 1 and 4 samples).
/// A query that fails says yes, and creation decides.
unsafe fn reservable(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    let mut f = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: desc.Format, ..Default::default() };
    if dev.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &mut f as *mut _ as *mut c_void, std::mem::size_of_val(&f) as u32).is_err() {
        return true;
    }
    // a format without TILED is not reservable whatever the sample count's query says
    if !can_reserve(f.Support2.0, 1, 0) {
        return false;
    }
    let samples = desc.SampleDesc.Count;
    let mut m = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: desc.Format, SampleCount: samples, Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE, NumQualityLevels: 0 };
    if samples > 1 && dev.CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &mut m as *mut _ as *mut c_void, std::mem::size_of_val(&m) as u32).is_err() {
        return true;
    }
    can_reserve(f.Support2.0, samples, m.NumQualityLevels)
}

/// Keys of `Pages::pages` for the title's physical pages (title addresses never have bit 63).
const PHYS_KEY: usize = 1 << 63;

/// The key of the page of video memory for title page address `a`: the physical page mapped
/// there (`phys`), else the address. A retail game maps one physical page under render targets
/// at two addresses (aliasing, as on the console): keyed by address, each got video memory of
/// its own (1.85 GB more in its first 40 s) and what the GPU wrote through one never showed in
/// the other.
fn page_key(a: usize, phys: Option<usize>) -> usize {
    phys.map_or(a, |p| PHYS_KEY | p)
}

/// The physical pages under a GPU-owned texture's owned pages (`require_aligned`), asked before
/// `PAGES` is taken; empty for others (keyed by address) and without the console size.
fn physical_pages_under(va: usize, console_bytes: Option<u64>, require_aligned: bool) -> Vec<Option<usize>> {
    // lab (`XCL_PHYS_KEYS_ALL=1`, the haze investigator's reading of G4): a texture WITHOUT target
    // flags at a page-aligned address is keyed by physical page too. Keyed by address it shares
    // with a flagged texture at its address only the pages that are at home: the reader of a
    // UAV-texture / sampled-texture pair saw its writer in part of the tiles only.
    static LAB_PHYS_ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !require_aligned && !(va % PAGE == 0 && lab_flag(&LAB_PHYS_ALL, "XCL_PHYS_KEYS_ALL")) {
        return Vec::new();
    }
    match console_bytes.and_then(|b| owned_pages(va, Some(b), usize::MAX)) {
        Some((first, n)) if n > 0 => crate::hle::kernelx::physical_pages(first, n),
        _ => Vec::new(),
    }
}

/// A texture at `va` as a reserved resource on the title's shared 64 KiB pages
/// (`console_bytes`: its size on the console; tiles past the owned page boundaries are shared by
/// `(va, tile_index)` so repeated placements at the same address do not leak). None: it cannot be.
#[allow(clippy::too_many_arguments)]
unsafe fn place_on_pages(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    require_aligned: bool,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    if desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D {
        return None;
    }
    if (require_aligned || console_bytes.is_none()) && va % PAGE != 0 {
        return None;
    }
    let phys = physical_pages_under(va, console_bytes, require_aligned);
    // reserved textures have the PC's 64 KiB tiles
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let log = |why: &str| {
        if require_aligned && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_info!("D3D12.X: GPU-owned texture at {va:#x} ({}x{}, format {}, {} samples) not reserved ({why}): in an arena", desc.Width, desc.Height, desc.Format.0, desc.SampleDesc.Count);
        }
    };
    let dev_key = dev.as_raw() as usize;
    let is_reservable = {
        let mut g = PAGES.lock().unwrap();
        let pages = g.entry(dev_key).or_insert_with(|| Pages::new(dev)).as_mut()?;
        *pages.reservable_cache.entry((tiled.Format.0, tiled.SampleDesc.Count)).or_insert_with(|| reservable(dev, &tiled))
    };
    if !is_reservable {
        log("no tiled support for its format or sample count");
        return None;
    }
    // CreateReservedResource and GetResourceTiling do not touch `PAGES`; keep them outside the
    // lock so concurrent streaming threads, `before_execute`, and `page_bytes` do not serialize.
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        log(&format!("{hr:#x}"));
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mut total = 0u32;
    dev.GetResourceTiling(res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
    let bound = {
        let mut g = PAGES.lock().unwrap();
        if let Some(Some(pages)) = g.get_mut(&dev_key) {
            if let Some((used, used_extra)) = pages.bind(dev, res, va, console_bytes, total, &phys) {
                pages.targets.insert(raw as usize, Target { va, console_bytes, total, mapped: true, used, used_extra, evictable: false, touched: std::time::Instant::now(), gpu_owned: require_aligned, seen: Seen::None });
                true
            } else {
                false
            }
        } else {
            false
        }
    };
    if !bound {
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    Some(hr)
}

/// Unmap a placed texture superseded by a placement with a different address range, returning its
/// unshared tiles (`extra_pages` and unshared tail pages) to the cooling pool immediately.
pub(super) unsafe fn unmap_superseded(res: &ID3D12Resource) {
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.get_mut(&key) else { continue };
        if !t.mapped {
            return;
        }
        let total = t.total;
        let used = std::mem::take(&mut t.used);
        let used_extra = std::mem::take(&mut t.used_extra);
        t.mapped = false;
        pages.unmap(res, total);
        pages.release_tiles(used, used_extra, now, true);
        return;
    }
}

/// Re-map a previously unmapped placed texture when the newer placement shadowing it is released.
pub(super) unsafe fn remap_if_unmapped(res: &ID3D12Resource) -> bool {
    let mut dev: Option<ID3D12Device> = None;
    if res.GetDevice(&mut dev).is_err() {
        return false;
    }
    let Some(dev) = dev else { return false };
    let key = res.as_raw() as usize;
    let target = |g: &mut std::sync::MutexGuard<'_, std::collections::BTreeMap<usize, Option<Pages>>>| {
        let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return None };
        pages.targets.get(&key).map(|t| (t.va, t.console_bytes, t.total, t.mapped, t.gpu_owned))
    };
    // (the physical pages are asked without `PAGES` held)
    let Some((va, console_bytes, _, mapped, gpu_owned)) = target(&mut PAGES.lock().unwrap()) else { return false };
    if mapped {
        return true;
    }
    let phys = physical_pages_under(va, console_bytes, gpu_owned);
    let mut g = PAGES.lock().unwrap();
    let Some((va, console_bytes, total, mapped, _)) = target(&mut g) else { return false };
    if mapped {
        return true;
    }
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return false };
    let Some((used, used_extra)) = pages.bind(&dev, res, va, console_bytes, total, &phys) else {
        return false;
    };
    if let Some(t) = pages.targets.get_mut(&key) {
        t.used = used;
        t.used_extra = used_extra;
        t.mapped = true;
        t.touched = std::time::Instant::now();
    }
    true
}

/// Page budget: `res` was evicted by the page budget.
pub(super) fn is_evicted(res: usize) -> bool {
    any_evicted() && EVICTED.lock().unwrap().as_ref().is_some_and(|s| s.contains(&res))
}

/// Page budget: map `res` again if the page budget evicted it. True when it was evicted and is
/// mapped again (the caller uploads it whole); a failed remap leaves it evicted, to be tried
/// again at its next write, transition or copy.
pub(super) unsafe fn revive(res: &ID3D12Resource) -> bool {
    let key = res.as_raw() as usize;
    if !is_evicted(key) || !remap_if_unmapped(res) {
        return false;
    }
    let mut g = EVICTED.lock().unwrap();
    let Some(s) = g.as_mut() else { return false };
    if !s.remove(&key) {
        return false;
    }
    EVICTED_LEN.store(s.len(), std::sync::atomic::Ordering::Relaxed);
    REMAPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Page budget: whether any texture is evicted (callers skip the lookup otherwise).
pub(super) fn any_evicted() -> bool {
    EVICTED_LEN.load(std::sync::atomic::Ordering::Relaxed) != 0
}

/// Whether the page budget follows the use of textures.
pub(super) fn tracking() -> bool {
    pool_cap() != 0
}

/// Page budget: the T# of the textures `visible` is in a shader-visible view heap (they may be
/// sampled: recent now), that of `cpu_only` only in CPU-only heaps. Returns the evicted ones of
/// `visible` (sorted), which the caller maps again (`revive`) and uploads before the title's
/// next work.
pub(super) fn seen(visible: &[usize], cpu_only: &[usize]) -> Vec<usize> {
    let now = std::time::Instant::now();
    // lab: `XCL_SEEN_TOUCH=0` – a T# in a view heap does not keep a texture (a retail title
    // has every streamed texture there: nothing was evicted), and an evicted one comes back
    // only with a transition or a CPU write
    let no_touch = lab_flag(&LAB_SEEN_NO_TOUCH, "XCL_SEEN_TOUCH");
    HEAPS_LOOKED.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut g = PAGES.lock().unwrap();
    for pages in g.values_mut().flatten() {
        for t in pages.targets.values_mut() {
            t.seen = Seen::None;
        }
        for r in visible {
            if let Some(t) = pages.targets.get_mut(r) {
                if !no_touch {
                    t.touched = now;
                }
                t.seen = Seen::Visible;
            }
        }
        for r in cpu_only {
            if let Some(t) = pages.targets.get_mut(r) {
                t.seen = Seen::CpuOnly;
            }
        }
    }
    drop(g);
    if !any_evicted() || no_touch {
        return Vec::new();
    }
    let e = EVICTED.lock().unwrap();
    let Some(e) = e.as_ref() else { return Vec::new() };
    let mut back: Vec<usize> = visible.iter().copied().filter(|r| e.contains(r)).collect();
    back.sort_unstable();
    back.dedup();
    back
}

/// Page budget: the textures `res` were used (a transition the title submitted, a CPU write
/// uploaded): later to be evicted.
pub(super) fn touch(res: impl IntoIterator<Item = usize>) {
    if !tracking() {
        return;
    }
    let now = std::time::Instant::now();
    let mut g = PAGES.lock().unwrap();
    for r in res {
        for pages in g.values_mut().flatten() {
            if let Some(t) = pages.targets.get_mut(&r) {
                t.touched = now;
                break;
            }
        }
    }
}

/// Page budget: `res` is a texture `title_textures` keeps in step with the title's memory, so
/// its texels can be uploaded again after an eviction.
pub(super) fn set_evictable(res: usize) {
    set_evictable_to(res, true);
}

/// Page budget: the GPU wrote `res`; its texels are not in the title's memory, so it is
/// never evicted.
pub(super) fn pin(res: usize) {
    set_evictable_to(res, false);
}

fn set_evictable_to(res: usize, on: bool) {
    if pool_cap() == 0 {
        return;
    }
    let mut g = PAGES.lock().unwrap();
    for pages in g.values_mut().flatten() {
        if let Some(t) = pages.targets.get_mut(&res) {
            t.evictable = on;
            return;
        }
    }
}

/// Page budget: (textures evicted, textures mapped again) so far.
pub fn eviction_counts() -> (u64, u64) {
    (EVICTIONS.load(std::sync::atomic::Ordering::Relaxed), REMAPS.load(std::sync::atomic::Ordering::Relaxed))
}

/// A resource released: its pages are no longer its.
fn forget_pages(resource: usize) {
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.remove(&resource) else { continue };
        if t.mapped {
            pages.release_tiles(t.used, t.used_extra, now, false);
        }
        drop(g);
        if let Some(s) = EVICTED.lock().unwrap().as_mut() {
            s.remove(&resource);
            EVICTED_LEN.store(s.len(), std::sync::atomic::Ordering::Relaxed);
        }
        return;
    }
}

/// Before GPU work on `queue`: have it wait for every tile mapping made so far on its device.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let mut dev: Option<ID3D12Device> = None;
    if queue.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return };
    let key = queue.as_raw() as usize;
    if pages.waited.get(&key).copied().unwrap_or(0) < pages.value {
        let _ = queue.Wait(&pages.fence, pages.value);
        pages.waited.insert(key, pages.value);
    }
}

#[cfg(test)]
pub(super) static VRAM_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Page budget, for tests: evict `res` now (as the least recently touched texture the CPU
/// fills); whether it was.
#[cfg(test)]
pub(super) unsafe fn evict_for_test(res: usize) -> bool {
    let mut g = PAGES.lock().unwrap();
    for pages in g.values_mut().flatten() {
        if let Some(t) = pages.targets.get_mut(&res) {
            t.evictable = true;
            t.touched = std::time::Instant::now().checked_sub(std::time::Duration::from_secs(600)).unwrap();
            pages.evict_idle_until = None;
            pages.evict_lru(1);
            return pages.targets.get(&res).is_some_and(|t| !t.mapped);
        }
    }
    false
}

#[cfg(test)]
pub(super) fn page_bytes_for(device: usize) -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.get(&device) else { return (0, 0) };
    ((p.pages.len() + p.extra_pages.len()) * PAGE, p.pools.len() * POOL_PAGES as usize * PAGE)
}

/// (bytes of pages in use by tiled textures and of small-texture slots, bytes of their heaps) – `XCL_ACTIVITY`
pub fn page_bytes() -> (usize, usize) {
    let (small, small_heaps) = small_bytes();
    let g = PAGES.lock().unwrap();
    g.values().flatten().fold((small, small_heaps), |(used, pools), p| {
        (used + (p.pages.len() + p.extra_pages.len()) * PAGE, pools + p.pools.len() * POOL_PAGES as usize * PAGE)
    })
}

/// lab: the pinned (never evicted) textures on pages by size, the 12 sizes that take most:
/// "count x MB (pages no other texture shares, MB)".
pub fn pinned_sizes() -> String {
    let g = PAGES.lock().unwrap();
    let mut by: std::collections::HashMap<u32, (u64, u64)> = std::collections::HashMap::new();
    let (mut alone_all, mut mapped_all) = (0u64, 0u64);
    for p in g.values().flatten() {
        for t in p.targets.values().filter(|t| t.mapped && !t.evictable) {
            let alone = t.used.iter().filter(|a| p.pages.get(a).is_some_and(|x| x.users <= 1)).count() + t.used_extra.iter().filter(|k| p.extra_pages.get(k).is_some_and(|x| x.users <= 1)).count();
            let e = by.entry(t.total).or_default();
            e.0 += 1;
            e.1 += alone as u64;
            alone_all += alone as u64;
            mapped_all += (t.used.len() + t.used_extra.len()) as u64;
        }
    }
    let mut v: Vec<(u32, (u64, u64))> = by.into_iter().collect();
    v.sort_unstable_by_key(|&(total, (n, _))| std::cmp::Reverse(u64::from(total) * n));
    let parts: Vec<String> = v.iter().take(12).map(|&(total, (n, alone))| format!("{n} x {:.1} MB ({} MB alone)", f64::from(total) / 16.0, alone >> 4)).collect();
    format!("pinned textures hold {} MB of page uses, {} MB of pages only one of them uses: {}", mapped_all >> 4, alone_all >> 4, parts.join(", "))
}

/// lab: what the pages hold, for the memory line of `XCL_ACTIVITY`: MB of tiles by kind of
/// texture (a page shared by textures counts for each), free and cooling tiles.
pub fn page_breakdown() -> String {
    let g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    // [evictable seen, evictable unseen, evictable CPU-only, pinned GPU-owned, pinned other, unmapped]
    let (mut n, mut tiles) = ([0u64; 6], [0u64; 6]);
    let (mut young, mut free, mut cooling, mut idle) = (0u64, 0usize, 0usize, 0usize);
    for p in g.values().flatten() {
        for t in p.targets.values() {
            let k = if !t.mapped {
                5
            } else if t.evictable {
                if now.duration_since(t.touched) < EVICT_MIN_AGE {
                    young += u64::from(t.total);
                }
                match t.seen {
                    Seen::Visible => 0,
                    Seen::None => 1,
                    Seen::CpuOnly => 2,
                }
            } else if t.gpu_owned {
                3
            } else {
                4
            };
            n[k] += 1;
            tiles[k] += u64::from(t.total);
        }
        free += p.pool.free_total;
        cooling += p.pool.cooling.len();
        idle += p.idle.len();
    }
    let mb = |t: u64| t >> 4;
    format!(
        "pages by texture: evictable seen {} MB ({}), unseen {} MB ({}), CPU-only {} MB ({}), of them touched in the last 5 s {} MB; pinned GPU-owned {} MB ({}), pinned other {} MB ({}); evicted {} MB ({}); free tiles {} MB, cooling {} MB, idle pages {} MB",
        mb(tiles[0]), n[0], mb(tiles[1]), n[1], mb(tiles[2]), n[2], mb(young), mb(tiles[3]), n[3], mb(tiles[4]), n[4], mb(tiles[5]), n[5], free >> 4, cooling >> 4, idle >> 4
    )
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

/// A sampled texture of the title at `va` (`desc`: the PC description, `console_bytes` its size on
/// the console) backed by shared 64 KiB video-memory pages; None when reserved placement is not
/// available (the caller falls back to a committed resource).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place_pages(
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) || std::env::var_os("XCL_VRAM_ARENAS").is_some() {
        return None;
    }
    if let Some(hr) = place_small(dev, this, va, desc, state, clear, console_bytes, riid, ppv) {
        return Some(hr);
    }
    place_on_pages(dev, this, va, desc, state, clear, console_bytes, false, riid, ppv)
}

/// A slot of the small-texture heaps (4 KiB).
const SLOT: usize = 0x1000;
/// slots of one small-texture heap (4 MiB)
const SLOTS_PER_HEAP: u32 = 1024;

struct SmallSlot {
    slot: u32,
    /// textures at its address
    users: u32,
}

/// The 4 KiB slots of the small-texture heaps: one per title address (textures at one address
/// share it, as they share a page's tile), given back slots reusable after `COOL`.
#[derive(Default)]
struct Slots {
    /// slots handed out so far (the next new one; a new heap at each `SLOTS_PER_HEAP`)
    next: u32,
    free: Vec<u32>,
    cooling: std::collections::VecDeque<(u32, std::time::Instant)>,
    by_va: std::collections::HashMap<usize, SmallSlot>,
}

impl Slots {
    /// The slot of `va` for one more texture there: (slot, whether its heap must be made first).
    fn take(&mut self, va: usize, now: std::time::Instant) -> (u32, bool) {
        if let Some(s) = self.by_va.get_mut(&va) {
            s.users += 1;
            return (s.slot, false);
        }
        while self.cooling.front().is_some_and(|(_, at)| now.duration_since(*at) >= COOL) {
            self.free.extend(self.cooling.pop_front().map(|(s, _)| s));
        }
        let (slot, new_heap) = match self.free.pop() {
            Some(s) => (s, false),
            None => {
                self.next += 1;
                (self.next - 1, (self.next - 1) % SLOTS_PER_HEAP == 0)
            }
        };
        self.by_va.insert(va, SmallSlot { slot, users: 1 });
        (slot, new_heap)
    }

    /// One texture at `va` fewer: its slot cools once no texture uses it.
    fn give_back(&mut self, va: usize, now: std::time::Instant) {
        if let Some(s) = self.by_va.get_mut(&va) {
            s.users -= 1;
            if s.users == 0 {
                let slot = s.slot;
                self.by_va.remove(&va);
                self.cooling.push_back((slot, now));
            }
        }
    }
}

/// The small-texture heaps of a device.
struct SmallHeaps {
    heaps: Vec<ID3D12Heap>,
    slots: Slots,
    /// resource → its address
    of: std::collections::HashMap<usize, usize>,
}

// D3D12 heaps are free-threaded
unsafe impl Send for SmallHeaps {}

static SMALL: Mutex<std::collections::BTreeMap<usize, SmallHeaps>> = Mutex::new(std::collections::BTreeMap::new());

/// A sampled texture of at most 4 KiB inside one 64 KiB page of the title's (no page boundary
/// within it, so on the pages it is one extra tile keyed by its address) as a 4 KiB-aligned
/// placement in a shared heap instead of that 64 KiB tile: a retail game makes ~12 000 views of
/// 16x4 textures of 1–2.5 KB while it loads its world – 768 MiB of tiles for 20 MiB. Textures
/// at one address share the slot as they share the tile. None: not such a texture, or the PC
/// wants more than 4 KiB for it.
#[allow(clippy::too_many_arguments)]
unsafe fn place_small(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let bytes = console_bytes?;
    if bytes > SLOT as u64 || desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.SampleDesc.Count > 1 || owned_pages(va, Some(bytes), usize::MAX)?.1 != 0 {
        return None;
    }
    let small = D3D12_RESOURCE_DESC { Alignment: SLOT as u64, ..*desc };
    let info = dev.GetResourceAllocationInfo(0, &[small]);
    if info.Alignment != SLOT as u64 || info.SizeInBytes == 0 || info.SizeInBytes > SLOT as u64 {
        return None;
    }
    let mut g = SMALL.lock().unwrap();
    let h = g.entry(dev.as_raw() as usize).or_insert_with(|| SmallHeaps { heaps: Vec::new(), slots: Slots::default(), of: std::collections::HashMap::new() });
    let now = std::time::Instant::now();
    let (slot, new_heap) = h.slots.take(va, now);
    if new_heap {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: SLOTS_PER_HEAP as u64 * SLOT as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES,
        };
        let mut made: Option<ID3D12Heap> = None;
        if super::title_textures::own_work(|| dev.CreateHeap(&desc, &mut made)).is_err() || made.is_none() {
            // the slot's heap was not made: hand that slot (and so that heap) out again next time
            h.slots.by_va.remove(&va);
            h.slots.next -= 1;
            return None;
        }
        h.heaps.push(made.unwrap());
    }
    let heap = &h.heaps[(slot / SLOTS_PER_HEAP) as usize];
    let offset = (slot % SLOTS_PER_HEAP) as usize * SLOT;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset, &small as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        h.slots.give_back(va, now);
        return None;
    }
    h.of.insert(*ppv as usize, va);
    drop(g);
    // aliased by address like the pages' textures
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    Some(hr)
}

/// A released small texture: its slot goes back when no texture at its address is left.
fn forget_small(resource: usize) {
    let mut g = SMALL.lock().unwrap();
    for h in g.values_mut() {
        if let Some(va) = h.of.remove(&resource) {
            h.slots.give_back(va, std::time::Instant::now());
            return;
        }
    }
}

/// (bytes of small-texture slots in use, bytes of small-texture heaps)
fn small_bytes() -> (usize, usize) {
    let g = SMALL.lock().unwrap();
    g.values().fold((0, 0), |(used, heaps), h| (used + h.slots.by_va.len() * SLOT, heaps + h.heaps.len() * SLOTS_PER_HEAP as usize * SLOT))
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description, `console_bytes` its
/// size on the console): on the title's pages there, else in the arena for that address (None:
/// neither – the caller makes a resource of its own).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        if let Some(hr) = place_on_pages(dev, this, va, desc, state, clear, console_bytes, true, riid, ppv) {
            return Some(hr);
        }
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || va as u64 % info.Alignment.max(1) != 0 {
        return None;
    }
    let align = (info.Alignment as usize).max(PAGE);
    let now = std::time::Instant::now();
    let (heap, offset, id) = ARENAS.lock().unwrap().take(va, info.SizeInBytes as usize, align, now, |size, align| arena_heap(dev, va, size, align))?;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        ARENAS.lock().unwrap().release(id, now);
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    ARENA_OF.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(*ppv as usize, id);
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const MIB: usize = 1 << 20;

    /// Small textures take one 4 KiB slot per address (shared there, as a page's tile is), 1024
    /// to a heap; a slot no texture uses any more is reused after `COOL`.
    #[test]
    fn small_texture_slots_are_per_address_and_come_back_cooled() {
        let mut s = Slots::default();
        let t0 = Instant::now();
        assert_eq!(s.take(0x1000, t0), (0, true));
        assert_eq!(s.take(0x1000, t0), (0, false), "a second texture at an address shares its slot");
        // 12 000 addresses: 12 heaps, not 12 000 pages
        let new_heaps = (1..12_000).filter(|k| s.take(0x1000 + k * 0x400, t0).1).count();
        assert_eq!((new_heaps, s.next), (11, 12_000));
        s.give_back(0x1000, t0);
        assert!(s.by_va.contains_key(&0x1000), "still used by the other texture there");
        s.give_back(0x1000, t0);
        assert_eq!(s.take(0x9_0000_0000, t0 + Duration::from_millis(100)), (12_000, false), "reused before COOL");
        assert_eq!(s.take(0xA_0000_0000, t0 + COOL), (0, false), "reused after COOL");
    }

    /// One call per heap: its tiles in runs, the other heaps' skipped.
    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)];
        assert_eq!(heap_ranges(0, &tiles), [(true, 5, 2), (false, 0, 2), (true, 9, 3)]);
        assert_eq!(heap_ranges(1, &tiles), [(false, 0, 2), (true, 0, 2), (false, 0, 3)]);
        assert_eq!(heap_ranges_trimmed(1, &tiles), [(false, 0, 2), (true, 0, 2)]);
        // shared pages out of order: runs break where the tiles do
        assert_eq!(heap_ranges(0, &[(0, 3), (0, 1), (0, 2)]), [(true, 3, 1), (true, 1, 2)]);
    }

    /// Streaming churn across 100+ 64 MiB pool heaps: cooling tiles returned in interleaved
    /// cross-heap order are regrouped per heap and sorted into contiguous ascending runs, so new
    /// textures stay inside 1 pool heap (1 UpdateTileMappings call, 1 contiguous range) instead of
    /// scattering across dozens of heaps.
    #[test]
    fn tile_pool_keeps_churned_allocations_in_single_heap_and_contiguous() {
        let t0 = Instant::now();
        let mut pool = TilePool::new();
        let heaps_made = std::cell::Cell::new(0u32);
        let mut make = || {
            let h = heaps_made.get();
            heaps_made.set(h + 1);
            Some(h)
        };

        // Fill 100 pools (64 tiles each for a fast unit test, same multi-heap topology as 1024).
        const P: u32 = 64;
        let mut live: Vec<Vec<Tile>> = Vec::new();
        for _ in 0..100 {
            live.push(pool.take_batch(P as usize, None, t0, P, &mut make).unwrap());
        }
        assert_eq!(heaps_made.get(), 100);
        assert_eq!(pool.free_total, 0);

        // Free 32 tiles from every pool in interleaved round-robin order (worst-case churn across
        // all 100 pools in `cooling`).
        for tile_idx in 0..32usize {
            for h in 0..100usize {
                pool.push_cooling(live[h][tile_idx], t0);
            }
        }
        assert_eq!(pool.cooling.len(), 3200);

        // After COOL, allocating 100 textures of 32 tiles each must:
        // - allocate ZERO new heaps,
        // - place EVERY texture into exactly 1 single pool heap (1 UpdateTileMappings call),
        // - yield 1 single contiguous tile range (0..32) per texture!
        let t1 = t0 + COOL + Duration::from_millis(1);
        for _ in 0..100 {
            let alloc = pool.take_batch(32, None, t1, P, &mut make).unwrap();
            assert_eq!(alloc.len(), 32);
            let h0 = alloc[0].0;
            assert!(alloc.iter().all(|&(h, _)| h == h0), "texture scattered across multiple heaps");
            let ranges = heap_ranges_trimmed(h0, &alloc);
            assert_eq!(ranges, [(true, 0, 32)], "expected 1 contiguous ascending range");
        }
        assert_eq!(heaps_made.get(), 100, "must reuse cooled tiles without making new heaps");
        assert_eq!(pool.free_total, 0);

        // Partial overlap preference: if heap 7 has free tiles and preferred=Some(7), use heap 7.
        for t in 10..18u32 {
            pool.push_cooling((7, t), t1);
            pool.push_cooling((3, t), t1);
        }
        let t2 = t1 + COOL + Duration::from_millis(1);
        let pref = pool.take_batch(8, Some(7), t2, P, &mut make).unwrap();
        assert!(pref.iter().all(|&(h, _)| h == 7), "preferred heap should be chosen");
        assert_eq!(heap_ranges_trimmed(7, &pref), [(true, 10, 8)]);

        // Rollback on OOM: requesting more tiles than available when make_heap fails returns every
        // popped tile cleanly so free_total is preserved.
        assert_eq!(pool.free_total, 8);
        let fail = pool.take_batch(16, None, t2, P, || None);
        assert!(fail.is_none());
        assert_eq!(pool.free_total, 8, "OOM rollback must return all popped tiles");
    }

    /// Idle pages go back after `IDLE`, and a sweep looks at the queue of idle pages alone: one
    /// used again stays, one idle again later goes when that is `IDLE` ago, and the sweeps of
    /// 20 000 placements beside 200 000 pages in use take no time to speak of (a look at every
    /// page at each of them: 4 000 000 000 looks).
    #[test]
    fn idle_pages_are_swept_from_their_queue() {
        use std::collections::{HashMap, VecDeque};
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut idle: VecDeque<(PageKey, Instant)> = VecDeque::new();
        let mut pages: HashMap<usize, Page> = (0..200_000usize).map(|i| (i * PAGE, Page { tile: (0, i as u32), users: 1, idle_since: None })).collect();
        let mut extra: HashMap<(usize, usize), Page> = HashMap::from([((7 * PAGE, 3), Page { tile: (1, 0), users: 1, idle_since: None })]);
        let mut cooling: VecDeque<(Tile, Instant)> = VecDeque::new();
        // what `release_tiles` and `bind` do to a page
        fn release(p: &mut Page, idle: &mut VecDeque<(PageKey, Instant)>, key: PageKey, when: Instant) {
            (p.users, p.idle_since) = (0, Some(when));
            idle.push_back((key, when));
        }
        fn use_again(p: &mut Page) {
            (p.users, p.idle_since) = (1, None);
        }
        // pages 1, 2, 3 idle at 0 ms and an extra page at 100 ms; page 2 is used again at
        // 500 ms; page 3 too, and is idle again at 1500 ms
        for a in [PAGE, 2 * PAGE, 3 * PAGE] {
            release(pages.get_mut(&a).unwrap(), &mut idle, PageKey::Own(a), at(0));
        }
        release(extra.get_mut(&(7 * PAGE, 3)).unwrap(), &mut idle, PageKey::Extra(7 * PAGE, 3), at(100));
        use_again(pages.get_mut(&(2 * PAGE)).unwrap());
        use_again(pages.get_mut(&(3 * PAGE)).unwrap());
        release(pages.get_mut(&(3 * PAGE)).unwrap(), &mut idle, PageKey::Own(3 * PAGE), at(1500));

        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(1900));
        assert!(cooling.is_empty() && pages.len() == 200_000 && idle.len() == 5, "nothing is idle for IDLE yet");
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2050));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1)], "page 1 alone: 2 is in use, 3 idle since 1500 ms");
        assert!(!pages.contains_key(&PAGE) && pages.contains_key(&(2 * PAGE)) && pages.contains_key(&(3 * PAGE)) && extra.len() == 1);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2150));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0)], "the extra page, idle since 100 ms");
        assert!(extra.is_empty() && pages.len() == 199_999);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3499));
        assert!(pages.contains_key(&(3 * PAGE)));
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3500));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0), (0, 3)]);
        assert!(idle.is_empty() && pages.len() == 199_998 && pages[&(2 * PAGE)].users == 1);

        // every placement sweeps: with nothing idle that is nothing, whatever is in use
        let t = Instant::now();
        for k in 0..20_000u64 {
            sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(4000 + k));
        }
        assert!(t.elapsed() < Duration::from_secs(2), "{:?} for 20 000 sweeps", t.elapsed());
        assert_eq!(pages.len(), 199_998);
    }

    /// Any 64 KiB page boundary inside `[va, va + bytes)` belongs to that range alone, so
    /// non-overlapping ranges never share an owned page even when only 4 KiB aligned.
    #[test]
    fn owned_pages_never_alias_disjoint_ranges() {
        assert_eq!(owned_pages(0x1_0000, Some(0x1_0000), 4), Some((0x1_0000, 1)));
        assert_eq!(owned_pages(0x1_0000, Some(0x1_4000), 4), Some((0x1_0000, 2)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4000), 4), Some((0x2_0000, 0)));
        assert_eq!(owned_pages(0x1_4000, Some(0x1_0000), 4), Some((0x2_0000, 1)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4_0000), 4), Some((0x2_0000, 4)));
        assert_eq!(owned_pages(0x1_4000, None, 4), None);
        // Exhaustively check 4 KiB-aligned disjoint pairs over 256 KiB.
        const K4: usize = 0x1000;
        for a in (0..16).map(|i| 0x10_0000 + i * K4) {
            for sa in [K4, 4 * K4, 16 * K4, 20 * K4] {
                let (fa, na) = owned_pages(a, Some(sa as u64), 8).unwrap();
                let pa: Vec<usize> = (0..na).map(|i| fa + i * PAGE).collect();
                for b in (0..16).map(|i| 0x10_0000 + i * K4) {
                    for sb in [K4, 4 * K4, 16 * K4, 20 * K4] {
                        if a + sa <= b || b + sb <= a {
                            let (fb, nb) = owned_pages(b, Some(sb as u64), 8).unwrap();
                            for j in 0..nb {
                                assert!(!pa.contains(&(fb + j * PAGE)), "disjoint [{a:#x}..{:#x}) and [{b:#x}..{:#x}) shared page", a + sa, b + sb);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Reserved only with TILED support, and for MSAA a tiled sample count (#940: the driver
    /// refused R32G8X24_TYPELESS reserved, at 1 and 4 samples, and the debug layer said so).
    #[test]
    fn reserved_needs_tiled_format_and_tiled_msaa() {
        let tiled = D3D12_FORMAT_SUPPORT2_TILED.0;
        assert!(can_reserve(tiled, 1, 0));
        assert!(!can_reserve(0, 1, 0), "format without TILED");
        assert!(can_reserve(tiled, 4, 1));
        assert!(!can_reserve(tiled, 4, 0), "no tiled quality level at 4 samples");
        assert!(!can_reserve(0, 4, 1));
    }

    /// The device's answer agrees with CreateReservedResource: what `reservable` turns away the
    /// driver refuses, what it lets through the driver makes (WARP or a hardware adapter).
    #[test]
    fn reservable_agrees_with_the_driver() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: windows::Win32::Graphics::Dxgi::IDXGIFactory4 = windows::Win32::Graphics::Dxgi::CreateDXGIFactory1().expect("DXGI factory");
                let Ok(warp) = factory.EnumWarpAdapter::<windows::Win32::Graphics::Dxgi::IDXGIAdapter>() else { return };
                let _ = D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev);
            }
            let Some(dev) = dev else { return };
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_err() || o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
                return;
            }
            for (format, samples) in [(DXGI_FORMAT_R32G8X24_TYPELESS, 1), (DXGI_FORMAT_R32G8X24_TYPELESS, 4), (DXGI_FORMAT_R32_TYPELESS, 1), (DXGI_FORMAT_R8G8B8A8_UNORM, 1)] {
                let flags = if format == DXGI_FORMAT_R8G8B8A8_UNORM { D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET } else { D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 128, Height: 128, DepthOrArraySize: 1, MipLevels: 1, Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, Flags: flags, ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                let made = dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).is_ok() && r.is_some();
                assert_eq!(reservable(&dev, &desc), made, "format {} x{samples}: reservable vs CreateReservedResource", format.0);
            }
        }
    }

    /// A placement's arena id, with heaps numbered as they are made.
    fn take(a: &mut Arenas<u32>, va: usize, bytes: usize, align: usize, now: Instant) -> u64 {
        let n = a.list.len() as u32;
        a.take(va, bytes, align, now, |_, _| Some(n)).unwrap().2
    }

    #[test]
    fn an_arena_is_shared_at_its_addresses_and_otherwise_its_placements_size() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let first = take(&mut a, a0, 70 * MIB, PAGE, t);
        assert_eq!(a.bytes(), 70 * MIB, "the placement's size, not 256 MiB");
        // the same address, and a smaller placement inside: the same memory
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t), first);
        assert_eq!(take(&mut a, a0 + 4 * MIB, 8 * MIB, PAGE, t), first);
        // one straddling its end gets an arena of its own size; so does an MSAA target that needs 4 MiB
        let straddle = take(&mut a, a0 + 64 * MIB, 70 * MIB, PAGE, t);
        let msaa = take(&mut a, a0 + 8 * MIB, 8 * MIB, 4 * MIB, t);
        assert!(straddle != first && msaa != first && msaa != straddle);
        assert_eq!(a.bytes(), (70 + 70 + 8) * MIB);
        // the newest arena holding an address is the one placements there share
        assert_eq!(take(&mut a, a0 + 64 * MIB, 4 * MIB, PAGE, t), straddle);
    }

    #[test]
    fn an_arena_no_resource_uses_goes_after_two_seconds() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let id = take(&mut a, a0, 70 * MIB, PAGE, t);
        a.release(id, t);
        // placed again within 2 s: the same memory (with what it holds)
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t + Duration::from_secs(1)), id);
        a.release(id, t + Duration::from_secs(1));
        take(&mut a, a0 + 512 * MIB, MIB, PAGE, t + Duration::from_secs(3) + Duration::from_millis(1));
        assert_eq!(a.bytes(), MIB, "the idle arena went");
    }

    /// The retail game's pattern (HANDOFF): targets packed back to back over 3.5 GB of its
    /// addresses and placed again and again, each round at the last round's addresses or a third
    /// of a target further on, so placements straddle earlier ones. Main's 256 MiB arenas merged
    /// straddled arenas into growing unions while the old ones lived on (15 GB in 2 s).
    #[test]
    fn streamed_targets_stay_bounded() {
        const S: usize = 64 * MIB; // a 3840x2160 D32S8, rounded
        const K: usize = 8; // targets a round
        const SPAN: usize = 3584 * MIB;
        let (t0, a0) = (Instant::now(), 0x1_0000_0000usize);
        let mut a = Arenas::new();
        let mut live: std::collections::VecDeque<Vec<u64>> = std::collections::VecDeque::new();
        let mut bytes = Vec::new();
        for r in 0..400usize {
            let now = t0 + Duration::from_millis(500 * r as u64);
            let start = ((r / 2) * S / 3) % (SPAN - K * S) & !(PAGE - 1);
            live.push_back((0..K).map(|k| take(&mut a, a0 + start + k * S, S, PAGE, now)).collect());
            // a target lives three rounds (1.5 s)
            if live.len() > 3 {
                for id in live.pop_front().unwrap() {
                    a.release(id, now);
                }
            }
            bytes.push(a.bytes());
        }
        // live: 3 rounds of targets; idle and kept (< 2 s): 4 more
        let most = *bytes.iter().max().unwrap();
        assert!(most <= 7 * K * S, "{most} bytes of arenas");
        let early = *bytes[20..120].iter().max().unwrap();
        let late = *bytes[300..].iter().max().unwrap();
        assert!(late <= early, "arenas grow over time: {early} then {late} bytes");
    }

    /// On the GPU: D32S8 targets 256 MiB apart take their own size of video memory, not an arena
    /// of 256 MiB each; depth and stencil survive a clear, CopyResource and a read of both planes;
    /// arenas no target uses go. Needs a hardware adapter (skips on WARP).
    /// Sampled textures of 4 KiB or less inside one page of the title's take 4 KiB slots of shared
    /// heaps (not a 64 KiB tile each), one per address; neighbours keep their own texels, and one
    /// over a page boundary stays on the pages.
    #[test]
    fn small_sampled_textures_take_slots_not_pages() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return crate::gpu::recomp::runner::skip_test("no GPU") };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return crate::gpu::recomp::runner::skip_test("no GPU") };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            // a retail game's: 16x4 R16G16B16A16_FLOAT, 1 KiB on the console, 256-byte aligned
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R16G16B16A16_FLOAT, SampleDesc: sample, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC, bytes: u64| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32, std::ptr::null(), Some(bytes), riid, &mut p);
                (hr.is_some_and(|hr| hr >= 0) && !p.is_null()).then(|| ID3D12Resource::from_raw(p))
            };
            let small = tex(16, 4);
            if dev.GetResourceAllocationInfo(0, &[D3D12_RESOURCE_DESC { Alignment: SLOT as u64, ..small }]).Alignment != SLOT as u64 {
                return crate::gpu::recomp::runner::skip_test("no 4 KiB texture placement on this adapter");
            }
            let key = dev.as_raw() as usize;
            let (pages0, _) = page_bytes_for(key);
            let used0 = SMALL.lock().unwrap().get(&key).map_or(0, |h| h.slots.by_va.len());
            let base = 0x50_0000_0000usize;
            let textures: Vec<ID3D12Resource> = (0..64).map(|k| place_at(base + k * PAGE + 0xa00, &small, 1024).expect("small texture")).collect();
            assert_eq!(page_bytes_for(key).0, pages0, "small textures took pages");
            assert_eq!(SMALL.lock().unwrap()[&key].slots.by_va.len() - used0, 64, "one slot per small texture");
            assert!(textures.iter().all(|t| is_placed_texture(t.as_raw() as usize)), "a small texture is aliased by address like the pages' ones");
            // a second texture at an address shares its slot
            let again = place_at(base + 0xa00, &small, 1024).expect("small texture again");
            assert_eq!(SMALL.lock().unwrap()[&key].slots.by_va.len() - used0, 64, "a second texture at an address took a slot");
            // over a page boundary: still the pages
            let across = place_at(base + 0x100 * PAGE + PAGE - 0x200, &small, 1024).expect("texture over a page boundary");
            assert!(is_placed_texture(across.as_raw() as usize));

            // two neighbours: each keeps what was copied into it
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
            let buffer = |t: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
                let d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 2 * total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: t, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, state, None, &mut r).unwrap();
                r.unwrap()
            };
            let (upload, readback) = (buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ), buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST));
            let mut p = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut p)).unwrap();
            for k in 0..2usize {
                std::slice::from_raw_parts_mut((p as *mut u8).add(k * total as usize), total as usize).fill(0x11 * (k as u8 + 1));
            }
            upload.Unmap(0, None);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let loc = |r: &ID3D12Resource, offset: u64| D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: offset, ..fp } } };
            let sub = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            for k in 0..2 {
                list.CopyTextureRegion(&sub(&textures[k]), 0, 0, 0, &loc(&upload, k as u64 * total), None);
            }
            let to_src = |r: &ID3D12Resource| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: D3D12_RESOURCE_STATE_COPY_DEST, StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ResourceBarrier(&[to_src(&textures[0]), to_src(&textures[1])]);
            for k in 0..2 {
                list.CopyTextureRegion(&loc(&readback, k as u64 * total), 0, 0, 0, &sub(&textures[k]), None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let t = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert_eq!(fence.GetCompletedValue(), 1, "GPU work did not finish");
            readback.Map(0, None, Some(&mut p)).unwrap();
            for k in 0..2usize {
                let row = std::slice::from_raw_parts((p as *const u8).add(k * total as usize), 16 * 8);
                assert!(row.iter().all(|&b| b == 0x11 * (k as u8 + 1)), "texture {k} read back {:02x?}", &row[..16]);
            }
            readback.Unmap(0, None);

            // released: the slots go back
            for t in textures.iter().chain([&again, &across]) {
                forget(t.as_raw() as usize);
            }
            assert_eq!(SMALL.lock().unwrap()[&key].slots.by_va.len(), used0);
        }
    }

    #[test]
    fn d32s8_targets_take_their_own_size_of_video_memory() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R32G8X24_TYPELESS, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default() };
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC| {
                let mut p = std::ptr::null_mut();
                let hr = place(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32, &clear as *const _ as *const c_void, None, riid, &mut p);
                assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "D32S8 target at {va:#x}");
                ID3D12Resource::from_raw(p)
            };

            // 14 4K targets over 3.5 GB of addresses
            let big = tex(3840, 2160);
            let size = dev.GetResourceAllocationInfo(0, &[big]).SizeInBytes as usize;
            let (used0, arenas0) = (local(), arena_bytes().0);
            let targets: Vec<ID3D12Resource> = (0..14).map(|k| place_at(0x40_0000_0000 + k * 256 * MIB, &big)).collect();
            let (used, arenas) = (local().saturating_sub(used0) as usize, arena_bytes().0 - arenas0);
            println!("14 D32S8 3840x2160 targets ({size} bytes each): {arenas} bytes of arenas, {used} bytes more local video memory");
            assert!(arenas <= 14 * size.next_multiple_of(PAGE), "{arenas} bytes of arenas");
            assert!(used < 14 * 256 * MIB, "{used} bytes of video memory");
            drop(targets);

            // clear one, copy it to another, read both planes back
            let small = tex(64, 64);
            let (a, b) = (place_at(0x48_0000_0000, &small), place_at(0x48_1000_0000, &small));
            let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&a, Some(&view), dsv);
            let mut layouts = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 2, 0, Some(layouts.as_mut_ptr()), None, None, Some(&mut total));
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25, 0x5A, None);
            list.ResourceBarrier(&[barrier(&a, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE), barrier(&b, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyResource(&b, &a);
            list.ResourceBarrier(&[barrier(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (plane, layout) in layouts.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *layout } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: plane as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            // no event: returns when the fence has the value
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let bytes = std::slice::from_raw_parts(p as *const u8, total as usize);
            let row = |plane: usize, y: usize| layouts[plane].Offset as usize + y * layouts[plane].Footprint.RowPitch as usize;
            for y in [0, 31, 63] {
                let depth = f32::from_le_bytes(bytes[row(0, y)..row(0, y) + 4].try_into().unwrap());
                assert_eq!((depth, bytes[row(1, y)]), (0.25, 0x5A), "row {y}: depth, stencil");
            }
            readback.Unmap(0, None);
            drop((a, b));

            // 2 s later the arenas no target uses are let go: one new target's is all that is left
            std::thread::sleep(IDLE + Duration::from_millis(100));
            let last = place_at(0x50_0000_0000, &small);
            let small_size = dev.GetResourceAllocationInfo(0, &[small]).SizeInBytes as usize;
            assert_eq!(arena_bytes().0 - arenas0, small_size.next_multiple_of(PAGE));
            drop(last);
        }
    }

    /// Reproduces the multi-pool streaming churn pattern on the active D3D12 driver (comparing
    /// scattered cross-heap `UpdateTileMappings` against single-heap contiguous
    /// `UpdateTileMappings`, both with `FLAG_NONE` and per-texture `Signal`):
    /// `cargo test -p xcl --lib -- --ignored --exact hle::d3d12x::vram::tests::churn_update_tile_mappings_timing --nocapture`
    #[test]
    #[ignore]
    fn churn_update_tile_mappings_timing() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("D3D12 device");
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            const HEAPS: usize = 64;
            const TILES_PER_TEX: usize = 32; // 2 MiB tiled texture
            const TEX_COUNT: usize = 256;
            let heap_desc = D3D12_HEAP_DESC {
                SizeInBytes: 64 * PAGE as u64,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_NONE,
            };
            let mut heaps = Vec::with_capacity(HEAPS);
            for _ in 0..HEAPS {
                let mut h: Option<ID3D12Heap> = None;
                dev.CreateHeap(&heap_desc, &mut h).unwrap();
                heaps.push(h.unwrap());
            }
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 1024,
                Height: 512,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let make_textures = || -> Vec<ID3D12Resource> {
                (0..TEX_COUNT)
                    .map(|_| {
                        let mut r: Option<ID3D12Resource> = None;
                        dev.CreateReservedResource(&tex_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                        r.unwrap()
                    })
                    .collect()
            };

            // BEFORE pattern: flat-FIFO cooling scatters each 32-tile texture across 32 heaps,
            // calling UpdateTileMappings(FLAG_NONE) per heap + Signal per texture.
            let old_res = make_textures();
            let mut fval = 0u64;
            let t0 = Instant::now();
            let mut old_calls = 0usize;
            for (k, r) in old_res.iter().enumerate() {
                let tiles: Vec<Tile> = (0..TILES_PER_TEX).map(|i| (((k + i) % HEAPS) as u32, (i % 64) as u32)).collect();
                let mut hs: Vec<u32> = tiles.iter().map(|t| t.0).collect();
                hs.sort_unstable();
                hs.dedup();
                for h in hs {
                    let ranges = heap_ranges(h, &tiles);
                    let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|x| if x.0 { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP }).collect();
                    let offsets: Vec<u32> = ranges.iter().map(|x| x.1).collect();
                    let counts: Vec<u32> = ranges.iter().map(|x| x.2).collect();
                    let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                    let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles.len() as u32, UseBox: false.into(), ..Default::default() };
                    queue.UpdateTileMappings(r, 1, Some(&coord), Some(&region), &heaps[h as usize], ranges.len() as u32, Some(flags.as_ptr()), Some(offsets.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
                    old_calls += 1;
                }
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
            }
            let old_submit = t0.elapsed();
            fence.SetEventOnCompletion(fval, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let old_total = t0.elapsed();

            // AFTER pattern: per-heap contiguous TilePool -> 1 heap, 1 contiguous range,
            // D3D12_TILE_MAPPING_FLAG_NONE, per-texture Signal.
            let new_res = make_textures();
            let t1 = Instant::now();
            let mut new_calls = 0usize;
            for (k, r) in new_res.iter().enumerate() {
                let h = k % HEAPS;
                let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                let region = D3D12_TILE_REGION_SIZE { NumTiles: TILES_PER_TEX as u32, UseBox: false.into(), ..Default::default() };
                let flag = D3D12_TILE_RANGE_FLAG_NONE;
                let off = 0u32;
                let cnt = TILES_PER_TEX as u32;
                queue.UpdateTileMappings(r, 1, Some(&coord), Some(&region), &heaps[h], 1, Some(&flag), Some(&off), Some(&cnt), D3D12_TILE_MAPPING_FLAG_NONE);
                new_calls += 1;
                fval += 1;
                queue.Signal(&fence, fval).unwrap();
            }
            let new_submit = t1.elapsed();
            fence.SetEventOnCompletion(fval, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let new_total = t1.elapsed();

            println!(
                "BEFORE ({TEX_COUNT} x 2MiB textures, {old_calls} UpdateTileMappings + {TEX_COUNT} Signals): submit={old_submit:?} gpu_done={old_total:?}"
            );
            println!(
                "AFTER  ({TEX_COUNT} x 2MiB textures, {new_calls} UpdateTileMappings + {TEX_COUNT} Signals): submit={new_submit:?} gpu_done={new_total:?}"
            );
            assert!(new_calls < old_calls);
        }
    }

    /// Page budget: a texture with no T# in a view heap goes first, and one whose pages are all
    /// shared with other mapped textures is passed over (#782).
    #[test]
    fn page_budget_takes_unseen_textures_first_and_passes_over_shared_pages() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 512, Height: 512, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(1 << 20), riid, &mut p);
                if !hr.is_some_and(|hr| hr >= 0) || p.is_null() {
                    return None;
                }
                Some(ID3D12Resource::from_raw(p))
            };
            // two textures at the same address share every page
            let (Some(a), Some(b), Some(unseen), Some(own)) = (place_at(0x62_0000_0000), place_at(0x62_0000_0000), place_at(0x62_1000_0000), place_at(0x62_2000_0000)) else {
                crate::gpu::recomp::runner::skip_test("no reserved textures");
                return;
            };
            let key = |r: &ID3D12Resource| r.as_raw() as usize;
            let dev_key = dev.as_raw() as usize;
            let now = Instant::now();
            HEAPS_LOOKED.store(true, std::sync::atomic::Ordering::Relaxed);
            {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                pages.evict_idle_until = None;
                // (`set_evictable` does nothing without XCL_PAGE_BUDGET_MB)
                for (r, age, seen) in [(&a, 60, Seen::CpuOnly), (&b, 60, Seen::CpuOnly), (&unseen, 3, Seen::None), (&own, 30, Seen::CpuOnly)] {
                    let t = pages.targets.get_mut(&key(r)).unwrap();
                    t.touched = now - Duration::from_secs(age);
                    t.seen = seen;
                    t.evictable = true;
                }
                pages.evict_lru(1);
                assert!(!pages.targets[&key(&unseen)].mapped, "a texture with no T# goes first, though younger");
                assert!(pages.targets[&key(&a)].mapped && pages.targets[&key(&b)].mapped && pages.targets[&key(&own)].mapped);
                pages.evict_lru(1);
                assert!(pages.targets[&key(&a)].mapped && pages.targets[&key(&b)].mapped, "shared pages give nothing back: passed over");
                assert!(!pages.targets[&key(&own)].mapped, "the next with pages of its own is evicted");
                assert!(pages.evict_idle_until.is_none(), "rounds that gave back what was wanted are not held back");
                pages.evict_lru(usize::MAX);
                assert!(pages.evict_idle_until.is_some_and(|t| t > Instant::now()), "a round that fell short holds the next back");
                assert!(pages.targets[&key(&a)].mapped && pages.targets[&key(&b)].mapped);
                pages.evict_idle_until = None;
            }
            HEAPS_LOOKED.store(false, std::sync::atomic::Ordering::Relaxed);
            assert!(is_evicted(key(&unseen)) && is_evicted(key(&own)));
            assert!(revive(&unseen) && revive(&own));
            drop((a, b, unseen, own));
        }
    }

    /// Page budget: the least recently touched CPU-filled texture gives its
    /// pages back first, a GPU-written one never does, and an evicted one maps again.
    #[test]
    fn page_budget_evicts_the_least_recently_touched_cpu_filled_texture() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 512, Height: 512, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(1 << 20), riid, &mut p);
                if !hr.is_some_and(|hr| hr >= 0) || p.is_null() {
                    return None;
                }
                Some(ID3D12Resource::from_raw(p))
            };
            let (Some(old), Some(newer), Some(written)) = (place_at(0x60_0000_0000), place_at(0x60_1000_0000), place_at(0x60_2000_0000)) else {
                crate::gpu::recomp::runner::skip_test("no reserved textures");
                return;
            };
            let key = |r: &ID3D12Resource| r.as_raw() as usize;
            let dev_key = dev.as_raw() as usize;
            let now = Instant::now();
            {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                // (`set_evictable` / `pin` do nothing without XCL_PAGE_BUDGET_MB)
                for (r, age, evictable) in [(&old, 30, true), (&newer, 20, true), (&written, 40, false)] {
                    let t = pages.targets.get_mut(&key(r)).unwrap();
                    t.touched = now - Duration::from_secs(age);
                    t.evictable = evictable;
                }
                let used_before = pages.pages.len();
                pages.evict_idle_until = None;
                assert!(pages.evict_lru(1) > 0, "tiles came back");
                assert!(!pages.targets[&key(&old)].mapped, "the oldest CPU-filled texture is evicted");
                assert!(pages.targets[&key(&newer)].mapped, "one is enough for one tile");
                assert!(pages.targets[&key(&written)].mapped, "a GPU-written texture is never evicted");
                assert!(pages.pages.len() < used_before, "its pages went back");
                // nothing left to evict: the round says so (take_tile then waits before the next)
                pages.targets.get_mut(&key(&newer)).unwrap().evictable = false;
                assert_eq!(pages.evict_lru(1), 0, "nothing evictable");
            }
            assert!(is_evicted(key(&old)) && !is_evicted(key(&newer)));
            assert!(revive(&old), "mapped again");
            assert!(!is_evicted(key(&old)) && !revive(&old), "once");
            assert!(PAGES.lock().unwrap().get(&dev_key).unwrap().as_ref().unwrap().targets[&key(&old)].mapped, "mapped again");
            drop((old, newer, written));
        }
    }

    /// Page budget: at the cap, one texture placed evicts one batch, and its other tiles come
    /// from that batch – not one more round per tile (#640 review: one placement evicted every
    /// eligible texture).
    #[test]
    fn page_budget_one_placement_at_the_cap_evicts_one_batch() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            // 512 x 512 RGBA8: 16 tiles each
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 512, Height: 512, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(1 << 20), riid, &mut p);
                if !hr.is_some_and(|hr| hr >= 0) || p.is_null() {
                    return None;
                }
                Some(ID3D12Resource::from_raw(p))
            };
            // 40 x 16 = 640 tiles: more than one batch of 512
            let old: Vec<ID3D12Resource> = (0..40).filter_map(|i| place_at(0x61_0000_0000 + i * 0x10_0000)).collect();
            if old.len() != 40 {
                crate::gpu::recomp::runner::skip_test("no reserved textures");
                return;
            }
            let key = |r: &ID3D12Resource| r.as_raw() as usize;
            let dev_key = dev.as_raw() as usize;
            let now = Instant::now();
            let (saved_free, saved_cooling) = {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                for (i, r) in old.iter().enumerate() {
                    let t = pages.targets.get_mut(&key(r)).unwrap();
                    t.touched = now - Duration::from_secs(100 - i as u64);
                    t.evictable = true;
                }
                // at the cap with no spare tile: the next tile has to come from an eviction
                pages.cap = pages.pools.len();
                pages.evict_idle_until = None;
                let free: Vec<Vec<u32>> = pages.pool.free_by_heap.iter_mut().map(std::mem::take).collect();
                pages.pool.free_total = 0;
                (free, std::mem::take(&mut pages.pool.cooling))
            };
            let new = place_at(0x61_8000_0000);
            let mapped: Vec<bool> = {
                let mut g = PAGES.lock().unwrap();
                let pages = g.get_mut(&dev_key).unwrap().as_mut().unwrap();
                let mapped = old.iter().map(|r| pages.targets[&key(r)].mapped).collect();
                pages.cap = 0;
                for (h, free) in saved_free.into_iter().enumerate() {
                    pages.pool.free_total += free.len();
                    pages.pool.free_by_heap[h].extend(free);
                    pages.pool.dirty_sort[h] = true;
                }
                pages.pool.cooling.extend(saved_cooling);
                mapped
            };
            assert!(new.is_some(), "placed from the evicted tiles");
            let evicted = mapped.iter().filter(|m| !**m).count();
            assert_eq!(evicted, 32, "one batch of 512 tiles: the 32 least recently touched");
            assert!(mapped[..32].iter().all(|m| !m) && mapped[32..].iter().all(|m| *m), "oldest first");
            for r in &old {
                let _ = revive(r);
            }
            drop((old, new));
        }
    }
}
