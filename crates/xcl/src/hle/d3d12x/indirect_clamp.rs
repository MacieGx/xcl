//! ExecuteIndirect's arguments, bounded on the GPU before the PC reads them.
//!
//! A title's own compute passes write most indirect arguments. A pass XCL skips (its pipeline
//! still being built, or untranslated) or translates wrongly leaves whatever the memory held,
//! and a draw of four billion vertices or a dispatch past the PC's limits runs until Windows
//! resets the display driver (0x887a0006). So, before an ExecuteIndirect with a translated
//! pipeline, a compute shader of ours (`indirect_clamp.hlsl`) copies the call's records into a
//! buffer of ours, clamps each record's draw or dispatch arguments, and empties the records past
//! a budget for the whole call; the PC call then reads that copy. The idea is VIctory's
//! (github.com/Marvisak/VIctory, `victory-gpu/src/indirect.rs`); this is XCL's own code.
//!
//! Only signatures without root arguments are bounded (those with them are skipped on
//! translated pipelines anyway), and only records in console buffers in unified memory; any
//! other call runs as before. `XCL_INDIRECT_CLAMP=0` turns it off.
//!
//! The limits (`indirect_clamp.hlsl`) are estimates, not measurements: 2^24 vertices per
//! instance, 2^26 vertex instances and 2^22 groups (65,535 per dimension) per record, 2^29
//! vertex instances or threads per call. A title that really needs more loses that work.
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::real;

/// Bytes of records one call may copy (larger calls run unbounded).
const MAX_BYTES: u64 = 16 << 20;
const CONFIG_WORDS: u32 = 7;
/// work one call may ask for, in threads (a group counts as 64, a mesh group as 128)
pub(super) const BUDGET: u32 = 1 << 29;
const NO_COUNT: u32 = u32::MAX;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_INDIRECT_CLAMP").is_ok_and(|v| v.trim() == "0"))
}

/// A command signature whose records can be bounded: bytes per record, where its draw or
/// dispatch arguments start in one, and their kind (0 draw, 1 indexed draw, 2 dispatch,
/// 3 mesh dispatch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Signature {
    stride: u32,
    action_at: u32,
    kind: u32,
}

static SIGNATURES: Mutex<Option<HashMap<usize, Signature>>> = Mutex::new(None);

impl Signature {
    /// From the PC argument list (`[type, …]` per argument, the action last): None for a
    /// signature that sets root arguments or whose action is not a draw or dispatch.
    fn parse(stride: u32, args: &[[u32; 4]]) -> Option<Self> {
        let ([action, ..], before) = args.split_last()?;
        let (kind, action_bytes) = match D3D12_INDIRECT_ARGUMENT_TYPE(*action as i32) {
            D3D12_INDIRECT_ARGUMENT_TYPE_DRAW => (0, 16),
            D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED => (1, 20),
            D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH => (2, 12),
            D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH => (3, 12),
            _ => return None,
        };
        let mut at = 0u32;
        for &[ty, ..] in before {
            at += match D3D12_INDIRECT_ARGUMENT_TYPE(ty as i32) {
                // D3D12_VERTEX_BUFFER_VIEW, D3D12_INDEX_BUFFER_VIEW
                D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW | D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW => 16,
                _ => return None,
            };
        }
        if stride == 0 || !stride.is_multiple_of(4) || at + action_bytes > stride {
            return None;
        }
        Some(Self { stride, action_at: at, kind })
    }
}

/// Note a command signature the title created (its PC argument list); one that cannot be
/// bounded is forgotten, so a new signature at a released one's address is never mistaken for it.
pub(super) fn note_signature(sig: usize, stride: u32, args: &[[u32; 4]]) {
    let mut g = SIGNATURES.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    match Signature::parse(stride, args) {
        Some(s) => map.insert(sig, s),
        None => map.remove(&sig),
    };
}

fn signature(sig: usize) -> Option<Signature> {
    SIGNATURES.lock().unwrap().as_ref()?.get(&sig).copied()
}

struct Clamper {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

/// per device; None: it could not be made (logged once, not tried again)
static CLAMPERS: Mutex<Option<HashMap<usize, Option<Arc<Clamper>>>>> = Mutex::new(None);

unsafe fn clamper(dev: &ID3D12Device) -> Option<Arc<Clamper>> {
    let mut g = CLAMPERS.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    let key = dev.as_raw() as usize;
    if let Some(c) = map.get(&key) {
        return c.clone();
    }
    let made = make_clamper(dev).map(Arc::new);
    if let Err(e) = &made {
        crate::xlog_warn!("D3D12.X: ExecuteIndirect arguments run unbounded: {e}");
    }
    let made = made.ok();
    map.insert(key, made.clone());
    made
}

unsafe fn make_clamper(dev: &ID3D12Device) -> Result<Clamper, String> {
    let dxil = crate::gpu::recomp::dxc::compile(include_str!("indirect_clamp.hlsl"), "main", "cs_6_0", &[])?;
    let uav = |reg| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: reg, RegisterSpace: 0 } },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let params = [
        uav(0),
        uav(1),
        uav(2),
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: CONFIG_WORDS } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), ..Default::default() };
    let mut blob = None;
    D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, None).map_err(|e| format!("root signature: {e}"))?;
    let blob = blob.ok_or("no root signature blob")?;
    let root: ID3D12RootSignature = dev
        .CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()))
        .map_err(|e| format!("root signature: {e}"))?;
    let mut pd = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let pso = dev.CreateComputePipelineState(&pd);
    ManuallyDrop::drop(&mut pd.pRootSignature);
    Ok(Clamper { root, pso: pso.map_err(|e| format!("pipeline: {e}"))? })
}

/// Bounded copies a list recorded, per allocator, until its reset; and those free again.
#[derive(Default)]
struct Scratch {
    held: HashMap<usize, Vec<(usize, u64, ID3D12Resource)>>,
    free: Vec<(usize, u64, ID3D12Resource)>,
    /// allocator → bytes taken of the last buffer it holds
    taken: HashMap<usize, u64>,
}

/// bytes of a buffer the copies of an allocator's calls are cut from. A buffer for each call
/// (64 KiB at least, a committed resource each) was thousands of resources a frame in a
/// retail title's world – made and released again with every allocator reset beyond the
/// `FREE_KEPT`: video memory and the working set grew by 2 GB in the 8 s after the world began.
const CHUNK: u64 = 1 << 20;
/// (bytes of buffers that exist, buffers made so far) – `XCL_ACTIVITY`
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// lab: (bytes of the copies' buffers that exist, buffers made so far)
pub fn scratch_stats() -> (u64, u64) {
    (LIVE[0].load(std::sync::atomic::Ordering::Relaxed), LIVE[1].load(std::sync::atomic::Ordering::Relaxed))
}

static SCRATCH: Mutex<Option<Scratch>> = Mutex::new(None);
/// free buffers kept for reuse (the rest are released)
const FREE_KEPT: usize = 64;

/// `bytes` of a buffer in UNORDERED_ACCESS (the buffer and the offset in it), held for
/// `allocator` until its reset: the next bytes of the buffer its last call took from, or a new
/// buffer of `CHUNK` at least. Calls of one allocator are recorded one after the other, each
/// leaves the buffer in UNORDERED_ACCESS again (`Bounded::finish`).
unsafe fn scratch(dev: &ID3D12Device, allocator: usize, bytes: u64) -> Option<(ID3D12Resource, u64)> {
    let key = dev.as_raw() as usize;
    let bytes = bytes.max(4).next_multiple_of(256);
    let mut g = SCRATCH.lock().unwrap();
    let s = g.get_or_insert_with(Default::default);
    if allocator != 0 {
        if let Some((d, size, r)) = s.held.get(&allocator).and_then(|v| v.last()) {
            let at = s.taken.get(&allocator).copied().unwrap_or(*size);
            if *d == key && at + bytes <= *size {
                let r = r.clone();
                s.taken.insert(allocator, at + bytes);
                return Some((r, at));
            }
        }
    }
    let reuse = s.free.iter().enumerate().filter(|(_, (d, n, _))| *d == key && *n >= bytes).min_by_key(|(_, (_, n, _))| *n).map(|(i, _)| i);
    let entry = match reuse {
        Some(i) => s.free.swap_remove(i),
        None => {
            let size = bytes.next_power_of_two().max(CHUNK);
            LIVE[0].fetch_add(size, std::sync::atomic::Ordering::Relaxed);
            LIVE[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut out: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                None,
                &mut out,
            )
            .ok()?;
            (key, size, out?)
        }
    };
    let r = entry.2.clone();
    s.held.entry(allocator).or_default().push(entry);
    s.taken.insert(allocator, bytes);
    Some((r, 0))
}

/// The GPU is done with what lists recorded from `allocator` (reset or released): its copies
/// can be reused.
pub(super) fn allocator_reset(allocator: usize) {
    // the buffers over `FREE_KEPT` are released after the lock: a buffer's last Release takes
    // the sparse state's lock (`real::res_release` → `sparse::forget_buffer`), and a thread
    // holding that one resets an allocator (`sparse::readback_after_execute`) and comes here –
    // the two stopped each other and every title thread behind them (lab: found in a world run)
    let mut over = Vec::new();
    {
        let mut g = SCRATCH.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        s.taken.remove(&allocator);
        if let Some(done) = s.held.remove(&allocator) {
            s.free.extend(done);
            if s.free.len() > FREE_KEPT {
                s.free.sort_by_key(|e| std::cmp::Reverse(e.1));
                over = s.free.split_off(FREE_KEPT);
            }
        }
    }
    LIVE[0].fetch_sub(over.iter().map(|e| e.1).sum::<u64>(), std::sync::atomic::Ordering::Relaxed);
    drop(over);
}

/// Where `bytes` at console offset `offset` of console buffer `res` are on the GPU: the GPU
/// address of the unified memory window holding them and their offset in it.
unsafe fn in_window(res: usize, offset: u64, bytes: u64) -> Option<(u64, u32)> {
    let (p, len) = real::console_buffer(res)?;
    if !offset.is_multiple_of(4) || offset.checked_add(bytes)? > len as u64 {
        return None;
    }
    let (window, at) = real::uma_window_buffer(p.checked_add(usize::try_from(offset).ok()?)?, usize::try_from(bytes).ok()?)?;
    let at = u32::try_from(at).ok()?;
    at.checked_add(u32::try_from(bytes).ok()?)?;
    let window = window as *mut c_void;
    Some((ID3D12Resource::from_raw_borrowed(&window)?.GetGPUVirtualAddress(), at))
}

/// The config words for a call: max records, where they are, the signature and the count.
fn config(sig: Signature, max: u32, source_at: u32, count_at: Option<u32>, budget: u32) -> [u32; CONFIG_WORDS as usize] {
    [max, source_at, sig.stride, sig.action_at, sig.kind | FOLD.with(|f| f.get()) << 8, count_at.unwrap_or(NO_COUNT), budget]
}

thread_local! {
    /// lab: groups a row for the dispatch records being bounded on this thread (`with_fold`)
    static FOLD: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Lab (`XCL_FOLD_GROUPS`): `f` with dispatch records of more than 65535 groups in x bounded
/// to rows of `row` groups (for a program that folds its group id); 0: cut, as always.
pub(super) fn with_fold<R>(row: u32, f: impl FnOnce() -> R) -> R {
    let was = FOLD.with(|c| c.replace(row));
    let r = f();
    FOLD.with(|c| c.set(was));
    r
}

unsafe fn transition(list: &ID3D12GraphicsCommandList, r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(r.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
        ..Default::default()
    };
    list.ResourceBarrier(std::slice::from_ref(&barrier));
    let mut t = ManuallyDrop::into_inner(barrier.Anonymous.Transition);
    ManuallyDrop::drop(&mut t.pResource);
}

/// A bounded copy of a call's records (the buffer and the copy's offset in it), in
/// INDIRECT_ARGUMENT until [`Bounded::finish`].
pub(super) struct Bounded(ID3D12Resource, u64);

impl Bounded {
    /// The copy's buffer, for the PC ExecuteIndirect (the copy is at [`Bounded::offset`]).
    pub(super) fn raw(&self) -> usize {
        self.0.as_raw() as usize
    }

    /// Where the copy is in its buffer.
    pub(super) fn offset(&self) -> u64 {
        self.1
    }

    /// After the PC call: back to UNORDERED_ACCESS (the state each use starts from; a list may
    /// run more than once).
    pub(super) unsafe fn finish(self, list: &ID3D12GraphicsCommandList) {
        transition(list, &self.0, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
}

/// Record the bounding of an ExecuteIndirect's records on `list` (offsets are console ones,
/// before `real::buffer_delta`). Changes the list's compute root signature, its root arguments
/// and its pipeline, so the caller binds its own after this. None: the call runs as it is.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn bound(
    list: &ID3D12GraphicsCommandList, allocator: usize, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64,
) -> Option<Bounded> {
    bound_within(list, allocator, sig, max, args, args_off, count_buf, count_off, BUDGET)
}

/// [`bound`] with the call's work limited to `budget` threads (a mesh group counts as 128):
/// a two-pass mesh call writes one record per group into a buffer of fixed size.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn bound_within(
    list: &ID3D12GraphicsCommandList, allocator: usize, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64, budget: u32,
) -> Option<Bounded> {
    if !enabled() || max == 0 {
        return None;
    }
    let s = signature(sig)?;
    let bytes = u64::from(max) * u64::from(s.stride);
    if bytes > MAX_BYTES {
        return None;
    }
    let (source, source_at) = in_window(args, args_off, bytes)?;
    let (counts, count_at) = match count_buf {
        0 => (source, None),
        c => match in_window(c, count_off, 4) {
            Some((w, at)) => (w, Some(at)),
            // a count we cannot read: every record takes from the budget
            None => (source, None),
        },
    };
    // lab: `XCL_CLAMP_SOURCE=args`: the records and the count are read through the title's own
    // PC buffers (as the PC ExecuteIndirect would) instead of through the window buffers
    static VIA_ARGS: OnceLock<bool> = OnceLock::new();
    let via_args = *VIA_ARGS.get_or_init(|| std::env::var("XCL_CLAMP_SOURCE").is_ok_and(|v| v.trim() == "args"));
    {
        // where the records are (the first 48 distinct 64 KiB pages)
        static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        if let Some((p, len)) = real::console_buffer(args) {
            let page = (p + args_off as usize) >> 16;
            let mut g = SEEN.lock().unwrap();
            if g.len() < 48 && !g.contains(&page) {
                g.push(page);
                let u = crate::gpu::uma::get();
                crate::xlog_info!(
                    "indirect clamp: records of kind {} (stride {}, action at {}, max {max}) at {:#x} (buffer {p:#x}, {len} bytes, offset {args_off:#x}), window {:?}, sparse {:?}, source at {source_at:#x}, count buffer {:x?} offset {count_off:#x}, budget {budget}, via args {via_args}",
                    s.kind,
                    s.stride,
                    s.action_at,
                    p + args_off as usize,
                    u.and_then(|u| u.window_of(p)),
                    u.map(|u| u.is_sparse(p)),
                    real::console_buffer(count_buf)
                );
            }
        }
    }
    let (source, source_at, counts, count_at) = if via_args {
        let raw = args as *mut c_void;
        let a = ID3D12Resource::from_raw_borrowed(&raw)?.GetGPUVirtualAddress() + args_off + real::buffer_delta(args);
        let (c, c_at) = match (count_buf, count_at) {
            (cb, Some(_)) if cb != 0 => {
                let raw = cb as *mut c_void;
                (ID3D12Resource::from_raw_borrowed(&raw)?.GetGPUVirtualAddress() + count_off + real::buffer_delta(cb), Some(0))
            }
            _ => (a, None),
        };
        (a, 0, c, c_at)
    } else {
        (source, source_at, counts, count_at)
    };
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let c = clamper(&dev)?;
    let (out, out_at) = scratch(&dev, allocator, bytes)?;
    let cfg = config(s, max, source_at, count_at, budget);
    // the title's writes to its records (through any view) before our reads
    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() }]);
    list.SetPipelineState(&c.pso);
    list.SetComputeRootSignature(&c.root);
    list.SetComputeRootUnorderedAccessView(0, source);
    list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress() + out_at);
    list.SetComputeRootUnorderedAccessView(2, counts);
    list.SetComputeRoot32BitConstants(3, CONFIG_WORDS, cfg.as_ptr().cast(), 0);
    list.Dispatch(1, 1, 1);
    transition(list, &out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    Some(Bounded(out, out_at))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_without_root_arguments_are_bounded() {
        let (draw, indexed, dispatch, mesh) = (
            D3D12_INDIRECT_ARGUMENT_TYPE_DRAW.0 as u32,
            D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED.0 as u32,
            D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.0 as u32,
            D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH.0 as u32,
        );
        let (vbv, ibv, constant) = (
            D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW.0 as u32,
            D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW.0 as u32,
            D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT.0 as u32,
        );
        assert_eq!(Signature::parse(16, &[[draw, 0, 0, 0]]), Some(Signature { stride: 16, action_at: 0, kind: 0 }));
        assert_eq!(Signature::parse(12, &[[dispatch, 0, 0, 0]]), Some(Signature { stride: 12, action_at: 0, kind: 2 }));
        assert_eq!(Signature::parse(16, &[[mesh, 0, 0, 0]]), Some(Signature { stride: 16, action_at: 0, kind: 3 }));
        // buffer views before the draw: 16 bytes each
        assert_eq!(
            Signature::parse(52, &[[vbv, 0, 0, 0], [ibv, 0, 0, 0], [indexed, 0, 0, 0]]),
            Some(Signature { stride: 52, action_at: 32, kind: 1 })
        );
        // root arguments, a stride too small or unaligned, no action, an action not last
        assert_eq!(Signature::parse(20, &[[constant, 0, 0, 1], [draw, 0, 0, 0]]), None);
        assert_eq!(Signature::parse(16, &[[indexed, 0, 0, 0]]), None);
        assert_eq!(Signature::parse(18, &[[draw, 0, 0, 0]]), None);
        assert_eq!(Signature::parse(16, &[]), None);
        assert_eq!(Signature::parse(32, &[[draw, 0, 0, 0], [vbv, 0, 0, 0]]), None);
    }

    #[test]
    fn a_signature_that_cannot_be_bounded_replaces_one_at_the_same_address() {
        let sig = 0x7300_0010usize;
        let draw = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW.0 as u32;
        note_signature(sig, 16, &[[draw, 0, 0, 0]]);
        assert!(signature(sig).is_some());
        note_signature(sig, 20, &[[D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT.0 as u32, 0, 0, 1], [draw, 0, 0, 0]]);
        assert!(signature(sig).is_none());
    }

    #[test]
    fn config_words_follow_the_shader() {
        let s = Signature { stride: 52, action_at: 32, kind: 1 };
        assert_eq!(config(s, 7, 128, Some(64), BUDGET), [7, 128, 52, 32, 1, 64, BUDGET]);
        assert_eq!(config(s, 7, 128, None, 99)[5], NO_COUNT);
        assert_eq!(config(s, 7, 128, None, 99)[6], 99);
    }

    /// lab: single records as a title's two-pass mesh calls pass them (max 1, small budgets)
    #[test]
    fn lab_single_records_with_small_budgets() {
        use windows::Win32::Foundation::WAIT_OBJECT_0;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            if crate::gpu::recomp::dxc::create_instance().is_none() {
                crate::gpu::recomp::runner::skip_test("no DXC");
                return;
            }
            let dev = dev.unwrap();
            let Some(c) = clamper(&dev) else { panic!("the clamp pipeline was not made") };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
            let buffer = |bytes: u64, heap, state, flags| -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..Default::default()
                };
                let mut out = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out).unwrap();
                out.unwrap()
            };
            // (kind, budget, record)
            let cases: [(u32, u32, [u32; 3]); 8] = [
                (2, BUDGET, [0x10, 1, 1]),
                (3, 102144, [1, 1, 1]),
                (3, 102144, [0, 1, 1]),
                (3, 833536, [0x1b, 1, 1]),
                (3, BUDGET, [4, 1, 1]),
                (2, 102144, [1, 1, 1]),
                (3, 93056, [2, 1, 1]),
                (3, 102144, [6, 1, 1]),
            ];
            let data: Vec<u32> = cases.iter().flat_map(|c| [c.2[0], c.2[1], c.2[2], 0xABCD]).collect();
            let bytes = (data.len() * 4) as u64;
            let upload = buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
            let mut mapped: *mut c_void = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut mapped)).unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast(), data.len());
            upload.Unmap(0, None);
            let src = buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            let out = buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            let readback = buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
            list.CopyBufferRegion(&src, 0, &upload, 0, bytes);
            transition(&list, &src, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            for (i, c0) in cases.iter().enumerate() {
                let s = Signature { stride: 12, action_at: 0, kind: c0.0 };
                let cfg = config(s, 1, 0, None, c0.1);
                list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() }]);
                list.SetPipelineState(&c.pso);
                list.SetComputeRootSignature(&c.root);
                list.SetComputeRootUnorderedAccessView(0, src.GetGPUVirtualAddress() + 16 * i as u64);
                list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress() + 16 * i as u64);
                list.SetComputeRootUnorderedAccessView(2, src.GetGPUVirtualAddress() + 16 * i as u64);
                list.SetComputeRoot32BitConstants(3, CONFIG_WORDS, cfg.as_ptr().cast(), 0);
                list.Dispatch(1, 1, 1);
            }
            transition(&list, &out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list.CopyBufferRegion(&readback, 0, &out, 0, bytes);
            list.Close().unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            queue.Signal(&fence, 1).unwrap();
            fence.SetEventOnCompletion(1, event).unwrap();
            assert_eq!(WaitForSingleObject(event, 10_000), WAIT_OBJECT_0);
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let got = std::slice::from_raw_parts(mapped.cast::<u32>(), data.len()).to_vec();
            readback.Unmap(0, None);
            for (i, c0) in cases.iter().enumerate() {
                eprintln!("case {i}: kind {} budget {} record {:x?} -> {:x?}", c0.0, c0.1, c0.2, &got[4 * i..4 * i + 3]);
            }
            for (i, c0) in cases.iter().enumerate() {
                assert_eq!(&got[4 * i..4 * i + 3], &c0.2, "case {i}");
            }
        }
    }

    /// The shader on the GPU: clamped records, and records past the budget emptied; everything
    /// else in a record copied as it is.
    #[test]
    fn gpu_records_are_clamped_and_budgeted() {
        use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            if crate::gpu::recomp::dxc::create_instance().is_none() {
                crate::gpu::recomp::runner::skip_test("no DXC");
                return;
            }
            let dev = dev.unwrap();
            let Some(c) = clamper(&dev) else { panic!("the clamp pipeline was not made") };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
            let buffer = |bytes: u64, heap, state, flags| -> ID3D12Resource {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..Default::default()
                };
                let mut out = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out).unwrap();
                out.unwrap()
            };
            // indexed draws after a 16-byte vertex buffer view: 36 bytes a record; 2,000 records
            // (two chunks of the shader's 1,024), the count buffer after them
            let (stride, n) = (36u32, 2000u32);
            let words = (stride / 4) as usize;
            let mut data = vec![0u32; words * n as usize + 1];
            for i in 0..n as usize {
                let r = &mut data[i * words..(i + 1) * words];
                r[..4].copy_from_slice(&[0xAAAA_0000 + i as u32, 0xBBBB, 0xCCCC, 0xDDDD]);
                // index count, instances, first index, base vertex, first instance
                r[4..9].copy_from_slice(&[3, 1, 7, 0xFFFF_FFFF, 9]);
            }
            data[words..words + 9][4..6].copy_from_slice(&[u32::MAX, 5]); // 1: too many indices
            data[2 * words + 4..2 * words + 6].copy_from_slice(&[1 << 20, 1 << 20]); // 2: too many instances
            // 1,000..1,500: 2^20 vertex instances each; after records 1 and 2 (2^26 each) and
            // 998 of 3, the budget (2^29) holds 383 of them: record 1,383 is the first past it
            for i in 1000..1500 {
                data[i * words + 4..i * words + 6].copy_from_slice(&[1 << 10, 1 << 10]);
            }
            data[words * n as usize] = 1600; // the count
            let bytes = (data.len() * 4) as u64;
            let upload = buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
            let mut mapped: *mut c_void = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut mapped)).unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast(), data.len());
            upload.Unmap(0, None);
            let src = buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            let out = buffer(u64::from(stride * n), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            let readback = buffer(u64::from(stride * n), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
            list.CopyBufferRegion(&src, 0, &upload, 0, bytes);
            transition(&list, &src, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            let s = Signature { stride, action_at: 16, kind: 1 };
            let cfg = config(s, n, 0, Some(stride * n), BUDGET);
            list.SetPipelineState(&c.pso);
            list.SetComputeRootSignature(&c.root);
            list.SetComputeRootUnorderedAccessView(0, src.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, src.GetGPUVirtualAddress());
            list.SetComputeRoot32BitConstants(3, CONFIG_WORDS, cfg.as_ptr().cast(), 0);
            list.Dispatch(1, 1, 1);
            transition(&list, &out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list.CopyBufferRegion(&readback, 0, &out, 0, u64::from(stride * n));
            list.Close().unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            queue.Signal(&fence, 1).unwrap();
            fence.SetEventOnCompletion(1, event).unwrap();
            assert_eq!(WaitForSingleObject(event, 10_000), WAIT_OBJECT_0);
            CloseHandle(event).unwrap();
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let got = std::slice::from_raw_parts(mapped.cast::<u32>(), words * n as usize).to_vec();
            readback.Unmap(0, None);
            let rec = |i: usize| &got[i * words..(i + 1) * words];
            assert_eq!(rec(0), &[0xAAAA_0000, 0xBBBB, 0xCCCC, 0xDDDD, 3, 1, 7, 0xFFFF_FFFF, 9], "an ordinary record is copied as it is");
            assert_eq!(&rec(1)[4..6], &[1 << 24, 4], "indices per instance clamped, instances to 2^26 vertex instances");
            assert_eq!(&rec(2)[4..6], &[1 << 20, 1 << 6], "instances clamped to 2^26 vertex instances");
            assert_eq!(&rec(1382)[4..6], &[1 << 10, 1 << 10], "within the budget");
            assert_eq!(&rec(1383)[4..6], &[1 << 10, 0], "past the budget: no instances");
            assert_eq!(&rec(1550)[4..6], &[3, 0], "every record after it is emptied too");
            assert_eq!(rec(1499)[0], 0xAAAA_0000 + 1499, "the rest of an emptied record is kept");
        }
    }
}
