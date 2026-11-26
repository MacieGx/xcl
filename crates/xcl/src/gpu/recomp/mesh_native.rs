//! Which console mesh (NGG) programs can run as a native D3D12 mesh shader instead of the
//! two-pass path (compute pass + pass-through mesh shader).
//!
//! D3D12 has no `WaveSize` for mesh shaders, so a native form cannot fix the hardware wave: it
//! must give the same result whatever lanes the driver puts a thread in. The form meant here
//! numbers the console's threads by `SV_GroupIndex` (thread = wave · 64 + lane, vertex and
//! primitive slot = thread) and keeps EXEC / VCC as ballots of the hardware wave, as vertex
//! and pixel shaders do. That is exact when the program
//! - only uses masks lane by lane (guards, selects), never as numbers (counts, first bit,
//!   compares of the whole mask, prefix counts over a mask that is not the lane-index idiom);
//! - reads other lanes only with `v_readfirstlane` (of a value every lane holds alike: not
//!   proven here) and has no lane permutes, DPP or LDS;
//! - jumps over code on an empty mask only where no scalar result of that code is read
//!   afterwards: a hardware wave of 32 can be empty where the console's wave of 64 was not, and
//!   would then keep the older scalar;
//! - fits the mesh limits: 128 threads, 256 vertices, 256 primitives, 32 KiB of outputs.
//!
//! This module only reports. Nothing is translated differently because of it.

use std::collections::{BTreeMap, BTreeSet};

use super::MsInterface;
use crate::gpu::rdna2::{self, Decoded, F_CONDITIONAL, F_TERMINATOR};

/// D3D12: bytes of vertex + primitive attributes one mesh group may output.
pub const OUTPUT_BYTES: u32 = 32768;

/// Bytes D3D12 accounts for `rows` float4 vertex rows (position included): the vertices are
/// counted in whole chunks of 32 (`emit::passthrough_max_vertices`).
pub fn output_bytes(max_vertices: u32, rows: u32) -> u32 {
    max_vertices.next_multiple_of(32) * rows * 16
}

/// Why a program cannot take the native form (empty: it can), and what it contains.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub blockers: BTreeSet<&'static str>,
    pub instructions: usize,
    /// forward branches on an empty mask (`s_cbranch_execz` and the like)
    pub mask_skips: usize,
    pub read_first_lane: usize,
    /// `v_readfirstlane` by what last wrote the VGPR it reads (`mesh_wave::LaneValueProvenance`)
    pub first_lane_of: BTreeMap<String, usize>,
    pub barriers: usize,
    pub allocations: usize,
    /// position, primitive, parameter exports
    pub exports: [usize; 3],
}

fn decode(code: &[u8]) -> (Vec<Decoded>, bool) {
    let (mut out, mut off) = (Vec::new(), 0usize);
    while off + 4 <= code.len() {
        let Ok(d) = rdna2::decode(&code[off..], off as u64) else { return (out, false) };
        off += d.len;
        out.push(d);
    }
    rdna2::resolve_far_jumps(&mut out);
    (out, off == code.len())
}

fn reads_scc(d: &Decoded) -> bool {
    d.ienc.operands.iter().any(|o| !o.output && o.ty.contains("SCC")) || matches!(d.inst.name, "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1")
}

fn writes_scc(d: &Decoded) -> bool {
    d.ienc.operands.iter().any(|o| o.output && o.ty.contains("SCC"))
}

/// SGPRs (and SCC as 128) live before each instruction.
fn liveness(insts: &[Decoded], pcs: &BTreeMap<u64, usize>) -> Vec<[u64; 3]> {
    let bit = |s: &mut [u64; 3], r: u64, on: bool| {
        let (w, b) = ((r / 64) as usize, r % 64);
        if on {
            s[w] |= 1 << b
        } else {
            s[w] &= !(1 << b)
        }
    };
    let mut live = vec![[0u64; 3]; insts.len() + 1];
    loop {
        let mut changed = false;
        for i in (0..insts.len()).rev() {
            let d = &insts[i];
            let mut out = [0u64; 3];
            let mut join = |j: usize| {
                for k in 0..3 {
                    out[k] |= live[j][k];
                }
            };
            match rdna2::branch_target(d) {
                Some(t) => {
                    if let Some(&j) = pcs.get(&t) {
                        join(j);
                    }
                    if d.inst.flags & F_CONDITIONAL != 0 {
                        join(i + 1);
                    }
                }
                None if d.inst.flags & F_TERMINATOR == 0 => join(i + 1),
                None => {}
            }
            let (r, w) = rdna2::sgpr_uses(d);
            for x in w {
                bit(&mut out, x, false);
            }
            if writes_scc(d) {
                bit(&mut out, 128, false);
            }
            for x in r {
                bit(&mut out, x, true);
            }
            if reads_scc(d) {
                bit(&mut out, 128, true);
            }
            // `s_sendmsg` reads M0
            if d.inst.name == "S_SENDMSG" {
                bit(&mut out, 124, true);
            }
            if out != live[i] {
                live[i] = out;
                changed = true;
            }
        }
        if !changed {
            return live;
        }
    }
}

pub fn report(code: &[u8], ms: &MsInterface) -> Report {
    let mut r = Report::default();
    let (insts, whole) = decode(code);
    r.instructions = insts.len();
    if !whole {
        r.blockers.insert("program not decoded in full");
    }
    if ms.threads > 128 {
        r.blockers.insert("more than 128 threads");
    }
    if ms.max_vertices > 256 || ms.max_primitives > 256 {
        r.blockers.insert("more than 256 vertices or primitives");
    }
    if ms.max_vertices > ms.threads || ms.max_primitives > ms.threads {
        r.blockers.insert("more vertices or primitives than threads");
    }
    // the wave-size hazards the split form is for; a branch on an empty mask is looked at below
    // (`v_readfirstlane` is in its "cross-half lane exchange": the other lane reads are named here)
    for h in super::geometry_wave32_hazards(code) {
        if !matches!(h, "branch on a whole-wave mask" | "cross-half lane exchange") {
            r.blockers.insert(h);
        }
    }
    for l in super::mesh_wave::analyze(code).lane_reads {
        if l.kind == super::mesh_wave::LaneReadKind::ReadFirst {
            *r.first_lane_of.entry(format!("{:?}{}", l.provenance, if l.source_vgpr == Some(5) { " v5" } else { "" })).or_default() += 1;
        }
    }
    let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
    let live = liveness(&insts, &pcs);
    for (i, d) in insts.iter().enumerate() {
        let n = d.inst.name;
        if d.enc.name == "ENC_DS" {
            r.blockers.insert("LDS");
        }
        match n {
            "S_BARRIER" => r.barriers += 1,
            "V_READFIRSTLANE_B32" => r.read_first_lane += 1,
            "V_READLANE_B32" | "V_WRITELANE_B32" | "V_PERMLANE16_B32" | "V_PERMLANEX16_B32" => {
                r.blockers.insert("lane read / write / permute");
            }
            _ if d.enc.name.contains("DPP") => {
                r.blockers.insert("DPP");
            }
            "S_SENDMSG" if rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) & 0xF == 9 => r.allocations += 1,
            "EXP" => match rdna2::field_value(d.enc, &d.words, "TARGET").unwrap_or(0) {
                12..=15 => r.exports[0] += 1,
                20 => r.exports[1] += 1,
                32..=63 => r.exports[2] += 1,
                _ => {}
            },
            _ => {}
        }
        if !matches!(n, "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ" | "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ") {
            continue;
        }
        let Some(&t) = rdna2::branch_target(d).and_then(|t| pcs.get(&t)) else {
            r.blockers.insert("mask branch out of the program");
            continue;
        };
        if t <= i {
            r.blockers.insert("loop on a whole-wave mask");
            continue;
        }
        r.mask_skips += 1;
        // scalars the jumped-over code writes that something after it may read
        let mut written = [0u64; 3];
        let mut inner_exit = false;
        for d in &insts[i + 1..t] {
            let (_, w) = rdna2::sgpr_uses(d);
            for x in w {
                // EXEC is put back where the branches meet; VCC is a mask (used lane by lane)
                if !matches!(x, 106 | 107 | 126 | 127) {
                    written[(x / 64) as usize] |= 1 << (x % 64);
                }
            }
            if writes_scc(d) {
                written[2] |= 1;
            }
            if d.inst.flags & F_TERMINATOR != 0 && d.inst.name.starts_with("S_ENDPGM") {
                inner_exit = true;
            }
        }
        let stale = (0..3).any(|k| written[k] & live[t][k] != 0);
        if stale && !inner_exit {
            r.blockers.insert("scalar written in code a mask branch jumps over is read after it");
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_bytes_counts_whole_chunks_of_vertices() {
        assert_eq!(output_bytes(128, 16), 32768);
        assert_eq!(output_bytes(80, 22), 96 * 22 * 16);
    }

    /// The pack's mesh programs against the native form's conditions: `XCL_SAMPLE_RPF`;
    /// `XCL_MESH_LIST=<file>` also writes one line per program (name, hash, limits, blockers).
    #[test]
    #[ignore]
    fn retail_mesh_native_survey() {
        let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
        let data = std::fs::read(path).unwrap();
        let mut total = 0usize;
        let mut fit = 0usize;
        let mut why: BTreeMap<String, usize> = BTreeMap::new();
        let mut shapes: BTreeMap<(u32, u32, u32), usize> = BTreeMap::new();
        let mut rows_hist: BTreeMap<u32, usize> = BTreeMap::new();
        let mut over_all = 0usize;
        let mut lines = Vec::new();
        let mut no_iface = 0usize;
        let mut first_of: BTreeMap<String, usize> = BTreeMap::new();
        let (mut wave32, mut skips, mut firsts, mut insts, mut allocs0, mut allocs_many) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        // (name, hash, interface, wave64, bytes, report) per program, on every core
        type Row = (String, u64, Option<MsInterface>, bool, usize, Report);
        let records = crate::gpu::pack::scan(&data, 0x40);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let rows: std::sync::Mutex<Vec<Row>> = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|sc| {
            for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
                sc.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(r) = records.get(i) else { break };
                    if !r.name.starts_with("MS") {
                        continue;
                    }
                    let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                    let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                    if sh.stage != "MS" {
                        continue;
                    }
                    for f in &sh.fssm {
                        if f.stages.first().map_or(0, |s| s.hw) != 7 {
                            continue;
                        }
                        let hash = crate::gpu::shader::fnv64(&blob);
                        let Some(crate::gpu::recomp::Mode::Mesh(ms)) = crate::gpu::recomp::interface(&blob, &sh, f) else {
                            rows.lock().unwrap().push((r.name.clone(), hash, None, false, 0, Report::default()));
                            continue;
                        };
                        let Some(p) = f.programs.first() else { continue };
                        let code = &f.data[p.offset..p.offset + p.len];
                        let rep = report(code, &ms);
                        rows.lock().unwrap().push((r.name.clone(), hash, Some(ms), crate::gpu::recomp::fssm_wave64(f, code), code.len(), rep));
                    }
                });
            }
        });
        let mut rows = rows.into_inner().unwrap();
        rows.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        for (name, hash, ms, wave64, bytes, rep) in rows {
            let Some(ms) = ms else {
                no_iface += 1;
                continue;
            };
            total += 1;
            let rows = 1 + crate::gpu::recomp::ranked(&ms.outputs, true).len() as u32;
            let all = output_bytes(ms.max_vertices, rows);
            *shapes.entry((ms.threads, ms.max_vertices, ms.max_primitives)).or_default() += 1;
            *rows_hist.entry(rows).or_default() += 1;
            if all > OUTPUT_BYTES {
                over_all += 1;
            }
            if !wave64 {
                wave32 += 1;
            }
            for (k, n) in &rep.first_lane_of {
                *first_of.entry(k.clone()).or_default() += n;
            }
            skips += rep.mask_skips;
            firsts += rep.read_first_lane;
            insts += rep.instructions;
            match rep.allocations {
                0 => allocs0 += 1,
                1 => {}
                _ => allocs_many += 1,
            }
            if rep.blockers.is_empty() {
                fit += 1;
            }
            for b in &rep.blockers {
                *why.entry(b.to_string()).or_default() += 1;
            }
            lines.push(format!(
                "{name}\t{hash:016x}\t{}\t{}\t{}\t{rows}\t{all}\t{bytes}\t{}",
                ms.threads,
                ms.max_vertices,
                ms.max_primitives,
                rep.blockers.iter().copied().collect::<Vec<_>>().join("; ")
            ));
        }
        println!("mesh programs: {total} ({no_iface} without an interface), native form possible: {fit}");
        println!("  wave32 programs: {wave32}; instructions {insts}; mask skips {skips}; v_readfirstlane {firsts}; no GS_ALLOC_REQ: {allocs0}, several: {allocs_many}");
        println!("  every output row declared is over 32 KiB: {over_all}");
        println!("  v_readfirstlane by the writer of its source: {first_of:?}");
        println!("  (threads, max vertices, max primitives): {shapes:?}");
        println!("  float4 rows with the position: {rows_hist:?}");
        for (b, n) in &why {
            println!("  {n:6}  {b}");
        }
        if let Ok(out) = std::env::var("XCL_MESH_LIST") {
            std::fs::write(out, lines.join("\n")).unwrap();
        }
    }
}
