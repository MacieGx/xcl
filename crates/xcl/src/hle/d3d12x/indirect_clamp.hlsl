// Bounds a title's indirect draw and dispatch records before ExecuteIndirect reads them
// (`indirect_clamp.rs`). One group walks every record in chunks of THREADS: each record is
// copied whole, its draw or dispatch arguments are clamped to per-record limits, and a running
// total of the work the records ask for (a prefix sum across the chunk) empties every record
// past BUDGET.

#define THREADS 1024
// per record: vertices (or indices) per instance, vertices times instances, groups per
// dimension, groups per dispatch; a group counts as 64 threads, a mesh group as 128
#define MAX_VERTICES (1u << 24)
#define MAX_VERTEX_INSTANCES (1u << 26)
#define MAX_GROUPS_DIM 65535u
#define MAX_GROUPS (1u << 22)

RWByteAddressBuffer source : register(u0);  // the UMA window holding the title's records
RWByteAddressBuffer bounded : register(u1); // our copy, which ExecuteIndirect reads
RWByteAddressBuffer counts : register(u2);  // the UMA window holding the count (if any)

#ifdef BATCH
// One group a call (`indirect_clamp.rs`, `XCL_CLAMP_BATCH`): the calls of a run are jobs in a
// table the CPU fills while the list is recorded, 8 words each - the 7 below, then where the
// call's copy starts in `bounded`. A job without records is none. The count the call may run
// (the title's, at most `records`) is written after the copy's last record.
ByteAddressBuffer jobs : register(t0);
static uint records, source_at, stride, action_at, kind, count_at, budget, out_at;
#else
cbuffer Config : register(b0) {
    uint records;   // the call's maximum command count
    uint source_at; // byte offset of record 0 in `source`
    uint stride;    // bytes per record (a multiple of 4)
    uint action_at; // byte offset of the draw or dispatch arguments in a record
    uint kind;      // 0 draw, 1 indexed draw, 2 dispatch, 3 mesh dispatch
    uint count_at;  // byte offset of the count in `counts`; 0xFFFFFFFF: no count buffer
    uint budget;    // work the whole call may ask for (threads; see clamp_args)
};
static const uint out_at = 0;
#endif

groupshared uint scan[THREADS];
groupshared uint carry;

uint sat_add(uint a, uint b) {
    uint s = a + b;
    return s < a ? 0xFFFFFFFFu : s;
}

// Clamps one record's arguments in place and returns the work they ask for.
uint clamp_args(inout uint3 a) {
    // lab: kind >> 8 = groups a row for a program with a folded group id (`kernel::FOLD_GROUPS`)
    uint fold = (kind >> 8) & 0x7FFFFFu;
    uint k = kind & 0xFF;
    // lab (`XCL_CLAMP_TIGHT=1`, bit 31 of kind): 2^18 groups a record. A garbage record clamped to
    // [65535, 64, 1] on a heavy program ran past the 2 s the driver gets (jessenic's runs on #1795:
    // 7 of 7 ended in 0x887a0006 with 2^22 / 2^29, 0 of 2 with 2^18 / 2^26); the title's widest
    // real dispatch seen is 115,200 groups
    uint max_groups = (kind >> 31) != 0 ? (1u << 18) : MAX_GROUPS;
    if (k <= 1) {
        a.x = min(a.x, MAX_VERTICES);
        a.y = min(a.y, MAX_VERTEX_INSTANCES / max(a.x, 1u));
        return a.x * a.y;
    }
    // a 1-D dispatch of more groups than D3D12 takes in a dimension, as whole rows (the last
    // row runs up to fold - 1 groups more than were asked for)
    if (fold != 0 && a.x > MAX_GROUPS_DIM && a.y == 1 && a.z == 1) {
        a.y = (a.x + fold - 1) / fold;
        a.x = fold;
    }
    a = min(a, uint3(MAX_GROUPS_DIM, MAX_GROUPS_DIM, MAX_GROUPS_DIM));
    a.y = min(a.y, max_groups / max(a.x, 1u));
    a.z = min(a.z, max_groups / max(a.x * a.y, 1u));
    return a.x * a.y * a.z * (k == 3 ? 128u : 64u);
}

[numthreads(THREADS, 1, 1)]
void main(uint t : SV_GroupIndex, uint3 group : SV_GroupID) {
#ifdef BATCH
    uint4 j0 = jobs.Load4(group.x * 32);
    uint4 j1 = jobs.Load4(group.x * 32 + 16);
    records = j0.x; source_at = j0.y; stride = j0.z; action_at = j0.w;
    kind = j1.x; count_at = j1.y; budget = j1.z; out_at = j1.w;
    if (records == 0)
        return;
#endif
    if (t == 0)
        carry = 0;
    uint live = records;
    if (count_at != 0xFFFFFFFFu)
        live = min(live, counts.Load(count_at));
#ifdef BATCH
    if (t == 0)
        bounded.Store(out_at + records * stride, live);
#endif
    GroupMemoryBarrierWithGroupSync();
    for (uint first = 0; first < records; first += THREADS) {
        uint i = first + t;
        uint3 a = uint3(0, 0, 0);
        uint work = 0;
        if (i < records) {
            uint src = source_at + i * stride;
            uint dst = out_at + i * stride;
            for (uint k = 0; k < stride; k += 4)
                bounded.Store(dst + k, source.Load(src + k));
            a = source.Load3(src + action_at);
            work = clamp_args(a);
        }
        // records past the count are never executed: they take none of the budget
        scan[t] = i < live ? work : 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint d = 1; d < THREADS; d <<= 1) {
            uint before = t >= d ? scan[t - d] : 0;
            GroupMemoryBarrierWithGroupSync();
            scan[t] = sat_add(scan[t], before);
            GroupMemoryBarrierWithGroupSync();
        }
        uint total = sat_add(carry, scan[t]);
        if (i < records) {
            if (total > budget) {
                if ((kind & 0xFF) <= 1)
                    a.y = 0;
                else
                    a = uint3(0, 0, 0);
            }
            bounded.Store3(out_at + i * stride + action_at, a);
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == THREADS - 1)
            carry = total;
        GroupMemoryBarrierWithGroupSync();
    }
}
