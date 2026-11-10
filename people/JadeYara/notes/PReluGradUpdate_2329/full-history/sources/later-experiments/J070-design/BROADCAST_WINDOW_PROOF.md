# J070 design: general compact-window broadcast Gather

Status: CPU-only design and validation. No NPU candidate, cloud action, production edit, or J069 edit.
Inputs read: J068/design/REPAIR_AND_GATHER.md and broadcast_layout_model.py; J067/reference-evidence/prelu_grad_update.py; J068/contract/probe-fixed.cpp; J069/candidate/code/op_kernel/p_relu_grad_update_gather.h.

## Result

For one **compact row-major source broadcast to a compact row-major output**, the image of any consecutive output interval is the union of **at most two exact contiguous source intervals**. Their combined element count is at most the output tile length. This includes arbitrary interleaved broadcast axes and rank 8. It is a theorem under the stated storage premise, not an assumed pattern.

A simpler first general fallback does not need two-window packing:
- Choose tile cap T (multiple of 64).
- Let P* be the smallest output suffix volume >= T, or total output volume when the latter is < T.
- End every tile at min(core end, z+T, next P* boundary).
- Every source's exact min/max hull for this tile has at most T elements.
- Copy that one hull, build bounded int32 relative indices, Gather, and run the already repaired TBE math.

This permits at most two pieces of any original length-T output interval, not one piece per channel or per lane. It has <=3 bulk input copies per piece for g/x/w, and no per-element GM load. Window planning is O(rank) scalar work, independent of the lane count. It is a general fallback design, not evidence of NPU speed.

## 1. Normalize the actual physical contract first

For each source, right-align the actual mapped shape with output shape D.
Each source extent must be 1 or D[j]. Compact source strides S[j] are zero on broadcast axes and ordinary row-major strides on retained axes. Define:
- P[j] = product(D[j+1:]), the output suffix volume.
- Q[j] = product(source_shape[j+1:]), the compact source suffix volume.
- f(z) = sum(((z // P[j]) % D[j]) * S[j]).
All products, global offsets, endpoints and GM element counts use checked uint64/int64 scalar arithmetic. Reject overflow. Empty output exits before any division/modulo or source copy.

Apply the actual TBE weight reshape from the reference; do not reshape solely by a guessed channel rule. Validate mapped **storage** element count and broadcast compatibility. ND and NCHW are ordinary row-major storage here.

NC1HWC0 uses physical [N,C1,H,W,C0] and physical blocked weight [1,C1,1,1,16]. For origin C=19, physical C1=2 and C0=16, hence 32 stored channel lanes, not 19. All physical lanes are part of the indexing model and existing raw-bit evidence. Do not discard padding lanes, modulus by origin C=19, read 19 raw weights as if 32 exist, or convert logical channel broadcast into an unverified storage mapping. Mixed-origin/layout transformations not represented by legal physical broadcast still require separate contract evidence.

## 2. Proof: at most two exact intervals

Consider a single output increment. Let j be the output digit that increments after all less-significant digits carry:
- If j is retained, f increases by exactly 1: the source stride Q[j] minus the reset of its entire source suffix Q[j]-1.
- If j is broadcast, f changes by -(Q[j]-1), possibly zero.
Thus an upward jump larger than one is impossible.

For an output tile of n elements, let Q be the largest Q[j] of any actual negative reset encountered, or 1 when none occurs.
Every smaller source suffix volume divides Q. Its reset stays within one Q-aligned source block. Therefore floor(f/Q) never decreases over the tile.

If the tile visits multiple Q-aligned source blocks, its source image is continuous: the first block's suffix reaches its top, middle blocks are fully traversed, and the last block's prefix starts at its bottom. Upward steps cannot skip an element. Hence hull length = distinct count <= n.

Otherwise it remains in one Q-aligned block. At an encountered maximum reset it moves from the top to the bottom of that block. The portion before this reset has visited one contiguous suffix ending at the top; the portion after it has visited one contiguous prefix starting at the bottom. Their union has at most two intervals.

Consequences:
1. Exact source image: <=2 intervals, combined length <=n.
2. Source hull length <= max(n,Q).
3. These claims do not assume source absolute offsets fit int32.

The exact intervals can have an enormous gap. Example output [2,2^32], source [1,2^32], tile z=2^32-2,n=4:
source offsets [2^32-2,2^32-1,0,1].
Two tiny copies suffice; a single min/max hull would be 2^32 elements.

## 3. Exact scalar window calculation without scanning lanes

The executable Layout.windows(z,n) is the specification. It uses O(rank) lexicographic boxes and a tiny scalar interval merge.

Let a=coords(z), b=coords(z+n-1); A[j]/B[j] are source offsets contributed by coordinates before j. Find the first differing coordinate k.
- If none differs, the image is the singleton f(z).
- Add singleton intervals f(z), f(z+n-1).
- For each j>k with a[j]+1<D[j], add:
  [ A[j]+(a[j]+1)*S[j], A[j]+(D[j]-1)*S[j]+Q[j]-1 ].
- For each j>k with b[j]>0, add:
  [ B[j], B[j]+(b[j]-1)*S[j]+Q[j]-1 ].
- If b[k]>a[k]+1, add:
  [ A[k]+(a[k]+1)*S[k], A[k]+(b[k]-1)*S[k]+Q[k]-1 ].

Each box fixes its prefix, ranges one digit, and fills the entire suffix. Its projected source image is exactly the indicated contiguous interval, including broadcast axes. Sort/merge the <=2*rank+1 intervals; the theorem guarantees <=2 results.

For the recommended **one-hull fallback**, do not sort or store the list: just update min(lo) and max(hi) as each box is generated. This is O(rank) scalar work and <=8-rank bounded metadata. Output coordinate decomposition can be shared by g/x/w; accumulate their prefix offsets together.

Merely mapping the two output endpoints is wrong. Example output [2,100], source [1,100], z=98,n=4 gives [98,99,0,1]. Endpoint min/max [1,98] misses both 0 and 99.

## 4. One-hull fallback and its bound

Choose common P* from output suffix volumes and T, independently of input broadcast masks:
P* = min({P[j] : P[j]>=T}), or total when this set is empty.
For z in the core's assigned global continuous output range:
n = min(T, core_end-z, P*-(z % P*)).

Every crossed output boundary has suffix period P[j]<T. Every corresponding source suffix Q[j]<=P[j]<T. Apply the hull bound from section 2:
hull length <= max(n,max_reset_Q) <= T.

If a preceding length-T tile crossed a large reset, the P* boundary divides it into at most two pieces. P*>=T whenever output total>=T, so this is a bounded split, not repeated channel segmentation. A core can maintain the remaining distance to its next P* boundary incrementally instead of doing a fresh large modulo on every tile.

For each source:
1. Compute exact min/max (lo,hi) with the box formulas.
2. Bulk copy source GM[lo:hi+1] to a 32B-aligned UB compact buffer.
3. Convert low-precision compact values to FP32 if required by the repaired math.
4. Build indices described below; Gather into the expanded g/x/w buffer.
5. Reuse the compact buffer for the next source only after Gather has completed.
6. Apply actual TBE coefficient/masked-x math, including quantized-dx zero propagation.

Preserve identity-source direct-copy and scalar-source UB Duplicate cases. A source with no broadcast does not need generic Gather. Existing same-shape fast paths remain useful and should not be slowed merely to route every shape through this fallback.

## 5. Bounded int32 vector index generation

There is no vector 64-bit division and no absolute source offset in a vector.

Precompute for j<rank-1:
C[j] = S[j] - D[j+1]*S[j+1].
Then the exact identity is:
f(z+i)-f(z) = S[last]*i
             + sum(C[j] * floor(((z % P[j])+i)/P[j])).

Only include terms P[j]<T. For all P[j]>=T, the tile does not cross a P* boundary, so their quotient difference is zero. Skip C[j]=0.

For 0<=i<n:
- S[last] is 0 or 1.
- |C[j]|<=P[j]<T.
- x=(z%P[j])+i <2*T.
- |C[j]*floor(x/P[j])|<2*T.
- The final element index is (f(z)-lo)+the relative expression, in [0,hi-lo].
All bases are subtracted in 64-bit scalar arithmetic first. Even huge source strides/offsets never enter int32 arithmetic.

Implement with the APIs already evidenced by J069:
- persistent CreateVecIndex<int32_t> ramp;
- Adds, Muls<int32_t>, ShiftRight with explicitly int32_t scalar shift;
- Add/Sub; final Muls(index,4);
- Gather<float> with byte offsets, source/destination distinct and 32B aligned.

Exact small-division constants for divisor d>1:
X=d+T-2;
k=ceil(log2(X*d));
m=ceil(2^k/d);
floor(x/d)=(x*m)>>k for all 0<=x<=X, provided X*m<=INT32_MAX.
The CPU script checks every x for d=1..4095 at T=4096, not only endpoints. Division by 1 and powers of two can use simpler code. No general float reciprocal is required.

The same quotient vector can be reused for all three input index accumulators if UB liveness permits; alternatively rebuild it per input using two scratch index buffers. This choice requires measurement, not extra shape-specialized routes.

## 6. Tail lanes, 32B padding, and core ownership

Let compute_count=AlignUp(n,64), with T a multiple of 64. First use:
I = min(ramp,n-1)
for **all** index generation. Thus padding lanes repeat a valid final source lane. Do not derive source windows using n and then extrapolate indices through compute_count: that can cross the P* boundary or end of GM.

Load exactly (hi-lo+1)*sizeof(input_type) bytes with DataCopyPad; zero-fill its UB right padding to a 32B boundary. Do not round the GM read beyond its physical allocation. Cast only valid compact elements; Gather's valid byte offsets remain within them. The expanded buffers have T lanes, including initialized virtual lanes.

When using the optional exact two-window packing, first UB span length is L0 and second starts at B=AlignUp(L0,32/sizeof(input_type)); remap by B, not by L0. Total padded compact allocation is at most n+2*(alignment-1) source elements. If conversion changes element width, keep the raw and FP32 compact segment offsets consistent; derive byte offsets from the actual FP32 layout.

Keep global multicore partitions aligned to 32B output ownership. P* cuts are internal to a core and may be unaligned; use exact-byte DataCopyPad output writes. Do not let padded virtual output lanes become intercore writes.

UB allocation must be recomputed for three expanded inputs, a reusable compact input, low-precision conversion, the existing repaired math, index scratch, ramp and alignment. T<=4096 here is an integer-model limit, not a claim that J069's UB budget supports the general fallback.

## 7. Boundaries of the theorem and useful counterexamples

The theorem needs compact row-major source storage, with each axis either retained or broadcast. It does not hold for arbitrary strided views or a logical-to-physical permutation. Example source strides [100,1] for output [3,3], tile z=1,n=8:
[1,2,100,101,102,200,201,202] requires 3 intervals. Treating NC1 as a logical flat [N,C,H,W] view would similarly invalidate the premise.

Two exact windows can still be separated when the output tile is larger than the reset source volume. Example output [2,100,1000], source [1,100,1], z=99900,n=200 visits 99 and 0 only. Its hull is 100<=n, but the exact image is two singleton intervals. Thus "two windows implies large reset>T" is false; use the proven hull rule, not that shortcut.

No new math policy is introduced. This design only expands the three inputs correctly. Actual numerical correctness must preserve the reference's mask multiplication, zero/NaN/Inf behavior, casts, and zero propagation, not replace it with numpy where.

## 8. Validation and next gate

Files:
- broadcast_windows.py: executable scalar planner and bounded SIMD lane model.
- validation.json: reproducible CPU results.
- session-card.md: isolated scope.

CPU checks include every source mask on dimensions 2/3 at ranks 1..4, every tile start and length up to 32; generic caps 4/8/16/32; randomized ranks 2..8 with unit dimensions and huge strides; adversarial 2^32 reset; physical NC1 C19 descriptor; all nonempty sources from the 17 J068 contract models; and exhaustive small-divisor exact integer division. A preceding independent JS enumeration covered rank<=5 and 2,898,526 tiles with maximum two intervals; the saved Python model independently validates its own exact box construction.

Next device gate, if parent chooses implementation: one independent general fallback with existing fast paths and repaired math, first descriptor/empty correctness, then actual TBE broadcast/physical-layout/NaN/tail tests, then targeted cost comparison. No performance conclusion follows from this CPU proof.
