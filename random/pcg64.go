package random

import "math/bits"

// PCG64: 128-bit LCG with XSL-RR output. Faithful port of
// numpy/random/src/pcg64/pcg64.h (pcg_setseq_128_xsl_rr_64_random_r) and
// numpy/random/_pcg64.pyx (seeding via SeedSequence.generate_state(4, uint64)).

// pcg128 arithmetic is [2]uint64{hi, lo}; the multiplier and increment below
// are PCG_DEFAULT_MULTIPLIER_128 and the seeding masks follow
// pcg_setseq_128_srandom_r (pcg64.h:174).
const (
	pcgMultHi uint64 = 2549297995355413924
	pcgMultLo uint64 = 4865540595714422341
)

// PCG64 is the permuted-congruential generator (numpy's default bit
// generator), 128-bit LCG state with 64-bit XSL-RR output. Use it through
// NewGenerator.
type PCG64 struct {
	stateHi, stateLo uint64
	incHi, incLo     uint64
	hasUint32        bool
	uinteger         uint32
}

// Compile-time check that PCG64 implements BitGenerator.
var _ BitGenerator = (*PCG64)(nil)

// NewPCG64 seeds PCG64 from a SeedSequence exactly like numpy: state key =
// seq[0:2], inc key = seq[2:4] (uint64 words), passed to
// pcg64_set_seed → pcg_setseq_128_srandom_r.
func NewPCG64(seq *SeedSequence) *PCG64 {
	key := seq.GenerateState64(4)
	p := &PCG64{}
	// pcg64_set_seed: initstate = key[0]<<64 | key[1], initseq = key[2]<<64 |
	// key[3]; pcg_setseq_128_srandom_r (pcg64.h:174): inc = (initseq<<1)|1.
	p.incHi = key[2]<<1 | key[3]>>63
	p.incLo = key[3]<<1 | 1
	p.step()
	p.add(p.stateHi, p.stateLo, key[0], key[1])
	p.step()
	return p
}

// mul128 computes hi*2^64+lo = a * b (128×128→128).
func mul128(aHi, aLo, bHi, bLo uint64) (hi, lo uint64) {
	h1 := aHi*bLo + aLo*bHi
	hi, lo = bits.Mul64(aLo, bLo)
	hi += h1
	return
}

func (p *PCG64) add(aHi, aLo, bHi, bLo uint64) {
	lo, carry := bits.Add64(aLo, bLo, 0)
	hi, _ := bits.Add64(aHi, bHi, carry)
	p.stateHi, p.stateLo = hi, lo
}

// step: pcg_setseq_128_step_r — state = state*multiplier + inc.
func (p *PCG64) step() {
	hi, lo := mul128(p.stateHi, p.stateLo, pcgMultHi, pcgMultLo)
	lo, carry := bits.Add64(lo, p.incLo, 0)
	hi, _ = bits.Add64(hi, p.incHi, carry)
	p.stateHi, p.stateLo = hi, lo
}

// Next64: pcg64_next64 — advance then XSL-RR output
// (pcg64.h:170, 186).
func (p *PCG64) Next64() uint64 {
	p.step()
	return bits.RotateLeft64(p.stateHi^p.stateLo, -int(p.stateHi>>58))
}

// Next32: pcg64_next32 — low 32 bits of the current draw, high 32 buffered
// (pcg64.h:386).
func (p *PCG64) Next32() uint32 {
	if p.hasUint32 {
		p.hasUint32 = false
		return p.uinteger
	}
	next := p.Next64()
	p.hasUint32 = true
	p.uinteger = uint32(next >> 32)
	return uint32(next)
}

// pcgAdvanceLcg128: pcg_advance_lcg_128 (pcg64.c:100) — fast exponentiation.
func pcgAdvanceLcg128(stateHi, stateLo, deltaHi, deltaLo, multHi, multLo, plusHi, plusLo uint64) (uint64, uint64) {
	accMultHi, accMultLo := uint64(0), uint64(1)
	accPlusHi, accPlusLo := uint64(0), uint64(0)
	for deltaHi != 0 || deltaLo != 0 {
		if deltaLo&1 != 0 {
			accMultHi, accMultLo = mul128(accMultHi, accMultLo, multHi, multLo)
			hi, lo := mul128(accPlusHi, accPlusLo, multHi, multLo)
			lo, c := bits.Add64(lo, plusLo, 0)
			accPlusHi, _ = bits.Add64(hi, plusHi, c)
			accPlusLo = lo
		}
		// curPlus = (curMult + 1) * curPlus; curMult *= curMult (old values)
		mlo, cm := bits.Add64(multLo, 1, 0)
		mhi, _ := bits.Add64(multHi, 0, cm)
		plusHi, plusLo = mul128(mhi, mlo, plusHi, plusLo)
		multHi, multLo = mul128(multHi, multLo, multHi, multLo)
		// delta >>= 1 (128-bit shift via 64-bit halves)
		deltaLo = deltaLo>>1 | deltaHi<<63
		deltaHi >>= 1
		_ = cm
	}
	hi, lo := mul128(accMultHi, accMultLo, stateHi, stateLo)
	lo2, c := bits.Add64(lo, accPlusLo, 0)
	hi2, _ := bits.Add64(hi, accPlusHi, c)
	return hi2, lo2
}

// Advance moves the generator state forward by delta draws without producing
// them (numpy PCG64.advance).
func (p *PCG64) Advance(delta uint64) {
	hi, lo := pcgAdvanceLcg128(p.stateHi, p.stateLo, 0, delta, pcgMultHi, pcgMultLo, p.incHi, p.incLo)
	p.stateHi, p.stateLo = hi, lo
}

// NextDouble: pcg64_next_double (numpy _pcg64.pyx:41) — uint64_to_double of
// the next 64-bit draw.
func (p *PCG64) NextDouble() float64 {
	return float64(p.Next64()>>11) * (1.0 / 9007199254740992.0)
}

// Jumped returns a new generator advanced by the golden-ratio jump constant
// (numpy PCG64.jumped: step = 0x9e3779b97f4a7c15f39cc0605cedc835).
func (p *PCG64) Jumped() BitGenerator {
	q := *p
	const (
		jumpHi = uint64(0x9e3779b97f4a7c15)
		jumpLo = uint64(0xf39cc0605cedc835)
	)
	hi, lo := pcgAdvanceLcg128(q.stateHi, q.stateLo, jumpHi, jumpLo, pcgMultHi, pcgMultLo, q.incHi, q.incLo)
	q.stateHi, q.stateLo = hi, lo
	q.hasUint32 = false
	q.uinteger = 0
	return &q
}
