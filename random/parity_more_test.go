package random

import (
	"testing"

	"github.com/aixfoundry/numgo"
)

// Parity of SeedSequence internals, spawn, advance/jump, and the
// sampling-based methods against numpy 2.5.1.

func TestSeedSequenceParity(t *testing.T) {
	ss := NewSeedSequence(12345)
	for i, v := range ss.pool {
		if uint64(v) != moreRef["pool"][i] {
			t.Fatalf("pool[%d]: %d != %d", i, v, moreRef["pool"][i])
		}
	}
	st := ss.GenerateState(8)
	for i, v := range st {
		if uint64(v) != moreRef["st32"][i] {
			t.Fatalf("st32[%d]: %d != %d", i, v, moreRef["st32"][i])
		}
	}
	kids := ss.Spawn(3)
	kid2 := kids[2].GenerateState64(2)
	for i := range kid2 {
		if kid2[i] != moreRef["kid2"][i] {
			t.Fatalf("kid2[%d]: %d != %d", i, kid2[i], moreRef["kid2"][i])
		}
	}
	gk := kids[0].Spawn(2)[1].GenerateState64(2)
	for i := range gk {
		if gk[i] != moreRef["gk1"][i] {
			t.Fatalf("gk1[%d]: %d != %d", i, gk[i], moreRef["gk1"][i])
		}
	}
	// entropy > 64 bits: numpy SeedSequence(2**64+5) → u32 words {5, 0, 1};
	// uint64 entropy words split low-first, so pass {5, 1} with the zero word
	// implied by the split of 5? No: numpy keeps the raw u32 words {5,0,1}.
	// Our uint64 API splits {5, 1} → u32 {5, 1}; provide exact u32 path:
	big := newSeedSequence([]uint64{5 | 1<<32}, nil, 4)
	// ^ {5,1} u32 words != numpy {5,0,1}; emulate via spawn-free assembled words
	big2 := newSeedSequenceU32([]uint32{5, 0, 1}, nil, 4).GenerateState64(2)
	if big2[0] != moreRef["big_st64"][0] || big2[1] != moreRef["big_st64"][1] {
		t.Fatalf("big_st64: %v != %v", big2, moreRef["big_st64"])
	}
	_ = big
}

func TestPCGAdvanceJumpParity(t *testing.T) {
	g := DefaultRng(12345)
	g.bg.Advance(1000003)
	raw := make([]uint64, 4)
	for i := range raw {
		raw[i] = g.bg.Next64()
	}
	for i := range raw {
		if raw[i] != moreRef["adv_raw"][i] {
			t.Fatalf("adv_raw[%d]: %d != %d", i, raw[i], moreRef["adv_raw"][i])
		}
	}
	g2 := DefaultRng(12345)
	j := g2.bg.Jumped()
	for i := range 4 {
		if v := j.Next64(); v != moreRef["jmp_raw"][i] {
			t.Fatalf("jmp_raw[%d]: %d != %d", i, v, moreRef["jmp_raw"][i])
		}
	}
}

func TestSamplingParity(t *testing.T) {
	g := DefaultRng(12345)
	p := []float64{0.1, 0.2, 0.05, 0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05}
	eqInts(t, "choice p", g.Choice(10, 100, true, p).Data(), moreRef["chpp100"])

	g = DefaultRng(12345)
	eqInts(t, "choice norep", g.Choice(50, 30, false, nil).Data(), moreRef["chnr100"])

	g = DefaultRng(12345)
	p60 := make([]float64, 60)
	for i := range p60 {
		p60[i] = 1.0 / 60
	}
	eqInts(t, "choice norep p", g.Choice(60, 25, false, p60).Data(), moreRef["chnrp50"])

	g = DefaultRng(12345)
	a := numgoArangeF(200)
	g.Shuffle(a)
	got := a.Data()
	for i := range got {
		if uint64(got[i]) != moreRef["shuf200"][i] {
			t.Fatalf("shuf200[%d]: %v != %v", i, got[i], float64(moreRef["shuf200"][i]))
		}
	}

	g = DefaultRng(12345)
	eqInts(t, "permutation", g.Permutation(200).Data(), moreRef["perm200"])
}

func numgoArangeF(n int) *arrayF {
	data := make([]float64, n)
	for i := range data {
		data[i] = float64(i)
	}
	return numgo.New1DArray(data)
}
