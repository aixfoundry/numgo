package random

import (
	"math"
	"testing"
)

func eqBits32(t *testing.T, name string, got []float32, want []uint64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != uint32(want[i]) {
			t.Fatalf("%s[%d]: got %x want %x", name, i, math.Float32bits(got[i]), uint32(want[i]))
		}
	}
}

func TestMT19937Parity(t *testing.T) {
	m := NewMT19937(NewSeedSequence(12345))
	for i := range 16 {
		if got := m.Next32(); got != uint32(mtRef["mt_raw32"][i]) {
			t.Fatalf("raw32[%d]: %x != %x", i, got, mtRef["mt_raw32"][i])
		}
	}
	m = NewMT19937(NewSeedSequence(12345))
	for i := range 8 {
		if got := m.Next64(); got != mtRef["mt_raw64"][i] {
			t.Fatalf("raw64[%d]: %x != %x", i, got, mtRef["mt_raw64"][i])
		}
	}
	j := NewMT19937(NewSeedSequence(12345)).Jumped().(*MT19937)
	for i := range 8 {
		if got := j.Next32(); got != uint32(mtRef["mt_jump_raw"][i]) {
			t.Fatalf("jump[%d]: %x != %x", i, got, mtRef["mt_jump_raw"][i])
		}
	}
}

func TestMT19937LegacySeeding(t *testing.T) {
	// RandomState(12345): mt19937_seed path
	l := NewMT19937LegacySeeding(12345, nil)
	for i := range 624 {
		if l.key[i] != uint32(mtRef["seed12345_key"][i]) {
			t.Fatalf("seed12345 key[%d]: %x != %x", i, l.key[i], mtRef["seed12345_key"][i])
		}
	}
	if l.pos != mtN {
		t.Fatalf("seed12345 pos: %d", l.pos)
	}
	// RandomState([1,2,3]): init_by_array path
	la := mt19937InitByArray([]uint32{1, 2, 3})
	for i := range mtN {
		if la.key[i] != uint32(mtRef["array123_key"][i]) {
			t.Fatalf("array123 key[%d]: %x != %x", i, la.key[i], mtRef["array123_key"][i])
		}
	}
}

func TestMTGeneratorParity(t *testing.T) {
	// Generator over MT19937: the Generator's double path uses next_uint64,
	// which is (a<<32)|b over two tempered draws.
	g := &Generator{bg: NewMT19937(NewSeedSequence(12345)), ss: NewSeedSequence(12345)}
	eqBits(t, "mt random", g.Random(16).Data(), mtRef["mt_rnd"])
	g = &Generator{bg: NewMT19937(NewSeedSequence(12345)), ss: NewSeedSequence(12345)}
	eqBits(t, "mt standard_normal", g.StandardNormal(16).Data(), mtRef["mt_sn"])
	g = &Generator{bg: NewMT19937(NewSeedSequence(12345)), ss: NewSeedSequence(12345)}
	eqInts(t, "mt integers", g.Integers(0, 1000000, 16).Data(), mtRef["mt_ints"])
}

func TestFloat32ParityPCG(t *testing.T) {
	g := DefaultRng(12345)
	eqBits32(t, "f32 random", g.Random32(16).Data(), mtRef["f32_random"])
	g = DefaultRng(12345)
	eqBits32(t, "f32 standard_normal", g.StandardNormal32(16).Data(), mtRef["f32_sn"])
	g = DefaultRng(12345)
	eqBits32(t, "f32 standard_exponential", g.StandardExponential32(16).Data(), mtRef["f32_se"])
	g = DefaultRng(12345)
	eqBits32(t, "f32 standard_gamma>1", g.StandardGamma32(5.5, 16).Data(), mtRef["f32_gamma"])
	g = DefaultRng(12345)
	eqBits32(t, "f32 standard_gamma<1", g.StandardGamma32(0.7, 16).Data(), mtRef["f32_gamma_lo"])
}

func TestFloat32ParityMT(t *testing.T) {
	mk := func() *Generator {
		ss := NewSeedSequence(12345)
		return &Generator{bg: NewMT19937(ss), ss: ss}
	}
	g := mk()
	eqBits32(t, "mt f32 random", g.Random32(16).Data(), mtRef["mt_f32_random"])
	g = mk()
	eqBits32(t, "mt f32 standard_normal", g.StandardNormal32(16).Data(), mtRef["mt_f32_sn"])
	g = mk()
	eqBits32(t, "mt f32 standard_exponential", g.StandardExponential32(16).Data(), mtRef["mt_f32_se"])
	g = mk()
	eqBits32(t, "mt f32 gamma>1", g.StandardGamma32(5.5, 16).Data(), mtRef["mt_f32_gamma"])
	g = mk()
	eqBits32(t, "mt f32 gamma<1", g.StandardGamma32(0.7, 16).Data(), mtRef["mt_f32_gamma_lo"])
}

// Stress parity: 800-sample float32 batches exercise the ziggurat tails and
// gamma rejection paths; int32 covers the Lemire-on-32-bit integers path.
func TestFloat32StressParity(t *testing.T) {
	g := DefaultRng(12345)
	eqBits32(t, "f32 random800", g.Random32(800).Data(), mtRef2["f32_random800"])
	g = DefaultRng(12345)
	eqBits32(t, "f32 sn800", g.StandardNormal32(800).Data(), mtRef2["f32_sn800"])
	g = DefaultRng(12345)
	eqBits32(t, "f32 se800", g.StandardExponential32(800).Data(), mtRef2["f32_se800"])
	g = DefaultRng(12345)
	eqBits32(t, "f32 gamma800", g.StandardGamma32(5.5, 800).Data(), mtRef2["f32_gamma800"])
	g = DefaultRng(12345)
	eqBits32(t, "f32 gammaL800", g.StandardGamma32(0.5, 800).Data(), mtRef2["f32_gammaL800"])
}

func TestFloat32MTStressParity(t *testing.T) {
	mk := func() *Generator {
		ss := NewSeedSequence(12345)
		return &Generator{bg: NewMT19937(ss), ss: ss}
	}
	g := mk()
	eqBits32(t, "mt f32 random800", g.Random32(800).Data(), mtRef2["mt_f32_random800"])
	g = mk()
	eqBits32(t, "mt f32 sn800", g.StandardNormal32(800).Data(), mtRef2["mt_f32_sn800"])
	g = mk()
	eqBits32(t, "mt f32 se800", g.StandardExponential32(800).Data(), mtRef2["mt_f32_se800"])
	g = mk()
	eqBits32(t, "mt f32 gamma800", g.StandardGamma32(5.5, 800).Data(), mtRef2["mt_f32_gamma800"])
	g = mk()
	eqBits32(t, "mt f32 gammaL800", g.StandardGamma32(0.5, 800).Data(), mtRef2["mt_f32_gammaL800"])
}

func TestIntegers32Parity(t *testing.T) {
	g := DefaultRng(12345)
	eqInts(t, "int32_100", g.Integers32(0, 1000000, 100).Data(), mtRef2["int32_100"])
	g = DefaultRng(12345)
	eqInts(t, "int32_rng100", g.Integers32(-5, 1000000, 100).Data(), mtRef2["int32_rng100"])
	ss := NewSeedSequence(12345)
	g = &Generator{bg: NewMT19937(ss), ss: ss}
	eqInts(t, "mt int32_100", g.Integers32(0, 1000000, 100).Data(), mtRef2["mt_int32_100"])
}

func TestMT19937RngConvenience(t *testing.T) {
	g := MT19937Rng(12345)
	if _, ok := g.bg.(*MT19937); !ok {
		t.Fatal("MT19937Rng did not build an MT19937 engine")
	}
	eqBits(t, "MT19937Rng random", g.Random(8).Data(), mtRef["mt_rnd"][:8])
	g2 := NewGeneratorMT(NewSeedSequence(12345))
	eqBits32(t, "NewGeneratorMT f32", g2.Random32(16).Data(), mtRef["mt_f32_random"])
}
