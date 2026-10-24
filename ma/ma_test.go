package ma

import (
	"math"
	"testing"

	"github.com/kaisawind/numgo"
)

// Golden tests against numpy 2.5.1 ma semantics (values generated with
// np.ma; see docs/design.md and numpy/ma/core.py references).

func approx(t *testing.T, name string, got []float64, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("%s[%d] = %g, want %g", name, i, got[i], want[i])
		}
	}
}

func checkMask(t *testing.T, name string, got []bool, want ...bool) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: mask length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s mask[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

// testArray mirrors the oracle array
// np.ma.array([[1,2,3],[4,5,6]], mask=[[0,1,0],[0,0,1]]):
// sum=13, mean=3.25, sum(axis=0)=[5,5,3], mean(axis=1)=[2,4.5].
func testArray() *Masked[float64] {
	return FromData(numgo.New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}}),
		[]bool{false, true, false, false, false, true})
}

func TestNewNomask(t *testing.T) {
	m := New(numgo.New2DArray([][]float64{{1, 2}, {3, 4}}))
	checkMask(t, "New", m.Mask(), false, false, false, false)
	if m.CountUnmasked() != 4 {
		t.Errorf("CountUnmasked = %d, want 4", m.CountUnmasked())
	}
}

func TestFromDataCopies(t *testing.T) {
	d := numgo.New1DArray([]float64{1, 2})
	mask := []bool{false, true}
	m := FromData(d, mask)
	d.Set([]int{0}, 99)
	mask[0] = true
	if m.Data().At(0) != 1 {
		t.Error("FromData must copy data")
	}
	if m.Mask()[0] {
		t.Error("FromData must copy mask")
	}
}

func TestSumMeanGolden(t *testing.T) {
	a := testArray()
	approx(t, "Sum", a.Sum().Data().Data(), []float64{13})
	approx(t, "Mean", a.Mean().Data().Data(), []float64{3.25})
	approx(t, "Sum0", a.Sum(0).Data().Data(), []float64{5, 5, 3})
	approx(t, "Mean1", a.Mean(1).Data().Data(), []float64{2, 4.5})
}

func TestSumKeepdims(t *testing.T) {
	a := testArray()
	k := a.SumKeepdims(0)
	if k.Data().Shape()[0] != 1 || k.Data().Shape()[1] != 3 {
		t.Fatalf("SumKeepdims(0) shape = %v", k.Data().Shape())
	}
	approx(t, "SumKeepdims0", k.Data().Data(), []float64{5, 5, 3})
	checkMask(t, "SumKeepdims0 mask", k.Mask(), false, false, false)
}

func TestSumAllMaskedRowNaN(t *testing.T) {
	// oracle: b.sum(axis=1) = [4.0, masked] (masked element reads as nan)
	b := FromData(numgo.New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}}),
		[]bool{false, true, false, true, true, true})
	s := b.Sum(1)
	checkMask(t, "Sum(1) all-masked row", s.Mask(), false, true)
	if !math.IsNaN(s.Data().At(1)) {
		t.Errorf("all-masked row sum = %g, want NaN", s.Data().At(1))
	}
	m := b.Mean(1)
	checkMask(t, "Mean(1) all-masked row", m.Mask(), false, true)
	if !math.IsNaN(m.Data().At(1)) {
		t.Errorf("all-masked row mean = %g, want NaN", m.Data().At(1))
	}
}

func TestStdVarGolden(t *testing.T) {
	// oracle: std()=1.479019945774904, var()=2.1875
	a := testArray()
	approx(t, "Std", a.Std().Data().Data(), []float64{1.479019945774904})
	approx(t, "Var", a.Var().Data().Data(), []float64{2.1875})
}

func TestDivZeroMasked(t *testing.T) {
	// oracle: true_divide(1, [0,2]) = [--, 0.5]
	one := New(numgo.Ones[float64](2))
	d := one.Div(New(numgo.New1DArray([]float64{0, 2})))
	if !math.IsInf(d.Data().At(0), 1) {
		t.Errorf("Div by zero data = %g, want +Inf (IEEE value stays)", d.Data().At(0))
	}
	approx(t, "Div data[1:]", d.Data().Data()[1:], []float64{0.5})
	checkMask(t, "Div mask", d.Mask(), true, false)
}

func TestSqrtNegativeMasked(t *testing.T) {
	// oracle: np.ma.sqrt([-1,4,9]) = [--, 2, 3]
	s := New(numgo.New1DArray([]float64{-1, 4, 9})).Sqrt()
	if !math.IsNaN(s.Data().At(0)) {
		t.Errorf("Sqrt(-1) data = %g, want NaN", s.Data().At(0))
	}
	approx(t, "Sqrt data[1:]", s.Data().Data()[1:], []float64{2, 3})
	checkMask(t, "Sqrt mask", s.Mask(), true, false, false)
}

func TestLogDomains(t *testing.T) {
	// oracle: np.ma.log([-1,1,e]) = [--, 0, 1]
	l := New(numgo.New1DArray([]float64{-1, 1, math.E})).Log()
	checkMask(t, "Log mask", l.Mask(), true, false, false)
	approx(t, "Log data[1:]", l.Data().Data()[1:], []float64{0, 1})
	// Log(0) stays -Inf unmasked (divide warning domain only).
	l0 := New(numgo.New1DArray([]float64{0, 1})).Log()
	checkMask(t, "Log0 mask", l0.Mask(), false, false)
	if !math.IsInf(l0.Data().At(0), -1) {
		t.Errorf("Log(0) = %g, want -Inf", l0.Data().At(0))
	}
	// domain per numpy ma.extras/domain rules: x < -1 masked.
	l1 := New(numgo.New1DArray([]float64{-2, -1, 0})).Log1p()
	checkMask(t, "Log1p mask", l1.Mask(), true, false, false)
}

func TestPassthroughMathPropagates(t *testing.T) {
	a := FromData(numgo.New1DArray([]float64{1, 2}), []bool{false, true})
	checkMask(t, "Exp", a.Exp().Mask(), false, true)
	checkMask(t, "Sin", a.Sin().Mask(), false, true)
	checkMask(t, "Cos", a.Cos().Mask(), false, true)
	checkMask(t, "Tan", a.Tan().Mask(), false, true)
	checkMask(t, "Floor", a.Floor().Mask(), false, true)
	checkMask(t, "Ceil", a.Ceil().Mask(), false, true)
	checkMask(t, "Round", a.Round().Mask(), false, true)
	// numpy computes on masked slots too (cosmetic data): Exp(2) present.
	approx(t, "Exp data", a.Exp().Data().Data(), []float64{math.E, math.Exp(2)})
}

func TestArithmeticMaskOR(t *testing.T) {
	a := FromData(numgo.New1DArray([]float64{1, 2, 3}), []bool{false, true, false})
	b := FromData(numgo.New1DArray([]float64{10, 20, 30}), []bool{false, false, true})
	// oracle: np.ma: mask ORs
	s := a.Add(b)
	approx(t, "Add", s.Data().Data(), []float64{11, 22, 33})
	checkMask(t, "Add mask", s.Mask(), false, true, true)
	approx(t, "Sub", a.Sub(b).Data().Data(), []float64{-9, -18, -27})
	checkMask(t, "Sub mask", a.Sub(b).Mask(), false, true, true)
	approx(t, "Mul", a.Mul(b).Data().Data(), []float64{10, 40, 90})
	// oracle: np.ma.power([2,3](mask=[0,1]), [2,2]) = [4, --]
	p := New(numgo.New1DArray([]float64{2, 3})).
		Pow(FromData(numgo.New1DArray([]float64{2, 2}), []bool{false, true}))
	approx(t, "Pow", p.Data().Data(), []float64{4, 9})
	checkMask(t, "Pow mask", p.Mask(), false, true)
	// oracle: -a mask [True False]
	n := FromData(numgo.New1DArray([]float64{1, 2}), []bool{true, false}).Neg()
	checkMask(t, "Neg mask", n.Mask(), true, false)
	approx(t, "Neg data", n.Data().Data(), []float64{-1, -2})
}

func TestValueOps(t *testing.T) {
	a := FromData(numgo.New1DArray([]float64{1, 2}), []bool{false, true})
	approx(t, "AddValue", a.AddValue(5).Data().Data(), []float64{6, 7})
	checkMask(t, "AddValue mask", a.AddValue(5).Mask(), false, true)
	approx(t, "SubValue", a.SubValue(1).Data().Data(), []float64{0, 1})
	approx(t, "MulValue", a.MulValue(2).Data().Data(), []float64{2, 4})
	d := a.DivValue(0)
	checkMask(t, "DivValue by zero masks", d.Mask(), true, true)
	// oracle: abs([--,2]) = [--,2]
	ab := FromData(numgo.New1DArray([]float64{-1, 2}), []bool{true, false}).Abs()
	approx(t, "Abs", ab.Data().Data(), []float64{1, 2})
	checkMask(t, "Abs mask", ab.Mask(), true, false)
}

func TestFilledAndDefault(t *testing.T) {
	// oracle: filled() = [1, 1e20, 3, 4]; filled(-7) = [1,-7,3,4]
	m := FromData(numgo.New1DArray([]float64{1, 2, 3, 4}), []bool{false, true, false, false})
	approx(t, "FilledDefault", m.FilledDefault().Data(), []float64{1, 1e20, 3, 4})
	approx(t, "Filled(-7)", m.Filled(-7).Data(), []float64{1, -7, 3, 4})
}

func TestCompressedOrder(t *testing.T) {
	// oracle: compressed([1,--,3,4]) = [1,3,4]
	m := FromData(numgo.New1DArray([]float64{1, 2, 3, 4}), []bool{false, true, false, false})
	approx(t, "Compressed", m.Compressed().Data(), []float64{1, 3, 4})
	empty := FromData(numgo.New1DArray([]float64{1}), []bool{true})
	if empty.Compressed().Size() != 0 {
		t.Error("all-masked compressed should be empty")
	}
}

func TestHardMaskSetRefused(t *testing.T) {
	// oracle: hard-masked [1,--]; hard[1]=99 leaves data/mask unchanged
	hard := FromData(numgo.New1DArray([]float64{1, 2}), []bool{false, true}).HardMask()
	hard.SetUnmasked([]int{1}, 99)
	if hard.Data().At(1) != 2 {
		t.Errorf("hard mask set: data = %g, want 2", hard.Data().At(1))
	}
	if !hard.Mask()[1] {
		t.Error("hard mask set: mask cleared")
	}
	// soft mask set succeeds and clears the bit
	soft := FromData(numgo.New1DArray([]float64{1, 2}), []bool{false, true}).SoftMask()
	soft.SetUnmasked([]int{1}, 9)
	if soft.Data().At(1) != 9 || soft.Mask()[1] {
		t.Error("soft mask set should write and unmask")
	}
}

func TestMaskedComparisons(t *testing.T) {
	a := numgo.New1DArray([]float64{1, 2, 3})
	checkMask(t, "MaskedEqual", MaskedEqual(a, 2).Mask(), false, true, false)
	checkMask(t, "MaskedNotEqual", MaskedNotEqual(a, 2).Mask(), true, false, true)
	checkMask(t, "MaskedLess", MaskedLess(a, 2).Mask(), true, false, false)
	checkMask(t, "MaskedLessEqual", MaskedLessEqual(a, 2).Mask(), true, true, false)
	checkMask(t, "MaskedGreater", MaskedGreater(a, 2).Mask(), false, false, true)
	checkMask(t, "MaskedGreaterEqual", MaskedGreaterEqual(a, 2).Mask(), false, true, true)
	checkMask(t, "MaskedInside", MaskedInside(numgo.New1DArray([]float64{1, 2, 3, 4}), 2, 3).Mask(),
		false, true, true, false)
	checkMask(t, "MaskedOutside", MaskedOutside(numgo.New1DArray([]float64{1, 2, 3, 4}), 2, 3).Mask(),
		true, false, false, true)
}

func TestMaskedWhereInvalid(t *testing.T) {
	checkMask(t, "MaskedWhere", MaskedWhere([]bool{true, false, true},
		numgo.New1DArray([]float64{1, 2, 3})).Mask(), true, false, true)
	checkMask(t, "MaskedInvalid float", MaskedInvalid(
		numgo.New1DArray([]float64{1, math.NaN(), math.Inf(1)})).Mask(), false, true, true)
	// oracle: complex NaN/Inf parts mask
	ci := MaskedInvalid(numgo.New1DArray([]complex128{
		1 + 2i, complex(math.NaN(), 0), complex(0, math.Inf(1))}))
	checkMask(t, "MaskedInvalid complex", ci.Mask(), false, true, true)
}

func TestConstructors(t *testing.T) {
	checkMask(t, "MaskedZeros", MaskedZeros[float64](2, 2).Mask(), false, false, false, false)
	checkMask(t, "MaskedOnes", MaskedOnes[float64](2, 3).Mask(),
		false, false, false, false, false, false)
	f := MaskedFull(7.0, 2, 2)
	if f.Data().At(1, 1) != 7 {
		t.Error("MaskedFull value")
	}
	checkMask(t, "MaskedFull", f.Mask(), false, false, false, false)
	ma := MaskedAll[float64](2, 2)
	checkMask(t, "MaskedAll", ma.Mask(), true, true, true, true)
}

func TestConcatenateMasks(t *testing.T) {
	// oracle: concat([1,--],[--,4]) = [1,--,--,4]
	c := Concatenate(0,
		FromData(numgo.New1DArray([]float64{1, 2}), []bool{false, true}),
		FromData(numgo.New1DArray([]float64{3, 4}), []bool{true, false}))
	approx(t, "Concat data", c.Data().Data(), []float64{1, 2, 3, 4})
	checkMask(t, "Concat mask", c.Mask(), false, true, true, false)
	// oracle: 2-D axis=1 masks merged row-wise
	c2 := Concatenate(1,
		FromData(numgo.New2DArray([][]float64{{1, 2}, {3, 4}}), []bool{false, true, true, false}),
		FromData(numgo.New2DArray([][]float64{{5, 6}, {7, 8}}), []bool{true, false, false, true}))
	checkMask(t, "Concat2d mask", c2.Mask(),
		false, true, true, false, true, false, false, true)
}

func TestStack(t *testing.T) {
	// oracle: stack axis=0 mask [[F,T],[T,F]]
	s := Stack(0,
		FromData(numgo.New1DArray([]float64{1, 2}), []bool{false, true}),
		FromData(numgo.New1DArray([]float64{3, 4}), []bool{true, false}))
	checkMask(t, "Stack0", s.Mask(), false, true, true, false)
	s1 := Stack(1,
		FromData(numgo.New1DArray([]float64{1, 2}), []bool{false, true}),
		FromData(numgo.New1DArray([]float64{3, 4}), []bool{true, false}))
	checkMask(t, "Stack1", s1.Mask(), false, true, true, false)
}

func TestTransposeReshapeSqueeze(t *testing.T) {
	m := FromData(numgo.New2DArray([][]float64{{1, 2}, {3, 4}}), []bool{false, true, true, false})
	// oracle: T mask [[F,T],[T,F]]
	tp := Transpose(m)
	checkMask(t, "T", tp.Mask(), false, true, true, false)
	// oracle: reshape(4) mask [F,T,T,F]
	checkMask(t, "reshape", Reshape(m, 4).Mask(), false, true, true, false)
	// oracle: squeeze([[1,2]]) mask [F,T]
	sq := Squeeze(FromData(numgo.New2DArray([][]float64{{1, 2}}), []bool{false, true}))
	checkMask(t, "squeeze", sq.Mask(), false, true)
}

func TestArgMaxMin(t *testing.T) {
	// oracle: x.argmax(axis=1) = [2,2], argmin(axis=0) = [0,1,0]
	x := testArray()
	if got, want := x.ArgMax(1), []int{2, 1}; !eqInts(got, want) {
		t.Errorf("ArgMax(1) = %v, want %v", got, want)
	}
	if got, want := x.ArgMin(0), []int{0, 1, 0}; !eqInts(got, want) {
		t.Errorf("ArgMin(0) = %v, want %v", got, want)
	}
	// all-masked channel → -1 (numpy returns a masked index; a channel with
	// unmasked entries still reports their coordinate)
	allm := FromData(numgo.New2DArray([][]float64{{1, 2}, {3, 4}}), []bool{true, false, true, false})
	if got, want := allm.ArgMax(0), []int{-1, 1}; !eqInts(got, want) {
		t.Errorf("ArgMax all-masked = %v, want %v", got, want)
	}
	pure := FromData(numgo.New1DArray([]float64{1, 2, 3}), []bool{true, true, true})
	if got := pure.ArgMax(0); !eqInts(got, []int{-1}) {
		t.Errorf("ArgMax fully-masked = %v, want [-1]", got)
	}
}

func TestAnyAll(t *testing.T) {
	// oracle: any()=False, all()=False for [1(masked),0]
	m := FromData(numgo.New1DArray([]float64{1, 0}), []bool{true, false})
	if m.Any() {
		t.Error("Any = true, want false (1 is masked)")
	}
	if m.All() {
		t.Error("All = true, want false")
	}
	// all-masked: Any=false, All=true (vacuous; documented deviation shape)
	allm := FromData(numgo.New1DArray([]float64{1, 0}), []bool{true, true})
	if m.Any() || !allm.All() {
		t.Error("all-masked: Any=false, All=true expected")
	}
}

func TestCumSum(t *testing.T) {
	// oracle: cumsum([1,--,3]) data = [1,1,4], mask = [F,T,F]
	c := FromData(numgo.New1DArray([]float64{1, 2, 3}), []bool{false, true, false}).CumSum()
	approx(t, "CumSum data", c.Data().Data(), []float64{1, 1, 4})
	checkMask(t, "CumSum mask", c.Mask(), false, true, false)
}

func TestBroadcastingOps(t *testing.T) {
	// 2x1 op 1x3 → 2x3 (P0-5 non-square)
	a := FromData(numgo.New2DArray([][]float64{{1}, {2}}), []bool{true, false})
	b := FromData(numgo.New2DArray([][]float64{{10, 20, 30}}), []bool{false, true, false})
	s := a.Add(b)
	if s.Data().Ndim() != 2 || s.Data().Shape()[0] != 2 || s.Data().Shape()[1] != 3 {
		t.Fatalf("broadcast shape = %v", s.Data().Shape())
	}
	checkMask(t, "broadcast mask", s.Mask(), true, true, true, false, true, false)
	approx(t, "broadcast data", s.Data().Data(), []float64{11, 21, 31, 12, 22, 32})
}

func TestFloat32(t *testing.T) {
	m := FromData(numgo.New1DArray([]float32{1, 2, 3}), []bool{false, true, false})
	// f4 default fill: 1e20
	f := m.FilledDefault()
	if got := f.At(1); math.Abs(float64(got)-1e20) > 1e14 {
		t.Errorf("f4 fill = %g, want ~1e20", got)
	}
	if got, want := m.Sum().Data().At(0), float32(4); got != want {
		t.Errorf("f32 sum = %g, want %g", got, want)
	}
}

func TestCopyIndependence(t *testing.T) {
	m := testArray()
	c := m.Copy()
	c.SetUnmasked([]int{0, 0}, 99)
	if m.Data().At(0, 0) == 99 {
		t.Error("Copy must be independent")
	}
}

func TestMaxMinGolden(t *testing.T) {
	// oracle: max=3, min=1 for [1,--,3]
	m := FromData(numgo.New1DArray([]float64{1, 2, 3}), []bool{false, true, false})
	approx(t, "Max", m.Max().Data().Data(), []float64{3})
	approx(t, "Min", m.Min().Data().Data(), []float64{1})
	// oracle: a.max(axis=0)=[4,5,3] (masked col 2 keeps 3); a.min(axis=1)=[1,4]
	a := testArray()
	approx(t, "Max0", a.Max(0).Data().Data(), []float64{4, 5, 3})
	approx(t, "Min1", a.Min(1).Data().Data(), []float64{1, 4})
}

func TestComplexMasked(t *testing.T) {
	// oracle: np.ma.sqrt([1+1j, -1+0j]) = [(1.09868+0.45509j), --] — complex
	// Sqrt uses the principal branch with no real-part domain.
	c := New(numgo.New1DArray([]complex128{1 + 1i, -1 + 0i})).Sqrt()
	approxNeed(t, "Sqrt complex", []complex128{c.Data().At(0), c.Data().At(1)},
		[]complex128{1.09868411346781 + 0.45508986056222733i, 6.123233995736766e-17 + 1i})
	checkMask(t, "Sqrt complex mask", c.Mask(), false, false)
	// oracle: np.ma.masked_invalid masks complex NaN parts
	iv := MaskedInvalid(numgo.New1DArray([]complex128{1, complex(math.NaN(), 1)}))
	checkMask(t, "complex MaskedInvalid", iv.Mask(), false, true)
	// complex Mul/Div mask ORs
	a := FromData(numgo.New1DArray([]complex128{1, 2}), []bool{false, true})
	checkMask(t, "complex Mul", a.Mul(New(numgo.New1DArray([]complex128{3, 4}))).Mask(),
		false, true)
}

// approxNeed compares complex values elementwise.
func approxNeed(t *testing.T, name string, got, want []complex128) {
	t.Helper()
	for i := range got {
		if math.Abs(real(got[i])-real(want[i])) > 1e-12 ||
			math.Abs(imag(got[i])-imag(want[i])) > 1e-12 {
			t.Errorf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func eqInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
