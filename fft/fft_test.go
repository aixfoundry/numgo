package fft

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/aixfoundry/numgo"
)

const tol = 1e-12

// expectC asserts element-wise closeness of two complex slices.
func expectC(t *testing.T, name string, got, want []complex128) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(real(got[i])-real(want[i])) > tol || math.Abs(imag(got[i])-imag(want[i])) > tol {
			t.Fatalf("%s: at %d got %v, want %v", name, i, got[i], want[i])
		}
	}
}

// expectR asserts element-wise closeness of two real slices.
func expectR(t *testing.T, name string, got, want []float64) {
	t.Helper()
	cc := make([]complex128, len(want))
	for i, v := range want {
		cc[i] = complex(v, 0)
	}
	expectC(t, name, realToC(got), cc)
}

func realToC(v []float64) []complex128 {
	out := make([]complex128, len(v))
	for i, x := range v {
		out[i] = complex(x, 0)
	}
	return out
}

// flat reads a C-order flat copy of the array.
func flat[T numgo.Number](a *numgo.Array[T]) []T {
	return a.Copy().Data()[:a.Size()]
}

var a23 = numgo.New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}})

// numpy: np.fft.fft(np.array([[1,2,3],[4,5,6]]))
var fft23Backward = [][]complex128{
	{6, cmplx.Conj(complex(-1.5, -0.8660254037844386)), complex(-1.5, -0.8660254037844386)},
	{15, cmplx.Conj(complex(-1.5, -0.8660254037844386)), complex(-1.5, -0.8660254037844386)},
}

// numpy: np.fft.fft(a, norm='ortho')
var fft23Ortho = [][]complex128{
	{3.4641016151377553 + 0i, -0.8660254037844388 + 0.5i, -0.8660254037844388 - 0.5i},
	{8.660254037844387 + 0i, -0.8660254037844388 + 0.5i, -0.8660254037844388 - 0.5i},
}

// numpy: np.fft.fft(a, norm='forward')
var fft23Forward = [][]complex128{
	{2 + 0i, -0.5 + 0.28867513459481287i, -0.5 - 0.28867513459481287i},
	{5 + 0i, -0.5 + 0.28867513459481287i, -0.5 - 0.28867513459481287i},
}

func flat2D(m [][]complex128) []complex128 {
	var out []complex128
	for _, row := range m {
		out = append(out, row...)
	}
	return out
}

func TestFFTNormsRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name string
		norm Norm
		want []complex128
	}{
		{"backward", Backward, flat2D(fft23Backward)},
		{"ortho", Ortho, flat2D(fft23Ortho)},
		{"forward", Forward, flat2D(fft23Forward)},
	} {
		f := FFT(a23, 0, tc.norm)
		expectC(t, "FFT "+tc.name, flat(f), tc.want)
		inv := IFFT(f, 0, tc.norm)
		expectC(t, "IFFT round-trip "+tc.name, flat(inv), []complex128{1, 2, 3, 4, 5, 6})
	}
}

func TestFFTPadTruncate(t *testing.T) {
	x := numgo.New1DArray([]float64{1, 2, 3, 4, 5})
	// numpy: np.fft.fft(x, 8)
	pad := FFT(x, 8, Backward)
	expectC(t, "pad8", flat(pad), []complex128{
		15, complex(-5.414213562373095, -7.242640687119286), complex(3, 2),
		complex(-2.585786437626905, -1.2426406871192857), 3,
		complex(-2.585786437626905, 1.2426406871192857), complex(3, -2),
		complex(-5.414213562373095, 7.242640687119286),
	})
	// ortho factor is 1/sqrt(8): padded length feeds the norm.
	ortho := FFT(x, 8, Ortho)
	expectC(t, "pad8 ortho", flat(ortho), []complex128{
		5.303300858899106, complex(-1.914213562373095, -2.5606601717798214),
		complex(1.0606601717798212, 0.7071067811865475),
		complex(-0.9142135623730951, -0.4393398282201786), 1.0606601717798212,
		complex(-0.9142135623730951, 0.4393398282201786),
		complex(1.0606601717798212, -0.7071067811865475),
		complex(-1.914213562373095, 2.5606601717798214),
	})
	// numpy: np.fft.fft(x, 3)
	trunc := FFT(x, 3, Backward)
	expectC(t, "trunc3", flat(trunc), []complex128{6, complex(-1.5, 0.8660254037844386), complex(-1.5, -0.8660254037844386)})
}

func TestFFTArbitraryN(t *testing.T) {
	// numpy n=12 and n=15 (mixed radix, first four bins).
	x12 := numgo.New1DArray([]float64{1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5})
	got12 := FFT(x12, 0, Backward).Data()[:4]
	expectC(t, "n=12", got12, []complex128{
		45, complex(-3, 11.196152422706632), complex(-3, 5.196152422706632), complex(-3, 3)})
	x15 := numgo.New1DArray([]float64{1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8})
	got15 := FFT(x15, 0, Backward).Data()[:4]
	expectC(t, "n=15", got15, []complex128{
		67.5, complex(-3.75, 17.64236291054420), complex(-3.75, 8.422637902140810), complex(-3.75, 5.161432201766901)})
}

func TestRFFTIrfftPair(t *testing.T) {
	a := numgo.New2DArray([][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}})
	r := RFFT(a, 0, Backward)
	// numpy: np.fft.rfft(a); last (Nyquist) bin included.
	expectC(t, "rfft 2x4", flat(r), []complex128{
		10, complex(-2, 2), -2, 26, complex(-2, 2), -2})
	ir := IRFFT(r, 0, Backward)
	expectR(t, "irfft round-trip", flat(ir), []float64{1, 2, 3, 4, 5, 6, 7, 8})
	if got := ir.Shape(); got[0] != 2 || got[1] != 4 {
		t.Fatalf("irfft shape %v, want [2 4]", got)
	}
	// ortho pair.
	ro := RFFT(a, 0, Ortho)
	expectC(t, "rfft ortho", flat(ro), []complex128{
		5, complex(-1, 1), -1, 13, complex(-1, 1), -1})
	expectR(t, "irfft ortho round-trip", flat(IRFFT(ro, 0, Ortho)), []float64{1, 2, 3, 4, 5, 6, 7, 8})
	// forward pair on 1-D with explicit odd/padded n.
	h := numgo.New1DArray([]complex128{10, complex(-2, 2), -2})
	expectR(t, "irfft n=6", flat(IRFFT(h, 6, Backward)), []float64{
		0.33333333333333315, 1.0893163974770408, 1.7559830641437078,
		1.6666666666666665, 2.9106836025229588, 2.2440169358562922})
	expectR(t, "irfft n=2", flat(IRFFT(h, 2, Backward)), []float64{4, 6})
}

func TestHfftIhfftPair(t *testing.T) {
	s := numgo.New1DArray([]complex128{15, -4, 0, -1, 0, -4})
	h := HFFT(s, 0, Backward)
	// numpy: np.fft.hfft(s) — default output length 2*(m-1) = 10.
	expectR(t, "hfft", flat(h), []float64{
		1, 13.145898033750315, 10.145898033750315, 19.854101966249687,
		16.854101966249687, 29, 16.854101966249687, 19.854101966249687,
		10.145898033750315, 13.145898033750315})
	ih := IHFFT(h, 0, Backward)
	expectC(t, "ihfft(hfft(s)) round-trip", flat(ih), []complex128{
		15, -4, 0, -1, 0, -4})
	// Full pair with explicit lengths, both norms.
	xr := numgo.New1DArray([]float64{1, 2, 3, 4, 3, 2})
	spec := IHFFT(xr, 0, Backward)
	expectC(t, "ihfft(xr)", flat(spec), []complex128{
		2.5, complex(-2.0/3, 0), 0, complex(-1.0/6, 0)})
	expectR(t, "hfft(ihfft(xr), 6)", flat(HFFT(spec, 6, Backward)), []float64{1, 2, 3, 4, 3, 2})
	// numpy ortho: ihfft(xr, norm='ortho') == rfft(xr, norm='forward') conj'd.
	so := IHFFT(xr, 0, Ortho)
	expectC(t, "ihfft ortho", flat(so), []complex128{
		6.1237243569579452, -1.6329931618554523, 0, -0.40824829046386307})
	expectR(t, "hfft(ihfft(xr), 6, ortho)", flat(HFFT(so, 6, Ortho)),
		[]float64{1, 2, 3, 4, 3, 2})
}

func TestFFT2AndFFTN(t *testing.T) {
	// numpy: np.fft.fft2(np.arange(12).reshape(3,4))
	f2 := FFT2(numgo.New2DArray([][]float64{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}}), nil, Backward)
	expectC(t, "fft2 3x4", flat(f2), []complex128{
		66, complex(-6, 6), -6, complex(-6, -6),
		complex(-24, 13.856406460551018), 0, 0, 0,
		complex(-24, -13.856406460551018), 0, 0, 0})
	// numpy: np.fft.fftn(np.arange(24).reshape(2,3,4))
	a3 := numgo.FromSlice(make([]float64, 24), 2, 3, 4)
	for i := range a3.Size() {
		a3.Set([]int{i / 12, (i % 12) / 4, i % 4}, float64(i))
	}
	fn := FFTN(a3, nil, Backward)
	expectC(t, "fftn 2x3x4 all axes", flat(fn), []complex128{
		276, complex(-12, 12), -12, complex(-12, -12),
		complex(-48, 27.712812921102035), 0, 0, 0,
		complex(-48, -27.712812921102035), 0, 0, 0,
		-144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	// numpy: np.fft.fftn(a3, axes=[0,2]) — explicit axis subset.
	fexp := FFTN(a3, nil, Backward, 0, 2)
	expectC(t, "fftn axes [0,2]", flat(fexp), []complex128{
		60, complex(-4, 4), -4, complex(-4, -4),
		92, complex(-4, 4), -4, complex(-4, -4),
		124, complex(-4, 4), -4, complex(-4, -4),
		-48, 0, 0, 0, -48, 0, 0, 0, -48, 0, 0, 0})
}

func TestRFFTN(t *testing.T) {
	a := numgo.New2DArray([][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}})
	// numpy: np.fft.rfft2(a) — output (2, n//2+1) = (2, 3).
	r2 := RFFT2(a, nil, Backward)
	expectC(t, "rfft2 2x4", flat(r2), []complex128{
		36, complex(-4, 4), -4, complex(-16, 0), 0, 0})
	back := IRFFT2(r2, nil, Backward)
	expectR(t, "irfft2 round-trip", flat(back), []float64{1, 2, 3, 4, 5, 6, 7, 8})
	// RFFTN on 2x3x4 with default axes: last axis half.
	a3 := numgo.FromSlice(make([]float64, 24), 2, 3, 4)
	for i := range a3.Size() {
		a3.Set([]int{i / 12, (i % 12) / 4, i % 4}, float64(i))
	}
	rn := RFFTN(a3, nil, Backward)
	if got := rn.Shape(); got[2] != 3 {
		t.Fatalf("rfftn last axis %d, want 3", got[2])
	}
	expectC(t, "rfftn plane 0", flat(numgo.AsArray(rn.Copy().Data()[:9], 3, 3)), []complex128{
		276, complex(-12, 12), -12,
		complex(-48, 27.712812921102035), 0, 0,
		complex(-48, -27.712812921102035), 0, 0})
	expectR(t, "irfftn round-trip", flat(IRFFTN(rn, nil, Backward)), flat(a3))
}

func TestFFTFreq(t *testing.T) {
	expectR(t, "fftfreq 6", flat(FFTFreq(6, 1)), []float64{0, 1.0 / 6, 1.0 / 3, -0.5, -1.0 / 3, -1.0 / 6})
	expectR(t, "fftfreq 5 d=0.1", flat(FFTFreq(5, 0.1)), []float64{0, 2, 4, -4, -2})
	expectR(t, "rfftfreq 6", flat(RFFTFreq(6, 1)), []float64{0, 1.0 / 6, 1.0 / 3, 0.5})
	expectR(t, "rfftfreq 5 d=2", flat(RFFTFreq(5, 2)), []float64{0, 0.1, 0.2})
}

func TestShift(t *testing.T) {
	x5 := numgo.New1DArray([]float64{0, 1, 2, 3, 4})
	expectR(t, "fftshift 5", flat(FFTShift(x5)), []float64{3, 4, 0, 1, 2})
	// numpy ifftshift(5): roll by -(5//2) = -2 → [2,3,4,0,1]
	shifted := FFTShift(x5)
	expectR(t, "ifftshift 5", flat(IFFTShift(shifted)), []float64{0, 1, 2, 3, 4})
	expectR(t, "ifftshift direct 5", flat(IFFTShift(x5)), []float64{2, 3, 4, 0, 1})
	x6 := numgo.New1DArray([]float64{0, 1, 2, 3, 4, 5})
	expectR(t, "fftshift 6", flat(FFTShift(x6)), []float64{3, 4, 5, 0, 1, 2})
	expectR(t, "ifftshift 6", flat(IFFTShift(x6)), []float64{3, 4, 5, 0, 1, 2})
	m := numgo.New2DArray([][]float64{{0, 1, 2}, {3, 4, 5}})
	got := FFTShift(m, 0)
	expectR(t, "fftshift 2d axis 0", flat(got), []float64{3, 4, 5, 0, 1, 2})
	// Pure copies: mutating the shift result must not touch the input.
	got.Set([]int{0, 0}, 99)
	if m.At(0, 0) != 0 {
		t.Fatal("FFTShift returned a view, want a copy")
	}
}

func TestComplexAndFloat32Input(t *testing.T) {
	z := numgo.New2DArray([][]complex128{
		{1 + 2i, 3 - 1i, 0.5i}, {2, -1i, 4 + 4i}})
	got := FFT(z, 0, Backward)
	expectC(t, "fft complex128", flat(got), []complex128{
		complex(4, 1.5), complex(-1.7990381056766580, -0.34807621135331601),
		complex(0.79903810567665801, 4.8480762113533160),
		complex(6, 3), complex(-4.3301270189221928, 1.9641016151377544),
		complex(4.3301270189221928, -4.9641016151377544)})
	expectC(t, "ifft round-trip", flat(IFFT(got, 0, Backward)),
		[]complex128{1 + 2i, 3 - 1i, 0.5i, 2, -1i, 4 + 4i})

	a32 := numgo.New2DArray([][]float32{{1, 2, 3}, {4, 5, 6}})
	got32 := FFT(a32, 0, Backward)
	expectC(t, "fft float32", flat(got32), []complex128{
		6, complex(-1.5, 0.866025403784439), complex(-1.5, -0.866025403784439),
		15, complex(-1.5, 0.866025403784439), complex(-1.5, -0.866025403784439)})
}

func TestZeroSizeAxis(t *testing.T) {
	// numpy returns empty output for an empty batch axis...
	e := numgo.Zeros[float64](0, 3)
	if got := FFT(e, 0, Backward); got.Size() != 0 || got.Shape()[1] != 3 {
		t.Fatalf("fft empty batch: shape %v", got.Shape())
	}
	// ...but raises Invalid number of FFT data points when the transformed
	// axis is empty and n does not supply a length.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("fft on size-0 transform axis should panic")
			} else if r != "numgo/fft: Invalid number of FFT data points (0) specified." {
				t.Fatalf("unexpected panic %q", r)
			}
		}()
		FFT(numgo.Zeros[float64](2, 0), 0, Backward)
	}()
	// Explicit n rescues an empty transform axis with an all-zero (padded)
	// result, matching numpy.
	got := FFT(numgo.Zeros[float64](2, 0), 4, Backward)
	if s := got.Shape(); s[0] != 2 || s[1] != 4 {
		t.Fatalf("fft (2,0) n=4: shape %v", s)
	}
}

func TestBadAxisPanic(t *testing.T) {
	defer func() {
		r := recover()
		if r != "numgo/fft: axis 2 is out of bounds for 2-d array" {
			t.Fatalf("bad axis panic %q", r)
		}
	}()
	FFT(a23, 0, Backward, 2)
}

func TestIRFFTRealDtype(t *testing.T) {
	h := numgo.New1DArray([]complex128{1, -1i, -1})
	r := IRFFT(h, 0, Backward)
	if got := r.Shape(); len(got) != 1 || got[0] != 4 {
		t.Fatalf("irfft shape %v", got)
	}
	// Real by construction: values are float64, no imaginary part can hide.
	// numpy: np.fft.irfft([1, -1j, -1]) == [0, 1, 0, 0].
	expectR(t, "irfft values", flat(r), []float64{0, 1, 0, 0})
}
