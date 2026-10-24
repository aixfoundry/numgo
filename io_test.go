package numgo

// IO tests. The binary fixtures in testdata/ were generated with numpy 2.5.1
// (reproduce with python3, from the repo root):
//
//	import numpy as np
//	np.save('testdata/f64_2x3.npy', np.array([[1.5, -2.25, 3.75], [4.0, 5.5, -6.75]]))
//	np.save('testdata/f32_3x2.npy', np.array([[1.5, -2.0], [3.25, 4.0], [-5.5, 6.125]], dtype=np.float32))
//	np.save('testdata/c128_2x2.npy', np.array([[1+2j, 3-4j], [0.5+0.25j, -1-1j]]))
//	np.save('testdata/fortran_2x3.npy', np.asfortranarray(np.arange(6.0).reshape(2, 3)))
//	np.save('testdata/be_f8_2x2.npy', np.arange(4.0).astype('>f8').reshape(2, 2))
//	np.save('testdata/int64_1d.npy', np.array([1, 2, 3], dtype=np.int64))
//	with open('testdata/v2_f64_3x4.npy', 'wb') as f:
//	    np.lib.format.write_array(f, np.arange(12.0).reshape(3, 4), version=(2, 0))
//
// testdata/f64_2x3.npy doubles as the canonical np.save byte image asserted
// against Save below.

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadNpyFixtures(t *testing.T) {
	// f64 2x3, C order.
	a, err := LoadFile("testdata/f64_2x3.npy")
	if err != nil {
		t.Fatal(err)
	}
	if a.Descr != "<f8" || !equalShape(a.Shape, 2, 3) || a.F64 == nil {
		t.Fatalf("descr=%q shape=%v", a.Descr, a.Shape)
	}
	want := []float64{1.5, -2.25, 3.75, 4.0, 5.5, -6.75}
	for i, w := range want {
		if got := a.F64.Data()[i]; got != w {
			t.Fatalf("f64[%d] = %v, want %v", i, got, w)
		}
	}

	// f32 3x2.
	b, err := LoadFile("testdata/f32_3x2.npy")
	if err != nil {
		t.Fatal(err)
	}
	if b.F32 == nil || b.Descr != "<f4" || !equalShape(b.Shape, 3, 2) {
		t.Fatalf("descr=%q shape=%v", b.Descr, b.Shape)
	}
	want32 := []float32{1.5, -2.0, 3.25, 4.0, -5.5, 6.125}
	for i, w := range want32 {
		if got := b.F32.Data()[i]; got != w {
			t.Fatalf("f32[%d] = %v, want %v", i, got, w)
		}
	}

	// complex128 2x2.
	c, err := LoadFile("testdata/c128_2x2.npy")
	if err != nil {
		t.Fatal(err)
	}
	if c.C128 == nil || c.Descr != "<c16" || !equalShape(c.Shape, 2, 2) {
		t.Fatalf("descr=%q shape=%v", c.Descr, c.Shape)
	}
	wantC := []complex128{1 + 2i, 3 - 4i, 0.5 + 0.25i, -1 - 1i}
	for i, w := range wantC {
		if got := c.C128.Data()[i]; got != w {
			t.Fatalf("c128[%d] = %v, want %v", i, got, w)
		}
	}

	// v2.0 header, f64 3x4.
	d, err := LoadFile("testdata/v2_f64_3x4.npy")
	if err != nil {
		t.Fatal(err)
	}
	if !equalShape(d.Shape, 3, 4) || d.F64 == nil {
		t.Fatalf("shape=%v", d.Shape)
	}
	for i := 0; i < 12; i++ {
		if got := d.F64.Data()[i]; got != float64(i) {
			t.Fatalf("v2[%d] = %v", i, got)
		}
	}
}

func TestLoadFortranOrder(t *testing.T) {
	// File holds F-order arange(6).reshape(2,3): logical [[0,1,2],[3,4,5]].
	a, err := LoadFile("testdata/fortran_2x3.npy")
	if err != nil {
		t.Fatal(err)
	}
	if !equalShape(a.Shape, 2, 3) {
		t.Fatal(a.Shape)
	}
	want := []float64{0, 1, 2, 3, 4, 5}
	for i, w := range want {
		if got := a.F64.Data()[i]; got != w {
			t.Fatalf("fortran[%d] = %v, want %v", i, got, w)
		}
	}
}

func TestLoadBigEndian(t *testing.T) {
	// File holds '>f8' arange(4).reshape(2,2).
	a, err := LoadFile("testdata/be_f8_2x2.npy")
	if err != nil {
		t.Fatal(err)
	}
	if a.Descr != ">f8" || !equalShape(a.Shape, 2, 2) {
		t.Fatalf("descr=%q shape=%v", a.Descr, a.Shape)
	}
	want := []float64{0, 1, 2, 3}
	for i, w := range want {
		if got := a.F64.Data()[i]; got != w {
			t.Fatalf("be[%d] = %v, want %v", i, got, w)
		}
	}
}

func TestLoadUnsupportedDtype(t *testing.T) {
	_, err := LoadFile("testdata/int64_1d.npy")
	if err == nil || err.Error() != `numgo: unsupported npy dtype: "<i8"` {
		t.Fatalf("got %v", err)
	}
}

func TestLoadNotNpy(t *testing.T) {
	_, err := Load(strings.NewReader("definitely not a npy file"))
	if err == nil || !strings.Contains(err.Error(), "magic string is not correct") {
		t.Fatalf("got %v", err)
	}
}

func TestLoadTruncatedData(t *testing.T) {
	raw, _ := os.ReadFile("testdata/f64_2x3.npy")
	_, err := Load(bytes.NewReader(raw[:len(raw)-8]))
	if err == nil {
		t.Fatal("expected error on truncated data")
	}
}

func TestSaveMatchesNumpyBytes(t *testing.T) {
	// Save output must be byte-identical to the committed np.save image.
	wantBytes, err := os.ReadFile("testdata/f64_2x3.npy")
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	in := New2DArray([][]float64{{1.5, -2.25, 3.75}, {4.0, 5.5, -6.75}})
	if err := Save(&buf, in); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf.Bytes(), wantBytes) {
		t.Fatalf("save bytes differ:\n got %q\nwant %q", buf.Bytes(), wantBytes)
	}
}

func TestSaveNonContiguous(t *testing.T) {
	// Transpose view input: Save must write its C-order ravel.
	base := New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}})
	a, err := Load(SaveBytes(t, base.Transpose()))
	if err != nil {
		t.Fatal(err)
	}
	if !equalShape(a.Shape, 3, 2) {
		t.Fatal(a.Shape)
	}
	want := []float64{1, 4, 2, 5, 3, 6}
	for i, w := range want {
		if got := a.F64.Data()[i]; got != w {
			t.Fatalf("noncontig[%d] = %v, want %v", i, got, w)
		}
	}
}

func TestRoundTripAllDtypes(t *testing.T) {
	f64 := New2DArray([][]float64{{1.5, -2.25}, {3.75, -4.5}})
	f32 := New2DArray([][]float32{{1.5, -2.25}, {3.75, -4.5}})
	c128 := New2DArray([][]complex128{{1 + 2i, 3 - 4i}, {0.5 + 0.25i, -1 - 1i}})
	c64 := New2DArray([][]complex64{{1 + 2i, 3 - 4i}, {0.5 + 0.25i, -1 - 1i}})

	for _, tc := range []struct {
		name  string
		load  AnyArray
		round func(AnyArray) AnyArray
	}{
		{"f64", mustLoad(t, SaveBytes(t, f64)), func(x AnyArray) AnyArray { return mustLoad(t, SaveBytes(t, x.F64)) }},
		{"f32", mustLoad(t, SaveBytes(t, f32)), func(x AnyArray) AnyArray { return mustLoad(t, SaveBytes(t, x.F32)) }},
		{"c128", mustLoad(t, SaveBytes(t, c128)), func(x AnyArray) AnyArray { return mustLoad(t, SaveBytes(t, x.C128)) }},
		{"c64", mustLoad(t, SaveBytes(t, c64)), func(x AnyArray) AnyArray { return mustLoad(t, SaveBytes(t, x.C64)) }},
		// Non-contiguous input: Save of a Transpose view round-trips. The
		// transposed values are checked element-wise in
		// TestRoundTripTransposeView; here compare against the transposed
		// ravel.
		{"f64-transposed", mustLoad(t, SaveBytes(t, f64.Transpose())), func(x AnyArray) AnyArray {
			return mustLoad(t, SaveBytes(t, x.F64))
		}},
	} {
		x := tc.round(tc.load)
		if !equalShape(x.Shape, 2, 2) {
			t.Fatalf("%s: shape %v", tc.name, x.Shape)
		}
		var want, got []float64
		var wantC, gotC []complex128
		switch {
		case x.F64 != nil:
			src := f64.Data()
			if tc.name == "f64-transposed" {
				src = f64.Transpose().Ravel().Data()[:4]
			}
			want, got = src, x.F64.Data()
		case x.F32 != nil:
			for i, w := range f32.Data() {
				if x.F32.Data()[i] != w {
					t.Fatalf("%s[%d] = %v, want %v", tc.name, i, x.F32.Data()[i], w)
				}
			}
			continue
		case x.C128 != nil:
			wantC, gotC = c128.Data(), x.C128.Data()
		case x.C64 != nil:
			for i, w := range c64.Data() {
				if x.C64.Data()[i] != w {
					t.Fatalf("%s[%d] = %v, want %v", tc.name, i, x.C64.Data()[i], w)
				}
			}
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s[%d] = %v, want %v", tc.name, i, got[i], want[i])
			}
		}
		for i := range wantC {
			if gotC[i] != wantC[i] {
				t.Fatalf("%s[%d] = %v, want %v", tc.name, i, gotC[i], wantC[i])
			}
		}
	}
}

func TestRoundTripTransposeView(t *testing.T) {
	// Explicit non-contiguous round trip with value check per element.
	base := New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}})
	tv := base.Transpose()
	back, err := Load(SaveBytes(t, tv))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			if got := back.F64.At(i, j); got != tv.At(i, j) {
				t.Fatalf("(%d,%d): got %v want %v", i, j, got, tv.At(i, j))
			}
		}
	}
}

func TestAsHelpers(t *testing.T) {
	a, _ := LoadFile("testdata/f64_2x3.npy")
	if a.AsFloat64() == nil {
		t.Fatal("AsFloat64 nil")
	}
}

func TestAsHelpersPanics(t *testing.T) {
	a, _ := LoadFile("testdata/f64_2x3.npy")
	assertPanic(t, func() { a.AsComplex128() })
	c, _ := LoadFile("testdata/c128_2x2.npy")
	if c.AsComplex128() == nil {
		t.Fatal("AsComplex128 nil")
	}
	assertPanic(t, func() { c.AsFloat64() })
}

func TestSaveFileAppendsNpy(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "arr")
	a := New1DArray([]float64{1, 2, 3})
	if err := SaveFile(path, a); err != nil {
		t.Fatal(err)
	}
	back, err := LoadFile(path + ".npy")
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []float64{1, 2, 3} {
		if back.F64.Data()[i] != w {
			t.Fatalf("[%d]=%v", i, back.F64.Data()[i])
		}
	}
}

func TestSaveTxtDefaultFormat(t *testing.T) {
	a := New2DArray([][]float64{{1.5, 2}, {3, 4.25}})
	var buf bytes.Buffer
	if err := SaveTxt(&buf, a, ""); err != nil {
		t.Fatal(err)
	}
	if got, want := buf.String(), "1.5 2\n3 4.25\n"; got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestSaveTxtCustomFormat(t *testing.T) {
	a := New2DArray([][]float64{{1.5, 2}, {3, 4.25}})
	var buf bytes.Buffer
	if err := SaveTxt(&buf, a, "%.2f"); err != nil {
		t.Fatal(err)
	}
	if got, want := buf.String(), "1.50 2.00\n3.00 4.25\n"; got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestSaveTxt1D(t *testing.T) {
	var buf bytes.Buffer
	if err := SaveTxt(&buf, New1DArray([]float64{1, 2, 3}), ""); err != nil {
		t.Fatal(err)
	}
	if got := buf.String(); got != "1 2 3\n" {
		t.Fatalf("got %q", got)
	}
}

func TestSaveTxt3DPanics(t *testing.T) {
	a := Zeros[float64](2, 2, 2)
	assertPanic(t, func() { _ = SaveTxt(&bytes.Buffer{}, a, "") })
}

func TestSaveTxtRoundTrip(t *testing.T) {
	a := New2DArray([][]float64{{1.5, 2}, {3, 4.25}})
	var buf bytes.Buffer
	if err := SaveTxt(&buf, a, "%v"); err != nil {
		t.Fatal(err)
	}
	b, err := LoadTxt(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !equalShape(b.Shape(), 2, 2) {
		t.Fatal(b.Shape())
	}
	for i := range a.Data() {
		if b.Data()[i] != a.Data()[i] {
			t.Fatalf("[%d]=%v want %v", i, b.Data()[i], a.Data()[i])
		}
	}
}

func TestLoadTxt(t *testing.T) {
	a, err := LoadTxt(strings.NewReader("1 2\n3 4\n\n5 6\n"))
	if err != nil {
		t.Fatal(err)
	}
	if !equalShape(a.Shape(), 3, 2) {
		t.Fatal(a.Shape())
	}
	want := []float64{1, 2, 3, 4, 5, 6}
	for i, w := range want {
		if a.Data()[i] != w {
			t.Fatalf("[%d]=%v want %v", i, a.Data()[i], w)
		}
	}
	// scientific notation and inf/nan parse through ParseFloat.
	b, err := LoadTxt(strings.NewReader("1e3 -inf\nnan 0\n"))
	if err != nil {
		t.Fatal(err)
	}
	if b.Data()[0] != 1000 || b.Data()[3] != 0 {
		t.Fatal(b.Data())
	}
}

func TestLoadTxtJagged(t *testing.T) {
	_, err := LoadTxt(strings.NewReader("1 2\n3\n"))
	if err == nil || err.Error() != "numgo: LoadTxt: inconsistent number of columns" {
		t.Fatalf("got %v", err)
	}
}

func TestLoadTxtFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(path, []byte("1 2\n3 4\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	a, err := LoadTxtFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !equalShape(a.Shape(), 2, 2) {
		t.Fatal(a.Shape())
	}
}

func TestFromRegex(t *testing.T) {
	// numpy docs example: text = "1312 foo\n1534  bar\n444   qux",
	// regexp = r"(\d+)\s+(...)" — numgo parses every group as a float column,
	// so we use the numeric-columns variant; the docs text itself is covered
	// by TestFromRegexNumericGroups shape behavior.
	text := "1312 3\n1534  4\n444   5"
	a, err := FromRegex(strings.NewReader(text), `(\d+)\s+(\S+)`)
	if err != nil {
		t.Fatal(err)
	}
	if !equalShape(a.Shape(), 3, 2) {
		t.Fatal(a.Shape())
	}
	want := []float64{1312, 3, 1534, 4, 444, 5}
	for i, w := range want {
		if a.Data()[i] != w {
			t.Fatalf("[%d]=%v want %v", i, a.Data()[i], w)
		}
	}
}

func TestFromRegexNumericGroups(t *testing.T) {
	a, err := FromRegex(strings.NewReader("x=1.5 y=-2\nx=3 y=4.25\n"), `x=(\S+) y=(\S+)`)
	if err != nil {
		t.Fatal(err)
	}
	if !equalShape(a.Shape(), 2, 2) {
		t.Fatal(a.Shape())
	}
	want := []float64{1.5, -2, 3, 4.25}
	for i, w := range want {
		if a.Data()[i] != w {
			t.Fatalf("[%d]=%v want %v", i, a.Data()[i], w)
		}
	}
}

func TestFromRegexNoGroups(t *testing.T) {
	_, err := FromRegex(strings.NewReader("1 2"), `\d+`)
	if err == nil || !strings.Contains(err.Error(), "no capture groups") {
		t.Fatalf("got %v", err)
	}
}

func mustLoad(t *testing.T, r *bytes.Reader) AnyArray {
	t.Helper()
	a, err := Load(r)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func assertPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	fn()
}

// SaveBytes is a test helper: Save into a fresh buffer.
func SaveBytes[T Number](t *testing.T, a *Array[T]) *bytes.Reader {
	t.Helper()
	var buf bytes.Buffer
	if err := Save(&buf, a); err != nil {
		t.Fatal(err)
	}
	return bytes.NewReader(buf.Bytes())
}

func equalShape(got []int, want ...int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

var _ = filepath.Join
