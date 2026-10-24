package numgo

import (
	"archive/zip"
	"bytes"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// npz + genfromtxt tests. Fixtures in testdata/ were generated with numpy
// 2.5.1 (python3, repo root):
//
//	np.savez('testdata/savez_mixed.npz', f64=np.arange(6.0).reshape(2, 3),
//	         f32=np.array([1.5, -2.25, 3.0], dtype=np.float32),
//	         c128=np.array([1 + 2j, -3 + 0.5j]))
//	np.savez_compressed('testdata/savez_compressed.npz',
//	                    a=np.array([[1.5, -2.0], [3.25, 4.0]]),
//	                    b=np.array([7, 8, 9], dtype=np.float32),
//	                    c=np.array([0.5 + 0.25j]))
//
// genfromtxt goldens (verified against np.genfromtxt with the same input):
// names/missing-nan fill: "a,b,c\n1,2,3\n4,,6\n7,8,-9.5\n" delimiter=',' names=True
// skip+custom fill:       "skip\n1 2\n3 -\n5 6\n# c\n7 8\n" skip_header=1 skip_footer=1
//                         missing_values='-' filling_values=-1
// ';' delimiter + names:  "x;y;z\n1;2;3\n4;;6\n" delimiter=';' names=True
// comments-names line:    "# a b c\n1 2 3\n4 5 6\n" names=True
// failed parse -> nan:    "1 2\n3 abc\n"

func TestAnyOfAllDtypes(t *testing.T) {
	f64 := AnyOf(New2DArray([][]float64{{1, 2}}))
	f32 := AnyOf(New1DArray([]float32{1.5}))
	c128 := AnyOf(New1DArray([]complex128{1 + 2i}))
	c64 := AnyOf(New1DArray([]complex64{3 + 4i}))
	for _, tc := range []struct {
		a     AnyArray
		descr string
		check func(AnyArray) bool
	}{{f64, "<f8", func(a AnyArray) bool { return a.F64 != nil }},
		{f32, "<f4", func(a AnyArray) bool { return a.F32 != nil }},
		{c128, "<c16", func(a AnyArray) bool { return a.C128 != nil }},
		{c64, "<c8", func(a AnyArray) bool { return a.C64 != nil }}} {
		if tc.a.Descr != tc.descr || !equalShape(tc.a.Shape, 1, 2) && tc.descr == "<f8" {
			t.Fatalf("descr=%q shape=%v", tc.a.Descr, tc.a.Shape)
		}
		if !tc.check(tc.a) {
			t.Fatalf("field not set for %s", tc.descr)
		}
		if tc.a.F64 != nil && tc.a.F32 != nil || tc.a.F64 != nil && tc.a.C128 != nil {
			t.Fatal("multiple fields set")
		}
	}
}

func TestSaveZRoundTripMixedDtypes(t *testing.T) {
	arrays := map[string]AnyArray{
		"f64":  AnyOf(New2DArray([][]float64{{1.5, -2.25, 3.75}, {4.0, 5.5, -6.75}})),
		"f32":  AnyOf(New1DArray([]float32{1.5, -2.25, 3.0})),
		"c128": AnyOf(New1DArray([]complex128{1 + 2i, -3 + 0.5i})),
	}
	var buf bytes.Buffer
	if err := SaveZ(&buf, arrays); err != nil {
		t.Fatal(err)
	}
	m, err := LoadNpz(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 3 {
		t.Fatalf("keys %v", m)
	}
	if got := m["f64"].F64.Data(); got[0] != 1.5 || got[5] != -6.75 || !equalShape(m["f64"].Shape, 2, 3) {
		t.Fatalf("f64: %v %v", m["f64"].Shape, got)
	}
	if got := m["f32"].F32.Data(); got[1] != -2.25 {
		t.Fatalf("f32: %v", got)
	}
	if got := m["c128"].C128.Data(); got[1] != -3+0.5i {
		t.Fatalf("c128: %v", got)
	}
}

func TestSaveZCompressedRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := SaveZCompressed(&buf, map[string]AnyArray{
		"a": AnyOf(New2DArray([][]float64{{1.5, -2.0}, {3.25, 4.0}})),
	}); err != nil {
		t.Fatal(err)
	}
	zr, err := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	if zr.File[0].Method != zip.Deflate {
		t.Fatalf("method %v", zr.File[0].Method)
	}
	m, err := LoadNpz(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if got := m["a"].F64; got.At(1, 1) != 4.0 {
		t.Fatal(got.Data())
	}
}

func TestSaveZFileSuffix(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data") // no .npz suffix yet
	if err := SaveZFile(path, false, map[string]AnyArray{
		"x": AnyOf(New1DArray([]float64{7, 8})),
	}); err != nil {
		t.Fatal(err)
	}
	m, err := LoadNpzFile(path + ".npz")
	if err != nil {
		t.Fatal(err)
	}
	if got := m["x"].F64.Data(); got[0] != 7 || got[1] != 8 {
		t.Fatal(got)
	}
}

func TestLoadNpzPythonFixtures(t *testing.T) {
	m, err := LoadNpzFile("testdata/savez_mixed.npz")
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 3 {
		t.Fatalf("keys %v", m)
	}
	// f64 = arange(6).reshape(2,3); f32 = [1.5,-2.25,3.0]; c128 = [1+2j,-3+0.5j]
	if !equalShape(m["f64"].Shape, 2, 3) || m["f64"].Descr != "<f8" {
		t.Fatalf("f64: %v %s", m["f64"].Shape, m["f64"].Descr)
	}
	for i := 0; i < 6; i++ {
		if got := m["f64"].F64.Data()[i]; got != float64(i) {
			t.Fatalf("f64[%d]=%v", i, got)
		}
	}
	if got := m["f32"].F32.Data(); got[0] != 1.5 || got[1] != -2.25 || got[2] != 3.0 {
		t.Fatalf("f32: %v", got)
	}
	if got := m["c128"].C128.Data(); got[0] != 1+2i || got[1] != -3+0.5i {
		t.Fatalf("c128: %v", got)
	}

	cm, err := LoadNpzFile("testdata/savez_compressed.npz")
	if err != nil {
		t.Fatal(err)
	}
	if len(cm) != 3 {
		t.Fatalf("keys %v", cm)
	}
	if got := cm["a"].F64; !equalShape(got.Shape(), 2, 2) || got.At(0, 1) != -2.0 || got.At(1, 0) != 3.25 {
		t.Fatal(got.Data())
	}
	if got := cm["b"].F32.Data(); got[2] != 9 {
		t.Fatal(got)
	}
	if got := cm["c"].C128.Data(); got[0] != 0.5+0.25i {
		t.Fatal(got)
	}
}

func TestLoadNpzCorruptAndEmpty(t *testing.T) {
	if _, err := LoadNpz(strings.NewReader("PK\x03\x04 not really a zip")); err == nil {
		t.Fatal("expected error on corrupt zip")
	}
	var empty bytes.Buffer
	zw := zip.NewWriter(&empty)
	if _, err := zw.Create("readme.txt"); err != nil {
		t.Fatal(err)
	}
	_ = zw.Close()
	if _, err := LoadNpz(bytes.NewReader(empty.Bytes())); err == nil {
		t.Fatal("expected error on zip without .npy members")
	}
}

func TestLoadFileNpzError(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.npz")
	if err := os.WriteFile(path, []byte("PK\x03\x04pretend zip"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadFile(path)
	if err == nil || err.Error() != "numgo: npz archive: use LoadNpzFile to load .npz files" {
		t.Fatalf("got %v", err)
	}
}

func TestSaveZNpyMembersLoadable(t *testing.T) {
	// Members must be plain npy streams byte-compatible with numpy's savez.
	var buf bytes.Buffer
	if err := SaveZ(&buf, map[string]AnyArray{"arr": AnyOf(New1DArray([]float64{1, 2, 3}))}); err != nil {
		t.Fatal(err)
	}
	zr, _ := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if zr.File[0].Method != zip.Store {
		t.Fatalf("savez must store uncompressed, got method %v", zr.File[0].Method)
	}
	rc, _ := zr.File[0].Open()
	defer func() { _ = rc.Close() }()
	a, err := Load(rc)
	if err != nil {
		t.Fatal(err)
	}
	if got := a.F64.Data(); got[2] != 3 {
		t.Fatal(got)
	}
}

// ---- GenFromTxt ----

func genCheck(t *testing.T, got *Array[float64], want []float64, shape ...int) {
	t.Helper()
	if !equalShape(got.Shape(), shape...) {
		t.Fatalf("shape %v, want %v", got.Shape(), shape)
	}
	for i, w := range want {
		g := got.Data()[i]
		if math.IsNaN(w) {
			if !math.IsNaN(g) {
				t.Fatalf("[%d]=%v, want NaN", i, g)
			}
			continue
		}
		if g != w {
			t.Fatalf("[%d]=%v, want %v", i, g, w)
		}
	}
}

func TestGenFromTxtHeaderNamesMissingNan(t *testing.T) {
	// numpy golden: names=True, delimiter=',' -> (1,2,3),(4,nan,6),(7,8,-9.5)
	a, names, err := GenFromTxt(strings.NewReader("a,b,c\n1,2,3\n4,,6\n7,8,-9.5\n"),
		GenConfig{Delimiter: ",", UseHeaderNames: true})
	if err != nil {
		t.Fatal(err)
	}
	if names == nil || strings.Join(names, ",") != "a,b,c" {
		t.Fatalf("names %v", names)
	}
	genCheck(t, a, []float64{1, 2, 3, 4, math.NaN(), 6, 7, 8, -9.5}, 3, 3)
}

func TestGenFromTxtExplicitNames(t *testing.T) {
	_, names, err := GenFromTxt(strings.NewReader("1 2\n3 4\n"), GenConfig{Names: []string{"x", "y"}})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(names, ",") != "x,y" {
		t.Fatalf("names %v", names)
	}
	// No names configured -> nil.
	if _, names, _ := GenFromTxt(strings.NewReader("1 2\n"), GenConfig{}); names != nil {
		t.Fatalf("names %v", names)
	}
}

func TestGenFromTxtSkipHeaderFooterCustomFill(t *testing.T) {
	// numpy golden: skip_header=1, skip_footer=1, missing='-', fill=-1
	a, names, err := GenFromTxt(strings.NewReader("skip me\n1 2\n3 -\n5 6\n# tail comment\n7 8\n"),
		GenConfig{SkipHeader: 1, SkipFooter: 1, Missing: "-", FillingValues: -1})
	if err != nil {
		t.Fatal(err)
	}
	if names != nil {
		t.Fatalf("names %v", names)
	}
	genCheck(t, a, []float64{1, 2, 3, -1, 5, 6}, 3, 2)
}

func TestGenFromTxtCustomDelimiterMissing(t *testing.T) {
	// numpy golden: "x;y;z\n1;2;3\n4;;6\n" delimiter=';' names=True
	a, names, err := GenFromTxt(strings.NewReader("x;y;z\n1;2;3\n4;;6\n"),
		GenConfig{Delimiter: ";", UseHeaderNames: true})
	if err != nil {
		t.Fatal(err)
	}
	if names == nil || strings.Join(names, ";") != "x;y;z" {
		t.Fatalf("names %v", names)
	}
	genCheck(t, a, []float64{1, 2, 3, 4, math.NaN(), 6}, 2, 3)
}

func TestGenFromTxtCommentedHeader(t *testing.T) {
	// numpy golden: "# a b c\n1 2 3\n4 5 6\n" names=True
	a, names, err := GenFromTxt(strings.NewReader("# a b c\n1 2 3\n4 5 6\n"),
		GenConfig{UseHeaderNames: true})
	if err != nil {
		t.Fatal(err)
	}
	if names == nil || strings.Join(names, ",") != "a,b,c" {
		t.Fatalf("names %v", names)
	}
	genCheck(t, a, []float64{1, 2, 3, 4, 5, 6}, 2, 3)
}

func TestGenFromTxtFailedParseBecomesNan(t *testing.T) {
	// numpy golden: "1 2\n3 abc\n" -> [[1,2],[3,nan]]
	a, _, err := GenFromTxt(strings.NewReader("1 2\n3 abc\n"), GenConfig{})
	if err != nil {
		t.Fatal(err)
	}
	genCheck(t, a, []float64{1, 2, 3, math.NaN()}, 2, 2)
}

func TestGenFromTxtEmpty(t *testing.T) {
	a, names, err := GenFromTxt(strings.NewReader(""), GenConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if !equalShape(a.Shape(), 0) || names != nil {
		t.Fatalf("shape %v names %v", a.Shape(), names)
	}
}

func TestGenFromTxtJaggedError(t *testing.T) {
	// numpy raises "Some errors were detected!\n Line #2 (got 1 columns
	// instead of 2)"; we match the LoadTxt-style wording instead.
	_, _, err := GenFromTxt(strings.NewReader("1 2\n3\n"), GenConfig{})
	if err == nil || !strings.Contains(err.Error(), "inconsistent number of columns") {
		t.Fatalf("got %v", err)
	}
}

func TestGenFromTxtSkipHeaderComments(t *testing.T) {
	// Header lines are skipped raw, before comment filtering.
	a, _, err := GenFromTxt(strings.NewReader("# meta\n# more\n1 2\n3 4\n"),
		GenConfig{SkipHeader: 2})
	if err != nil {
		t.Fatal(err)
	}
	genCheck(t, a, []float64{1, 2, 3, 4}, 2, 2)
}
