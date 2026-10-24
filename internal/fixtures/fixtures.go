// Package fixtures provides golden-data helpers for numpy comparison tests.
package fixtures

import (
	"encoding/json"
	"math/cmplx"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/aixfoundry/numgo"
)

// LoadJSON reads a JSON fixture from testdata/<path> into v.
func LoadJSON(path string, v any) error {
	if !filepath.IsAbs(path) {
		path = filepath.Join(testdataDir(), path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// AssertClose fails t when got's elements differ from want by more than tol.
func AssertClose[T numgo.Number](t *testing.T, got *numgo.Array[T], want []float64, tol float64) {
	t.Helper()
	if got.Size() != len(want) {
		t.Fatalf("size = %d, want %d", got.Size(), len(want))
	}
	shape := got.Shape()
	coords := make([]int, len(shape))
	for i := range want {
		rem := i
		for k := len(shape) - 1; k >= 0; k-- {
			coords[k] = rem % shape[k]
			rem /= shape[k]
		}
		re, im := fixtureParts(got.At(coords...))
		if d := cmplx.Abs(complex(re-want[i], im)); d > tol {
			t.Fatalf("at %v: got %v, want %v (diff %v > %v)", coords, got.At(coords...), want[i], d, tol)
		}
	}
}

func fixtureParts[T numgo.Number](v T) (re, im float64) {
	rv := reflect.ValueOf(v)
	if rv.Kind() == reflect.Complex64 || rv.Kind() == reflect.Complex128 {
		c := rv.Complex()
		return real(c), imag(c)
	}
	return rv.Float(), 0
}

func testdataDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "testdata")
}
