package strings

import (
	"fmt"
	"strconv"

	"github.com/aixfoundry/numgo"
	"github.com/aixfoundry/numgo/internal/nditer"
)

// Conversions bridging StrArray and numeric arrays (numpy's str→num ufuncs and
// the reverse via format strings).

// Pure: ParseFloat converts every element with strconv.ParseFloat 64-bit. The
// returned array shares the input's shape; the error reports the C-order
// linear index of the first failing element.
func ParseFloat(a *StrArray) (*numgo.Array[float64], error) {
	out := numgo.Zeros[float64](a.Shape()...)
	lin := 0
	var firstErr error
	forEach(a, func(v string) {
		if firstErr != nil {
			lin++
			return
		}
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			out.Data()[lin] = f
		} else if firstErr == nil {
			firstErr = &ElementError{Index: lin, Err: err}
		}
		lin++
	})
	if firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// Pure: Atoi converts every element with strconv.ParseInt base 10 into a
// plain []int64 in C order — int64 is not a numgo.Number, so no Array type
// exists for it (same precedent as the random package's arrayI). The error
// reports the C-order linear index of the first failing element.
func Atoi(a *StrArray) ([]int64, error) {
	out := make([]int64, a.Size())
	lin := 0
	var firstErr error
	forEach(a, func(v string) {
		if firstErr != nil {
			lin++
			return
		}
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			out[lin] = n
		} else {
			firstErr = &ElementError{Index: lin, Err: err}
		}
		lin++
	})
	if firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// ElementError reports the C-order linear index of the element that failed to
// parse, wrapping strconv's own error text.
type ElementError struct {
	Index int
	Err   error
}

// Error formats the failure with the element's linear index.
func (e *ElementError) Error() string {
	return "numgo/strings: element " + strconv.Itoa(e.Index) + ": " + e.Err.Error()
}

// Unwrap returns the underlying parse error.
func (e *ElementError) Unwrap() error { return e.Err }

// Pure: FromFloat formats every element of a with the format verb (a
// fmt.Sprintf verb like "%g" or "%.3f") into a new StrArray of the same shape.
func FromFloat(a *numgo.Array[float64], format string) *StrArray {
	shape := a.Shape()
	dst := newStrArray(shape, nditer.COrderStrides(shape), 0, make([]string, nditer.Size(shape)))
	i := 0
	for _, v := range a.Data() {
		dst.data[i] = fmt.Sprintf(format, v)
		i++
	}
	return dst
}
