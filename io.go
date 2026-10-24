package numgo

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Save writes a as a NumPy .npy 1.0 stream (2.0 only if the header length
// field would overflow uint16). Data is written C-order (fortran_order false);
// non-contiguous input is raveled first, matching numpy's save of a strided
// view.
func Save[T Number](w io.Writer, a *Array[T]) error {
	if a == nil {
		panicNilArray("Save")
	}
	if _, err := w.Write(wrapNpyHeader(npyHeaderString(npyDescrFor[T](), a.Shape(), false))); err != nil {
		return err
	}
	return npyWriteElems(w, npyDescrFor[T](), a.Ravel().Data()[:a.Size()])
}

// SaveFile writes a to path as .npy, appending the .npy suffix when missing
// (numpy np.save).
func SaveFile[T Number](path string, a *Array[T]) error {
	if a == nil {
		panicNilArray("SaveFile")
	}
	if !strings.HasSuffix(path, ".npy") {
		path += ".npy"
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Save(f, a); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// AnyArray is the dynamic-dtype result of Load: exactly one typed field is
// non-nil and matches Shape. Use AsFloat64/AsComplex128 to unwrap.
type AnyArray struct {
	F64   *Array[float64]
	F32   *Array[float32]
	C128  *Array[complex128]
	C64   *Array[complex64]
	Shape []int
	Descr string
}

// AsFloat64 returns the loaded real array; it panics on complex data.
func (a AnyArray) AsFloat64() *Array[float64] {
	if a.F64 == nil {
		panicAsReal()
	}
	return a.F64
}

// AsComplex128 returns the loaded complex array; it panics on real data.
func (a AnyArray) AsComplex128() *Array[complex128] {
	if a.C128 == nil {
		panicAsComplex()
	}
	return a.C128
}

// Load reads a NumPy .npy stream. Supported dtypes: float32/float64/
// complex64/complex128, either endianness. fortran_order data is transposed
// into C order on read.
func Load(r io.Reader) (AnyArray, error) {
	br := bufio.NewReader(r)
	h, err := readNpyHeader(br)
	if err != nil {
		return AnyArray{}, err
	}
	size, ok := npyDescrSize(h.descr)
	if !ok {
		return AnyArray{}, npyUnsupportedDtype(h.descr)
	}
	if len(h.shape) == 0 {
		return AnyArray{}, errors.New("numgo: 0-d npy arrays are not supported")
	}
	raw := make([]byte, h.count()*size)
	if _, err := io.ReadFull(br, raw); err != nil {
		return AnyArray{}, fmt.Errorf("numgo: reading npy data: %w", err)
	}
	return h.decode(raw)
}

// LoadFile reads path as .npy; a zip-magic file (.npz) is an error pointing
// at LoadNpzFile, since archives hold multiple arrays.
func LoadFile(path string) (AnyArray, error) {
	f, err := os.Open(path)
	if err != nil {
		return AnyArray{}, err
	}
	defer func() { _ = f.Close() }()
	br := bufio.NewReader(f)
	if magic, _ := br.Peek(4); !bytes.Equal(magic, []byte("PK\x03\x04")) {
		a, err := Load(br)
		return a, err
	}
	return AnyArray{}, errNpz
}

// SaveTxt writes a row per line with space-delimited format-formatted values
// (numpy savetxt defaults: delimiter ' ', fmt '%g'). Rank above 2 panics.
func SaveTxt[T Number](w io.Writer, a *Array[T], format string) error {
	if a == nil {
		panicNilArray("SaveTxt")
	}
	if a.Ndim() > 2 {
		panicSavetxtND(a.Ndim())
	}
	if format == "" {
		format = "%g"
	}
	var b strings.Builder
	writeRow := func(row *Array[T]) {
		for j, v := range row.Ravel().Data()[:row.Size()] {
			if j > 0 {
				b.WriteByte(' ')
			}
			fmt.Fprintf(&b, format, v)
		}
		b.WriteByte('\n')
	}
	switch a.Ndim() {
	case 1:
		writeRow(a)
	case 2:
		n := a.Shape()[0]
		for i := range n {
			writeRow(a.Slice(R(i, i+1, 1)))
		}
	}
	_, err := io.WriteString(w, b.String())
	return err
}

// SaveTxtFile writes a to path as text; an empty format means "%g".
func SaveTxtFile[T Number](path string, a *Array[T], format string) error {
	if a == nil {
		panicNilArray("SaveTxtFile")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := SaveTxt(f, a, format); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// LoadTxt parses whitespace-delimited float text (numpy loadtxt defaults).
// Blank lines are skipped; the first data row fixes the column count. A ragged
// file is an error, not a panic — IO failures are returned.
func LoadTxt(r io.Reader) (*Array[float64], error) {
	sc := bufio.NewScanner(r)
	cols := -1
	var vals []float64
	rows := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if cols < 0 {
			cols = len(fields)
		} else if len(fields) != cols {
			return nil, errors.New("numgo: LoadTxt: inconsistent number of columns")
		}
		for _, f := range fields {
			v, err := strconv.ParseFloat(f, 64)
			if err != nil {
				return nil, fmt.Errorf("numgo: LoadTxt: %w", err)
			}
			vals = append(vals, v)
		}
		rows++
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if cols <= 0 {
		return Zeros[float64](0), nil
	}
	return FromSlice(vals, rows, cols), nil
}

// LoadTxtFile reads path as whitespace-delimited float text.
func LoadTxtFile(path string) (*Array[float64], error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return LoadTxt(f)
}

// FromRegex parses every regex match's capture groups as float columns
// (numpy fromregex, with all groups taken as numeric fields).
func FromRegex(r io.Reader, regex string) (*Array[float64], error) {
	re, err := regexp.Compile(regex)
	if err != nil {
		return nil, err
	}
	n := re.NumSubexp()
	if n == 0 {
		return nil, errors.New("numgo: FromRegex: regexp has no capture groups")
	}
	sc := bufio.NewScanner(r)
	var vals []float64
	for sc.Scan() {
		for _, m := range re.FindAllStringSubmatch(sc.Text(), -1) {
			for _, g := range m[1:] {
				v, err := strconv.ParseFloat(strings.TrimSpace(g), 64)
				if err != nil {
					return nil, fmt.Errorf("numgo: FromRegex: %w", err)
				}
				vals = append(vals, v)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return FromSlice(vals, len(vals)/n, n), nil
}
