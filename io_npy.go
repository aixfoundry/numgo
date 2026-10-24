package numgo

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// npy header protocol (numpy/lib/_format_impl.py): \x93NUMPY + major.minor,
// then the header length (u16 for v1.0, u32 for v2.0), then a space-padded
// Python dict literal with keys descr, fortran_order, shape; the total header
// length is a multiple of 64.

const (
	npyMagic = "\x93NUMPY"
	npyAlign = 64
)

// npyDescr maps a Number type to its little-endian npy descriptor string.
func npyDescrFor[T Number]() string {
	var v T
	switch any(v).(type) {
	case float32:
		return "<f4"
	case float64:
		return "<f8"
	case complex64:
		return "<c8"
	default: // complex128; the Number type set has no other members
		return "<c16"
	}
}

// npyHeaderString renders the header dict exactly as numpy writes it:
// alphabetically sorted keys, repr values, plus the growth-axis spare spaces.
func npyHeaderString(descr string, shape []int, fortran bool) string {
	h := fmt.Sprintf("{'descr': '%s', 'fortran_order': %s, 'shape': %s, }",
		descr, npyBoolRepr(fortran), npyShapeRepr(shape))
	if len(shape) > 0 {
		idx := 0
		if fortran {
			idx = len(shape) - 1
		}
		h += strings.Repeat(" ", max(0, 21-len(strconv.Itoa(shape[idx]))))
	}
	return h
}

// npyShapeRepr renders shape as a Python tuple repr: (), (3,), (2, 3).
func npyShapeRepr(shape []int) string {
	switch len(shape) {
	case 0:
		return "()"
	case 1:
		return "(" + strconv.Itoa(shape[0]) + ",)"
	}
	var b strings.Builder
	b.WriteByte('(')
	for i, d := range shape {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(strconv.Itoa(d))
	}
	b.WriteByte(')')
	return b.String()
}

// wrapNpyHeader prepends magic+version, pads to a 64-byte boundary and
// terminates with a newline. v1.0 unless the length field would overflow u16.
func wrapNpyHeader(header string) []byte {
	build := func(major byte, lenSize int) []byte {
		hlen := len(header) + 1
		pad := npyAlign - (len(npyMagic)+2+lenSize+hlen)%npyAlign
		total := hlen + pad
		buf := make([]byte, 0, len(npyMagic)+2+lenSize+total)
		buf = append(buf, npyMagic...)
		buf = append(buf, major, 0)
		if lenSize == 2 {
			buf = binary.LittleEndian.AppendUint16(buf, uint16(total))
		} else {
			buf = binary.LittleEndian.AppendUint32(buf, uint32(total))
		}
		buf = append(buf, header...)
		buf = append(buf, bytes.Repeat([]byte{' '}, pad)...)
		return append(buf, '\n')
	}
	if l := len(header) + 1 + npyAlign; l > math.MaxUint16 {
		return build(2, 4)
	}
	return build(1, 2)
}

// npyBoolRepr renders a Python bool literal.
func npyBoolRepr(b bool) string {
	if b {
		return "True"
	}
	return "False"
}

// npyDescrSize reports the byte size of one element for the four supported
// dtypes (either endianness); ok is false for every other descriptor.
func npyDescrSize(descr string) (size int, ok bool) {
	switch descr {
	case "<f4", ">f4":
		return 4, true
	case "<c8", ">c8":
		return 8, true
	case "<f8", ">f8":
		return 8, true
	case "<c16", ">c16":
		return 16, true
	}
	return 0, false
}

// npyHeader is the parsed npy header.
type npyHeader struct {
	descr   string
	shape   []int
	fortran bool
}

func (h *npyHeader) count() int {
	n := 1
	for _, d := range h.shape {
		n *= d
	}
	return n
}

var (
	npyDescrRe   = regexp.MustCompile(`'descr':\s*'([^']*)'`)
	npyFortranRe = regexp.MustCompile(`'fortran_order':\s*(True|False)`)
	npyShapeRe   = regexp.MustCompile(`'shape':\s*\(([^)]*)\)`)
)

// readNpyHeader reads magic, version and the dict header from br.
func readNpyHeader(br *bufio.Reader) (*npyHeader, error) {
	magic := make([]byte, len(npyMagic)+2)
	if _, err := io.ReadFull(br, magic); err != nil {
		return nil, fmt.Errorf("numgo: reading npy magic: %w", err)
	}
	if string(magic[:len(npyMagic)]) != npyMagic {
		return nil, fmt.Errorf("numgo: the magic string is not correct; expected %q, got %q",
			npyMagic, magic[:len(npyMagic)])
	}
	var hlen int
	switch major := magic[len(npyMagic)]; major {
	case 1:
		var u16 uint16
		if err := binary.Read(br, binary.LittleEndian, &u16); err != nil {
			return nil, fmt.Errorf("numgo: reading npy header length: %w", err)
		}
		hlen = int(u16)
	case 2, 3:
		var u32 uint32
		if err := binary.Read(br, binary.LittleEndian, &u32); err != nil {
			return nil, fmt.Errorf("numgo: reading npy header length: %w", err)
		}
		hlen = int(u32)
	default:
		return nil, fmt.Errorf("numgo: unsupported npy format version: %d", major)
	}
	raw := make([]byte, hlen)
	if _, err := io.ReadFull(br, raw); err != nil {
		return nil, fmt.Errorf("numgo: reading npy header: %w", err)
	}
	s := string(raw)
	m := npyDescrRe.FindStringSubmatch(s)
	sm := npyShapeRe.FindStringSubmatch(s)
	if m == nil || sm == nil {
		return nil, fmt.Errorf("numgo: cannot parse npy header: %q", s)
	}
	h := &npyHeader{descr: m[1]}
	if fm := npyFortranRe.FindStringSubmatch(s); fm != nil {
		h.fortran = fm[1] == "True"
	}
	// Fields are "2" or "2, 3"; the trailing comma of 1-D tuples and any
	// Python-2 long suffixes are stripped.
	for f := range strings.SplitSeq(sm[1], ",") {
		f = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(f), "L"))
		if f == "" {
			continue
		}
		n, err := strconv.Atoi(f)
		if err != nil {
			return nil, fmt.Errorf("numgo: cannot parse npy header: %q", s)
		}
		h.shape = append(h.shape, n)
	}
	return h, nil
}

// decode builds the typed array from the raw element bytes. The caller has
// already validated descr via npyDescrSize, so only the four supported forms
// (plus a leading '>' endianness mark) reach this switch.
func (h *npyHeader) decode(raw []byte) (AnyArray, error) {
	le := h.descr[:1] != ">"
	rd32 := func(i int) uint32 { // i-th 4-byte word
		b := raw[i*4 : i*4+4]
		if le {
			return binary.LittleEndian.Uint32(b)
		}
		return binary.BigEndian.Uint32(b)
	}
	rd64 := func(i int) uint64 { // i-th 8-byte word
		b := raw[i*8 : i*8+8]
		if le {
			return binary.LittleEndian.Uint64(b)
		}
		return binary.BigEndian.Uint64(b)
	}
	n := h.count()
	switch h.descr[1:] {
	case "f4":
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(rd32(i))
		}
		return AnyArray{F32: npyArray(out, h.shape, h.fortran), Shape: h.shape, Descr: h.descr}, nil
	case "c8":
		out := make([]complex64, n)
		for i := range out {
			out[i] = complex(math.Float32frombits(rd32(2*i)), math.Float32frombits(rd32(2*i+1)))
		}
		return AnyArray{C64: npyArray(out, h.shape, h.fortran), Shape: h.shape, Descr: h.descr}, nil
	case "f8":
		out := make([]float64, n)
		for i := range out {
			out[i] = math.Float64frombits(rd64(i))
		}
		return AnyArray{F64: npyArray(out, h.shape, h.fortran), Shape: h.shape, Descr: h.descr}, nil
	default: // "c16", the only remaining supported descr
		out := make([]complex128, n)
		for i := range out {
			out[i] = complex(math.Float64frombits(rd64(2*i)), math.Float64frombits(rd64(2*i+1)))
		}
		return AnyArray{C128: npyArray(out, h.shape, h.fortran), Shape: h.shape, Descr: h.descr}, nil
	}
}

// npyArray builds the C-order array, transposing fortran-order file data: the
// bytes form a compact array of the reversed shape whose transpose is the
// logical array; the copy runs through the regular Zip engine.
func npyArray[T Number](data []T, shape []int, fortran bool) *Array[T] {
	if !fortran {
		return FromSlice(data, shape...)
	}
	rev := make([]int, len(shape))
	for i := range shape {
		rev[i] = shape[len(shape)-1-i]
	}
	return FromSlice(data, rev...).Transpose().Copy()
}

// npyWriteElems encodes values little-endian into w.
func npyWriteElems[T Number](w io.Writer, descr string, data []T) error {
	switch descr {
	case "<f4":
		buf := make([]byte, 4*len(data))
		for i, v := range data {
			re, _ := parts(v)
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(float32(re)))
		}
		_, err := w.Write(buf)
		return err
	case "<c8":
		buf := make([]byte, 8*len(data))
		for i, v := range data {
			re, im := parts(v)
			binary.LittleEndian.PutUint32(buf[8*i:], math.Float32bits(float32(re)))
			binary.LittleEndian.PutUint32(buf[8*i+4:], math.Float32bits(float32(im)))
		}
		_, err := w.Write(buf)
		return err
	case "<f8":
		buf := make([]byte, 8*len(data))
		for i, v := range data {
			re, _ := parts(v)
			binary.LittleEndian.PutUint64(buf[8*i:], math.Float64bits(re))
		}
		_, err := w.Write(buf)
		return err
	default: // <c16
		buf := make([]byte, 16*len(data))
		for i, v := range data {
			re, im := parts(v)
			binary.LittleEndian.PutUint64(buf[16*i:], math.Float64bits(re))
			binary.LittleEndian.PutUint64(buf[16*i+8:], math.Float64bits(im))
		}
		_, err := w.Write(buf)
		return err
	}
}

// panicSavetxtND reports a SaveTxt array of rank > 2 (numpy wording).
func panicSavetxtND(ndim int) {
	panic(fmt.Sprintf("numgo: SaveTxt: expected 1D or 2D array, got %dD array instead", ndim))
}

// panicAsReal reports AsFloat64 on complex data.
func panicAsReal() {
	panic("numgo: AsFloat64 on a complex array; use AsComplex128")
}

// panicAsComplex reports AsComplex128 on real data.
func panicAsComplex() {
	panic("numgo: AsComplex128 on a real array; use AsFloat64")
}

// panicNilArray reports a nil *Array argument.
func panicNilArray(op string) {
	panic(fmt.Sprintf("numgo: %s: nil array", op))
}

// npyUnsupportedDtype is the error for dtypes outside the four supported.
func npyUnsupportedDtype(descr string) error {
	return fmt.Errorf("numgo: unsupported npy dtype: %q", descr)
}
