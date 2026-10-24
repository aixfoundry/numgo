package numgo

import (
	"slices"
	"testing"
	"unsafe"
)

func TestUnsafeDataRoundTrip(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	p := (*float64)(a.UnsafeData())
	for i := range 6 {
		if got := unsafe.Slice(p, 6)[i]; got != float64(i+1) {
			t.Fatalf("UnsafeData[%d] = %v, want %v", i, got, i+1)
		}
	}
	if a.ByteSize() != 6*unsafe.Sizeof(float64(0)) {
		t.Fatalf("ByteSize = %d", a.ByteSize())
	}
	// A view's ByteSize covers the view, not the backing store.
	v := a.Slice(R(1, 2, 1)) // second row: offset 3, 3 elements
	if got := (*float64)(v.UnsafeData()); *got != 4 {
		t.Fatalf("view first element = %v, want 4", *got)
	}
	if v.ByteSize() != 3*unsafe.Sizeof(float64(0)) {
		t.Fatalf("view ByteSize = %d", v.ByteSize())
	}
	if !v.IsContiguous() {
		t.Fatal("row view should be contiguous")
	}
}

func TestUnsafeDataNonContiguousPanics(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	tv := a.Transpose()
	mustPanic(t, "numgo: UnsafeData: array is not C-order contiguous; use Copy or Ravel first",
		func() { tv.UnsafeData() })
}

func TestBorrowSliceAliasing(t *testing.T) {
	data := []float64{1, 2, 3}
	a := BorrowSlice(data, 3)
	a.Set([]int{1}, 42)
	if data[1] != 42 {
		t.Fatalf("BorrowSlice copied: data = %v", data)
	}
	data[2] = 7
	if a.At(2) != 7 {
		t.Fatalf("slice mutation not visible: %v", a.At(2))
	}
	// Shape validation still applies.
	mustPanic(t, "numgo: cannot reshape array of size 3 into shape (2)", func() { BorrowSlice(data, 2) })
	// Doc contract: the array's Data must be the caller's slice itself.
	if &a.Data()[0] != &data[0] {
		t.Fatal("BorrowSlice did not alias the input slice")
	}
	if !slices.Equal(a.Shape(), []int{3}) {
		t.Fatalf("shape = %v", a.Shape())
	}
}

func TestFromByteSlice(t *testing.T) {
	// float32 round trip.
	f := []float32{1.5, -2.25}
	b := make([]byte, 0, 8)
	for _, v := range f {
		b = binaryLittleEndian(v, b)
	}
	a := FromByteSlice[float32](b)
	if a.At(0) != 1.5 || a.At(1) != -2.25 {
		t.Fatalf("FromByteSlice float32 = %v, %v", a.At(0), a.At(1))
	}
	if a.ByteSize() != uintptr(len(b)) {
		t.Fatalf("ByteSize = %d, want %d", a.ByteSize(), len(b))
	}

	// complex128 round trip via a float64 slice reinterpreted as bytes
	// (guaranteed 8-aligned, exercises the alias path).
	cs := []complex128{1 + 2i, 3 + 4i}
	fb := make([]float64, 0, 4)
	for _, c := range cs {
		fb = append(fb, real(c), imag(c))
	}
	cbuf := *(*[]byte)(unsafe.Pointer(&struct {
		p      *float64
		n, cap int
	}{&fb[0], len(fb) * 8, cap(fb) * 8}))
	if len(cbuf) != 8*len(fb) {
		t.Fatalf("reinterpretation failed: len=%d want=%d", len(cbuf), 8*len(fb))
	}
	ca := FromByteSlice[complex128](cbuf)
	if ca.At(0) != 1+2i || ca.At(1) != 3+4i {
		t.Fatalf("FromByteSlice complex128 = %v, %v", ca.At(0), ca.At(1))
	}

	// Length not a multiple of sizeof(T).
	mustPanic(t, "numgo: FromByteSlice: buffer length is not a multiple of the element size",
		func() { FromByteSlice[float64](make([]byte, 3)) })
	// Shape mismatch.
	mustPanic(t, "numgo: cannot reshape array of size 2 into shape (3)",
		func() { FromByteSlice[float32](b, 3) })
}

func TestFromByteSliceMisalignedCopies(t *testing.T) {
	// byte[1:] is misaligned for float64: the result must be a copy that
	// reads correctly (and does not alias the misaligned buffer).
	buf := make([]byte, 9)
	for i := range buf {
		buf[i] = byte(i)
	}
	mis := buf[1:]
	a := FromByteSlice[float64](mis)
	if a.Size() != 1 {
		t.Fatalf("size = %d", a.Size())
	}
	want := binaryDecodeFloat64(mis)
	if a.At(0) != want {
		t.Fatalf("misaligned decode = %v, want %v", a.At(0), want)
	}
}
