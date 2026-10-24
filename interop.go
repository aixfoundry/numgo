package numgo

import (
	"slices"
	"unsafe"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// Interop primitives for handing array memory to foreign code (cgo, syscalls,
// binary decoders) — the Go-side analog of numpy's __array_interface__ /
// ctypeslib. These are the ONLY ways this package aliases caller memory: every
// constructor otherwise copies its input (design.md P0-2).
//
// cgo example (a is *Array[float64], C-order contiguous):
//
//	p := (*C.double)(a.UnsafeData())
//	C.process(p, C.int(a.Size())) // ByteSize() bytes, C float64 == Go float64
//
// The pointer stays valid while a is reachable (the GC keeps the backing
// array alive) and its data is not resized.
func (a *Array[T]) UnsafeData() unsafe.Pointer {
	if a.Size() == 0 {
		return nil
	}
	if !a.IsContiguous() {
		panic("numgo: UnsafeData: array is not C-order contiguous; use Copy or Ravel first")
	}
	return unsafe.Pointer(&a.data[a.offset])
}

// Pure: ByteSize returns the number of bytes from UnsafeData to the end of the
// VIEW: Size()·sizeof(T) — NOT the size of the whole backing store, which may
// be longer (a view starting at offset > 0 covers less than len(data)).
func (a *Array[T]) ByteSize() uintptr {
	return uintptr(a.Size()) * unsafe.Sizeof(*new(T))
}

// Shares data: BorrowSlice wraps a caller-owned slice ZERO-COPY (numpy
// asarray semantics — the one explicit exception to the constructors-copy
// rule, named Borrow* to flag the aliasing). Mutations through the array write
// through to the slice and vice versa. The caller guarantees the slice is not
// resized or reused for the array's lifetime. With no shape the result is 1-D.
func BorrowSlice[T Number](data []T, shape ...int) *Array[T] {
	if len(shape) == 0 {
		shape = []int{len(data)}
	}
	if nditer.Size(shape) != len(data) {
		panicReshape(len(data), shape)
	}
	return newArray(slices.Clone(shape), nditer.COrderStrides(shape), 0, data)
}

// Shares data: FromByteSlice reinterprets raw LITTLE-ENDIAN bytes as T
// elements (numpy frombuffer semantics). When len(b) is a multiple of
// sizeof(T) AND b is suitably aligned for T, it is a zero-copy
// reinterpretation that ALIASES b (mutations flow both ways); misaligned
// input is silently copied into aligned storage instead, so sharing only
// holds on the aligned path. Panics when len(b) is not a sizeof(T) multiple
// or product(shape) mismatches the element count.
func FromByteSlice[T Number](b []byte, shape ...int) *Array[T] {
	sizeof := unsafe.Sizeof(*new(T))
	if len(b)%int(sizeof) != 0 {
		panic("numgo: FromByteSlice: buffer length is not a multiple of the element size")
	}
	n := len(b) / int(sizeof)
	if len(shape) == 0 {
		shape = []int{n}
	}
	if nditer.Size(shape) != n {
		panicReshape(n, shape)
	}
	ptr := unsafe.Pointer(unsafe.SliceData(b))
	if uintptr(ptr)%unsafe.Alignof(*new(T)) == 0 {
		return newArray(slices.Clone(shape), nditer.COrderStrides(shape), 0,
			unsafe.Slice((*T)(ptr), n))
	}
	elems := make([]T, n)
	dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(elems))), len(b))
	copy(dst, b)
	return newArray(slices.Clone(shape), nditer.COrderStrides(shape), 0, elems)
}
