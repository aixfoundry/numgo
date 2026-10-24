package ma

import "fmt"

// Panic messages, wording mirrored from numpy (design §4/§7 P0-4).

func panicMaskedSize() {
	panic("numgo: mask length does not match data size")
}

func panicIndex(coords, shape []int) {
	if len(coords) != len(shape) {
		panic(fmt.Sprintf("numgo: expected %d indices for a %d-d array, got %d",
			len(shape), len(shape), len(coords)))
	}
	panic(fmt.Sprintf("numgo: index %v is out of bounds for shape %v", coords, shape))
}

func panicAxis(axis, ndim int) {
	panic(fmt.Sprintf("numgo: axis %d is out of bounds for %d-d array", axis, ndim))
}
