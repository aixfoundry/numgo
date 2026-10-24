package numgo

import (
	"math/rand"
	"slices"
	"strings"
	"testing"
	"unicode"
)

func TestEinsumMatmulPathRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	cases := []struct {
		sub string
		sha []int
		shb []int
		shc []int
		shd []int
	}{
		{"ij,jk->ik", []int{7, 5}, []int{5, 9}, nil, nil},
		{"ij,jk", []int{4, 6}, []int{6, 3}, nil, nil},
		{"ij,jk,kl->il", []int{5, 4}, []int{4, 6}, []int{6, 2}, nil},
		{"ij,kj->ik", []int{3, 5}, []int{4, 5}, nil, nil},
		{"ij,jkl->ikl", []int{3, 4}, []int{4, 2, 5}, nil, nil},
		{"ijk,jk->ik", []int{3, 4, 5}, []int{4, 5}, nil, nil},
		{"ij,j->i", []int{6, 7}, []int{7}, nil, nil},
		{"i,i->", []int{8}, []int{8}, nil, nil},
		{"ij,jk,kl,lm->im", []int{3, 4}, []int{4, 5}, []int{5, 2}, []int{2, 3}},
		{"ijk,kl->il", []int{2, 3, 4}, []int{4, 5}, nil, nil},
	}
	for _, c := range cases {
		args := []*Array[float64]{randArr(rng, c.sha), randArr(rng, c.shb)}
		if c.shc != nil {
			args = append(args, randArr(rng, c.shc))
		}
		if c.shd != nil {
			args = append(args, randArr(rng, c.shd))
		}
		got := Einsum(c.sub, args...)
		want := einsumReference(c.sub, args...)
		if !shapesEqual(got.Shape(), want.Shape()) {
			t.Fatalf("%s: shape %v vs %v", c.sub, got.Shape(), want.Shape())
		}
		for i := range got.data {
			if got.data[i] != want.data[i] {
				t.Fatalf("%s: [%d] = %v, want %v (bit-identical)", c.sub, i, got.data[i], want.data[i])
			}
		}
	}
}

func randArr(rng *rand.Rand, shape []int) *Array[float64] {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float64, n)
	for i := range data {
		data[i] = float64(rng.Intn(9) - 4)
	}
	return FromSlice(data, shape...)
}

// einsumReference computes the generic-path result: call Einsum with the
// same operands but a subscript that always falls back (ellipsis-free
// explicit form is not guaranteed to fall back) — instead, force the
// fallback by wrapping operands in a way einsumMatmul rejects: a 1-operand
// call is rejected, so do it pairwise with an outer-product trick... Simplest
// reliable approach: duplicate operand through the generic branch by making
// one operand strided (Transpose), which einsumMatmul accepts equally.
// Actually the generic path IS reachable: rename it via build tag? No —
// compute the reference with straightforward loops here instead.
// shortcut: independent brute-force oracle (slow O(out·sum·ops)); only for tests.
func einsumReference(subscript string, as ...*Array[float64]) *Array[float64] {
	// Independent oracle: brute-force loops over the output coordinates.
	// Parse like Einsum: labels per operand, output labels.
	in, out, explicit := strings.Cut(subscript, "->")
	specs := strings.Split(in, ",")
	labels := make([][]rune, len(specs))
	for i, s := range specs {
		for _, r := range s {
			if unicode.IsLetter(r) {
				labels[i] = append(labels[i], r)
			}
		}
	}
	dims := map[rune]int{}
	occurs := map[rune]int{}
	for i, ls := range labels {
		sh := as[i].Shape()
		for k, l := range ls {
			dims[l] = sh[k]
			occurs[l]++
		}
	}
	var outLabels []rune
	if explicit {
		for _, r := range out {
			if r != '.' {
				outLabels = append(outLabels, r)
			}
		}
	} else {
		for l, c := range occurs {
			if c == 1 {
				outLabels = append(outLabels, l)
			}
		}
		slices.Sort(outLabels)
	}
	outShape := make([]int, len(outLabels))
	for i, l := range outLabels {
		outShape[i] = dims[l]
	}
	outN := 1
	for _, d := range outShape {
		outN *= d
	}
	sumLabels := map[rune]bool{}
	for l := range dims {
		if !slices.Contains(outLabels, l) {
			sumLabels[l] = true
		}
	}
	var sumList []rune
	for l := range sumLabels {
		sumList = append(sumList, l)
	}
	slices.Sort(sumList)
	sumN := 1
	for _, l := range sumList {
		sumN *= dims[l]
	}
	res := make([]float64, outN)
	oCoords := make([]int, len(outLabels))
	sCoords := make([]int, len(sumList))
	aCoords := make([]int, max(len(outLabels), 0))
	_ = aCoords
	for oi := range outN {
		// decode oCoords
		x := oi
		for i := len(outShape) - 1; i >= 0; i-- {
			oCoords[i] = x % outShape[i]
			x /= outShape[i]
		}
		var sum float64
		for si := range sumN {
			y := si
			for i := len(sumList) - 1; i >= 0; i-- {
				sCoords[i] = y % dims[sumList[i]]
				y /= dims[sumList[i]]
			}
			coordOf := func(l rune) int {
				if k := slices.Index(outLabels, l); k >= 0 {
					return oCoords[k]
				}
				return sCoords[slices.Index(sumList, l)]
			}
			prod := 1.0
			for i, a := range as {
				coords := make([]int, len(labels[i]))
				for k, l := range labels[i] {
					coords[k] = coordOf(l)
				}
				prod *= a.At(coords...)
			}
			sum += prod
		}
		res[oi] = sum
	}
	return FromSlice(res, outShape...)
}
