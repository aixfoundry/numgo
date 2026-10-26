package random

import (
	"fmt"
	"math"
	"sort"

	"github.com/aixfoundry/numgo"
)

// Generator is the numpy.random.Generator equivalent: a distribution layer
// over a BitGenerator. Zero value is not usable; construct via DefaultRng.
type Generator struct {
	bg BitGenerator
	ss *SeedSequence
}

// Spawn, Choice, Shuffle and Permutation live here.

// DefaultRng mirrors numpy.random.default_rng: PCG64 seeded from
// SeedSequence(seed).
func DefaultRng(seed ...uint64) *Generator {
	var ss *SeedSequence
	if len(seed) == 0 {
		// No OS entropy source in this port; entropy 0 keeps the constructor
		// total. shortcut: non-reproducible seeding, upgrade when the project
		// wants crypto/rand-backed defaults.
		ss = NewSeedSequence(0)
	} else {
		ss = NewSeedSequence(seed...)
	}
	return &Generator{bg: NewPCG64(ss), ss: ss}
}

// NewGenerator builds a Generator over an explicit SeedSequence.
func NewGenerator(ss *SeedSequence) *Generator {
	return &Generator{bg: NewPCG64(ss), ss: ss}
}

// NewGeneratorMT builds a Generator over an MT19937 engine seeded from ss
// (numpy Generator(MT19937(SeedSequence(...)))).
func NewGeneratorMT(ss *SeedSequence) *Generator {
	return &Generator{bg: NewMT19937(ss), ss: ss}
}

// MT19937Rng is the MT19937 analogue of DefaultRng: a Generator over
// MT19937(SeedSequence(seed...)).
func MT19937Rng(seed ...uint64) *Generator {
	var ss *SeedSequence
	if len(seed) == 0 {
		ss = NewSeedSequence(0)
	} else {
		ss = NewSeedSequence(seed...)
	}
	return NewGeneratorMT(ss)
}

// BitGenerator returns the underlying engine (numpy .bit_generator).
func (g *Generator) BitGenerator() BitGenerator { return g.bg }

// SeedSequence returns the generator's seed sequence (numpy ._seed_seq).
func (g *Generator) SeedSequence() *SeedSequence { return g.ss }

// Spawn returns child generators (numpy Generator.spawn).
func (g *Generator) Spawn(n int) []*Generator {
	kids := g.ss.Spawn(n)
	out := make([]*Generator, n)
	for i, k := range kids {
		out[i] = NewGenerator(k)
	}
	return out
}

// kahanSum: numpy _common.pyx kahan_sum, used to validate p.
func kahanSum(p []float64) float64 {
	var sum, c float64
	for _, x := range p {
		t := sum + x
		if math.Abs(sum) >= math.Abs(x) {
			c += (sum - t) + x
		} else {
			c += (x - t) + sum
		}
		sum = t
	}
	return sum + c
}

// Choice mirrors numpy Generator.choice for the a-population-as-int case:
// replace with or without p, or Floyd/interval sampling without replacement.
func (g *Generator) Choice(a int, size int, replace bool, p []float64) *arrayI {
	pop := int64(a)
	if pop <= 0 && size != 0 {
		panicAPositive()
	}
	out := newI(size)
	if replace {
		if p != nil {
			g.validateP(p, pop)
			cdf := make([]float64, len(p))
			var sum float64
			for i, v := range p {
				sum += v
				cdf[i] = sum
			}
			last := cdf[len(cdf)-1]
			for i := range cdf {
				cdf[i] /= last
			}
			for i := range size {
				u := g.nextDouble()
				// searchsorted(cdf, u, side='right')
				out.Data()[i] = int64(sort.SearchFloat64s(cdf, u))
			}
		} else {
			idx := g.integersRange(0, pop, false, size)
			copy(out.Data(), idx.Data())
		}
		return out
	}
	// replace=False
	if size > int(pop) {
		panicSampleTooLarge()
	}
	if p != nil {
		g.validateP(p, pop)
		if countNonZero(p) < size {
			panicFewerNonZero()
		}
		p2 := append([]float64(nil), p...)
		found := make([]int64, 0, size)
		for len(found) < size {
			x := g.Random(size - len(found)).Data()
			if len(found) > 0 {
				for _, f := range found {
					p2[f] = 0
				}
			}
			var sum float64
			cdf := make([]float64, len(p2))
			for i, v := range p2 {
				sum += v
				cdf[i] = sum
			}
			for i := range cdf {
				cdf[i] /= sum
			}
			seen := make(map[int64]bool)
			for _, u := range x {
				idx := int64(sort.SearchFloat64s(cdf, u))
				if !seen[idx] {
					seen[idx] = true
					found = append(found, idx)
				}
			}
		}
		copy(out.Data(), found[:size])
		return out
	}
	// No p: Floyd's algorithm variant from _generator.pyx:969.
	// smallest power of 2 larger than 1.2 * size
	setSize := uint64(1.2 * float64(size))
	mask := genMask(setSize)
	setSize = mask + 1
	const empty = ^uint64(0)
	hashSet := make([]uint64, setSize)
	for i := range hashSet {
		hashSet[i] = empty
	}
	data := out.Data()
	for j := pop - int64(size); j < pop; j++ {
		val := g.randomBoundedUint64(0, uint64(j), 0, false)
		loc := val & mask
		for hashSet[loc] != empty && hashSet[loc] != val {
			loc = (loc + 1) & mask
		}
		if hashSet[loc] == empty {
			hashSet[loc] = val
			data[j-pop+int64(size)] = int64(val)
		} else {
			loc := uint64(j) & mask
			for hashSet[loc] != empty {
				loc = (loc + 1) & mask
			}
			hashSet[loc] = uint64(j)
			data[j-pop+int64(size)] = j
		}
	}
	// shuffle the result (shuffle=True default, _shuffle_int uses
	// random_bounded_uint64(0, i, 0, 0), NOT random_interval)
	g.shuffleBoundedInt64(data)
	return out
}

func (g *Generator) validateP(p []float64, pop int64) {
	if len(p) != int(pop) {
		panic("a and p must have same size")
	}
	s := kahanSum(p)
	if math.IsNaN(s) {
		panicProbNaN()
	}
	for _, v := range p {
		if v < 0 {
			panicProbNegative()
		}
	}
	atol := math.Sqrt(2.220446049250313e-16) // sqrt(eps)
	if math.Abs(s-1.0) > atol {
		panicProbSum()
	}
}

func countNonZero(p []float64) int {
	n := 0
	for _, v := range p {
		if v > 0 {
			n++
		}
	}
	return n
}

// Shuffle permutes a 1-D float64 array in place with Fisher-Yates using
// random_interval (numpy _generator.pyx shuffle fast path, _shuffle_raw:
// for i in reversed(range(1, n)): j = random_interval(i); swap).
func (g *Generator) Shuffle(a *numgo.Array[float64]) {
	data := a.Data()
	for i := len(data) - 1; i >= 1; i-- {
		j := g.randomInterval(uint64(i))
		data[i], data[j] = data[j], data[i]
	}
}

// ShuffleInt64 permutes an int64 array in place.
func (g *Generator) ShuffleInt64(a *arrayI) {
	g.shuffleInt64(a.Data())
}

func (g *Generator) shuffleInt64(data []int64) {
	for i := len(data) - 1; i >= 1; i-- {
		j := g.randomInterval(uint64(i))
		data[i], data[j] = data[j], data[i]
	}
}

// shuffleBoundedInt64 is numpy _shuffle_int: j = bounded(0, i) Lemire, not
// random_interval (used by choice without replacement).
func (g *Generator) shuffleBoundedInt64(data []int64) {
	for i := len(data) - 1; i >= 1; i-- {
		j := g.randomBoundedUint64(0, uint64(i), 0, false)
		data[i], data[j] = data[j], data[i]
	}
}

// Permutation returns a shuffled arange(n); Permutation of an array
// returns a shuffled copy. Integer form matches numpy permutation(int).
func (g *Generator) Permutation(n int) *arrayI {
	arr := arangeI(n)
	g.shuffleInt64(arr.Data())
	return arr
}

// PermutationArray returns a shuffled copy of a 1-D array.
func (g *Generator) PermutationArray(a *numgo.Array[float64]) *numgo.Array[float64] {
	cp := a.Copy()
	g.Shuffle(cp)
	return cp
}

// Permuted permutes x independently along axis (numpy Generator.permuted):
// each slice along the other axes is shuffled with the same Fisher-Yates
// walk as Shuffle (_shuffle_raw: for i in reversed(range(1, n)):
// j = random_interval(i); swap). The iteration order follows the pyx: an
// iterator over all-but-axis dimensions runs the slices in C order, so the
// random stream matches numpy's for the same seed. x is not modified; a
// copy is returned. Negative axes normalize (axis=-1 is the last axis).
func (g *Generator) Permuted(x *numgo.Array[float64], axis int) *numgo.Array[float64] {
	out := x.Copy()
	ax := axis
	if ax < 0 {
		ax += out.Ndim()
	}
	if ax < 0 || ax >= out.Ndim() {
		panic(fmt.Sprintf("axis %d is out of bounds for array of dimension %d", axis, out.Ndim()))
	}
	n := out.Shape()[ax]
	rest := append(append([]int{}, out.Shape()[:ax]...), out.Shape()[ax+1:]...)
	coords := make([]int, out.Ndim())
	ci := make([]int, out.Ndim())
	cj := make([]int, out.Ndim())
	odometerPermuted(rest, func(c []int) {
		copy(coords[:ax], c[:ax])
		copy(coords[ax+1:], c[ax:])
		copy(ci, coords)
		copy(cj, coords)
		for i := n - 1; i >= 1; i-- {
			j := int(g.randomInterval(uint64(i)))
			ci[ax] = i
			cj[ax] = j
			vi, vj := out.At(ci...), out.At(cj...)
			out.Set(ci, vj)
			out.Set(cj, vi)
		}
	})
	return out
}

// odometerPermuted iterates every coordinate of shape in C order (the
// root package's odometer is unexported, so this is the local copy).
func odometerPermuted(shape []int, f func(coords []int)) {
	if len(shape) == 0 {
		f(nil)
		return
	}
	idx := make([]int, len(shape))
	for {
		f(idx)
		k := len(idx) - 1
		for k >= 0 {
			idx[k]++
			if idx[k] < shape[k] {
				break
			}
			idx[k] = 0
			k--
		}
		if k < 0 {
			return
		}
	}
}

// Bytes returns n random bytes (numpy Generator.bytes): ceil(n/4) uint32
// draws from the bitgen via integers(0, 2^32) laid out little-endian,
// truncated to n.
func (g *Generator) Bytes(n int) []byte {
	if n < 0 {
		panicNegativeEntropy()
	}
	nUint32 := 0
	if n > 0 {
		nUint32 = (n-1)/4 + 1
	}
	// numpy: integers(0, 2**32, dtype=uint32) — endpoint excluded, so rng =
	// 0xFFFFFFFF hits randomBoundedUint64's direct next_uint32 path.
	words := g.integersRange(0, 1<<32, false, nUint32).Data()
	out := make([]byte, n)
	for i := range out {
		out[i] = byte(words[i/4] >> (8 * (i % 4)))
	}
	return out
}
