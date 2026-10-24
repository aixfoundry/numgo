package ma

import (
	"math"

	"github.com/aixfoundry/numgo"
)

// Element-wise operations (numpy ma: mask = OR of operand masks). All are
// Pure: they return a new Masked and never alias operand data.

// zip2 applies op element-wise over a and b with right-aligned broadcasting;
// the result mask ORs the operand masks.
//
// Fast path: both data arrays contiguous with the full result shape — the
// per-coordinate walk collapses to two flat slices (data via the root zip
// engine, masks as a plain byte-wise OR). The generic walk stays for strided
// or broadcast layouts. Data results are bit-identical (same op on the same
// values); the mask OR is order-independent.
func zip2[T numgo.Number](a, b *Masked[T], op func(x, y T) T) *Masked[T] {
	shape := broadcastShape(a.Shape(), b.Shape())
	n := sizeOf(shape)
	out := numgo.Zeros[T](shape...)
	mask := make([]bool, n)
	if slicesEqual(shape, a.a.Shape()) && a.a.IsContiguous() &&
		slicesEqual(shape, b.a.Shape()) && b.a.IsContiguous() && n > 0 {
		ad, bd, od := a.a.Data()[:n], b.a.Data()[:n], out.Data()[:n]
		for i := range od {
			od[i] = op(ad[i], bd[i])
		}
		for i := range mask {
			mask[i] = a.mask[i] || b.mask[i]
		}
		return newMasked(out, mask)
	}
	walk(shape, func(coords []int) {
		i := flat(shape, coords)
		out.Set(coords, op(valueAt(a.a, coords), valueAt(b.a, coords)))
		mask[i] = maskAtB(a.mask, a.Shape(), coords) || maskAtB(b.mask, b.Shape(), coords)
	})
	return newMasked(out, mask)
}

// slicesEqual reports identical shapes.
func slicesEqual(x, y []int) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

// Pure: Add returns the element-wise sum with the masks ORed.
func (m *Masked[T]) Add(b *Masked[T]) *Masked[T] {
	return zip2(m, b, func(x, y T) T { return x + y })
}

// Pure: Sub returns the element-wise difference with the masks ORed.
func (m *Masked[T]) Sub(b *Masked[T]) *Masked[T] {
	return zip2(m, b, func(x, y T) T { return x - y })
}

// Pure: Mul returns the element-wise product with the masks ORed.
func (m *Masked[T]) Mul(b *Masked[T]) *Masked[T] {
	return zip2(m, b, func(x, y T) T { return x * y })
}

// Pure: Div returns the element-wise quotient. Positions where the divisor is
// zero are additionally masked (numpy ma: divide-by-zero warning becomes a
// mask bit; the IEEE result stays in the data).
func (m *Masked[T]) Div(b *Masked[T]) *Masked[T] {
	out := zip2(m, b, func(x, y T) T { return x / y })
	bShape := b.Shape()
	walk(out.Shape(), func(coords []int) {
		if maskAtB(b.mask, bShape, coords) {
			return // already masked; keep the operand's mask bit
		}
		y := valueAt(b.a, coords)
		re, im := parts(y)
		if re == 0 && im == 0 {
			out.mask[flat(out.Shape(), coords)] = true
		}
	})
	return out
}

// Pure: Neg returns the element-wise negation.
func (m *Masked[T]) Neg() *Masked[T] {
	return m.unary(func(x T) T { return -x }, nil)
}

// Pure: AddValue adds the scalar v to every element.
func (m *Masked[T]) AddValue(v T) *Masked[T] {
	return m.unary(func(x T) T { return x + v }, nil)
}

// Pure: SubValue subtracts the scalar v from every element.
func (m *Masked[T]) SubValue(v T) *Masked[T] {
	return m.unary(func(x T) T { return x - v }, nil)
}

// Pure: MulValue multiplies every element by the scalar v.
func (m *Masked[T]) MulValue(v T) *Masked[T] {
	return m.unary(func(x T) T { return x * v }, nil)
}

// Pure: DivValue divides every element by the scalar v (zero v masks, like Div).
func (m *Masked[T]) DivValue(v T) *Masked[T] {
	re, im := parts(v)
	out := m.unary(func(x T) T { return x / v }, nil)
	if re == 0 && im == 0 {
		for i := range out.mask {
			out.mask[i] = true
		}
	}
	return out
}

// Pure: Pow raises m to each element of b with the masks ORed.
func (m *Masked[T]) Pow(b *Masked[T]) *Masked[T] {
	return zip2(m, b, powValue[T])
}

// Pure: Abs returns the element-wise absolute value (real |x|, complex |z|).
func (m *Masked[T]) Abs() *Masked[T] {
	return m.unary(func(x T) T { return fromParts[T](math.Abs(realOf(x)), 0) }, nil)
}

// powValue computes x**y for any Number pair (real or complex).
func powValue[T numgo.Number](x, y T) T {
	xr, xi := parts(x)
	yr, yi := parts(y)
	if xi == 0 && yi == 0 {
		return fromParts[T](math.Pow(xr, yr), 0)
	}
	w := cmplxPow(complex(xr, xi), complex(yr, yi))
	return fromParts[T](real(w), imag(w))
}

// unary maps f over EVERY element including masked ones' data (numpy ma keeps
// computing on masked slots — they print as --, so their data is cosmetic);
// domain marks extra positions as masked (numpy ma domain: warning → mask).
func (m *Masked[T]) unary(f func(T) T, domain func(T) bool) *Masked[T] {
	out := m.a.Copy()
	d := out.Data()
	mask := slicesClone(m.mask)
	for i, v := range d {
		if !mask[i] && domain != nil && domain(v) {
			mask[i] = true
		}
		d[i] = f(v)
	}
	return newMasked(out, mask)
}
