package polynomial

import (
	"math"
	"slices"

	"github.com/aixfoundry/numgo"
)

// HermiteE is the probabilists' Hermite series c[0]*He0 + c[1]*He1 + ...
// (numpy polynomial.hermite_e, class HermiteE). Default domain/window [-1,1].
type HermiteE[T numgo.Float] struct{ Base[T] }

// hermeOps ports numpy polynomial.hermite_e's coefficient kernels.
type hermeOps[T numgo.Float] struct{}

// NewHermiteE builds a probabilists' Hermite series from coefficients.
func NewHermiteE[T numgo.Float](coef []T) *HermiteE[T] {
	return &HermiteE[T]{newBase(hermeOps[T]{}, coef)}
}

// NewHermiteEDomain is NewHermiteE with an explicit domain and window.
func NewHermiteEDomain[T numgo.Float](coef []T, domain, window [2]T) *HermiteE[T] {
	h := NewHermiteE(coef)
	h.domain, h.window = domain, window
	return h
}

// NewHermiteEFromRoots builds the monic HermiteE series with the given roots.
func NewHermiteEFromRoots[T numgo.Float](roots []T) *HermiteE[T] {
	return &HermiteE[T]{newBaseFromRoots(hermeOps[T]{}, roots)}
}

func (h *HermiteE[T]) derived(coef []T) *HermiteE[T] { return &HermiteE[T]{h.Base.with(coef)} }

func (hermeOps[T]) domainWindow() ([2]T, [2]T) { return [2]T{-1, 1}, [2]T{-1, 1} }
func (hermeOps[T]) basisName() string          { return "HermiteE" }

// val is hermeval: Clenshaw recurrence.
func (hermeOps[T]) val(x, c []T) []T {
	out := make([]T, len(x))
	n := len(c)
	for i, xv := range x {
		var c0, c1 T
		switch {
		case n == 1:
			c0 = c[0]
		case n == 2:
			c0, c1 = c[0], c[1]
		default:
			nd := n
			c0, c1 = c[n-2], c[n-1]
			for k := 3; k <= n; k++ {
				tmp := c0
				nd--
				c0 = c[n-k] - c1*T(nd-1)
				c1 = tmp + c1*xv
			}
		}
		out[i] = c0 + c1*xv
	}
	return out
}

// vander is hermevander: v[k] = x*v[k-1] - (k-1)*v[k-2].
func (hermeOps[T]) vander(x []T, deg int) []T {
	if deg < 0 {
		panicNegDeg()
	}
	dof := deg + 1
	v := make([]T, len(x)*dof)
	for i, xv := range x {
		v[i*dof] = 1
		if deg > 0 {
			v[i*dof+1] = xv
			for d := 2; d <= deg; d++ {
				v[i*dof+d] = v[i*dof+d-1]*xv - v[i*dof+d-2]*T(d-1)
			}
		}
	}
	return v
}

func (hermeOps[T]) add(c1, c2 []T) []T { return coefAdd(c1, c2) }
func (hermeOps[T]) sub(c1, c2 []T) []T { return coefSub(c1, c2) }

// mulx is hermemulx.
func (hermeOps[T]) mulx(c []T) []T {
	if len(c) == 1 && c[0] == 0 {
		return c
	}
	prd := make([]T, len(c)+1)
	prd[1] = c[0]
	for i := 1; i < len(c); i++ {
		prd[i+1] += c[i]
		prd[i-1] += c[i] * T(i)
	}
	return prd
}

// mul is hermemul.
func (z hermeOps[T]) mul(c1, c2 []T) []T {
	c1 = trimseq(slices.Clone(c1))
	c2 = trimseq(slices.Clone(c2))
	c, xs := c1, c2
	if len(c1) > len(c2) {
		c, xs = c2, c1
	}
	var c0, c1s []T
	switch len(c) {
	case 1:
		c0, c1s = scale(xs, c[0]), []T{0}
	case 2:
		c0, c1s = scale(xs, c[0]), scale(xs, c[1])
	default:
		nd := len(c)
		c0, c1s = scale(xs, c[len(c)-2]), scale(xs, c[len(c)-1])
		for i := 3; i <= len(c); i++ {
			tmp := c0
			nd--
			c0 = z.sub(scale(xs, c[len(c)-i]), scale(c1s, T(nd-1)))
			c1s = z.add(tmp, z.mulx(c1s))
		}
	}
	return z.add(c0, z.mulx(c1s))
}

func (z hermeOps[T]) div(c1, c2 []T) (quo, rem []T) { return divByMul(z.mul, c1, c2) }
func (z hermeOps[T]) pow(c []T, n int) []T          { return powCoef(z.mul, c, n) }

// der is hermeder: c'[j] = (j+1)*c[j+1].
func (hermeOps[T]) der(c []T, m int, scl T) []T {
	c = slices.Clone(c)
	n := len(c)
	if m == 0 {
		return c
	}
	if m >= n {
		return []T{0}
	}
	for range m {
		n--
		for j := range c {
			c[j] *= scl
		}
		dst := make([]T, n)
		for j := n; j >= 1; j-- {
			dst[j-1] = T(j) * c[j]
		}
		c = dst
	}
	return c
}

// integ is hermeint.
func (z hermeOps[T]) integ(c []T, m int, k []T, lbnd, scl T) []T {
	c = slices.Clone(c)
	for i := range m {
		n := len(c)
		for j := range c {
			c[j] *= scl
		}
		if n == 1 && c[0] == 0 {
			c[0] += k[i]
			continue
		}
		tmp := make([]T, n+1)
		tmp[1] = c[0]
		for j := 1; j < n; j++ {
			tmp[j+1] = c[j] / T(j+1)
		}
		tmp[0] += k[i] - z.val([]T{lbnd}, tmp)[0]
		c = tmp
	}
	return c
}

// line is hermeline: off + scl*He1.
func (hermeOps[T]) line(off, scl T) []T {
	if scl != 0 {
		return []T{off, scl}
	}
	return []T{off}
}

// companion is hermecompanion.
func (hermeOps[T]) companion(c []T) []T {
	c = trimseq(slices.Clone(c))
	n := len(c) - 1
	if n < 1 {
		panicDegAtLeast1()
	}
	if n == 1 {
		return []T{-c[0] / c[1]}
	}
	mat := make([]T, n*n)
	for k := 0; k <= n-2; k++ {
		v := T(math.Sqrt(f64(T(k + 1))))
		mat[k*n+k+1] = v   // top[k]
		mat[(k+1)*n+k] = v // bot[k]
	}
	// scl = cumprod(1, 1/sqrt(n-1), ..., 1/sqrt(1)) reversed.
	scl := make([]float64, n)
	scl[n-1] = 1
	acc := 1.0
	for i := 1; i < n; i++ {
		acc /= math.Sqrt(f64(T(n - i)))
		scl[n-1-i] = acc
	}
	for i := range n {
		mat[i*n+n-1] -= T(scl[i]) * c[i] / c[n]
	}
	return mat
}

func (z hermeOps[T]) fromroots(roots []T) []T { return fromRootsCoef(z.line, z.mul, roots) }

// roots is hermeroots: eigenvalues of the rotated companion matrix.
func (z hermeOps[T]) roots(c []T) []complex128 {
	c = trimseq(slices.Clone(c))
	switch len(c) {
	case 0, 1:
		return nil
	case 2:
		return []complex128{-cx(c[0] / c[1])}
	}
	return rootsEig(rot180(z.companion(c)), len(c)-1)
}

// compose advances the HermiteE recurrence with the series L.
func (z hermeOps[T]) compose(c []T, off, scl T) []T {
	L := z.line(off, scl)
	acc := []T{c[0]}
	if len(c) > 1 {
		acc = coefAdd(acc, z.mul([]T{c[1]}, L))
	}
	v2, v1 := []T{T(1)}, L
	for d := 2; d < len(c); d++ {
		v := coefSub(z.mul(v1, L), scale(v2, T(d-1)))
		acc = coefAdd(acc, z.mul([]T{c[d]}, v))
		v2, v1 = v1, v
	}
	return acc
}

// Add returns h + o in the HermiteE basis.
func (h *HermiteE[T]) Add(o *HermiteE[T]) *HermiteE[T] {
	return h.derived(coefAdd(h.coef, o.coef))
}

// Sub returns h - o in the HermiteE basis.
func (h *HermiteE[T]) Sub(o *HermiteE[T]) *HermiteE[T] {
	return h.derived(coefSub(h.coef, o.coef))
}

// Mul returns the HermiteE product.
func (h *HermiteE[T]) Mul(o *HermiteE[T]) *HermiteE[T] {
	return h.derived(h.ops.mul(h.coef, o.coef))
}

// Div returns the quotient and remainder of h / o.
func (h *HermiteE[T]) Div(o *HermiteE[T]) (*HermiteE[T], *HermiteE[T]) {
	q, r := h.ops.div(h.coef, o.coef)
	return h.derived(q), h.derived(r)
}

// Pow raises the series to the n-th power.
func (h *HermiteE[T]) Pow(n int) *HermiteE[T] { return h.derived(h.ops.pow(h.coef, n)) }

// Deriv returns the m-th derivative (m >= 1).
func (h *HermiteE[T]) Deriv(m int) *HermiteE[T] {
	_, scl := mapparms(h.domain, h.window)
	return h.derived(h.ops.der(h.coef, m, scl))
}

// Integ returns the m-th integral with integration constants k.
func (h *HermiteE[T]) Integ(m int, k ...T) *HermiteE[T] {
	off, scl := mapparms(h.domain, h.window)
	return h.derived(h.ops.integ(h.coef, m, padConsts(m, k), off, 1/scl))
}

// Trim drops trailing coefficients with |c| <= tol.
func (h *HermiteE[T]) Trim(tol T) *HermiteE[T] { return h.derived(trimcoef(h.coef, tol)) }

// Truncate drops coefficients beyond the first size.
func (h *HermiteE[T]) Truncate(size int) *HermiteE[T] {
	trimTruncate(size)
	if size >= len(h.coef) {
		return h.Copy()
	}
	return h.derived(slices.Clone(h.coef[:size]))
}

// Convert re-parameterizes the same function onto domain/window.
func (h *HermiteE[T]) Convert(domain, window [2]T) *HermiteE[T] {
	off, scl := h.convertParms(domain, window)
	q := h.derived(h.ops.compose(h.coef, off, scl))
	q.domain, q.window = domain, window
	return q
}

// Copy returns a deep copy.
func (h *HermiteE[T]) Copy() *HermiteE[T] { return h.derived(slices.Clone(h.coef)) }

// Fit least-squares fits degree deg to (x, y) with numpy default domain.
func (h *HermiteE[T]) Fit(x, y *numgo.Array[T], deg int) *HermiteE[T] {
	if x.Ndim() != 1 {
		panicDim1D("x")
	}
	coef, domain := fitCoef(h.ops, x.Copy().Data(), y.Copy().Data(), deg)
	q := h.derived(coef)
	q.domain = domain
	return q
}
