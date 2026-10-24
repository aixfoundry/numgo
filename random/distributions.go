package random

import (
	"github.com/aixfoundry/numgo"
	"math"
	"math/bits"

	"github.com/aixfoundry/numgo/internal/cmath"
)

// arrayF aliases the root float64 array.
type arrayF = numgo.Array[float64]

// arrayI is a concrete int64 vector. int64 is not a numgo.Number, so this is
// a minimal wrapper instead of an alias; Data exposes the storage.
type arrayI struct {
	data []int64
}

// Data exposes the backing slice.
func (a *arrayI) Data() []int64 { return a.data }

// newF builds a C-order float64 array of n elements.
func newF(n int) *arrayF { return numgo.Zeros[float64](n) }

// newI builds an int64 array of n elements.
func newI(n int) *arrayI { return &arrayI{data: make([]int64, n)} }

func sizeOr1(size []int) int {
	if len(size) == 0 {
		return 1
	}
	n := 1
	for _, d := range size {
		n *= d
	}
	return n
}

// Port of numpy/random/src/distributions/distributions.c. Call order against
// the bit generator is preserved exactly; bit-exactness depends on it.

var signDouble = [2]float64{1.0, -1.0}

// nextDouble: the bit generator's next_double slot (numpy bitgen.h:15) —
// PCG64: (next64 >> 11) * 2^-53; MT19937: mt19937_next_double (mt19937.h:56).
func (g *Generator) nextDouble() float64 {
	return g.bg.NextDouble()
}

// StandardUniform returns a double in [0, 1)
// (distributions.c random_standard_uniform).
func (g *Generator) StandardUniform() float64 { return g.nextDouble() }

// Random fills size draws of the standard uniform; no size = one draw.
func (g *Generator) Random(size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = g.nextDouble()
	}
	return out
}

// standardExponentialUnlikely: distributions.c:54.
func (g *Generator) standardExponentialUnlikely(idx uint8, x float64) float64 {
	if idx == 0 {
		// 1.0 - U to avoid log(0.0), GH 13361
		return expR - cmath.Log1p(-g.nextDouble())
	} else if (fe[idx-1]-fe[idx])*g.nextDouble()+fe[idx] < cmath.Exp(-x) {
		return x
	}
	return g.StandardExponentialScalar()
}

// StandardNormal returns a single standard normal draw.
func (g *Generator) StandardNormalScalar() float64 {
	for {
		r := g.bg.Next64()
		idx := int(r & 0xff)
		r >>= 8
		rabs := (r >> 1) & 0x000fffffffffffff
		x := float64(rabs) * wi[idx] * signDouble[r&0x1]
		if rabs < ki[idx] {
			return x
		}
		if idx == 0 {
			for {
				xx := -norInvR * cmath.Log1p(-g.nextDouble())
				yy := -cmath.Log1p(-g.nextDouble())
				if yy+yy > xx*xx {
					if (rabs>>8)&0x1 != 0 {
						return -(norR + xx)
					}
					return norR + xx
				}
			}
		} else {
			if (fi[idx-1]-fi[idx])*g.nextDouble()+fi[idx] < cmath.Exp(-0.5*x*x) {
				return x
			}
		}
	}
}

// StandardNormal fills size draws (distributions.c random_standard_normal).
func (g *Generator) StandardNormal(size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = g.StandardNormalScalar()
	}
	return out
}

// StandardExponential returns a single standard exponential draw.
func (g *Generator) StandardExponentialScalar() float64 {
	ri := g.bg.Next64() >> 3
	idx := uint8(ri & 0xFF)
	ri >>= 8
	x := float64(ri) * we[idx]
	if ri < ke[idx] {
		return x
	}
	return g.standardExponentialUnlikely(idx, x)
}

// StandardExponential fills size draws (distributions.c:68 ziggurat).
func (g *Generator) StandardExponential(size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = g.StandardExponentialScalar()
	}
	return out
}

// Normal: loc + scale*standard_normal (distributions.c:386).
func (g *Generator) Normal(loc, scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = loc + scale*g.StandardNormalScalar()
	}
	return out
}

// Uniform: lower + range*next_double where range = high-lower
// (distributions.c:394; numpy Generator.uniform(low, high)).
func (g *Generator) Uniform(lower, high float64, size ...int) *arrayF {
	rng := high - lower
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = lower + rng*g.nextDouble()
	}
	return out
}

// Exponential: scale * standard_exponential (distributions.c:390).
func (g *Generator) Exponential(scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = scale * g.StandardExponentialScalar()
	}
	return out
}

// StandardGamma returns a single draw (scalar convenience).
func (g *Generator) StandardGammaScalar(shape float64) float64 {
	var b, c, U, V, X, Y float64
	switch {
	case shape == 1.0:
		return g.StandardExponentialScalar()
	case shape == 0.0:
		return 0.0
	case shape < 1.0:
		for {
			U = g.nextDouble()
			V = g.StandardExponentialScalar()
			if U <= 1.0-shape {
				X = cmath.Pow(U, 1.0/shape)
				if X <= V {
					return X
				}
			} else {
				Y = -cmath.Log((1 - U) / shape)
				X = cmath.Pow(1.0-shape+shape*Y, 1.0/shape)
				if X <= V+Y {
					return X
				}
			}
		}
	default:
		b = shape - 1.0/3.0
		c = 1.0 / math.Sqrt(9*b)
		for {
			for {
				X = g.StandardNormalScalar()
				V = 1.0 + c*X
				if V > 0.0 {
					break
				}
			}
			V = V * V * V
			U = g.nextDouble()
			if U < 1.0-0.0331*(X*X)*(X*X) {
				return b * V
			}
			// log(0.0) ok here
			if cmath.Log(U) < 0.5*X*X+b*(1.0-V+cmath.Log(V)) {
				return b * V
			}
		}
	}
}

// StandardGammaArray fills size draws.
func (g *Generator) StandardGamma(shape float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = g.StandardGammaScalar(shape)
	}
	return out
}

// Gamma: scale * standard_gamma (distributions.c:398).
func (g *Generator) Gamma(shape, scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = scale * g.StandardGammaScalar(shape)
	}
	return out
}

const betaTinyThreshold = 3e-103

// Beta: distributions.c:411. Assumes a != 0 and b != 0.
// Beta fills size draws (distributions.c:411). Assumes a != 0 and b != 0.
func (g *Generator) Beta(a, b float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = g.betaScalar(a, b)
	}
	return out
}

func (g *Generator) betaScalar(a, b float64) float64 {
	if a <= 1.0 && b <= 1.0 {
		if a < betaTinyThreshold && b < betaTinyThreshold {
			U := g.nextDouble()
			if (a+b)*U < a {
				return 1
			}
			return 0
		}
		// Johnk's algorithm
		for {
			U := g.nextDouble()
			V := g.nextDouble()
			X := cmath.Pow(U, 1.0/a)
			Y := cmath.Pow(V, 1.0/b)
			XpY := X + Y
			if XpY <= 1.0 && U+V > 0.0 {
				if X > 0 && Y > 0 {
					return X / XpY
				}
				// X or Y underflowed; work in logs (distributions.c:442).
				logX := cmath.Log(U) / a
				logY := cmath.Log(V) / b
				delta := logX - logY
				if delta > 0 {
					return cmath.Exp(-cmath.Log1p(cmath.Exp(-delta)))
				}
				return cmath.Exp(delta - cmath.Log1p(cmath.Exp(delta)))
			}
		}
	}
	Ga := g.StandardGammaScalar(a)
	Gb := g.StandardGammaScalar(b)
	return Ga / (Ga + Gb)
}

// loggam: random_loggam (distributions.c:344).
func loggam(x float64) float64 {
	a := [10]float64{8.333333333333333e-02, -2.777777777777778e-03,
		7.936507936507937e-04, -5.952380952380952e-04,
		8.417508417508418e-04, -1.917526917526918e-03,
		6.410256410256410e-03, -2.955065359477124e-02,
		1.796443723688307e-01, -1.39243221690590e+00}
	var n int64
	if x == 1.0 || x == 2.0 {
		return 0.0
	} else if x < 7.0 {
		n = int64(7 - x)
	}
	x0 := x + float64(n)
	x2 := (1.0 / x0) * (1.0 / x0)
	lg2pi := 1.8378770664093453e+00
	gl0 := a[9]
	for k := 8; k >= 0; k-- {
		gl0 *= x2
		gl0 += a[k]
	}
	gl := gl0/x0 + 0.5*lg2pi + (x0-0.5)*cmath.Log(x0) - x0
	if x < 7.0 {
		for k := int64(1); k <= n; k++ {
			gl -= cmath.Log(x0 - 1.0)
			x0 -= 1.0
		}
	}
	return gl
}

// poissonMult: distributions.c:548.
func (g *Generator) poissonMult(lam float64) int64 {
	enlam := cmath.Exp(-lam)
	var X int64
	prod := 1.0
	for {
		U := g.nextDouble()
		prod *= U
		if prod > enlam {
			X++
		} else {
			return X
		}
	}
}

// poissonPTRS: transformed rejection, distributions.c:573.
func (g *Generator) poissonPTRS(lam float64) int64 {
	slam := math.Sqrt(lam)
	loglam := cmath.Log(lam)
	b := 0.931 + 2.53*slam
	a := -0.059 + 0.02483*b
	invalpha := 1.1239 + 1.1328/(b-3.4)
	vr := 0.9277 - 3.6224/(b-2)
	for {
		U := g.nextDouble() - 0.5
		V := g.nextDouble()
		us := 0.5 - math.Abs(U)
		k := int64(math.Floor((2*a/us+b)*U + lam + 0.43))
		if us >= 0.07 && V <= vr {
			return k
		}
		if k < 0 || (us < 0.013 && V > us) {
			continue
		}
		if cmath.Log(V)+cmath.Log(invalpha)-cmath.Log(a/(us*us)+b) <=
			-lam+float64(k)*loglam-loggam(float64(k)+1) {
			return k
		}
	}
}

// Poisson: distributions.c:604.
func (g *Generator) Poisson(lam float64, size ...int) *arrayI {
	n := sizeOr1(size)
	out := newI(n)
	for i := 0; i < n; i++ {
		switch {
		case lam >= 10:
			out.Data()[i] = g.poissonPTRS(lam)
		case lam == 0:
			out.Data()[i] = 0
		default:
			out.Data()[i] = g.poissonMult(lam)
		}
	}
	return out
}

// binomialState caches BTPE parameters across calls (binomial_t).
type binomialState struct {
	hasBinomial          bool
	nsave                int64
	psave                float64
	r, q, fm, p1, xm, xl float64
	xr, c, laml, lamr    float64
	p2, p3, p4           float64
	m                    int64
}

// binomialBTPE: distributions.c:620.
func (g *Generator) binomialBTPE(n int64, p float64, st *binomialState) int64 {
	var a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x float64
	var y, k, i int64

	if !st.hasBinomial || st.nsave != n || st.psave != p {
		st.nsave = n
		st.psave = p
		st.hasBinomial = true
		st.r = math.Min(p, 1.0-p)
		st.q = 1.0 - st.r
		st.fm = float64(n)*st.r + st.r
		st.m = int64(math.Floor(st.fm))
		st.p1 = math.Floor(2.195*math.Sqrt(float64(n)*st.r*st.q)-4.6*st.q) + 0.5
		st.xm = float64(st.m) + 0.5
		st.xl = st.xm - st.p1
		st.xr = st.xm + st.p1
		st.c = 0.134 + 20.5/(15.3+float64(st.m))
		a = (st.fm - st.xl) / (st.fm - st.xl*st.r)
		st.laml = a * (1.0 + a/2.0)
		a = (st.xr - st.fm) / (st.xr * st.q)
		st.lamr = a * (1.0 + a/2.0)
		st.p2 = st.p1 * (1.0 + 2.0*st.c)
		st.p3 = st.p2 + st.c/st.laml
		st.p4 = st.p3 + st.c/st.lamr
	}
	r, q := st.r, st.q
	m := st.m
	p1, xm, xl, xr := st.p1, st.xm, st.xl, st.xr
	c, laml, lamr := st.c, st.laml, st.lamr
	p2, p3, p4 := st.p2, st.p3, st.p4

step10:
	nrq = float64(n) * r * q
	u = g.nextDouble() * p4
	v = g.nextDouble()
	if u > p1 {
		goto step20
	}
	y = int64(math.Floor(xm - p1*v + u))
	goto step60

step20:
	if u > p2 {
		goto step30
	}
	x = xl + (u-p1)/c
	v = v*c + 1.0 - math.Abs(float64(m)-x+0.5)/p1
	if v > 1.0 {
		goto step10
	}
	y = int64(math.Floor(x))
	goto step50

step30:
	if u > p3 {
		goto step40
	}
	y = int64(math.Floor(xl + cmath.Log(v)/laml))
	if y < 0 || v == 0.0 {
		goto step10
	}
	v = v * (u - p2) * laml
	goto step50

step40:
	y = int64(math.Floor(xr - cmath.Log(v)/lamr))
	if y > n || v == 0.0 {
		goto step10
	}
	v = v * (u - p3) * lamr

step50:
	k = abs64(y - m)
	if k > 20 && float64(k) < (nrq/2.0-1) {
		goto step52
	}
	s = r / q
	a = s * (float64(n) + 1)
	F = 1.0
	if m < y {
		for i = m + 1; i <= y; i++ {
			F *= (a/float64(i) - s)
		}
	} else if m > y {
		for i = y + 1; i <= m; i++ {
			F /= (a/float64(i) - s)
		}
	}
	if v > F {
		goto step10
	}
	goto step60

step52:
	rho = (float64(k) / nrq) * ((float64(k)*(float64(k)/3.0+0.625)+0.16666666666666666)/nrq + 0.5)
	t = -float64(k) * float64(k) / (2 * nrq)
	A = cmath.Log(v)
	if A < t-rho {
		goto step60
	}
	if A > t+rho {
		goto step10
	}
	x1 = float64(y) + 1
	f1 = float64(m) + 1
	z = float64(n) + 1 - float64(m)
	w = float64(n) - float64(y) + 1
	x2 = x1 * x1
	f2 = f1 * f1
	z2 = z * z
	w2 = w * w
	if A > (xm*cmath.Log(f1/x1) + (float64(n)-float64(m)+0.5)*cmath.Log(z/w) +
		float64(y-m)*cmath.Log(w*r/(x1*q)) +
		(13860.-(462.-(132.-(99.-140./f2)/f2)/f2)/f2)/f1/166320. +
		(13860.-(462.-(132.-(99.-140./z2)/z2)/z2)/z2)/z/166320. -
		(13860.-(462.-(132.-(99.-140./x2)/x2)/x2)/x2)/x1/166320. -
		(13860.-(462.-(132.-(99.-140./w2)/w2)/w2)/w2)/w/166320.) {
		goto step10
	}

step60:
	if p > 0.5 {
		y = n - y
	}
	return y
}

// binomialInversion: distributions.c:769.
func (g *Generator) binomialInversion(n int64, p float64, st *binomialState) int64 {
	var X int64
	if !st.hasBinomial || st.nsave != n || st.psave != p {
		st.nsave = n
		st.psave = p
		st.hasBinomial = true
		st.q = 1.0 - p
		st.r = cmath.Exp(float64(n) * cmath.Log1p(-p))
		st.c = float64(n) * p
		st.m = min64(n, int64(math.Min(float64(n), st.c+10.0*math.Sqrt(st.c*st.q+1))))
	}
	q, qn := st.q, st.r
	bound := st.m
	X = 0
	px := qn
	U := g.nextDouble()
	for U > px {
		X++
		if X > bound {
			X = 0
			px = qn
			U = g.nextDouble()
		} else {
			U -= px
			px = (float64(n-X+1) * p * px) / (float64(X) * q)
		}
	}
	return X
}

// Binomial: dispatch per distributions.c:806.
func (g *Generator) Binomial(n int64, p float64, size ...int) *arrayI {
	cnt := sizeOr1(size)
	out := newI(cnt)
	st := &binomialState{}
	for i := 0; i < cnt; i++ {
		out.Data()[i] = g.binomialScalar(n, p, st)
	}
	return out
}

func (g *Generator) binomialScalar(n int64, p float64, st *binomialState) int64 {
	if n == 0 || p == 0.0 {
		return 0
	}
	if p <= 0.5 {
		if p*float64(n) <= 30.0 {
			return g.binomialInversion(n, p, st)
		}
		return g.binomialBTPE(n, p, st)
	}
	q := 1.0 - p
	if q*float64(n) <= 30.0 {
		return n - g.binomialInversion(n, q, st)
	}
	return n - g.binomialBTPE(n, q, st)
}

// genMask: smallest bit mask >= max (distributions.c:1110).
func genMask(max uint64) uint64 {
	mask := max
	mask |= mask >> 1
	mask |= mask >> 2
	mask |= mask >> 4
	mask |= mask >> 8
	mask |= mask >> 16
	mask |= mask >> 32
	return mask
}

// boundedLemireUint64: Lemire rejection, distributions.c:1246.
func (g *Generator) boundedLemireUint64(rng uint64) uint64 {
	rngExcl := rng + 1
	hi, lo := bits.Mul64(g.bg.Next64(), rngExcl)
	if lo < rngExcl {
		threshold := (math.MaxUint64 - rng) % rngExcl
		for lo < threshold {
			hi, lo = bits.Mul64(g.bg.Next64(), rngExcl)
		}
	}
	return hi
}

// boundedMaskedUint64: distributions.c:1150.
func (g *Generator) boundedMaskedUint64(rng, mask uint64) uint64 {
	var val uint64
	for {
		val = g.bg.Next64() & mask
		if val <= rng {
			return val
		}
	}
}

// boundedLemireUint32: Lemire rejection on 32-bit draws
// (distributions.c buffered_bounded_lemire_uint32:1291; numpy's uint16/8 bit
// buffers only matter for 8/16-bit dtypes, which this batch does not port).
func (g *Generator) boundedLemireUint32(rng uint32) uint32 {
	rngExcl := rng + 1
	m := uint64(g.next32()) * uint64(rngExcl)
	leftover := uint32(m)
	if leftover < rngExcl {
		threshold := (math.MaxUint32 - rng) % rngExcl
		for leftover < threshold {
			m = uint64(g.next32()) * uint64(rngExcl)
			leftover = uint32(m)
		}
	}
	return uint32(m >> 32)
}

// boundedMaskedUint32: masked rejection (distributions.c:1162).
func (g *Generator) boundedMaskedUint32(rng, mask uint32) uint32 {
	for {
		val := g.next32() & mask
		if val <= rng {
			return val
		}
	}
}

// next32 aliases the bit generator draw; numpy's uint16/8 bit buffers are not
// needed by any ported distribution (every Lemire/masked 32-bit path consumes
// whole draws).
func (g *Generator) next32() uint32 { return g.bg.Next32() }

// randomBoundedUint64: distributions.c:1455 (scalar path).
func (g *Generator) randomBoundedUint64(off, rng, mask uint64, useMasked bool) uint64 {
	switch {
	case rng == 0:
		return off
	case rng <= 0xFFFFFFFF:
		if rng == 0xFFFFFFFF {
			return off + uint64(g.bg.Next32())
		}
		if useMasked {
			return off + uint64(g.boundedMaskedUint32(uint32(rng), uint32(mask)))
		}
		return off + uint64(g.boundedLemireUint32(uint32(rng)))
	case rng == math.MaxUint64:
		return off + g.bg.Next64()
	default:
		if useMasked {
			return off + g.boundedMaskedUint64(rng, mask)
		}
		return off + g.boundedLemireUint64(rng)
	}
}

// randomInterval: legacy shuffle path (distributions.c:1082).
func (g *Generator) randomInterval(max uint64) uint64 {
	if max == 0 {
		return 0
	}
	mask := genMask(max)
	if max <= 0xFFFFFFFF {
		for {
			v := uint64(g.bg.Next32()) & mask
			if v <= max {
				return v
			}
		}
	}
	for {
		v := g.bg.Next64() & mask
		if v <= max {
			return v
		}
	}
}

// Integers returns random int64 in [low, high) via Lemire
// (numpy Generator.integers default dtype int64; _bounded_integers.pyx.in
// _rand_int64 with use_masked=False). Single-arg low means [0, low).
func (g *Generator) Integers(low, high int64, size ...int) *arrayI {
	return g.integersRange(low, high, false, size...)
}

// IntegersEndpoint samples [low, high] (numpy endpoint=True).
func (g *Generator) IntegersEndpoint(low, high int64, size ...int) *arrayI {
	return g.integersRange(low, high, true, size...)
}

func (g *Generator) integersRange(low, high int64, closed bool, size ...int) *arrayI {
	if !closed {
		// subtract 1 since the internal generator is on the closed interval
		if high == math.MinInt64 {
			panicBounds("high", "<", high) // high-1 underflows; numpy raises
		}
		high--
	}
	if low > high {
		panicBoundsError(low, high+1, closed)
	}
	cnt := sizeOr1(size)
	out := newI(cnt)
	rng := uint64(high) - uint64(low)
	off := uint64(low)
	mask := genMask(rng)
	for i := 0; i < cnt; i++ {
		out.Data()[i] = int64(g.randomBoundedUint64(off, rng, mask, false))
	}
	return out
}

func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
