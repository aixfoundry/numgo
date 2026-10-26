package random

import (
	"math"

	"github.com/aixfoundry/numgo"
)

// float32 sampling paths, ported from
// numpy/random/src/distributions/distributions.c (_f variants). Unlike the
// float64 paths, these genuinely compute in float32: the ziggurat uses the
// separate *_float tables (ziggurat_constants.h), next_float is
// (next_uint32 >> 8) * 2^-24 with a float multiply, and standard_gamma_f
// evaluates Marsaglia-Tsang with powf/logf/sqrtf on floats. The two rejection
// tails use double exp() (distributions.c:207 uses exp, not expf) — preserved.

// Single-precision math via float64 promotion through Go's math package;
// results can differ from correctly-rounded float32 kernels by 1-2 ULP.
func expf(x float32) float32 { return float32(math.Exp(float64(x))) }
func logf(x float32) float32 { return float32(math.Log(float64(x))) }
func log1pf(x float32) float32 {
	return float32(math.Log1p(float64(x)))
}
func powf(x, y float32) float32 { return float32(math.Pow(float64(x), float64(y))) }

// arrayF32 aliases the root float32 array (numgo.Array is generic over
// Number, which includes float32).
type arrayF32 = numgo.Array[float32]

// newF32 builds a C-order float32 array of n elements.
func newF32(n int) *arrayF32 { return numgo.Zeros[float32](n) }

// nextFloat: distributions.c:20 — (next_uint32 >> 8) * 2^-24, float arithmetic.
func (g *Generator) nextFloat() float32 {
	return float32(g.next32()>>8) * (1.0 / 16777216.0)
}

var signFloat = [2]float32{1.0, -1.0}

// StandardUniform32 is random_standard_uniform_f (distributions.c:32).
func (g *Generator) StandardUniform32() float32 { return g.nextFloat() }

// Random32 fills size float32 uniforms in [0,1); no size = one draw
// (numpy Generator.random(dtype=float32) → random_standard_uniform_fill_f).
func (g *Generator) Random32(size ...int) *arrayF32 {
	n := sizeOr1(size)
	out := newF32(n)
	for i := range n {
		out.Data()[i] = g.nextFloat()
	}
	return out
}

// standardExponentialUnlikelyF: distributions.c:91. expf tail;
// log1pf routes through log1pf.
func (g *Generator) standardExponentialUnlikelyF(idx uint8, x float32) float32 {
	if idx == 0 {
		// 1.0 - U to avoid log(0.0), GH 13361
		return zigguratExpRf - log1pf(-g.nextFloat())
	}
	// barrier: no arm64 FMA fusion
	probe := (feFloat[idx-1] - feFloat[idx]) * g.nextFloat()
	if probe+feFloat[idx] < expf(-x) {
		return x
	}
	return g.StandardExponential32Scalar()
}

// StandardExponential32Scalar is random_standard_exponential_f (distributions.c:105).
// ri is uint32: next_uint32>>1, idx = ri&0xff, ri >>= 8 — the mantissa consumes
// 23 bits, sign+idx the low 9.
func (g *Generator) StandardExponential32Scalar() float32 {
	ri := g.next32() >> 1
	idx := uint8(ri & 0xFF)
	ri >>= 8
	x := float32(ri) * weFloat[idx]
	if ri < keFloat[idx] {
		return x
	}
	return g.standardExponentialUnlikelyF(idx, x)
}

// StandardExponential32 fills size draws (numpy
// standard_exponential(dtype=float32) → random_standard_exponential_fill_f).
func (g *Generator) StandardExponential32(size ...int) *arrayF32 {
	n := sizeOr1(size)
	out := newF32(n)
	for i := range n {
		out.Data()[i] = g.StandardExponential32Scalar()
	}
	return out
}

// StandardNormal32Scalar is random_standard_normal_f (distributions.c:183).
// r = n23sb8: idx = r&0xff, rabs = (r>>9)&0x7fffff, sign = (r>>8)&1.
// The idx==0 tail uses double exp(-0.5*x*x) (distributions.c:207).
func (g *Generator) StandardNormal32Scalar() float32 {
	for {
		r := g.next32()
		idx := uint8(r & 0xff)
		rabs := (r >> 9) & 0x0007fffff
		x := float32(rabs) * wiFloat[idx] * signFloat[(r>>8)&0x1]
		if rabs < kiFloat[idx] {
			return x // 99.3% of the time
		}
		if idx == 0 {
			for {
				// 1.0 - U to avoid log(0.0), GH 13361
				xx := -zigguratNorInvRf * log1pf(-g.nextFloat())
				yy := -log1pf(-g.nextFloat())
				if yy+yy > xx*xx {
					if (rabs>>8)&0x1 != 0 {
						return -(zigguratNorRf + xx)
					}
					return zigguratNorRf + xx
				}
			}
		} else {
			// distributions.c:207 uses double exp here (not expf).
			// barrier: no arm64 FMA fusion
			probe := (fiFloat[idx-1] - fiFloat[idx]) * g.nextFloat()
			if probe+fiFloat[idx] <
				float32(math.Exp(-0.5*float64(x)*float64(x))) {
				return x
			}
		}
	}
}

// StandardNormal32 fills size draws (numpy
// standard_normal(dtype=float32) → random_standard_normal_fill_f).
func (g *Generator) StandardNormal32(size ...int) *arrayF32 {
	n := sizeOr1(size)
	out := newF32(n)
	for i := range n {
		out.Data()[i] = g.StandardNormal32Scalar()
	}
	return out
}

// StandardGamma32Scalar is random_standard_gamma_f (distributions.c:266).
// Entirely float32 arithmetic: powf/logf/sqrtf on floats via float64
// promotion (see the helpers above; sqrtf == float32(math.Sqrt) is exactly rounded).
func (g *Generator) StandardGamma32Scalar(shape float32) float32 {
	var b, c, U, V, X, Y float32
	switch {
	case shape == 1.0:
		return g.StandardExponential32Scalar()
	case shape == 0.0:
		return 0.0
	case shape < 1.0:
		for {
			U = g.nextFloat()
			V = g.StandardExponential32Scalar()
			if U <= 1.0-shape {
				X = powf(U, 1.0/shape)
				if X <= V {
					return X
				}
			} else {
				Y = -logf((1.0 - U) / shape)
				shapeY := shape * Y // barrier: no arm64 FMA fusion
				X = powf(1.0-shape+shapeY, 1.0/shape)
				if X <= V+Y {
					return X
				}
			}
		}
	default:
		b = shape - 1.0/3.0
		c = 1.0 / sqrtf(9.0*b)
		for {
			for {
				X = g.StandardNormal32Scalar()
				V = 1.0 + c*X
				if V > 0.0 {
					break
				}
			}
			V = V * V * V
			U = g.nextFloat()
			// barriers: original association kept, mul/add split into
			// statements (gc fuses a*b+c on arm64).
			x2 := X * X
			scaled := 0.0331 * x2
			quart := scaled * x2
			if U < 1.0-quart {
				return b * V
			}
			// logf(0.0) ok here (distributions.c:311)
			quad := 0.5 * x2
			bTail := b * (1.0 - V + logf(V))
			cmp := quad + bTail
			if logf(U) < cmp {
				return b * V
			}
		}
	}
}

// sqrtf is exactly-rounded float sqrt (IEEE, matches C sqrtf).
func sqrtf(x float32) float32 { return float32(math.Sqrt(float64(x))) }

// StandardGamma32 fills size draws (numpy
// standard_gamma(shape, dtype=float32) → cont_f path; shape is cast to
// float32 before sampling, numpy _common.pyx cont_f:1043).
func (g *Generator) StandardGamma32(shape float32, size ...int) *arrayF32 {
	n := sizeOr1(size)
	out := newF32(n)
	for i := range n {
		out.Data()[i] = g.StandardGamma32Scalar(shape)
	}
	return out
}

// Integers32 returns random int32 in [low, high) (numpy integers(dtype=int32):
// _rand_int32 → random_bounded_uint32_fill, Lemire on 32-bit draws via
// buffered_bounded_lemire_uint32, distributions.c:1308). numpy has no dtype
// argument on Generator.integers's Go-facing convenience here, so this is the
// explicit int32-width entry point.
func (g *Generator) Integers32(low, high int32, size ...int) *arrayI {
	// numpy scalar path works on the closed interval [low, high-1]; int32
	// ranges always fit in uint32.
	if high == math.MinInt32 {
		panicBounds("high", "<", int64(high)) // high-1 underflows; numpy raises
	}
	highClosed := high - 1
	if low > highClosed {
		panicBoundsError(int64(low), int64(high), false)
	}
	cnt := sizeOr1(size)
	out := newI(cnt)
	rng := uint32(highClosed) - uint32(low)
	off := uint32(low)
	mask := uint32(genMask(uint64(rng)))
	for i := range cnt {
		out.Data()[i] = int64(boundedUint32(off, rng, mask, g))
	}
	return out
}

// boundedUint32: random_buffered_bounded_uint32 (distributions.c:1462).
func boundedUint32(off, rng, mask uint32, g *Generator) uint32 {
	if rng == 0 {
		return off
	}
	if rng == 0xFFFFFFFF {
		// Lemire32 doesn't support rng = 0xFFFFFFFF.
		return off + g.next32()
	}
	return off + g.boundedLemireUint32(rng)
}

// Normal32 returns loc + scale*standard_normal_f as float32 (numpy has no
// dtype=float32 normal(); this is the float32 analogue for convenience and
// consumes the same stream as StandardNormal32).
func (g *Generator) Normal32(mu, sigma float32, size ...int) *arrayF32 {
	n := sizeOr1(size)
	out := newF32(n)
	for i := range n {
		term := sigma * g.StandardNormal32Scalar() // barrier: no arm64 FMA fusion
		out.Data()[i] = mu + term
	}
	return out
}

// Uniform32 returns lower + range*next_float as float32 (numpy has no dtype=float32
// uniform(); float32 analogue consuming the same stream as Random32).
func (g *Generator) Uniform32(low, high float32, size ...int) *arrayF32 {
	rng := high - low
	n := sizeOr1(size)
	out := newF32(n)
	for i := range n {
		term := rng * g.nextFloat() // barrier: no arm64 FMA fusion
		out.Data()[i] = low + term
	}
	return out
}

// Beta32 is pending: numpy has no float32 beta kernel (random_beta is
// double-only); a float32 variant would need a chosen downsampling policy.

// Gamma32 returns scale * standard_gamma_f, both float32 (numpy has no dtype
// argument on gamma(); float32 analogue of random_gamma_f, distributions.c:402).
func (g *Generator) Gamma32(shape, scale float32, size ...int) *arrayF32 {
	n := sizeOr1(size)
	out := newF32(n)
	for i := range n {
		out.Data()[i] = scale * g.StandardGamma32Scalar(shape)
	}
	return out
}
