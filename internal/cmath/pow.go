// Pow: port of the GNU C Library __ieee754_pow (sysdeps/ieee754/dbl-64/
// e_pow.c), which derives from ARM optimized-routines (MIT), Copyright (c)
// 2018-2019 Arm Limited; GNU C Library portions Copyright (C) 2018-2024 Free
// Software Foundation, Inc.
//
// Worst-case error of the reference: 0.54 ULP; arithmetic order matches the
// glibc 2.39 x86-64 FMA code path (fused r in log_inline, fused y*hi+tail
// recombination, non-fused poly), validated bit-exactly against the platform
// libm (see cmath_test.go).
package cmath

import "math"

const (
	powLn2hi = 0x1.62e42fefa3800p-1
	powLn2lo = 0x1.ef35793c76730p-45
	powOFF   = 0x3fe6955500000000 // bit pattern of 0x1.69555p-1-ish boundary
	// powSignBias biases the scale exponent by 0x800<<7; set for x<0 with odd y.
	powSignBias = uint32(0x800 << 7)
)

// checkint returns 0 if y is not an integer, 1 if odd, 2 if even
// (iy is the bit pattern of a non-zero finite value).
func checkint(iy uint64) int {
	e := (iy >> 52) & 0x7ff
	if e < 0x3ff {
		return 0
	}
	if e > 0x3ff+52 {
		return 2
	}
	if iy&((1<<(0x3ff+52-e))-1) != 0 {
		return 0
	}
	if iy&(1<<(0x3ff+52-e)) != 0 {
		return 1
	}
	return 2
}

// zeroinfnan reports whether i is the bit pattern of 0, Inf or NaN.
func zeroinfnan(i uint64) bool {
	return 2*i-1 >= 2*math.Float64bits(math.Inf(1))-1
}

// powLogInline computes hi+tail = log(x) for normalized ix.
func powLogInline(ix uint64) (hi, tail float64) {
	tmp := ix - powOFF
	i := uint32(tmp>>45) & 0x7f
	k := int64(tmp) >> 52
	iz := ix - (tmp & (0xfff << 52))
	kd := float64(k)
	z := math.Float64frombits(iz)

	invc := powLogTab[i][0]
	logc := powLogTab[i][1]
	logctail := powLogTab[i][2]

	// r = z/c - 1, exactly representible by table construction.
	r := math.FMA(z, invc, -1.0)

	t1 := kd*powLn2hi + logc
	t2 := t1 + r
	lo1 := kd*powLn2lo + logctail
	lo2 := t1 - t2 + r

	ar := -0.5 * r
	ar2 := r * ar
	ar3 := r * ar2
	hi = t2 + ar2
	lo3 := math.FMA(ar, r, -ar2)
	lo4 := t2 - hi + ar2
	// Scaled polynomial coefficients (from e_pow_log_data.c).
	p := ar3 * (-0x1.5555555555560p-1 +
		r*0x1.0000000000006p-1 +
		ar2*(0x1.999999959554ep-1+
			r*-0x1.555555529a47ap-1+
			ar2*(-0x1.2495b9b4845e9p+0+
				r*0x1.0002b8b263fc3p+0)))
	lo := lo1 + lo2 + lo3 + lo4 + p
	y := hi + lo
	return y, hi - y + lo
}

// powExpInline computes sign*exp(x+xtail) where |xtail| < 2^-8/128.
func powExpInline(x, xtail float64, signBias uint32) float64 {
	ux := math.Float64bits(x)
	abstop := uint32(ux>>52) & 0x7ff
	if abstop >= 0x409 && abstop <= 0x7fe {
		// |x| >= 1024: definite overflow or underflow.
		if ux>>63 != 0 {
			if signBias != 0 {
				return math.Copysign(0, -1)
			}
			return 0
		}
		y := math.Inf(1)
		if signBias != 0 {
			return -y
		}
		return y
	}
	if abstop-0x3c9 >= 0x408-0x3c9 {
		if abstop-0x3c9 >= 0x80000000 {
			// |x| < 0x1p-1022: avoid spurious underflow.
			one := 1.0 + x
			if signBias != 0 {
				return -one
			}
			return one
		}
		if abstop >= 0x7ff {
			// ±Inf and NaN.
			if ux == 0xfff0000000000000 {
				if signBias != 0 {
					return math.Copysign(0, -1)
				}
				return 0
			}
			one := 1.0 + x
			if signBias != 0 {
				return -one
			}
			return one
		}
		// |x| in [512, 1024): the scale exponent may leave the normal
		// range; route through the specialcase (glibc handles overflow to
		// ±Inf there).
		z := expInvln2N * x
		kd := z + expShift
		ki := math.Float64bits(kd)
		kd -= expShift
		r := x + kd*expNegLn2hiN + kd*expNegLn2loN
		r += xtail
		idx := ki % 128
		tail := math.Float64frombits(expTab[idx][0])
		sbits := expTab[idx][1] + ((ki + uint64(signBias)) << (52 - 7))
		r2 := r * r
		tmp := tail + r + r2*(expC2+r*expC3) + r2*r2*(expC4+r*expC5)
		return expSpecialcase(tmp, sbits, ki)
	}
	z := expInvln2N * x
	kd := z + expShift
	ki := math.Float64bits(kd)
	kd -= expShift
	r := x + kd*expNegLn2hiN + kd*expNegLn2loN
	r += xtail
	idx := ki % 128
	tail := math.Float64frombits(expTab[idx][0])
	sbits := expTab[idx][1] + ((ki + uint64(signBias)) << (52 - 7))
	r2 := r * r
	tmp := tail + r + r2*(expC2+r*expC3) + r2*r2*(expC4+r*expC5)
	scale := math.Float64frombits(sbits)
	return math.FMA(scale, tmp, scale)
}

// Pow returns x**y, bit-identical to C pow on glibc x86-64.
func Pow(x, y float64) float64 {
	var signBias uint32
	ix := math.Float64bits(x)
	iy := math.Float64bits(y)
	topx := uint32(ix >> 52)
	topy := uint32(iy >> 52)
	if topx-0x001 >= 0x7ff-0x001 || (topy&0x7ff)-0x3be >= 0x43e-0x3be {
		// Special cases: (x < 0x1p-126 or inf or nan) or
		// (|y| < 0x1p-65 or |y| >= 0x1p63 or nan).
		if zeroinfnan(iy) {
			if 2*iy == 0 {
				return 1.0 // pow(±0, ±0)=1 per glibc main branch
			}
			if ix == math.Float64bits(1.0) {
				return 1.0
			}
			if 2*ix > 2*math.Float64bits(math.Inf(1)) || 2*iy > 2*math.Float64bits(math.Inf(1)) {
				return x + y // NaN propagation
			}
			if 2*ix == 2*math.Float64bits(1.0) {
				return 1.0
			}
			if (2*ix < 2*math.Float64bits(1.0)) == (iy>>63 == 0) {
				return 0.0
			}
			return y * y
		}
		if zeroinfnan(ix) {
			x2 := x * x
			if ix>>63 != 0 && checkint(iy) == 1 {
				x2 = -x2
				signBias = 1
			}
			if 2*ix == 0 && iy>>63 != 0 {
				// pow(±0, -y) = +Inf
				return math.Inf(1)
			}
			if iy>>63 != 0 {
				return 1 / x2
			}
			return x2
		}
		// Finite x < 0.
		if ix>>63 != 0 {
			yint := checkint(iy)
			if yint == 0 {
				return math.NaN()
			}
			if yint == 1 {
				signBias = powSignBias
			}
			ix &= 0x7fffffffffffffff
			topx &= 0x7ff
		}
		if (topy&0x7ff)-0x3be >= 0x43e-0x3be {
			// |y| tiny or huge with finite non-zero x.
			if ix == math.Float64bits(1.0) {
				return 1.0
			}
			if (topy & 0x7ff) < 0x3be {
				// |y| < 2^-65: x**y ~= 1 + y*log(x).
				if ix > math.Float64bits(1.0) {
					return 1.0 + y
				}
				return 1.0 - y
			}
			cond := (ix > math.Float64bits(1.0)) == (topy < 0x800)
			sg := 1.0
			if signBias != 0 {
				sg = -1.0
			}
			if cond {
				yv := math.Inf(1)
				if signBias != 0 {
					return -yv
				}
				return yv
			}
			_ = sg
			if signBias != 0 {
				return math.Copysign(0, -1)
			}
			return 0
		}
		if topx == 0 {
			// Normalize subnormal x.
			ix = math.Float64bits(x * 0x1p52)
			ix &= 0x7fffffffffffffff
			ix -= 52 << 52
		}
	}

	hi, lo := powLogInline(ix)
	ehi := y * hi
	elo := y*lo + math.FMA(y, hi, -ehi)
	return powExpInline(ehi, elo, signBias)
}
