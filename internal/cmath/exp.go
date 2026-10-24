// Exp, Log and Log1p: ports of the GNU C Library x86-64 implementations
// (sysdeps/ieee754/dbl-64/e_exp.c, e_log.c and s_log1p.c), which derive from
// ARM optimized-routines (MIT), Copyright (c) 2018-2019 Arm Limited; the GNU C
// Library portions are Copyright (C) 2018-2024 Free Software Foundation, Inc.;
// s_log1p derives from fdlibm, Copyright (C) 1993 Sun Microsystems, Inc.
//
// Arithmetic order matches the glibc 2.39 x86-64 code paths, validated
// bit-exactly against the platform libm (see cmath_test.go).
package cmath

import "math"

const (
	expInvln2N   = 0x1.71547652b82fep+7 // 128/ln2
	expShift     = 0x1.8p52
	expNegLn2hiN = -0x1.62e42fefa0000p-8  // -ln2/128, hi
	expNegLn2loN = -0x1.cf79abc9e3b3ap-47 // -ln2/128, lo
	expC2        = 0x1.ffffffffffdbdp-2
	expC3        = 0x1.555555555543cp-3
	expC4        = 0x1.55555cf172b91p-5
	expC5        = 0x1.1111167a4d017p-7
)

// expCore implements the glibc x86-64 __ieee754_exp body. signBias is 0 or
// powSignBias; xtail is the low-order correction (only used by Pow). Results
// whose scale exponent could leave the normal range (|x| >= 512 or subnormal
// scale) go through expSpecialcase, exactly as in glibc.
func expCore(x, xtail float64, signBias uint32) float64 {
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
	ux := math.Float64bits(x)
	abstop := uint32(ux>>52) & 0x7ff
	switch {
	case abstop-0x3c9 >= 0x408-0x3c9 && abstop-0x3c9 < 0x80000000 && abstop < 0x7ff:
		if abstop == 0x408 {
			// |x| in [512, 1024): the scale exponent may leave the normal
			// range; route through the specialcase (handles ±Inf overflow).
			return expSpecialcase(tmp, sbits, ki)
		}
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
	scale := math.Float64frombits(sbits)
	return math.FMA(scale, tmp, scale)
}

// expSpecialcase is the glibc specialcase for results that may overflow or
// underflow in scale*(1+tmp) computation.
func expSpecialcase(tmp float64, sbits, ki uint64) float64 {
	var scale, y float64
	if ki&0x80000000 == 0 {
		// k > 0: exponent of scale might have overflowed by <= 460.
		sbits -= 1009 << 52
		scale = math.Float64frombits(sbits)
		return 0x1p1009 * math.FMA(scale, tmp, scale)
	}
	// k < 0: need special care in the subnormal range.
	sbits += 1022 << 52
	scale = math.Float64frombits(sbits)
	y = scale + scale*tmp
	if math.Abs(y) < 1.0 {
		// Round y to full precision before scaling into the subnormal range.
		var hi, lo float64
		one := 1.0
		if y < 0 {
			one = -1.0
		}
		lo = scale - y + scale*tmp
		hi = one + y
		lo = one - hi + y + lo
		y = hi + lo - one
		if y == 0 {
			y = math.Float64frombits(sbits & 0x8000000000000000)
		}
	}
	return 0x1p-1022 * y
}

// Exp returns e**x, bit-identical to C exp on glibc x86-64.
func Exp(x float64) float64 {
	ux := math.Float64bits(x)
	abstop := uint32(ux>>52) & 0x7ff
	if abstop-0x3c9 >= 0x408-0x3c9 {
		if abstop-0x3c9 >= 0x80000000 {
			// |x| < 0x1p-1022: avoid spurious underflow; 0 is common.
			return 1.0 + x
		}
		if abstop >= 0x7ff {
			// ±Inf and NaN.
			if ux == 0xfff0000000000000 {
				return 0
			}
			return 1.0 + x
		}
	}
	return expCore(x, 0, 0)
}

const (
	logLn2hi = 0x1.62e42fefa3800p-1
	logLn2lo = 0x1.ef35793c76730p-45
	logOFF   = 0x3fe6000000000000 // bit pattern of 0.875
)

// Log returns the natural logarithm, bit-identical to C log on glibc x86-64.
// Port of __ieee754_log (e_log.c): fused argument reduction (r = fma(z,invc,-1)),
// fused Horner polynomial, non-fused w = kd*Ln2hi + logc — the order used by
// glibc 2.39 on x86-64.
func Log(x float64) float64 {
	ix := math.Float64bits(x)
	// Handle close to 1.0 inputs separately.
	lo1 := math.Float64bits(1.0 - 0x1p-4)
	hi1 := math.Float64bits(1.0 + 0x1.09p-4)
	if ix-lo1 < hi1-lo1 {
		if ix == math.Float64bits(1.0) {
			return 0
		}
		r := x - 1.0
		r2 := r * r
		r3 := r * r2
		b1 := 0x1.5555555555577p-2
		b2 := -0x1.ffffffffffdcbp-3
		b3 := 0x1.999999995dd0cp-3
		b4 := -0x1.55555556745a7p-3
		b5 := 0x1.24924a344de3p-3
		b6 := -0x1.fffffa4423d65p-4
		b7 := 0x1.c7184282ad6cap-4
		b8 := -0x1.999eb43b068ffp-4
		b9 := 0x1.78182f7afd085p-4
		b10 := -0x1.5521375d145cdp-4
		y := r3 * (b1 + r*b2 + r2*b3 + r3*(b4+r*b5+r2*b6+r3*(b7+r*b8+r2*b9+r3*b10)))
		w := r * 0x1p27
		rhi := r + w - w
		rlo := r - rhi
		w = rhi * rhi * -0.5
		hi := r + w
		lo := r - hi + w
		lo += -0.5 * rlo * (rhi + r)
		y += lo
		y += hi
		return y
	}
	top := uint32(ix >> 48)
	if top-0x0010 >= 0x7ff0-0x0010 {
		if ix*2 == 0 {
			return math.Inf(-1) // log(±0)
		}
		if ix == 0x7ff0000000000000 {
			return x // log(+Inf)
		}
		if top&0x8000 != 0 || top&0x7ff0 == 0x7ff0 {
			return math.NaN() // log(negative) or log(NaN)
		}
		// Subnormal: normalize.
		ix = math.Float64bits(x * 0x1p52)
		ix -= 52 << 52
	}
	return logCore(ix)
}

func logCore(ix uint64) float64 {
	tmp := ix - logOFF
	i := uint32(tmp>>45) & 0x7f
	k := int64(tmp) >> 52 // arithmetic shift
	iz := ix - (tmp & (0xfff << 52))
	invc := logTab[i][0]
	logc := logTab[i][1]
	z := math.Float64frombits(iz)
	kd := float64(k)

	// log(x) = k*Ln2 + log(c) + log1p(z/c-1); r = z/c-1.
	r := math.FMA(z, invc, -1.0)
	w := kd*logLn2hi + logc
	hi := w + r
	lo := w - hi + r + kd*logLn2lo
	r2 := r * r
	r3 := r2 * r
	// Horner evaluation fused as in glibc.
	a0 := -0x1.0000000000001p-1
	a1 := 0x1.555555551305bp-2
	a2 := -0x1.fffffffeb459p-3
	a3 := 0x1.999b324f10111p-3
	a4 := -0x1.55575e506c89fp-3
	in1 := math.FMA(r, a2, a1)
	in2 := math.FMA(r, a4, a3)
	p := math.FMA(in2, r2, in1)
	lo = math.FMA(r2, a0, lo)
	return math.FMA(r3, p, lo) + hi
}

// Log1p returns log(1+x), bit-identical to C log1p on glibc x86-64.
// Port of s_log1p.c (fdlibm), Sun Microsystems, 1993.
func Log1p(x float64) float64 {
	const (
		ln2Hi = 6.93147180369123816490e-01
		ln2Lo = 1.90821492927058770002e-10
		two54 = 1.80143985094819840000e+16
	)
	var lp = [8]float64{
		0.0,
		6.666666666666735130e-01,
		3.999999999940941908e-01,
		2.857142874366239149e-01,
		2.222219843214978396e-01,
		1.818357216161805012e-01,
		1.531383769920937332e-01,
		1.479819860511658591e-01,
	}

	var f, c, u float64
	var k, hx, hu, ax int32

	hx = int32(math.Float64bits(x) >> 32)
	ax = hx & 0x7fffffff

	k = 1
	if hx < 0x3FDA827A { // x < 0.41422
		if ax >= 0x3ff00000 { // x <= -1.0
			if x == -1.0 {
				return math.Inf(-1) // -inf
			}
			return (x - x) / (x - x) // NaN
		}
		if ax < 0x3e200000 { // |x| < 2**-29
			if ax < 0x3c900000 { // |x| < 2**-54
				return x
			}
			return x - x*x*0.5
		}
		if hx > 0 || hx <= -1078464313 { // (int32)0xbfd2bec3, signed compare
			k = 0
			f = x
			hu = 1
		} // 0 < x < 0.41422 or x <= -0.45613
	} else if uint32(hx) >= 0x7ff00000 {
		return x + x
	}
	if k != 0 {
		if hx < 0x43400000 {
			u = 1.0 + x
			hu = int32(math.Float64bits(u) >> 32)
			k = (hu >> 20) - 1023
			if k > 0 {
				c = 1.0 - (u - x)
			} else {
				c = x - (u - 1.0)
			}
			c /= u
		} else {
			u = x
			hu = int32(math.Float64bits(u) >> 32)
			k = (hu >> 20) - 1023
			c = 0
		}
		hu &= 0x000fffff
		if hu < 0x6a09e {
			// normalize u
			bits := math.Float64bits(u)&0xFFFFFFFF | uint64(uint32(hu|0x3ff00000))<<32
			u = math.Float64frombits(bits)
		} else {
			k++
			bits := math.Float64bits(u)&0xFFFFFFFF | uint64(uint32(hu|0x3fe00000))<<32
			u = math.Float64frombits(bits)
			hu = (0x00100000 - hu) >> 2
		}
		f = u - 1.0
	}
	hfsq := 0.5 * f * f
	if hu == 0 { // |f| < 2**-20
		if f == 0 {
			if k == 0 {
				return 0
			}
			c += float64(k) * ln2Lo
			return float64(k)*ln2Hi + c
		}
		rr := hfsq * (1.0 - 0.66666666666666666*f)
		if k == 0 {
			return f - rr
		}
		return float64(k)*ln2Hi - ((rr - (float64(k)*ln2Lo + c)) - f)
	}
	s := f / (2.0 + f)
	z := s * s
	z2 := z * z
	z4 := z2 * z2
	z6 := z4 * z2
	r1 := z * lp[1]
	r2 := lp[2] + z*lp[3]
	r3 := lp[4] + z*lp[5]
	r4 := lp[6] + z*lp[7]
	rsum := r1 + z2*r2 + z4*r3 + z6*r4
	if k == 0 {
		return f - (hfsq - s*(hfsq+rsum))
	}
	kf := float64(k)
	return kf*ln2Hi - ((hfsq - (s*(hfsq+rsum) + (kf*ln2Lo + c))) - f)
}
