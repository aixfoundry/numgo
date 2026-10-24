package cmath

import "math"

// Single-precision libm kernels (log1pf, logf, expf, powf) bit-identical to
// glibc 2.39 x86-64 (sysdeps/ieee754/flt-32), used by the numpy float32
// distribution paths. glibc's FMA and non-FMA builds agree for expf/logf/powf
// (their data is shared); log1pf is the fdlibm-derived routine with no FMA
// variant. All intermediate arithmetic is double (float_t == double on
// x86-64); the final round to float32 is explicit.

func f32bits(x float32) uint32     { return math.Float32bits(x) }
func f32frombits(b uint32) float32 { return math.Float32frombits(b) }

// Log1pf returns float32(log1p(x)), bit-identical to glibc log1pf
// (s_log1pf.c, fdlibm lineage; no FMA dependence).
func Log1pf(x float32) float32 {
	hx := int32(f32bits(x))
	ax := hx & 0x7fffffff

	k := int32(1)
	var hu uint32
	var f, c float32
	if hx < 0x3ed413d7 { // x < 0.41422
		if ax >= 0x3f800000 { // x <= -1.0
			if x == -1.0 {
				return float32(math.Inf(-1))
			}
			// (x-x)/(x-x): x86 SSE divss returns the first operand quieted
			// for NaN inputs; finite x gives the indefinite QNaN.
			if ax >= 0x7f800000 {
				return x
			}
			return f32frombits(0xffc00000)
		}
		if ax < 0x31000000 { // |x| < 2**-29
			if ax < 0x24800000 { // |x| < 2**-54
				return x
			}
			return x - x*x*0.5
		}
		if hx > 0 || hx <= -1097468385 { // hx <= (int32_t)0xbe95f61f (signed)
			k = 0
			f = x
			hu = 1
		}
	}
	if hx >= 0x7f800000 {
		return x + x
	}
	var u float32
	if k != 0 {
		if hx < 0x5a000000 {
			u = 1.0 + x
			hu = f32bits(u)
			k = int32(int32(hu>>23)) - 127
			// correction term
			if k > 0 {
				c = 1.0 - (u - x)
			} else {
				c = x - (u - 1.0)
			}
			c /= u
		} else {
			u = x
			hu = f32bits(u)
			k = int32(int32(hu>>23)) - 127
			c = 0
		}
		hu &= 0x007fffff
		if hu < 0x3504f7 {
			u = f32frombits(hu | 0x3f800000) // normalize u
		} else {
			k++
			u = f32frombits(hu | 0x3f000000) // normalize u/2
			hu = (0x00800000 - hu) >> 2
		}
		f = u - 1.0
	}
	hfsq := 0.5 * f * f
	if hu == 0 { // |f| < 2**-20
		if f == 0 {
			if k == 0 {
				return 0
			}
			c += float32(k) * 9.0580006145e-06
			return float32(k)*6.9313812256e-01 + c
		}
		R := hfsq * (1.0 - float32(0.66666666666666666)*f)
		if k == 0 {
			return f - R
		}
		return float32(k)*6.9313812256e-01 - ((R - (float32(k)*9.0580006145e-06 + c)) - f)
	}
	s := f / (2.0 + f)
	z := s * s
	R := z * (6.6666668653e-01 + z*(4.0000000596e-01+z*(2.8571429849e-01+
		z*(2.2222198546e-01+z*(1.8183572590e-01+z*(1.5313838422e-01+1.4798198640e-01*z))))))
	if k == 0 {
		return f - (hfsq - s*(hfsq+R))
	}
	return float32(k)*6.9313812256e-01 - ((hfsq - (s*(hfsq+R) +
		(float32(k)*9.0580006145e-06 + c))) - f)
}

// exp2fData is glibc __exp2f_data (e_exp2f_data.c).
var exp2fTab = [32]uint64{
	0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
	0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
	0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
	0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
	0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
	0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
	0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
	0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
}

const (
	exp2fInvLn2N  = 0x1.71547652b82fep+0 * 32 // invln2_scaled
	exp2fShift    = 0x1.8p+52                 // .shift
	exp2fShiftS   = 0x1.8p+52 / 32            // .shift_scaled
	exp2fPolyC0   = 0x1.c6af84b912394p-5      // unscaled poly (powf exp2_inline)
	exp2fPolyC1   = 0x1.ebfce50fac4f3p-3
	exp2fPolyC2   = 0x1.62e42ff0c52d6p-1
	exp2fPolyS0   = 0x1.c6af84b912394p-5 / 32 / 32 / 32 // poly_scaled (expf)
	exp2fPolyS1   = 0x1.ebfce50fac4f3p-3 / 32 / 32
	exp2fPolyS2   = 0x1.62e42ff0c52d6p-1 / 32
	expfTop12Of88 = 0x42b // top12(88.0f)
	expfTop12Inf  = 0x7f8
)

// expfCore is the body of glibc __expf (e_expf.c) for finite x; the
// overflow/underflow branches reproduce __math_oflowf/__math_uflowf values.
func expfCore(xd float64) float32 {
	z := exp2fInvLn2N * xd
	kd := z + exp2fShift // round via double
	ki := math.Float64bits(kd)
	kd -= exp2fShift
	r := z - kd
	// C: t = T[ki % N]; t += ki << (52-TABLE_BITS) — the shift uses the full
	// ki and truncates to 64 bits (high bits carry exponent multiples of 32).
	t := exp2fTab[ki%32] + ki<<(52-5)
	s := math.Float64frombits(t)
	z2 := exp2fPolyS0*r + exp2fPolyS1
	r2 := r * r
	y := exp2fPolyS2*r + 1
	y = z2*r2 + y
	y = y * s
	return float32(y)
}

// Expf returns e**x as float32, bit-identical to glibc expf.
func Expf(x float32) float32 {
	ux := f32bits(x)
	xd := float64(x)
	abstop := uint32(uint64(ux)>>20) & 0x7ff
	if abstop >= expfTop12Of88 {
		// |x| >= 88 or nan
		if ux == 0xff800000 {
			return 0
		}
		if abstop >= expfTop12Inf {
			return x + x
		}
		if x > 0x1.62e42ep+6 { // x > log(0x1p128)
			return float32(math.Inf(1))
		}
		if x < -0x1.9fe368p+6 { // x < log(0x1p-150)
			return 0
		}
	}
	return expfCore(xd)
}

// logfData is glibc __logf_data (e_logf_data.c).
type logfTabEnt struct {
	invc, logc float64
}

var logfTab = [16]logfTabEnt{
	{0x1.661ec79f8f3bep+0, -0x1.57bf7808caadep-2},
	{0x1.571ed4aaf883dp+0, -0x1.2bef0a7c06ddbp-2},
	{0x1.49539f0f010bp+0, -0x1.01eae7f513a67p-2},
	{0x1.3c995b0b80385p+0, -0x1.b31d8a68224e9p-3},
	{0x1.30d190c8864a5p+0, -0x1.6574f0ac07758p-3},
	{0x1.25e227b0b8eap+0, -0x1.1aa2bc79c81p-3},
	{0x1.1bb4a4a1a343fp+0, -0x1.a4e76ce8c0e5ep-4},
	{0x1.12358f08ae5bap+0, -0x1.1973c5a611cccp-4},
	{0x1.0953f419900a7p+0, -0x1.252f438e10c1ep-5},
	{0x1p+0, 0x0p+0},
	{0x1.e608cfd9a47acp-1, 0x1.aa5aa5df25984p-5},
	{0x1.ca4b31f026aap-1, 0x1.c5e53aa362eb4p-4},
	{0x1.b2036576afce6p-1, 0x1.526e57720db08p-3},
	{0x1.9c2d163a1aa2dp-1, 0x1.bc2860d22477p-3},
	{0x1.886e6037841edp-1, 0x1.1058bc8a07ee1p-2},
	{0x1.767dcf5534862p-1, 0x1.4043057b6ee09p-2},
}

const logfLn2 = 0x1.62e42fefa39efp-1

var logfPolyA = [3]float64{-0x1.00ea348b88334p-2, 0x1.5575b0be00b6ap-2, -0x1.ffffef20a4123p-2}

const logfOff = 0x3f330000

// Logf returns ln(x) as float32, bit-identical to glibc logf (e_logf.c).
func Logf(x float32) float32 {
	ix := f32bits(x)
	if ix == 0x3f800000 {
		return 0
	}
	if ix-0x00800000 >= 0x7f800000-0x00800000 {
		// x < 0x1p-126 or inf or nan
		if ix*2 == 0 {
			return float32(math.Inf(-1)) // log(+0) = -inf (errno ignored)
		}
		if ix == 0x7f800000 {
			return x
		}
		if ix&0x80000000 != 0 || ix*2 >= 0xff000000 {
			return float32(math.NaN())
		}
		// subnormal: normalize
		ix = f32bits(x * 0x1p23)
		ix -= 23 << 23
	}
	tmp := ix - logfOff
	i := (tmp >> (23 - 4)) % 16
	k := int32(tmp) >> 23 // arithmetic shift
	iz := ix - (tmp & (0x1ff << 23))
	invc := logfTab[i].invc
	logc := logfTab[i].logc
	z := float64(f32frombits(iz))
	r := z*invc - 1
	y0 := logc + float64(k)*logfLn2
	r2 := r * r
	y := logfPolyA[1]*r + logfPolyA[2]
	y = logfPolyA[0]*r2 + y
	y = y*r2 + (y0 + r)
	return float32(y)
}

// powfLog2Data is glibc __powf_log2_data (e_powf_log2_data.c) with
// TOINT_INTRINSICS=0 → POWF_SCALE_BITS=0, POWF_SCALE=1 (unscaled entries).
type powfTabEnt struct {
	invc, logc float64
}

var powfTab = [16]powfTabEnt{
	{0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2},
	{0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2},
	{0x1.49539f0f010bp+0, -0x1.7418b0a1fb77bp-2},
	{0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2},
	{0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2},
	{0x1.25e227b0b8eap+0, -0x1.97c1d1b3b7afp-3},
	{0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3},
	{0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4},
	{0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5},
	{0x1p+0, 0x0p+0},
	{0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4},
	{0x1.ca4b31f026aap-1, 0x1.476a9543891bap-3},
	{0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3},
	{0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2},
	{0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2},
	{0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2},
}

var powfPolyA = [5]float64{
	0x1.27616c9496e0bp-2, -0x1.71969a075c67ap-2,
	0x1.ec70a6ca7baddp-2, -0x1.7154748bef6c8p-1,
	0x1.71547652ab82bp0,
}

// exp2fInline is e_powf.c exp2_inline (unscaled poly, shift_scaled).
func exp2fInline(xd float64, signBias uint32) float64 {
	kd := xd + exp2fShiftS
	ki := math.Float64bits(kd)
	kd -= exp2fShiftS
	r := xd - kd
	t := exp2fTab[ki%32]
	ski := ki + uint64(signBias)
	t += ski << (52 - 5)
	s := math.Float64frombits(t)
	z := exp2fPolyC0*r + exp2fPolyC1
	r2 := r * r
	y := exp2fPolyC2*r + 1
	y = z*r2 + y
	return y * s
}

func powfCheckint(iy uint32) int {
	e := int(iy>>23) & 0xff
	if e < 0x7f {
		return 0
	}
	if e > 0x7f+23 {
		return 2
	}
	if iy&((1<<(0x7f+23-e))-1) != 0 {
		return 0
	}
	if iy&(1<<(0x7f+23-e)) != 0 {
		return 1
	}
	return 2
}

func powfZeroinfnan(ix uint32) bool {
	return 2*ix-1 >= 2*0x7f800000-1
}

func powfLog2Inline(ix uint32) float64 {
	tmp := ix - logfOff
	i := (tmp >> (23 - 4)) % 16
	// top = tmp & (0x1ff << 23) (C precedence: shift binds tighter than &)
	top := tmp & (0x1ff << 23)
	iz := ix - top
	k := int32(top) >> 23 // arithmetic shift, POWF_SCALE_BITS=0
	invc := powfTab[i].invc
	logc := powfTab[i].logc
	z := float64(f32frombits(iz))
	r := z*invc - 1
	y0 := logc + float64(k)
	r2 := r * r
	y := powfPolyA[0]*r + powfPolyA[1]
	p := powfPolyA[2]*r + powfPolyA[3]
	r4 := r2 * r2
	q := powfPolyA[4]*r + y0
	q = p*r2 + q
	return y*r4 + q
}

// Powf returns x**y as float32, bit-identical to glibc powf (e_powf.c).
// Errno side effects are ignored (values match; fenv is not modeled).
func Powf(x, y float32) float32 {
	signBias := uint32(0)
	ix, iy := f32bits(x), f32bits(y)
	if ix-0x00800000 >= 0x7f800000-0x00800000 || powfZeroinfnan(iy) {
		if powfZeroinfnan(iy) {
			if 2*iy == 0 {
				return 1
			}
			if ix == 0x3f800000 {
				return 1
			}
			if 2*ix > 2*0x7f800000 || 2*iy > 2*0x7f800000 {
				return x + y
			}
			if 2*ix == 2*0x3f800000 {
				return 1
			}
			if (2*ix < 2*0x3f800000) == (iy&0x80000000 == 0) {
				return 0
			}
			return y * y
		}
		if powfZeroinfnan(ix) {
			x2 := x * x
			if ix&0x80000000 != 0 && powfCheckint(iy) == 1 {
				x2 = -x2
			}
			if iy&0x80000000 != 0 {
				return 1 / x2
			}
			return x2
		}
		// x and y non-zero finite
		if ix&0x80000000 != 0 {
			yint := powfCheckint(iy)
			if yint == 0 {
				// __math_invalidf: x86 SSE default indefinite QNaN.
				return f32frombits(0xffc00000)
			}
			if yint == 1 {
				signBias = 1 << (5 + 11) // SIGN_BIAS = 1 << (EXP2F_TABLE_BITS + 11)
			}
			ix &= 0x7fffffff
		}
		if ix < 0x00800000 {
			ix = f32bits(x * 0x1p23)
			ix &= 0x7fffffff
			ix -= 23 << 23
		}
	}
	logx := powfLog2Inline(ix)
	ylogx := float64(y) * logx
	if math.Float64bits(ylogx)>>47&0xffff >= math.Float64bits(126.0)>>47 {
		// |y*log(x)| >= 126: oflow/uflow with sign applied (WANT_ERRNO paths
		// reduced to value-only; the rounding-mode check is round-to-nearest).
		if ylogx > 0x1.fffffffd1d571p+6 {
			r := float32(math.Inf(1))
			if signBias != 0 {
				r = -r
			}
			return r
		}
		if ylogx <= -150.0 {
			if signBias != 0 {
				return float32(math.Copysign(0, -1))
			}
			return 0
		}
	}
	// The sign for odd integer y rides bit 63 via signBias<<47 inside exp2_inline.
	return float32(exp2fInline(ylogx, signBias))
}
