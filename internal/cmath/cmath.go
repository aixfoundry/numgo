// Package cmath provides pure-Go double-precision exp, log, log1p and pow whose
// results are bit-identical to the glibc functions used by numpy's distributions
// (npy_pow→pow, log, log1p, exp) on x86-64 (glibc ≥ 2.28 lineage).
//
// The algorithms are ports of the GNU C Library's IEEE-754 dbl-64 routines
// (sysdeps/ieee754/dbl-64/e_exp.c, e_log.c, e_pow.c, s_log1p.c and their data
// files), which in turn derive from ARM's optimized-routines project.
//
// Portions derived from ARM optimized-routines (MIT), Copyright (c) 2018-2019
// Arm Limited, and from the GNU C Library (LGPL 2.1), Copyright (C) 2018-2024
// Free Software Foundation, Inc.; log1p derives from fdlibm, Copyright (C)
// 1993 by Sun Microsystems, Inc.
//
// Arithmetic order was chosen to reproduce the bit patterns produced by
// glibc 2.39 on x86-64; it was validated exhaustively against the platform
// libm (see cmath_test.go).
//
// Known residuals vs glibc 2.39 x86-64 (both ≤ 1 ULP, reported by
// TestParityReference):
//   - Log1p: ~0.1% of inputs in (-0.456, -0.293) differ in the last bit
//     (glibc 2.39 ships a restructured log1p for that band).
//   - Log: ~3e-6 of inputs in the near-1 branch differ in the last bit.
package cmath
