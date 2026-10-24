// Package cmath wraps the C math library functions numpy's distributions use
// (npy_pow→pow, log, log1p, exp) so results are bit-identical to numpy on the
// same platform. Go's pure-Go math.Pow differs from glibc by 1 ULP on some
// inputs, which would break bit-exact parity of gamma/beta-derived draws.
package cmath

// #include <math.h>
// #cgo LDFLAGS: -lm
import "C"

// Pow wraps C pow (numpy npy_pow is #defined to pow).
func Pow(x, y float64) float64 { return float64(C.pow(C.double(x), C.double(y))) }

// Log wraps C log.
func Log(x float64) float64 { return float64(C.log(C.double(x))) }

// Log1p wraps C log1p (numpy npy_log1p).
func Log1p(x float64) float64 { return float64(C.log1p(C.double(x))) }

// Exp wraps C exp.
func Exp(x float64) float64 { return float64(C.exp(C.double(x))) }
