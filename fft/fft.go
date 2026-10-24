// Package fft provides numpy.fft-semantics discrete Fourier transforms over
// numgo arrays, backed by an own FFT: mixed-radix Cooley-Tukey + Bluestein
// for large primes.
//
// Semantics follow numpy.fft (pocketfft): the forward transform is
// A_k = Σ_j a_j·exp(-2πi·jk/n), the last axis is transformed by default,
// arbitrary (mixed-radix) transform lengths are supported, n == 0 means "use
// the axis length" (Go's zero value stands in for numpy's None) and n > 0
// zero-pads or truncates, and norm follows numpy's backward/ortho/forward
// modes with factors applied per transformed axis over the padded length n.
// IRFFT-family results are returned as float64 (the imaginary part is zero by
// construction, matching numpy's real output dtype). FFTShift/IFFTShift return
// pure copies, never views.
package fft

import (
	"math"
	"slices"

	"github.com/aixfoundry/numgo"
)

// Norm selects the normalization of a forward/backward transform pair,
// mirroring numpy.fft's norm argument. Backward (numpy's default) scales the
// forward transform by 1 and the inverse by 1/n; Forward is the mirror image;
// Ortho scales both by 1/√n. n is the resolved transform length, including
// any zero padding.
type Norm int

const (
	// Backward is numpy's default: 1 on the forward transform, 1/n on the inverse.
	Backward Norm = iota
	// Ortho scales both directions by 1/√n.
	Ortho
	// Forward scales the forward transform by 1/n and the inverse by 1.
	Forward
)

// swap returns the norm numpy applies to the inverse direction of a pair.
func (n Norm) swap() Norm {
	switch n {
	case Forward:
		return Backward
	case Backward:
		return Forward
	default:
		return Ortho
	}
}

// factor returns the multiplier applied to one transform of length nn.
func (n Norm) factor(nn int) float64 {
	switch n {
	case Ortho:
		return 1 / math.Sqrt(float64(nn))
	case Forward:
		return 1 / float64(nn)
	default:
		return 1
	}
}

// Pure: FFT transforms a along one axis (default last). n == 0 uses the axis
// length; n > 0 zero-pads or truncates. Full-spectrum complex128 output.
func FFT[T numgo.Number](a *numgo.Array[T], n int, norm Norm, axes ...int) *numgo.Array[complex128] {
	ax := axisOrLast(a.Shape(), axes)
	nn := defaultN(n, a.Shape()[ax])
	return fullAxis(numgo.AsType[T, complex128](a), nn, ax, true, norm.factor(nn))
}

// Pure: IFFT is the inverse of FFT. n == 0 uses the axis length; n > 0
// zero-pads or truncates the input.
func IFFT[T numgo.Number](a *numgo.Array[T], n int, norm Norm, axes ...int) *numgo.Array[complex128] {
	ax := axisOrLast(a.Shape(), axes)
	nn := defaultN(n, a.Shape()[ax])
	return fullAxis(numgo.AsType[T, complex128](a), nn, ax, false, norm.swap().factor(nn))
}

// Pure: RFFT is the FFT of real input along one axis, returning the
// non-negative frequency half-spectrum of length n//2+1 (including the
// Nyquist bin).
func RFFT[T numgo.Float](a *numgo.Array[T], n int, norm Norm, axes ...int) *numgo.Array[complex128] {
	ax := axisOrLast(a.Shape(), axes)
	nn := defaultN(n, a.Shape()[ax])
	return halfAxisForward(numgo.AsType[T, complex128](a), nn, ax, norm.factor(nn))
}

// Pure: IRFFT inverts RFFT: complex half-spectrum input, real float64 output
// of length n (n == 0 means 2*(m-1) for input length m along the axis).
func IRFFT(a *numgo.Array[complex128], n int, norm Norm, axes ...int) *numgo.Array[float64] {
	ax := axisOrLast(a.Shape(), axes)
	nn := defaultN(n, 2*(a.Shape()[ax]-1))
	return halfAxisInverse(a, nn, ax, norm.swap().factor(nn))
}

// Pure: HFFT is the inverse of IHFFT: a complex Hermitian half-spectrum in,
// real float64 output of length n (n == 0 means 2*(m-1)).
func HFFT(a *numgo.Array[complex128], n int, norm Norm, axes ...int) *numgo.Array[float64] {
	ax := axisOrLast(a.Shape(), axes)
	nn := defaultN(n, 2*(a.Shape()[ax]-1))
	return halfAxisInverse(conjAll(a), nn, ax, norm.factor(nn))
}

// Pure: IHFFT is the inverse of HFFT: real input, complex conjugated
// half-spectrum output of length n//2+1 (numpy's ihfft; the input must be
// real, as in numpy where complex input raises TypeError).
func IHFFT[T numgo.Float](a *numgo.Array[T], n int, norm Norm, axes ...int) *numgo.Array[complex128] {
	ax := axisOrLast(a.Shape(), axes)
	nn := defaultN(n, a.Shape()[ax])
	return conjAll(halfAxisForward(numgo.AsType[T, complex128](a), nn, ax, norm.swap().factor(nn)))
}

// Pure: FFT2 transforms the last two axes (numpy fft2's default (-2,-1)).
// s == nil or s[i] <= 0 uses the axis length; positive s[i] pads/truncates.
func FFT2[T numgo.Number](a *numgo.Array[T], s []int, norm Norm) *numgo.Array[complex128] {
	return FFTN(a, s, norm, -2, -1)
}

// Pure: IFFT2 is the two-dimensional inverse of FFT2 along the last two axes.
func IFFT2[T numgo.Number](a *numgo.Array[T], s []int, norm Norm) *numgo.Array[complex128] {
	return IFFTN(a, s, norm, -2, -1)
}

// Pure: RFFT2 is the two-dimensional real FFT along the last two axes; the
// last output axis holds the half-spectrum n//2+1.
func RFFT2[T numgo.Float](a *numgo.Array[T], s []int, norm Norm) *numgo.Array[complex128] {
	return RFFTN(a, s, norm, -2, -1)
}

// Pure: IRFFT2 is the two-dimensional inverse of RFFT2; the result is a real
// float64 array.
func IRFFT2(a *numgo.Array[complex128], s []int, norm Norm) *numgo.Array[float64] {
	return IRFFTN(a, s, norm, -2, -1)
}

// Pure: FFTN transforms along the given axes (default: all axes). s aligns
// with axes: s == nil or s[i] <= 0 uses the axis length; positive s[i]
// zero-pads or truncates that axis.
func FFTN[T numgo.Number](a *numgo.Array[T], s []int, norm Norm, axes ...int) *numgo.Array[complex128] {
	axs, ns := ndAxes(a.Shape(), s, axes, false)
	cur := numgo.AsType[T, complex128](a)
	for i, ax := range axs {
		cur = fullAxis(cur, ns[i], ax, true, norm.factor(ns[i]))
	}
	return cur
}

// Pure: IFFTN is the inverse of FFTN with the same s/axes conventions.
func IFFTN[T numgo.Number](a *numgo.Array[T], s []int, norm Norm, axes ...int) *numgo.Array[complex128] {
	axs, ns := ndAxes(a.Shape(), s, axes, false)
	cur := numgo.AsType[T, complex128](a)
	for i, ax := range axs {
		cur = fullAxis(cur, ns[i], ax, false, norm.swap().factor(ns[i]))
	}
	return cur
}

// Pure: RFFTN is the n-dimensional real FFT: RFFT along the last listed axis,
// then FFT along the others. Like numpy, the real-input RFFT runs first so
// the remaining transforms operate on the complex half-spectrum.
func RFFTN[T numgo.Float](a *numgo.Array[T], s []int, norm Norm, axes ...int) *numgo.Array[complex128] {
	axs, ns := ndAxes(a.Shape(), s, axes, false)
	cur := halfAxisForward(numgo.AsType[T, complex128](a), ns[len(axs)-1], axs[len(axs)-1], norm.factor(ns[len(axs)-1]))
	for i := len(axs) - 2; i >= 0; i-- {
		cur = fullAxis(cur, ns[i], axs[i], true, norm.factor(ns[i]))
	}
	return cur
}

// Pure: IRFFTN is the inverse of RFFTN; the last listed axis is the
// half-spectrum (default output length there is 2*(m-1)), the rest are full
// complex axes. The result is a real float64 array.
func IRFFTN(a *numgo.Array[complex128], s []int, norm Norm, axes ...int) *numgo.Array[float64] {
	axs, ns := ndAxes(a.Shape(), s, axes, true)
	cur := a
	for i := 0; i < len(axs)-1; i++ {
		cur = fullAxis(cur, ns[i], axs[i], false, norm.swap().factor(ns[i]))
	}
	last := len(axs) - 1
	return halfAxisInverse(cur, ns[last], axs[last], norm.swap().factor(ns[last]))
}

// Pure: FFTFreq returns n sample frequencies in cycles per unit of the sample
// spacing d: [0, 1, ..., n/2-1, -n/2, ..., -1] / (n*d).
func FFTFreq(n int, d float64) *numgo.Array[float64] {
	if n < 1 {
		panicInvalidN(n)
	}
	val := 1 / (float64(n) * d)
	half := (n-1)/2 + 1
	data := make([]float64, n)
	for i := range half {
		data[i] = float64(i) * val
	}
	for i := half; i < n; i++ {
		data[i] = float64(i-n) * val
	}
	return numgo.New1DArray(data)
}

// Pure: RFFTFreq returns the non-negative frequencies of RFFT output:
// [0, 1, ..., n/2] / (n*d).
func RFFTFreq(n int, d float64) *numgo.Array[float64] {
	if n < 1 {
		panicInvalidN(n)
	}
	val := 1 / (float64(n) * d)
	data := make([]float64, n/2+1)
	for i := range data {
		data[i] = float64(i) * val
	}
	return numgo.New1DArray(data)
}

// Pure: FFTShift shifts the zero-frequency component to the center by rolling
// each listed axis (default: all) by n//2. The result is a pure copy.
func FFTShift[T numgo.Number](a *numgo.Array[T], axes ...int) *numgo.Array[T] {
	out := a
	for _, ax := range shiftAxes(a, axes) {
		out = numgo.Roll(out, out.Shape()[ax]/2, ax)
	}
	return out
}

// Pure: IFFTShift is the inverse of FFTShift: it rolls each listed axis by
// -n/2. Identical to FFTShift for even-length axes. The result is a pure copy.
func IFFTShift[T numgo.Number](a *numgo.Array[T], axes ...int) *numgo.Array[T] {
	out := a
	for _, ax := range shiftAxes(a, axes) {
		out = numgo.Roll(out, -out.Shape()[ax]/2, ax)
	}
	return out
}

// normAxis resolves a possibly negative axis, reporting the original value on
// failure (numpy normalize_axis_index wording).
func normAxis(axis, ndim int) int {
	ax := axis
	if ax < 0 {
		ax += ndim
	}
	if ax < 0 || ax >= ndim {
		panicAxis(axis, ndim)
	}
	return ax
}

// axisOrLast resolves the single optional axis of the 1-D transform family.
func axisOrLast(shape []int, axes []int) int {
	if len(axes) > 1 {
		panicSingleAxis()
	}
	ax := -1
	if len(axes) == 1 {
		ax = axes[0]
	}
	return normAxis(ax, len(shape))
}

// defaultN resolves the n parameter: 0 selects def; the result must be >= 1
// (numpy reports the resolved value in its error).
func defaultN(n, def int) int {
	nn := n
	if nn == 0 {
		nn = def
	}
	if nn < 1 {
		panicInvalidN(nn)
	}
	return nn
}

// ndAxes resolves transform axes and their lengths for the n-d family. With
// irreal the last listed axis is a half-spectrum axis whose default output
// length is 2*(m-1).
func ndAxes(shape []int, s []int, axes []int, irreal bool) ([]int, []int) {
	axs := axes
	if len(axs) == 0 {
		axs = make([]int, len(shape))
		for i := range axs {
			axs[i] = i
		}
	}
	if len(s) > 0 && len(s) != len(axs) {
		panicShapeAxes()
	}
	ns := make([]int, len(axs))
	for i, ax := range axs {
		axs[i] = normAxis(ax, len(shape))
		ns[i] = shape[axs[i]]
		if i < len(s) {
			if s[i] > 0 {
				ns[i] = s[i]
			} else if s[i] < -1 {
				panicInvalidN(s[i])
			}
		}
		if ns[i] < 1 {
			panicInvalidN(ns[i])
		}
	}
	if irreal {
		last := len(axs) - 1
		if len(s) == 0 || s[last] <= 0 {
			ns[last] = 2 * (shape[axs[last]] - 1)
			if ns[last] < 1 {
				panicInvalidN(ns[last])
			}
		}
	}
	return axs, ns
}

// shiftAxes resolves the axes of FFTShift/IFFTShift, defaulting to all.
func shiftAxes[T numgo.Number](a *numgo.Array[T], axes []int) []int {
	if len(axes) == 0 {
		axes = make([]int, a.Ndim())
		for i := range axes {
			axes[i] = i
		}
	}
	out := make([]int, len(axes))
	for i, ax := range axes {
		out[i] = normAxis(ax, a.Ndim())
	}
	return out
}

// conjAll returns a pure complex-conjugated copy of a.
func conjAll(a *numgo.Array[complex128]) *numgo.Array[complex128] {
	if a.IsContiguous() {
		src := a.Data()[:a.Size()]
		return numgo.New1DArray(conjInto(src, make([]complex128, len(src))))
	}
	return numgo.FromFunction(a.Shape(), func(coords ...int) complex128 {
		c := a.At(coords...)
		return complex(real(c), -imag(c))
	})
}

// conjInto conjugates every element of src into dst (same lengths).
func conjInto(src, dst []complex128) []complex128 {
	for i, c := range src {
		dst[i] = complex(real(c), -imag(c))
	}
	return dst
}

// fullAxis transforms one resolved axis of a C-order complex array. m may be
// smaller or larger than n: values beyond n are dropped, the rest zero-padded.
func fullAxis(in *numgo.Array[complex128], n, axis int, forward bool, fct float64) *numgo.Array[complex128] {
	inShape := in.Shape()
	m := inShape[axis]
	outShape := slices.Clone(inShape)
	outShape[axis] = n
	out := numgo.Zeros[complex128](outShape...)
	if out.Size() == 0 {
		return out
	}
	seq := make([]complex128, n)
	res := make([]complex128, n)
	scratch := make([]complex128, 2*n)
	line := make([]int, len(inShape))
	iterBatch(outShape, axis, func(batch []int) {
		copy(line, batch)
		for i := 0; i < m && i < n; i++ {
			line[axis] = i
			seq[i] = in.At(line...)
		}
		for i := m; i < n; i++ {
			seq[i] = 0
		}
		if forward {
			fftScratch(res, seq, false, scratch)
		} else {
			fftScratch(res, seq, true, scratch)
		}
		if fct != 1 {
			f := complex(fct, 0)
			for i := range res {
				res[i] *= f
			}
		}
		for i := range res {
			line[axis] = i
			out.Set(line, res[i])
		}
	})
	return out
}

// halfAxisForward transforms one axis of real data to a half-spectrum of
// length n/2+1.
func halfAxisForward(in *numgo.Array[complex128], n, axis int, fct float64) *numgo.Array[complex128] {
	inShape := in.Shape()
	m := inShape[axis]
	outShape := slices.Clone(inShape)
	outShape[axis] = n/2 + 1
	out := numgo.Zeros[complex128](outShape...)
	if out.Size() == 0 {
		return out
	}
	seq := make([]complex128, n)
	full := make([]complex128, n)
	res := make([]complex128, n/2+1)
	scratch := make([]complex128, 2*n)
	line := make([]int, len(inShape))
	iterBatch(outShape, axis, func(batch []int) {
		copy(line, batch)
		for i := 0; i < m && i < n; i++ {
			line[axis] = i
			seq[i] = complex(real(in.At(line...)), 0)
		}
		for i := m; i < n; i++ {
			seq[i] = 0
		}
		fftScratch(full, seq, false, scratch)
		copy(res, full[:n/2+1])
		if fct != 1 {
			f := complex(fct, 0)
			for i := range res {
				res[i] *= f
			}
		}
		for i := range res {
			line[axis] = i
			out.Set(line, res[i])
		}
	})
	return out
}

// halfAxisInverse transforms a half-spectrum axis of length m into real
// output of length n, cropping or zero-padding high frequencies.
func halfAxisInverse(in *numgo.Array[complex128], n, axis int, fct float64) *numgo.Array[float64] {
	inShape := in.Shape()
	m := inShape[axis]
	half := n/2 + 1
	outShape := slices.Clone(inShape)
	outShape[axis] = n
	out := numgo.Zeros[float64](outShape...)
	if out.Size() == 0 {
		return out
	}
	coeff := make([]complex128, half)
	full := make([]complex128, n)
	res := make([]float64, n)
	scratch := make([]complex128, 2*n)
	inData := in.Data()
	line := make([]int, len(inShape))
	if len(inShape) == 1 {
		// 1-D fast path: the transform line IS the data slice — read from
		// inData directly and write res back as one copy (no At/Set, no
		// per-line iteration).
		for i := 0; i < m && i < half; i++ {
			coeff[i] = inData[i]
		}
		for i := m; i < half; i++ {
			coeff[i] = 0
		}
		irfftUnnormBuf(res, coeff, full, scratch)
		if fct != 1 {
			for i := range res {
				res[i] *= fct
			}
		}
		copy(out.Data(), res)
		return out
	}
	iterBatch(outShape, axis, func(batch []int) {
		copy(line, batch)
		for i := 0; i < m && i < half; i++ {
			line[axis] = i
			coeff[i] = in.At(line...)
		}
		for i := m; i < half; i++ {
			coeff[i] = 0
		}
		irfftUnnormBuf(res, coeff, full, scratch)
		if fct != 1 {
			for i := range res {
				res[i] *= fct
			}
		}
		for i := range res {
			line[axis] = i
			out.Set(line, res[i])
		}
	})
	return out
}

// iterBatch calls fn once per C-order coordinate combination of shape with
// coords[axis] left at zero (the transform loop overwrites it).
func iterBatch(shape []int, axis int, fn func(line []int)) {
	line := make([]int, len(shape))
	var rec func(k int)
	rec = func(k int) {
		if k == len(shape) {
			fn(line)
			return
		}
		if k == axis {
			rec(k + 1)
			return
		}
		for i := 0; i < shape[k]; i++ {
			line[k] = i
			rec(k + 1)
		}
	}
	rec(0)
}
