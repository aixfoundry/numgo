package numgo

// Discrete 1-D convolution and correlation (numpy convolve / correlate).
// Both flatten their inputs (numpy accepts 1-D only) and run a direct
// O(n·m) accumulation, with numpy's swap when v is longer than a.

// convolveCore computes the numpy-style sliding inner product of the raw
// ravelled slices; n1 >= n2 after numpy's swap. Windows verified against
// numpy 2.5.1: "full" is the naive convolution over k = -(n2-1)..n1-1;
// "same" is full[(n2-1)/2 : +n1] (length n1 = max(n,m)); "valid" is
// full[n2-1 : n1] (length n1-n2+1). Note numpy's "same" trims (n2-1)/2
// from the left of the full output, not n2/2 — the C code computes
// correlate(a, v[::-1]) so the n2/2 right-trim lands on the other side.
func convolveCore[T Number](a, v []T, mode string) *Array[T] {
	n1, n2 := len(a), len(v)
	if n1 == 0 {
		panic("numgo: a cannot be empty")
	}
	if n2 == 0 {
		panic("numgo: v cannot be empty")
	}
	if n1 < n2 {
		a, v = v, a
		n1, n2 = n2, n1
	}
	full := func(k int) T { // Σ a[m]·v[k−m] over valid m
		var sum T
		lo, hi := max(0, k-(n2-1)), min(n1-1, k)
		for m := lo; m <= hi; m++ {
			sum += a[m] * v[k-m]
		}
		return sum
	}
	off := 0
	length := n1 + n2 - 1
	switch mode {
	case "full":
	case "same":
		off = (n2 - 1) / 2
		length = n1
	case "valid":
		off = n2 - 1
		length = n1 - n2 + 1
	default:
		panic("numgo: mode must be one of 'full', 'same', 'valid'")
	}
	out := make([]T, length)
	for i := range out {
		out[i] = full(off + i)
	}
	return New1DArray(out)
}

// Convolve returns the discrete linear convolution of two sequences
// (numpy convolve): out[n] = Σ a[m]·v[n−m], with v effectively reversed.
// Inputs are flattened; when v is longer than a they are swapped first
// (numpy does the same). mode selects the output window, matching numpy's
// mode argument: "full" (length n+m−1, numpy's default but explicit here),
// "same" (length max(n,m)) or "valid" (length |n−m|+1). Empty inputs or an
// unknown mode panic (numpy raises ValueError).
func Convolve[T Number](a, v *Array[T], mode string) *Array[T] {
	return convolveCore(a.Ravel().Data(), v.Ravel().Data(), mode)
}

// Correlate returns the discrete cross-correlation of two sequences
// (numpy correlate, i.e. multiarray.correlate2): like Convolve but with v
// conjugated and not reversed — out[k] = Σ a[m+k]·conj(v[m]), the signal
// processing definition numpy documents. Conj is the identity for real
// arrays. Swapping, flattening, mode semantics and panics match Convolve.
func Correlate[T Number](a, v *Array[T], mode string) *Array[T] {
	// correlate(a, v) == convolve(a, reverse(conj(v))); reuse the core.
	fv := Flip(v.Ravel().Conj(), 0)
	return convolveCore(a.Ravel().Data(), fv.Data(), mode)
}
