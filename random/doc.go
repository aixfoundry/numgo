// Package random is the numgo equivalent of numpy.random: SeedSequence +
// PCG64 bit generation and the Generator distribution layer, ported for
// bit-exact agreement with numpy 2.x for identical seeds.
//
// Architecture (mirrors numpy.random):
//
//   - SeedSequence — entropy pool hashing and ChaCha-free state generation
//     (numpy/random/bit_generator.pyx SeedSequence). Spawn(n) builds the
//     spawn-key tree used for parallel streams.
//   - BitGenerator — minimal interface {Next64, Next32, Jumped, Advance};
//     documented in bitgen.go. PCG64 is the only engine this batch
//     (numpy default_rng's engine); MT19937 is planned next batch.
//   - Generator — distributions ported from
//     numpy/random/src/distributions/distributions.c preserving call order
//     against the bit generator (bit-exactness depends on it).
//
// Scope: Random, Integers(+Endpoint), StandardNormal, StandardExponential,
// Normal, Uniform, Exponential, StandardGamma, Gamma, Beta, Poisson,
// Binomial, Choice, Shuffle, Permutation. Remaining distributions
// (chisquare, f, hypergeometric, …) are the next batch.
//
// Differences from numpy, documented by design:
//   - No size argument returns a 1-element array (numpy returns a scalar;
//     Go has no scalar/array duality).
//   - float32 sampling paths (numpy dtype=float32, 24-bit) are not ported.
//   - Integers is int64-only this batch (numpy supports 8/16/32/64 widths).
//   - Choice takes the population as an int (numpy also accepts arrays).
//   - Transcendental functions (pow/log/log1p/exp) route through glibc via
//     internal/cmath so ULP behavior matches numpy's C code; the pure-Go
//     math.Pow differs from glibc by 1 ULP on some inputs.
package random
