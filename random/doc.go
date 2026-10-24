// Package random is the numgo equivalent of numpy.random: SeedSequence +
// bit-generator engines + the Generator distribution layer, ported for
// bit-exact agreement with numpy 2.x for identical seeds.
//
// Architecture (mirrors numpy.random):
//
//   - SeedSequence — entropy pool hashing and state generation
//     (numpy/random/bit_generator.pyx SeedSequence). Spawn(n) builds the
//     spawn-key tree used for parallel streams.
//   - BitGenerator — interface {Next64, Next32, NextDouble, Jumped, Advance};
//     NextDouble is numpy's per-engine next_double slot (bitgen.h:15): PCG64
//     uses (next64>>11)*2^-53, MT19937 uses the mt19937.h:56 formula over two
//     tempered 32-bit draws. PCG64 is default_rng's engine; MT19937
//     (mt19937.go) ports mt19937.c/mt19937.h 1:1 including SeedSequence
//     seeding (_mt19937.pyx:130) and the 2**128 jump polynomial
//     (mt19937-jump.c). NewMT19937LegacySeeding reproduces RandomState
//     seeding (mt19937_seed / init_by_array).
//   - Generator — distributions ported from
//     numpy/random/src/distributions/distributions.c preserving call order
//     against the bit generator (bit-exactness depends on it).
//
// Scope: Random, Integers(+Endpoint, +Integers32), StandardNormal(+32),
// StandardExponential(+32), Normal, Uniform, Exponential, StandardGamma(+32),
// Gamma, Beta, Poisson, Binomial, Choice, Shuffle, Permutation; float32
// sampling paths in float32.go (random/standard_normal/standard_exponential/
// standard_gamma with dtype=float32 use numpy's genuine float32 kernels —
// separate ziggurat tables, powf/logf/expf via internal/cmath's glibc-parity
// float routines). Remaining distributions (chisquare, f, hypergeometric, …)
// live in distributions2.go; float64 gamma/beta/… float32 variants beyond the
// four above are pending.
//
// Differences from numpy, documented by design:
//   - No size argument returns a 1-element array (numpy returns a scalar;
//     Go has no scalar/array duality).
//   - Integers is int64-only plus the explicit Integers32 int32 entry point;
//     8/16-bit widths (and their numpy bit buffers) are not ported.
//   - Choice takes the population as an int (numpy also accepts arrays).
//   - MT19937 has Jumped (2**128-step polynomial) but numpy exposes no
//     MT19937.advance; Advance falls back to consuming delta draws.
//   - Transcendental functions (pow/log/log1p/exp and the float32 kernels)
//     route through internal/cmath so ULP behavior matches numpy's C code;
//     the pure-Go math package differs from glibc by 1 ULP on some inputs.
package random
