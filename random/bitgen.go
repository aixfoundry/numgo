package random

// BitGenerator is the minimal interface the Generator consumes. It supports
// 64/32-bit raw draws only; float paths are derived in the Generator
// layer exactly as numpy's C distributions do. Implementations must preserve
// numpy's next_uint64/next_uint32 buffering contract (pcg64.h:382-396).
type BitGenerator interface {
	Next64() uint64
	Next32() uint32
	// NextDouble is numpy's per-bitgen next_double slot (bitgen.h:15):
	// PCG64 uses uint64_to_double(next64) = (x>>11)*2^-53, MT19937 uses
	// mt19937_next_double = (a>>5)*2^26 + (b>>6) over 2^53 (mt19937.h:56).
	NextDouble() float64
	// Jumped returns an independent generator (numpy "jumped").
	Jumped() BitGenerator
	// Advance skips delta draws.
	Advance(delta uint64)
}
