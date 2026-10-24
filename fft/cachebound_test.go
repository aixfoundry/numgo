package fft

import "testing"

func TestCacheBoundEvicts(t *testing.T) {
	// Fill the twiddle cache past the bound directly (white box): each
	// distinct n allocates a fresh table, so after >maxCacheEntries distinct
	// sizes the caches must have been cleared and the counter reset.
	for n := 33; n <= 33+maxCacheEntries+10; n++ {
		twiddles(n, false)
	}
	if c := cacheEntryCount(); c > maxCacheEntries {
		t.Fatalf("cacheEntries = %d, want <= %d after eviction", c, maxCacheEntries)
	}
	if _, ok := twiddleCache.Load(twiddleKey{33, false}); ok {
		t.Fatal("oldest twiddle entry survived eviction")
	}
	// The cache still works: a fresh lookup repopulates and is used.
	tw := twiddles(64, false)
	if len(tw) == 0 {
		t.Fatal("twiddles after eviction returned empty table")
	}
	if _, ok := twiddleCache.Load(twiddleKey{64, false}); !ok {
		t.Fatal("post-eviction insert not cached")
	}
	// Bluestein kernels and work pools share the same bound.
	for n := 41; n <= 41+maxCacheEntries+10; n++ {
		bluesteinKernelFor(n, false)
	}
	if c := cacheEntryCount(); c > maxCacheEntries {
		t.Fatalf("cacheEntries = %d after bluestein fill, want <= %d", c, maxCacheEntries)
	}
	if _, ok := bluesteinCache.Load(twiddleKey{41, false}); ok {
		t.Fatal("oldest bluestein entry survived eviction")
	}
	if _, ok := bluesteinWork.Load(1 << 20); ok {
		t.Fatal("bluestein work pool survived eviction")
	}
}
