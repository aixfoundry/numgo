package random

// Panic message table, aligned with numpy's wording (design.md §7 P0-4).
import "fmt"

func panicBounds(kind, comp string, v int64) {
	panic(fmt.Sprintf("%s %s %d", kind, comp, v))
}

func panicBoundsError(low, high int64, closed bool) {
	// numpy: "low >= high" / "low > high" wrapped as a bounds error.
	comp := ">="
	if closed {
		comp = ">"
	}
	panic(fmt.Sprintf("low %s high: %d %d", comp, low, high))
}

func panicProbNaN()      { panic("Probabilities contain NaN") }
func panicProbNegative() { panic("Probabilities are not non-negative") }
func panicProbSum() {
	panic("Probabilities do not sum to 1. See Notes section of docstring for more information.")
}

func panicSampleTooLarge() {
	panic("Cannot take a larger sample than population when replace is False")
}
func panicFewerNonZero()     { panic("Fewer non-zero entries in p than size") }
func panicAPositive()        { panic("a must be a positive integer unless no samples are taken") }
func panicSeedSequenceSize() { panic("The size of the entropy pool should be at least 4") }
func panicNegativeEntropy()  { panic("expected non-negative integer") }
func panicChildren()         { panic("n_children must be non-negative") }
