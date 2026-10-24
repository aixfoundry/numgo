package polynomial

import "errors"

// Panic messages, wording mirrored from numpy (design §4/§7 P0-4).

func panicNegDeg()           { panic("deg must be non-negative") }
func panicNegPower()         { panic("Power must be a non-negative integer.") }
func panicNegOrder()         { panic("The order of derivation must be non-negative") }
func panicZeroDivide()       { panic(errors.New("polynomial division by zero")) }
func panicTol()              { panic("tol must be non-negative") }
func panicTruncate()         { panic("size must be a positive integer") }
func panicFitLen()           { panic("expected x and y to have same length") }
func panicDegenerateDomain() { panic(errors.New("degenerate domain")) }
func panicDegAtLeast1()      { panic("Series must have maximum degree of at least 1.") }
func panicTooManyConsts()    { panic("Too many integration constants") }
func panicDim1D(name string) { panic("expected 1D vector for " + name) }
