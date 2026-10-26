// Package polynomial is a Go port of numpy.polynomial: the six polynomial
// classes sharing the ABCPolyBase API (numpy _polybase.py), each with its own
// basis arithmetic ported from the corresponding numpy module.
//
// Translated in part from NumPy (numpy/polynomial), BSD-3-Clause; see
// LICENSES/numpy-BSD3-CLAUSE.txt.
//
//	class        numpy module    basis
//	Polynomial   polynomial.py   power basis x^k
//	Chebyshev    chebyshev.py    T_k, first kind
//	Legendre     legendre.py     P_k
//	Hermite      hermite.py      He_k(x/2)·..., physicists' H_k
//	HermiteE     hermite_e.py    probabilists' He_k
//	Laguerre     laguerre.py     L_k
//
// Coefficients are ordered lowest degree first, like numpy. Each class maps
// its domain onto its window with a linear map (polyutils.mapdomain) before
// evaluating; Call applies the map, Val evaluates raw window coordinates.
//
// Skipped relative to the numpy 72-method class surface (documented API cut):
// val2d/val3d/valnd/grid2d/grid3d, the public mapparms/getdomain functions,
// cross-kind Convert (only same-kind domain/window conversion), fromroots
// with explicit domain/window arguments, integ with explicit lbnd, Fit with
// explicit domain/window/rcond/full/w, basis/basisLine/interpolate/linspace/
// cutdeg/degree/has_samecoef and the legacy np.poly1d function family.
package polynomial
