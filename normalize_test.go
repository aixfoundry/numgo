package numgo

import (
	"reflect"
	"testing"
)

func TestMinMaxScaler(t *testing.T) {
	a := New1DArray([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
	scaled := a.MinMaxScaler()
	expected := []float64{0, 0.25, 0.5, 0.75, 1}
	if !reflect.DeepEqual(scaled.Data(), expected) {
		t.Errorf("Expected %v, got %v", expected, scaled.Data())
		t.FailNow()
	}
}
