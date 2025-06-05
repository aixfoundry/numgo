package numgo

import (
	"math"
	"testing"
)

func TestNewRGBA(t *testing.T) {
	a, err := LoadFromFile("testdata/feature.txt")
	if err != nil {
		t.Error(err)
		t.FailNow()
	}
	for i := range 10 {
		b := NewValueArray(a.r, 1, float64(i*100))
		a.Cat(b, 1)
	}
	err = a.SaveToFile("testdata/feature.png")
	if err != nil {
		t.Error(err)
		t.FailNow()
	}
}

func Test_calcTicks(t *testing.T) {
	type args struct {
		min float64
		max float64
	}
	tests := []struct {
		name      string
		args      args
		wantTicks []float64
	}{
		{
			name: "零值测试",
			args: args{
				min: 0,
				max: 0,
			},
			wantTicks: []float64{0},
		},
		{
			name: "小范围测试",
			args: args{
				min: 0,
				max: 1,
			},
			wantTicks: []float64{0, 0.2, 0.4, 0.6, 0.8, 1},
		},
		{
			name: "常规测试",
			args: args{
				min: 0,
				max: 5,
			},
			wantTicks: []float64{0, 1, 2, 3, 4, 5},
		},
		{
			name: "整十测试",
			args: args{
				min: 0,
				max: 10,
			},
			wantTicks: []float64{0, 2, 4, 6, 8, 10},
		},
		{
			name: "大范围测试",
			args: args{
				min: 0,
				max: 100,
			},
			wantTicks: []float64{0, 20, 40, 60, 80, 100},
		},
		{
			name: "非常规范围",
			args: args{
				min: 0,
				max: 3,
			},
			wantTicks: []float64{0, 0.5, 1, 1.5, 2, 2.5, 3},
		},
		{
			name: "负值测试",
			args: args{
				min: -5,
				max: 15,
			},
			wantTicks: []float64{-5, 0, 5, 10, 15},
		},
		{
			name: "大数测试",
			args: args{
				min: 0,
				max: 1234,
			},
			wantTicks: []float64{0, 250, 500, 750, 1000, 1250},
		},
		{
			name: "小数测试",
			args: args{
				min: 0,
				max: 0.001,
			},
			wantTicks: []float64{0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001},
		},
		{
			name: "全负测试",
			args: args{
				min: -10,
				max: -1,
			},
			wantTicks: []float64{-10, -8, -6, -4, -2, 0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotTicks := calcTicks(tt.args.min, tt.args.max)
			if len(gotTicks) != len(tt.wantTicks) {
				t.Errorf("calcTicks() = %v, want %v", gotTicks, tt.wantTicks)
			} else {
				for i, tick := range gotTicks {
					if math.Abs(tick-tt.wantTicks[i]) > 1e-9 {
						t.Errorf("calcTicks() = %v, want %v", gotTicks, tt.wantTicks)
						break
					}
				}
			}
		})
	}
}
