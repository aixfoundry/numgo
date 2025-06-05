package numgo

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"math/rand"
	"strconv"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// Color represents a 32-bit color value.
type Color int32

const (
	ColorBlack        Color = 0x000000
	ColorMaroon       Color = 0x800000
	ColorGreen        Color = 0x008000
	ColorOlive        Color = 0x808000
	ColorNavy         Color = 0x000080
	ColorPurple       Color = 0x800080
	ColorTeal         Color = 0x008080
	ColorSilver       Color = 0xC0C0C0
	ColorGray         Color = 0x808080
	ColorRed          Color = 0xFF0000
	ColorLime         Color = 0x00FF00
	ColorYellow       Color = 0xFFFF00
	ColorBlue         Color = 0x0000FF
	ColorFuchsia      Color = 0xFF00FF
	ColorAqua         Color = 0x00FFFF
	ColorWhite        Color = 0xFFFFFF
	ColorBrightRed    Color = 0xFF6666 // 亮红色
	ColorBrightGreen  Color = 0x66FF66 // 亮绿色
	ColorBrightBlue   Color = 0x66B2FF // 亮蓝色
	ColorBrightOrange Color = 0xFFB266 // 亮橙色
	ColorBrightPurple Color = 0xCC99FF // 亮紫色
	ColorBrightCyan   Color = 0x66FFFF // 亮青色
	ColorBrightPink   Color = 0xFF66FF // 亮粉红
	ColorBrightYellow Color = 0xFFFF66 // 亮黄色
	ColorLavender     Color = 0xE5CCFF // 淡紫色
	ColorTurquoise    Color = 0x66FFB2 // 蓝绿色
)

// RGBA converts the Color to a color.RGBA value.
func (c Color) RGBA() (color color.RGBA) {
	color.R = uint8(c >> 16)
	color.G = uint8(c >> 8)
	color.B = uint8(c)
	color.A = 0xFF
	return
}

// HSVColor converts HSV color values to RGB and returns a Color.
//
// Example:
//
//	color := HSVColor(120, 100, 100) // Returns a Color value representing a pure green
func HSVColor(h int, s, v float64) Color {
	s /= 100
	v /= 100

	c := v * s
	x := c * (1 - math.Abs(float64(h/60%2-1)))
	m := v - c

	var r, g, b float64
	switch {
	case h < 60:
		r, g, b = c, x, 0
	case h < 120:
		r, g, b = x, c, 0
	case h < 180:
		r, g, b = 0, c, x
	case h < 240:
		r, g, b = 0, x, c
	case h < 300:
		r, g, b = x, 0, c
	default:
		r, g, b = c, 0, x
	}
	return Color(int32((r+m)*255)<<16 | int32((g+m)*255)<<8 | int32((b+m)*255))
}

// calcTicks calculates the tick positions for a given range.
func calcTicks(min, max float64) (ticks []float64) {
	// 强制包含0值
	if min > 0 {
		min = 0
	} else if max < 0 {
		max = 0
	}
	// 处理特殊情况
	if min == 0 && max == 0 {
		ticks = append(ticks, 0)
		return
	}

	// 计算数据范围
	dataRange := max - min

	// 1. 计算初始步长（目标5-8个刻度）
	targetTicks := 6.0
	roughStep := dataRange / (targetTicks - 1)

	// 2. 计算数量级
	exponent := math.Floor(math.Log10(roughStep))
	magnitude := math.Pow(10, exponent)

	// 3. 归一化步长
	normalizedStep := roughStep / magnitude

	// 4. 定义友好步长候选集
	candidateSteps := []float64{0.1, 0.15, 0.2, 0.25, 0.5, 1, 1.5, 2, 2.5, 5, 10}

	// 5. 选择最合适的步长
	var bestStep float64
	bestScore := math.MaxFloat64
	for _, candidate := range candidateSteps {
		score := math.Abs(candidate - normalizedStep)
		if score < bestScore || (score == bestScore && candidate > bestStep) {
			bestScore = score
			bestStep = candidate
		}
	}

	// 6. 计算实际步长
	step := bestStep * magnitude

	// 7. 计算刻度边界
	firstTick := math.Floor(min/step) * step
	lastTick := math.Ceil(max/step) * step

	// 8. 生成刻度序列
	numTicks := int(math.Round((lastTick-firstTick)/step)) + 1
	for i := range numTicks {
		ticks = append(ticks, firstTick+step*float64(i))
	}
	return
}

func drawText(img draw.Image, xy image.Point, text string, color color.RGBA) {
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(xy.X, xy.Y),
	}
	d.DrawString(text)
}

func drawLine(img draw.Image, xy1, xy2 image.Point, color color.RGBA, stribe int) {
	ymin, ymax := float64(min(xy1.Y, xy2.Y)), float64(max(xy1.Y, xy2.Y))
	xmin, xmax := float64(min(xy1.X, xy2.X)), float64(max(xy1.X, xy2.X))
	if xmax == xmin {
		for y := ymin; y <= ymax; y++ {
			for i := range stribe {
				img.Set(int(xmin)+i, int(y), color)
			}
		}
	} else if ymax == ymin {
		for x := xmin; x <= xmax; x++ {
			for i := range stribe {
				img.Set(int(x), int(ymin)+i, color)
			}
		}
	} else {
		a := (ymax - ymin) / (xmax - xmin)
		for x, y := xmin, ymin; x <= xmax && y <= ymax; x, y = x+1, y+a {
			for i := range stribe {
				img.Set(int(x)+i, int(y), color)
			}
		}
	}
}

func (a *Array) drawXAxes(img draw.Image) (step float64, xmin, xmax float64) {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	xpad := width / 10
	ypad := height / 10

	// 画X轴直线
	f, t := image.Pt(xpad, height-ypad), image.Pt(width-(xpad/2), height-ypad)
	drawLine(img, f, t, ColorGray.RGBA(), 2)

	xlen := float64(t.X) - float64(f.X) - (float64(xpad) / 2.0) - 10.0 // 计算label之间宽度
	ticks := calcTicks(0, float64(a.r))
	numTicks := len(ticks)
	xsplit := float64(numTicks - 1)
	xstep := xlen / xsplit
	for i := range numTicks {
		x := f.X + int(xstep*float64(i)) + 10                                    // pad left 10 pixels
		drawLine(img, image.Pt(x, f.Y), image.Pt(x, f.Y+10), ColorRed.RGBA(), 2) // label 15 pixels high
		text := strconv.Itoa(int(ticks[i]))
		textWidth := basicfont.Face7x13.Width * len(text)
		drawText(img, image.Pt(x-(textWidth/2), f.Y+23), text, ColorWhite.RGBA())
	}
	xmin = ticks[0]
	xmax = ticks[numTicks-1]
	step = xlen / (xmax - xmin)
	return
}

func (a *Array) drawYAxes(img draw.Image) (step float64, ymin, ymax float64) {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	xpad, ypad := width/10, height/10

	// 画Y轴直线
	f, t := image.Pt(xpad, ypad/2), image.Pt(xpad, height-ypad)
	drawLine(img, f, t, ColorGray.RGBA(), 2)

	ylen := float64(t.Y) - float64(f.Y) - (float64(ypad) / 2.0) - 10.0 // 计算label之间宽度
	vmin, vmax := a.Min(-1).At(0, 0), a.Max(-1).At(0, 0)
	ticks := calcTicks(vmin, vmax)
	numTicks := len(ticks)
	ysplit := float64(numTicks - 1)
	ystep := ylen / ysplit
	for i := range numTicks {
		y := t.Y - int(ystep*float64(i)) - 10                                    // pad left 10 pixels
		drawLine(img, image.Pt(f.X-10, y), image.Pt(f.X, y), ColorRed.RGBA(), 2) // label 10 pixels high
		text := strconv.Itoa(int(ticks[i]))
		textWidth := basicfont.Face7x13.Width * len(text)
		textHeight := basicfont.Face7x13.Height
		drawText(img, image.Pt(f.X-17-textWidth, y+textHeight/2), text, ColorWhite.RGBA())
	}
	ymin = ticks[0]
	ymax = ticks[numTicks-1]
	step = ylen / (ymax - ymin)
	return
}

// randColors generates a list of random colors.
func randColors(count int) (colors []Color) {
	bases := []Color{
		ColorBrightRed,
		ColorBrightGreen,
		ColorBrightBlue,
		ColorBrightOrange,
		ColorBrightPurple,
		ColorBrightCyan,
		ColorBrightPink,
		ColorBrightYellow,
		ColorLavender,
		ColorTurquoise,
	}
	if count <= len(bases) {
		colors = bases[:count]
		return
	}

	colors = make([]Color, count)
	copy(colors, bases)

	// Generate additional colors to ensure visibility on a black background
	for i := len(bases); i < count; i++ {
		// Use the HSV model for better control over brightness and saturation
		hue := rand.Intn(360)
		saturation := 70 + rand.Intn(30) // 70-100% saturation
		value := 80 + rand.Intn(20)      // 80-100% brightness
		colors[i] = HSVColor(hue, float64(saturation), float64(value))
	}
	return
}

// drawPoints draws the data points on the image.
func (a *Array) drawPoints(img draw.Image, xstep, ystep float64, vmin float64) {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	xpad, ypad := width/10, height/10
	xstart := xpad + 10
	ystart := height - ypad - 10

	colors := randColors(a.c)

	for k, v := range a.All() {
		// Calculate the position of the point
		x := xstart + int(xstep*float64(k[0]))
		y := ystart - int(ystep*(v-vmin))
		img.Set(x, y, colors[k[1]].RGBA())
	}
}

// Draw renders the graph on the provided image.
func (a *Array) Draw(img draw.Image) {
	xstep, _, _ := a.drawXAxes(img)
	ystep, ymin, _ := a.drawYAxes(img)
	a.drawPoints(img, xstep, ystep, ymin)
}
