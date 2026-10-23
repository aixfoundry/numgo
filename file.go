package numgo

import (
	"bufio"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path"
	"strings"
)

// SaveToText saves the array to a text file with each element separated by spaces.
//
// Example:
//
//	a := New2DArray([][]float64{{1.0, 2.0}, {3.0, 4.0}})
//	err := a.SaveToText("output.txt")
//
//	if err != nil {
//	    fmt.Println("Error:", err)
//	}
func (a *Array[T]) SaveToText(filename string) (err error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return
	}
	defer file.Close()

	for i := range a.r {
		row := a.Row(i)
		for k, v := range row.All() {
			s := fmt.Sprintf("%g ", toF64(v))
			if k[1] == row.c-1 {
				s = fmt.Sprintf("%g", toF64(v))
			}
			_, err = file.WriteString(s)
			if err != nil {
				return
			}
		}
		_, err = file.WriteString("\n")
		if err != nil {
			return
		}
	}
	return
}

// SaveToText saves the array to a text file with each element separated by spaces.
//
// Example:
//
//	a := New2DArray([][]float64{{1.0, 2.0}, {3.0, 4.0}})
//	err := SaveToText(a, "output.txt")
//
//	if err != nil {
//	    fmt.Println("Error:", err)
//	}
func SaveToText[T Float](a *Array[T], filename string) (err error) {
	return a.SaveToText(filename)
}

// LoadFromText loads an array from a space-delimited text file.
//
// Example:
//
//	a, err := LoadFromText("input.txt")
//
//	if err != nil {
//	    fmt.Println("Error:", err)
//	}
func LoadFromText(filename string) (a *Array[float64], err error) {
	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()

	data := [][]float64{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		text := scanner.Text()
		// Assuming the file is space-delimited
		values := []float64{}
		for _, s := range strings.Fields(text) {
			var f float64
			_, err := fmt.Sscanf(s, "%f", &f)
			if err != nil {
				return nil, err
			}
			values = append(values, f)
		}
		data = append(data, values)
	}
	err = scanner.Err()
	if err != nil {
		return
	}
	a = New2DArray(data)
	return
}

// SaveToCsv saves the array to a CSV file with each element separated by commas.
//
// Example:
//
//	a := New2DArray([][]float64{{1.0, 2.0}, {3.0, 4.0}})
//	err := a.SaveToCsv("output.csv")
//
//	if err != nil {
//	    fmt.Println("Error:", err)
//	}
func (a *Array[T]) SaveToCsv(filename string) (err error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return
	}
	defer file.Close()

	for i := range a.r {
		row := a.Row(i)
		for k, v := range row.All() {
			s := fmt.Sprintf("%g,", toF64(v))
			if k[1] == row.c-1 {
				s = fmt.Sprintf("%g", toF64(v))
			}
			_, err = file.WriteString(s)
			if err != nil {
				return
			}
		}
		_, err = file.WriteString("\n")
		if err != nil {
			return
		}
	}
	return
}

// SaveToCsv saves the array to a CSV file with each element separated by commas.
//
// Example:
//
//	a := New2DArray([][]float64{{1.0, 2.0}, {3.0, 4.0}})
//	err := SaveToCsv(a, "output.csv")
//
//	if err != nil {
//	    fmt.Println("Error:", err)
//	}
func SaveToCsv[T Float](a *Array[T], filename string) (err error) {
	return a.SaveToCsv(filename)
}

// LoadFromCsv loads an array from a comma-delimited CSV file.
//
// Example:
//
//	a, err := LoadFromCsv("input.csv")
//
//	if err != nil {
//	    fmt.Println("Error:", err)
//	}
func LoadFromCsv(filename string) (a *Array[float64], err error) {
	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()

	data := [][]float64{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		text := scanner.Text()
		// Assuming the file is comma-delimited
		values := []float64{}
		for _, s := range strings.Split(text, ",") {
			var f float64
			_, err := fmt.Sscanf(s, "%f", &f)
			if err != nil {
				return nil, err
			}
			values = append(values, f)
		}
		data = append(data, values)
	}
	err = scanner.Err()
	if err != nil {
		return
	}
	a = New2DArray(data)
	return
}

// LoadFromFile reads data from a file and returns an Array and an error.
// The function supports .txt and .csv file extensions.
// For .txt files, it uses [LoadFromText], and for .csv files, it uses [LoadFromCsv].
// If the file extension is unsupported, it returns an error.
//
// Example:
//
//	array, err := LoadFromFile("data.txt")
//	if err != nil {
//	    log.Fatalf("Failed to load data: %v", err)
//	}
//	fmt.Println(array)
func LoadFromFile(filename string) (a *Array[float64], err error) {
	ext := strings.ToLower(path.Ext(filename))
	switch ext {
	case ".txt":
		a, err = LoadFromText(filename)
	case ".csv":
		a, err = LoadFromCsv(filename)
	default:
		err = fmt.Errorf("unsupported file extension: %s", ext)
	}
	return
}

// SaveToFile writes the contents of the Array to a file with the given filename.
// It supports writing to files with .txt, .csv and .png extensions.
// For other extensions, it returns an error.
//
// Example:
//
//	a := RangeArray(1, 10)
//	err := a.SaveToFile("data.txt")
//
//	if err != nil {
//	    log.Fatalf("Failed to save array: %v", err)
//	}
func (a *Array[T]) SaveToFile(filename string) (err error) {
	ext := strings.ToLower(path.Ext(filename))
	switch ext {
	case ".txt":
		err = a.SaveToText(filename)
	case ".csv":
		err = a.SaveToCsv(filename)
	case ".png":
		err = a.SaveToPng(filename)
	case ".jpg", ".jpeg":
		err = a.SaveToJpeg(filename)
	default:
		err = fmt.Errorf("unsupported file extension: %s", ext)
	}
	return
}

// SaveToFile is a utility function that calls the SaveToFile method on the Array.
//
// Example:
//
//	a := RangeArray(1, 10)
//	err := SaveToFile(a, "data.csv")
//
//	if err != nil {
//	    log.Fatalf("Failed to save array: %v", err)
//	}
func SaveToFile[T Float](a *Array[T], filename string) (err error) {
	return a.SaveToFile(filename)
}

// SaveToPng saves the content of the Array to a PNG file with the given filename.
// It creates a new RGBA image with dimensions 800x600, draws the array onto it, and encodes it as a PNG.
//
// Example:
//
//	a := RangeArray(1, 10)
//	err := a.SaveToPng("output.png")
//
//	if err != nil {
//	    log.Fatalf("Failed to save array to PNG: %v", err)
//	}
func (a *Array[T]) SaveToPng(filename string) (err error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return
	}
	defer file.Close()
	img := image.NewRGBA(image.Rect(0, 0, 800, 600))
	a.Draw(img)
	return png.Encode(file, img)
}

// SaveToPng is a helper function that calls the SaveToPng method on the Array instance.
//
// Example:
//
//	a := RangeArray(1, 10)
//	err := SaveToPng(a, "output.png")
//
//	if err != nil {
//	    log.Fatalf("Failed to save array to PNG: %v", err)
//	}
func SaveToPng[T Float](a *Array[T], filename string) (err error) {
	return a.SaveToPng(filename)
}

// SaveToJpeg saves the contents of the Array to a JPEG file with the given filename.
// It creates a new RGBA image with a size of 800x600 pixels, draws the Array onto this image,
// and then encodes it into a JPEG file with maximum quality.
//
// Example:
//
//	arr := RangeArray(1, 10)
//	err := arr.SaveToJpeg("output.jpg")
//	if err != nil {
//	    log.Fatalf("Failed to save image: %v", err)
//	}
func (a *Array[T]) SaveToJpeg(filename string) (err error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return
	}
	defer file.Close()
	img := image.NewRGBA(image.Rect(0, 0, 800, 600))
	a.Draw(img)
	return jpeg.Encode(file, img, &jpeg.Options{Quality: 100})
}

// SaveToJpeg is a convenience function that calls the SaveToJpeg method on the Array.
//
// Example:
//
//	arr := RangeArray(1, 10)
//	err := SaveToJpeg(arr, "output.jpg")
//	if err != nil {
//	    log.Fatalf("Failed to save image: %v", err)
//	}
func SaveToJpeg[T Float](a *Array[T], filename string) (err error) {
	return a.SaveToJpeg(filename)
}
