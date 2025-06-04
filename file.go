package numgo

import (
	"bufio"
	"fmt"
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
func (a *Array) SaveToText(filename string) (err error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return
	}
	defer file.Close()

	for i := range a.r {
		row := a.Row(i)
		for k, v := range row.All() {
			s := fmt.Sprintf("%g ", v)
			if k[1] == row.c-1 {
				s = fmt.Sprintf("%g", v)
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
func SaveToText(a *Array, filename string) (err error) {
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
func LoadFromText(filename string) (a *Array, err error) {
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
func (a *Array) SaveToCsv(filename string) (err error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return
	}
	defer file.Close()

	for i := range a.r {
		row := a.Row(i)
		for k, v := range row.All() {
			s := fmt.Sprintf("%g,", v)
			if k[1] == row.c-1 {
				s = fmt.Sprintf("%g", v)
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
func SaveToCsv(a *Array, filename string) (err error) {
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
func LoadFromCsv(filename string) (a *Array, err error) {
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
func LoadFromFile(filename string) (a *Array, err error) {
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
// It supports writing to files with .txt and .csv extensions.
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
func (a *Array) SaveToFile(filename string) (err error) {
	ext := strings.ToLower(path.Ext(filename))
	switch ext {
	case ".txt":
		err = a.SaveToText(filename)
	case ".csv":
		err = a.SaveToCsv(filename)
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
func SaveToFile(a *Array, filename string) (err error) {
	return a.SaveToFile(filename)
}
