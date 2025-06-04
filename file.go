package numgo

import (
	"bufio"
	"fmt"
	"os"
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
