package numgo

import "fmt"

func ExampleArray_SaveToText() {
	a := RangeArray(0, 10)
	a.Reshape(2, 5)
	err := SaveToText(a, "testdata/output.txt")
	if err != nil {
		return
	}
	// Output:
}

func ExampleLoadFromText() {
	a, err := LoadFromText("testdata/output.txt")
	if err != nil {
		return
	}
	fmt.Println(a)
	// Output:
	// ⎡0  1  2  3  4⎤
	// ⎣5  6  7  8  9⎦
}

func ExampleSaveToCsv() {
	a := RangeArray(0, 10)
	a.Reshape(2, 5)
	err := SaveToCsv(a, "testdata/output.csv")
	if err != nil {
		return
	}
	// Output:
}

func ExampleLoadFromCsv() {
	a, err := LoadFromCsv("testdata/output.csv")
	if err != nil {
		return
	}
	fmt.Println(a)
	// Output:
	// ⎡0  1  2  3  4⎤
	// ⎣5  6  7  8  9⎦
}

func ExampleLoadFromFile_csv() {
	a, err := LoadFromFile("testdata/output.csv")
	if err != nil {
		return
	}
	fmt.Println(a)
	// Output:
	// ⎡0  1  2  3  4⎤
	// ⎣5  6  7  8  9⎦
}

func ExampleLoadFromFile_txt() {
	a, err := LoadFromFile("testdata/output.txt")
	if err != nil {
		return
	}
	fmt.Println(a)
	// Output:
	// ⎡0  1  2  3  4⎤
	// ⎣5  6  7  8  9⎦
}

func ExampleSaveToFile_csv() {
	a := RangeArray(0, 10)
	a.Reshape(2, 5)
	err := SaveToFile(a, "testdata/output.csv")
	if err != nil {
		return
	}
	// Output:
}

func ExampleSaveToFile_txt() {
	a := RangeArray(0, 10)
	a.Reshape(2, 5)
	err := SaveToFile(a, "testdata/output.txt")
	if err != nil {
		return
	}
	// Output:
}

func ExampleSaveToFile_png() {
	a := RangeArray(0, 10)
	a.Reshape(2, 5)
	err := SaveToFile(a, "testdata/output.png")
	if err != nil {
		return
	}
	// Output:
}

func ExampleSaveToPng() {
	a := RangeArray(0, 10)
	a.Reshape(2, 5)
	err := SaveToPng(a, "testdata/output.png")
	if err != nil {
		return
	}
	// Output:
}

func ExampleSaveToFile_jpg() {
	a := RangeArray(0, 10)
	a.Reshape(2, 5)
	err := SaveToFile(a, "testdata/output.jpg")
	if err != nil {
		return
	}
	// Output:
}

func ExampleSaveToJpeg() {
	a := RangeArray(0, 10)
	a.Reshape(2, 5)
	err := SaveToJpeg(a, "testdata/output.jpg")
	if err != nil {
		return
	}
	// Output:
}
