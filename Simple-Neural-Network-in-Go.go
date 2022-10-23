package main

import (
	. "fmt"
	. "math"
)

func main() {

	var (
		numberOfInputVal int
		iterations       int
		outputLog        int
		eta              float64
		meanSquaredError float64
		Input            []float64
		W1               [][]float64
		B2               []float64
		W2               [][]float64
		B3               []float64
		Y                []float64
	)

	Input, B2, B3, Y, W1, W2 = inputData(&numberOfInputVal, &iterations, &outputLog, &eta, &Input, &B2, &B3, &Y, &W1, &W2)

	for l := 0; l < iterations; l++ {

		// Forward Propagation

		Hidden := HiddenLayer(numberOfInputVal, &Input, &B2, &W1)

		Output := OutputLayer(numberOfInputVal, Hidden, &B3, &W2)

		// Backward Propagation

		// W2

		d := d(numberOfInputVal, Output, &Y)

		dW2 := dW2(numberOfInputVal, d, Hidden)

		// B3

		dB3 := dB3(numberOfInputVal, d)

		// W1

		dW2A := dW2A(numberOfInputVal, d, &W2)

		dW1 := dW1(numberOfInputVal, dW2A, &Input, Hidden)

		// B2

		dB2 := dB2(numberOfInputVal, dW2A, Hidden)

		// Output Log

		meanSquaredError = 0

		if l != 0 && l%outputLog == 0 {
			Println()

			Println("Epoch: ", l)

			for i := 0; i < numberOfInputVal; i++ {
				meanSquaredError += Pow(Y[i]-(*Output)[i], 2)
			}

			Printf("Error: %.9f\n", meanSquaredError/float64(numberOfInputVal))

			Println()

			Println("=================== Weights W1 ===================")

			for i := 0; i < numberOfInputVal; i++ {
				for j := 0; j < numberOfInputVal; j++ {
					Printf("%.9f ", W1[i][j])
				}
				Println()
			}

			Println()

			Println("=================== Weights W2 ===================")

			for i := 0; i < numberOfInputVal; i++ {
				for j := 0; j < numberOfInputVal; j++ {
					Printf("%.9f ", W2[i][j])
				}
				Println()
			}

			Println()

			Println("===================   Output   ===================")

			for i := 0; i < numberOfInputVal; i++ {
				Printf("%.9f ", (*Output)[i])
			}

			Println()
		}

		// Updating Weights

		for i := 0; i < numberOfInputVal; i++ {
			for j := 0; j < numberOfInputVal; j++ {
				W2[i][j] += eta * (*dW2)[i][j]
			}
		}

		for i := 0; i < numberOfInputVal; i++ {
			B3[i] += eta * (*dB3)[i]
		}

		for i := 0; i < numberOfInputVal; i++ {
			for j := 0; j < numberOfInputVal; j++ {
				W1[i][j] += eta * (*dW1)[i][j]
			}
		}

		for i := 0; i < numberOfInputVal; i++ {
			B2[i] += eta * (*dB2)[i]
		}
	}
}

func HiddenLayer(numberOfInputVal int, Input, B2 *[]float64, W1 *[][]float64) *[]float64 {

	var Hidden = make([]float64, numberOfInputVal, numberOfInputVal)

	for i := 0; i < numberOfInputVal; i++ {
		for j := 0; j < numberOfInputVal; j++ {
			Hidden[i] += (*W1)[i][j] * (*Input)[j]
		}
		Hidden[i] = 1 / (1 + (Exp(-Hidden[i] - (*B2)[i])))
	}

	return &Hidden
}

func OutputLayer(numberOfInputVal int, Hidden, B3 *[]float64, W2 *[][]float64) *[]float64 {

	var Output = make([]float64, numberOfInputVal, numberOfInputVal)

	for i := 0; i < numberOfInputVal; i++ {
		for j := 0; j < numberOfInputVal; j++ {
			Output[i] += (*W2)[i][j] * (*Hidden)[j]
		}
		Output[i] = 1 / (1 + (Exp(-Output[i] - (*B3)[i])))
	}

	return &Output
}

func d(numberOfInputVal int, Output, Y *[]float64) *[]float64 {

	var d = make([]float64, numberOfInputVal, numberOfInputVal)

	for i := 0; i < numberOfInputVal; i++ {
		d[i] = ((*Y)[i] - (*Output)[i]) * (*Output)[i] * (1 - (*Output)[i])
	}

	return &d
}

func dW2(numberOfInputVal int, d, Hidden *[]float64) *[][]float64 {

	var dW2 = make([][]float64, numberOfInputVal, numberOfInputVal)

	for i := 0; i < numberOfInputVal; i++ {
		dW2[i] = make([]float64, numberOfInputVal)
	}

	for i := 0; i < numberOfInputVal; i++ {
		for j := 0; j < numberOfInputVal; j++ {
			dW2[i][j] = 2 * ((*d)[i] * (*Hidden)[j]) / float64(numberOfInputVal)
		}
	}

	return &dW2
}

func dB3(numberOfInputVal int, d *[]float64) *[]float64 {

	var dB3 = make([]float64, numberOfInputVal, numberOfInputVal)

	for i := 0; i < numberOfInputVal; i++ {
		dB3[i] = 2 * (*d)[i] / float64(numberOfInputVal)
	}

	return &dB3
}

func dW2A(numberOfInputVal int, d *[]float64, W2 *[][]float64) *[]float64 {

	var dW2A = make([]float64, numberOfInputVal, numberOfInputVal)

	for i := 0; i < numberOfInputVal; i++ {
		for j := 0; j < numberOfInputVal; j++ {
			dW2A[i] += (*d)[j] * (*W2)[j][i]
		}
	}

	return &dW2A
}

func dW1(numberOfInputVal int, dW2A, Input, Hidden *[]float64) *[][]float64 {

	var dW1 = make([][]float64, numberOfInputVal, numberOfInputVal)

	for i := 0; i < numberOfInputVal; i++ {
		dW1[i] = make([]float64, numberOfInputVal)
	}

	for i := 0; i < numberOfInputVal; i++ {
		for j := 0; j < numberOfInputVal; j++ {
			dW1[i][j] = 2 * ((*dW2A)[i] * (*Hidden)[i] * (1 - (*Hidden)[i]) * (*Input)[j]) / float64(numberOfInputVal)
		}
	}

	return &dW1
}

func dB2(numberOfInputVal int, dW2A, Hidden *[]float64) *[]float64 {

	var dB2 = make([]float64, numberOfInputVal, numberOfInputVal)

	for i := 0; i < numberOfInputVal; i++ {
		dB2[i] = 2 * ((*dW2A)[i] * (*Hidden)[i] * (1 - (*Hidden)[i]) * 1) / float64(numberOfInputVal)
	}

	return &dB2
}

func inputData(numberOfInputVal, iterations, outputLog *int, eta *float64, Input, B2, B3, Y *[]float64, W1, W2 *[][]float64) ([]float64, []float64, []float64, []float64, [][]float64, [][]float64) {

	var (
		defData string
		err     error
	)

	for ok := true; ok; ok = defData != "Y" && defData != "y" && defData != "N" && defData != "n" {
		Print("Use default data (Y/N) ")
		_, err = Scan(&defData)
		if err != nil {
			Println(err)
		}
	}

	if defData == "Y" || defData == "y" {

		// Default data

		*numberOfInputVal = 3
		*eta = 0.01
		*iterations = 10001
		*outputLog = 1000
		Input = &[]float64{0.03, 0.72, 0.49}
		W1 = &[][]float64{{0.88, 0.39, 0.9}, {0.37, 0.14, 0.41}, {0.96, 0.5, 0.6}}
		B2 = &[]float64{0.23, 0.89, 0.08}
		W2 = &[][]float64{{0.29, 0.57, 0.36}, {0.73, 0.53, 0.68}, {0.01, 0.02, 0.58}}
		B3 = &[]float64{0.78, 0.83, 0.80}
		Y = &[]float64{0.93, 0.74, 0.17}

	} else if defData == "N" || defData == "n" {

		for ok := true; ok; ok = err != nil {
			Print("Number of input values? ")
			_, err = Scan(&*numberOfInputVal)
			if err != nil {
				Println(err)
			}
		}

		for ok := true; ok; ok = err != nil {
			Print("Learning rate? ")
			_, err = Scan(&*eta)
			if err != nil {
				Println(err)
			}
		}

		for ok := true; ok; ok = err != nil {
			Print("Number of iterations? ")
			_, err = Scan(&*iterations)
			if err != nil {
				Println(err)
			}
		}

		for ok := true; ok; ok = err != nil {
			Print("Value 'outputLog'? (Depends on the number of iterations.) ")
			// Example. If "iterations" = 101 and "outputLog" = 10: iterations / outputLog = 10 log messages.
			_, err = Scan(&*outputLog)
			if err != nil {
				Println(err)
			}
		}

		Println()

		*Input = make([]float64, *numberOfInputVal, *numberOfInputVal)
		*W1 = make([][]float64, *numberOfInputVal, *numberOfInputVal)
		*B2 = make([]float64, *numberOfInputVal, *numberOfInputVal)
		*W2 = make([][]float64, *numberOfInputVal, *numberOfInputVal)
		*B3 = make([]float64, *numberOfInputVal, *numberOfInputVal)
		*Y = make([]float64, *numberOfInputVal, *numberOfInputVal)

		for i := 0; i < *numberOfInputVal; i++ {
			(*W1)[i] = make([]float64, *numberOfInputVal)
		}

		for i := 0; i < *numberOfInputVal; i++ {
			(*W2)[i] = make([]float64, *numberOfInputVal)
		}

		for i := 0; i < *numberOfInputVal; i++ {
			Print("Input_", i, " ")
			_, err = Scan(&(*Input)[i])
			if err != nil {
				Println(err)
				i -= 1
			}
		}

		Println()

		for i := 0; i < *numberOfInputVal; i++ {
			for j := 0; j < *numberOfInputVal; j++ {
				Print("Weight matrix W1_", i, "", j, " ")
				_, err = Scan(&(*W1)[i][j])
				if err != nil {
					Println(err)
					j -= 1
				}
			}
		}

		Println()

		for i := 0; i < *numberOfInputVal; i++ {
			Print("Bias B2_", i, " ")
			_, err = Scan(&(*B2)[i])
			if err != nil {
				Println(err)
				i -= 1
			}
		}

		Println()

		for i := 0; i < *numberOfInputVal; i++ {
			for j := 0; j < *numberOfInputVal; j++ {
				Print("Weight matrix W2_", i, "", j, " ")
				_, err = Scan(&(*W2)[i][j])
				if err != nil {
					Println(err)
					j -= 1
				}
			}
		}

		Println()

		for i := 0; i < *numberOfInputVal; i++ {
			Print("Bias B3_", i, " ")
			_, err = Scan(&(*B3)[i])
			if err != nil {
				Println(err)
				i -= 1
			}
		}

		Println()

		for i := 0; i < *numberOfInputVal; i++ {
			Print("Target values Y_", i, " ")
			_, err = Scan(&(*Y)[i])
			if err != nil {
				Println(err)
				i -= 1
			}
		}

		Println()
	}

	return *Input, *B2, *B3, *Y, *W1, *W2
}
