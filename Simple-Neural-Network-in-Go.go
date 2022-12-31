package main

import (
	. "fmt"
	. "math"
)

const (
	inputValNumber = 3
	epochs         = 10001
	outputLog      = 1000
	eta            = 0.01
)

func hiddenLayer(x, biases2 *[]float64, weights1 *[][]float64) *[]float64 {

	var hidden = make([]float64, inputValNumber, inputValNumber)

	for i := 0; i < inputValNumber; i++ {
		for j := 0; j < inputValNumber; j++ {
			hidden[i] += (*weights1)[i][j] * (*x)[j]
		}
		hidden[i] = 1 / (1 + (Exp(-hidden[i] - (*biases2)[i])))
	}

	return &hidden
}

func outputLayer(hidden, biases3 *[]float64, weights2 *[][]float64) *[]float64 {

	var output = make([]float64, inputValNumber, inputValNumber)

	for i := 0; i < inputValNumber; i++ {
		for j := 0; j < inputValNumber; j++ {
			output[i] += (*weights2)[i][j] * (*hidden)[j]
		}
		output[i] = 1 / (1 + (Exp(-output[i] - (*biases3)[i])))
	}

	return &output
}

func d(output, y *[]float64) *[]float64 {

	var d = make([]float64, inputValNumber, inputValNumber)

	for i := 0; i < inputValNumber; i++ {
		d[i] = ((*y)[i] - (*output)[i]) * (*output)[i] * (1 - (*output)[i])
	}

	return &d
}

func dW2(d, hidden *[]float64) *[][]float64 {

	var dW2 = make([][]float64, inputValNumber, inputValNumber)

	for i := 0; i < inputValNumber; i++ {
		dW2[i] = make([]float64, inputValNumber)
	}

	for i := 0; i < inputValNumber; i++ {
		for j := 0; j < inputValNumber; j++ {
			dW2[i][j] = 2 * ((*d)[i] * (*hidden)[j]) / float64(inputValNumber)
		}
	}

	return &dW2
}

func dB3(d *[]float64) *[]float64 {

	var dB3 = make([]float64, inputValNumber, inputValNumber)

	for i := 0; i < inputValNumber; i++ {
		dB3[i] = 2 * (*d)[i] / float64(inputValNumber)
	}

	return &dB3
}

func dW2A(d *[]float64, weights2 *[][]float64) *[]float64 {

	var dW2A = make([]float64, inputValNumber, inputValNumber)

	for i := 0; i < inputValNumber; i++ {
		for j := 0; j < inputValNumber; j++ {
			dW2A[i] += (*d)[j] * (*weights2)[j][i]
		}
	}

	return &dW2A
}

func dW1(dW2A, x, hidden *[]float64) *[][]float64 {

	var dW1 = make([][]float64, inputValNumber, inputValNumber)

	for i := 0; i < inputValNumber; i++ {
		dW1[i] = make([]float64, inputValNumber)
	}

	for i := 0; i < inputValNumber; i++ {
		for j := 0; j < inputValNumber; j++ {
			dW1[i][j] = 2 * ((*dW2A)[i] * (*hidden)[i] * (1 - (*hidden)[i]) * (*x)[j]) / float64(inputValNumber)
		}
	}

	return &dW1
}

func dB2(dW2A, hidden *[]float64) *[]float64 {

	var dB2 = make([]float64, inputValNumber, inputValNumber)

	for i := 0; i < inputValNumber; i++ {
		dB2[i] = 2 * ((*dW2A)[i] * (*hidden)[i] * (1 - (*hidden)[i]) * 1) / float64(inputValNumber)
	}

	return &dB2
}

func main() {

	var (
		X                = []float64{0.03, 0.72, 0.49}
		Y                = []float64{0.93, 0.74, 0.17}
		weights1         = [][]float64{{0.88, 0.39, 0.9}, {0.37, 0.14, 0.41}, {0.96, 0.5, 0.6}}
		biases2          = []float64{0.23, 0.89, 0.08}
		weights2         = [][]float64{{0.29, 0.57, 0.36}, {0.73, 0.53, 0.68}, {0.01, 0.02, 0.58}}
		biases3          = []float64{0.78, 0.83, 0.80}
		meanSquaredError float64
	)

	for l := 0; l < epochs; l++ {

		// Forward Propagation

		hidden := hiddenLayer(&X, &biases2, &weights1)

		output := outputLayer(hidden, &biases3, &weights2)

		// Backward Propagation

		// weights2

		d := d(output, &Y)

		dW2 := dW2(d, hidden)

		// biases3

		dB3 := dB3(d)

		// weights1

		dW2A := dW2A(d, &weights2)

		dW1 := dW1(dW2A, &X, hidden)

		// biases2

		dB2 := dB2(dW2A, hidden)

		// Output Log

		meanSquaredError = 0

		if l != 0 && l%outputLog == 0 {
			Println()

			Println("Epoch: ", l)

			for i := 0; i < inputValNumber; i++ {
				meanSquaredError += Pow(Y[i]-(*output)[i], 2)
			}

			Printf("Error: %.9f\n", meanSquaredError/float64(inputValNumber))

			Println()

			Println("=================== Weights 1 ===================")

			for i := 0; i < inputValNumber; i++ {
				for j := 0; j < inputValNumber; j++ {
					Printf("%.9f ", weights1[i][j])
				}
				Println()
			}

			Println()

			Println("=================== Weights 2 ===================")

			for i := 0; i < inputValNumber; i++ {
				for j := 0; j < inputValNumber; j++ {
					Printf("%.9f ", weights2[i][j])
				}
				Println()
			}

			Println()

			Println("===================   Output   ===================")

			for i := 0; i < inputValNumber; i++ {
				Printf("%.9f ", (*output)[i])
			}

			Println()
		}

		// Updating Weights

		for i := 0; i < inputValNumber; i++ {
			for j := 0; j < inputValNumber; j++ {
				weights2[i][j] += eta * (*dW2)[i][j]
			}
		}

		for i := 0; i < inputValNumber; i++ {
			biases3[i] += eta * (*dB3)[i]
		}

		for i := 0; i < inputValNumber; i++ {
			for j := 0; j < inputValNumber; j++ {
				weights1[i][j] += eta * (*dW1)[i][j]
			}
		}

		for i := 0; i < inputValNumber; i++ {
			biases2[i] += eta * (*dB2)[i]
		}
	}
}
