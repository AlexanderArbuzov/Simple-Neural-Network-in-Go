package main

import (
	. "fmt"
	. "math"
)

func main() {

	var (
		NUMBEROFINPUTVAL = 3
		ITERATIONS       = 10001
		OUTPUTLOG        = 1000
		ETA              = 0.01
		X                = []float64{0.03, 0.72, 0.49}
		Y                = []float64{0.93, 0.74, 0.17}
		weights1         = [][]float64{{0.88, 0.39, 0.9}, {0.37, 0.14, 0.41}, {0.96, 0.5, 0.6}}
		biases2          = []float64{0.23, 0.89, 0.08}
		weights2         = [][]float64{{0.29, 0.57, 0.36}, {0.73, 0.53, 0.68}, {0.01, 0.02, 0.58}}
		biases3          = []float64{0.78, 0.83, 0.80}
		meanSquaredError float64
	)

	for l := 0; l < ITERATIONS; l++ {

		// Forward Propagation

		hidden := hiddenLayer(NUMBEROFINPUTVAL, &X, &biases2, &weights1)

		output := outputLayer(NUMBEROFINPUTVAL, hidden, &biases3, &weights2)

		// Backward Propagation

		// weights2

		d := d(NUMBEROFINPUTVAL, output, &Y)

		dW2 := dW2(NUMBEROFINPUTVAL, d, hidden)

		// biases3

		dB3 := dB3(NUMBEROFINPUTVAL, d)

		// weights1

		dW2A := dW2A(NUMBEROFINPUTVAL, d, &weights2)

		dW1 := dW1(NUMBEROFINPUTVAL, dW2A, &X, hidden)

		// biases2

		dB2 := dB2(NUMBEROFINPUTVAL, dW2A, hidden)

		// Output Log

		meanSquaredError = 0

		if l != 0 && l%OUTPUTLOG == 0 {
			Println()

			Println("Epoch: ", l)

			for i := 0; i < NUMBEROFINPUTVAL; i++ {
				meanSquaredError += Pow(Y[i]-(*output)[i], 2)
			}

			Printf("Error: %.9f\n", meanSquaredError/float64(NUMBEROFINPUTVAL))

			Println()

			Println("=================== Weights 1 ===================")

			for i := 0; i < NUMBEROFINPUTVAL; i++ {
				for j := 0; j < NUMBEROFINPUTVAL; j++ {
					Printf("%.9f ", weights1[i][j])
				}
				Println()
			}

			Println()

			Println("=================== Weights 2 ===================")

			for i := 0; i < NUMBEROFINPUTVAL; i++ {
				for j := 0; j < NUMBEROFINPUTVAL; j++ {
					Printf("%.9f ", weights2[i][j])
				}
				Println()
			}

			Println()

			Println("===================   Output   ===================")

			for i := 0; i < NUMBEROFINPUTVAL; i++ {
				Printf("%.9f ", (*output)[i])
			}

			Println()
		}

		// Updating Weights

		for i := 0; i < NUMBEROFINPUTVAL; i++ {
			for j := 0; j < NUMBEROFINPUTVAL; j++ {
				weights2[i][j] += ETA * (*dW2)[i][j]
			}
		}

		for i := 0; i < NUMBEROFINPUTVAL; i++ {
			biases3[i] += ETA * (*dB3)[i]
		}

		for i := 0; i < NUMBEROFINPUTVAL; i++ {
			for j := 0; j < NUMBEROFINPUTVAL; j++ {
				weights1[i][j] += ETA * (*dW1)[i][j]
			}
		}

		for i := 0; i < NUMBEROFINPUTVAL; i++ {
			biases2[i] += ETA * (*dB2)[i]
		}
	}
}

func hiddenLayer(NUMBEROFINPUTVAL int, X, biases2 *[]float64, weights1 *[][]float64) *[]float64 {

	var hidden = make([]float64, NUMBEROFINPUTVAL, NUMBEROFINPUTVAL)

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		for j := 0; j < NUMBEROFINPUTVAL; j++ {
			hidden[i] += (*weights1)[i][j] * (*X)[j]
		}
		hidden[i] = 1 / (1 + (Exp(-hidden[i] - (*biases2)[i])))
	}

	return &hidden
}

func outputLayer(NUMBEROFINPUTVAL int, hidden, biases3 *[]float64, weights2 *[][]float64) *[]float64 {

	var output = make([]float64, NUMBEROFINPUTVAL, NUMBEROFINPUTVAL)

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		for j := 0; j < NUMBEROFINPUTVAL; j++ {
			output[i] += (*weights2)[i][j] * (*hidden)[j]
		}
		output[i] = 1 / (1 + (Exp(-output[i] - (*biases3)[i])))
	}

	return &output
}

func d(NUMBEROFINPUTVAL int, output, Y *[]float64) *[]float64 {

	var d = make([]float64, NUMBEROFINPUTVAL, NUMBEROFINPUTVAL)

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		d[i] = ((*Y)[i] - (*output)[i]) * (*output)[i] * (1 - (*output)[i])
	}

	return &d
}

func dW2(NUMBEROFINPUTVAL int, d, hidden *[]float64) *[][]float64 {

	var dW2 = make([][]float64, NUMBEROFINPUTVAL, NUMBEROFINPUTVAL)

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		dW2[i] = make([]float64, NUMBEROFINPUTVAL)
	}

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		for j := 0; j < NUMBEROFINPUTVAL; j++ {
			dW2[i][j] = 2 * ((*d)[i] * (*hidden)[j]) / float64(NUMBEROFINPUTVAL)
		}
	}

	return &dW2
}

func dB3(NUMBEROFINPUTVAL int, d *[]float64) *[]float64 {

	var dB3 = make([]float64, NUMBEROFINPUTVAL, NUMBEROFINPUTVAL)

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		dB3[i] = 2 * (*d)[i] / float64(NUMBEROFINPUTVAL)
	}

	return &dB3
}

func dW2A(NUMBEROFINPUTVAL int, d *[]float64, weights2 *[][]float64) *[]float64 {

	var dW2A = make([]float64, NUMBEROFINPUTVAL, NUMBEROFINPUTVAL)

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		for j := 0; j < NUMBEROFINPUTVAL; j++ {
			dW2A[i] += (*d)[j] * (*weights2)[j][i]
		}
	}

	return &dW2A
}

func dW1(NUMBEROFINPUTVAL int, dW2A, X, hidden *[]float64) *[][]float64 {

	var dW1 = make([][]float64, NUMBEROFINPUTVAL, NUMBEROFINPUTVAL)

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		dW1[i] = make([]float64, NUMBEROFINPUTVAL)
	}

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		for j := 0; j < NUMBEROFINPUTVAL; j++ {
			dW1[i][j] = 2 * ((*dW2A)[i] * (*hidden)[i] * (1 - (*hidden)[i]) * (*X)[j]) / float64(NUMBEROFINPUTVAL)
		}
	}

	return &dW1
}

func dB2(NUMBEROFINPUTVAL int, dW2A, hidden *[]float64) *[]float64 {

	var dB2 = make([]float64, NUMBEROFINPUTVAL, NUMBEROFINPUTVAL)

	for i := 0; i < NUMBEROFINPUTVAL; i++ {
		dB2[i] = 2 * ((*dW2A)[i] * (*hidden)[i] * (1 - (*hidden)[i]) * 1) / float64(NUMBEROFINPUTVAL)
	}

	return &dB2
}
