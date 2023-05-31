package main

import (
	"fmt"
	"runtime"
	"xgb_test/xgbmod/go-xgboost"
)

func main() {
	file := "./model/xgb_20230531_v1.pkl"
	mod, err := xgboost.NewPredictor(file, runtime.NumCPU(), 0, 0, -1)
	if err != nil {
		panic(err)
	}
	features := []float32{11, 0, 9,
		5, 9, 9,
		3, 0, 0,
		0, -1, 1130,
		-1, -1, 12,
		-1, -1, -1,
		-1, 1, 1,
		0, 0, 0,
		0, 0, 0,
		0, 3966, 8777,
		16549, 24135, 0.87842226,
		0.83284336, 0.8662653, 0.8793305,
		0.8835481, 1277, 195,
		255, 366, 566,
		870}
	score, err := mod.Predict(xgboost.FloatSliceVector(features))
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("score: ", score)
}
