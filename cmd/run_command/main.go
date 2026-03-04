package main

import (
	"bufio"
	"fmt"
	"log"
	"os"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

func main() {
	// Extract embedded model files to a temp directory
	modelDir, err := extractModel()
	if err != nil {
		log.Fatalf("Failed to extract embedded model: %v", err)
	}
	defer os.RemoveAll(modelDir)

	// Initialize Hugot with the ONNX Runtime backend
	onnxLibPath := os.Getenv("ONNX_LIB_PATH")
	if onnxLibPath == "" {
		onnxLibPath = "/usr/lib/"
	}
	session, err := hugot.NewORTSession(
		options.WithOnnxLibraryPath(onnxLibPath),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Destroy()

	// Load the quantized command model from the extracted temp dir
	config := hugot.TextClassificationConfig{
		ModelPath: modelDir,
		Name:      "linux_command_analysis",
	}

	commandPipeline, err := hugot.NewPipeline(session, config)
	if err != nil {
		log.Fatal(err)
	}

	// Read from stdin
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		cmd := scanner.Text()
		if cmd == "" {
			continue
		}

		// Run Inference on single command
		batch := []string{cmd}
		results, err := commandPipeline.Run(batch)
		if err != nil {
			log.Printf("Error running inference on %q: %v\n", cmd, err)
			continue
		}

		// Extract highest scoring label
		var bestLabel string
		var highestScore float32 = -1.0

		for _, prediction := range results.GetOutput()[0].([]pipelines.ClassificationOutput) {
			if prediction.Score > highestScore {
				highestScore = prediction.Score
				bestLabel = prediction.Label
			}
		}

		fmt.Println(bestLabel)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading standard input: %v\n", err)
	}
}
