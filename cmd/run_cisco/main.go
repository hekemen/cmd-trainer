package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

func main() {
	// 1. NewSession was moved to hugot.NewSession()
	// but check that you have the right package import
	session, err := hugot.NewORTSession(
		options.WithOnnxLibraryPath("/lib/hugot/"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Destroy()

	// 2. Setup the configuration
	// Note: The task-specific configs are now in the pipelines package
	config := hugot.TextClassificationConfig{
		ModelPath: "./models/distilbert-cisco-onnx-int8",
		Name:      "command_analysis",
	}

	// 3. NewTextClassificationPipeline is now called directly
	// from the pipelines package, passing the session as the first argument
	commandPipeline, err := hugot.NewPipeline(session, config)
	if err != nil {
		log.Fatal(err)
	}

	// 4. Run Inference
	batch := []string{"This updated library is a bit tricky but very powerful!", "worst than ever", "wtf top of top", "shutdown", "init 0"}

	// Hugot's Run method returns a slice of results
	results, err := commandPipeline.Run(batch)
	if err != nil {
		log.Fatal(err)
	}

	s, err := json.Marshal(results)

	fmt.Println(string(s))
}
