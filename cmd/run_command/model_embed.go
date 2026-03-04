package main

import (
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

//go:embed model_data/*
var modelFS embed.FS

// extractModel extracts the embedded model files to a temporary directory
// and returns the path to the extracted model directory.
// The caller is responsible for cleaning up with os.RemoveAll.
func extractModel() (string, error) {
	tmpDir, err := os.MkdirTemp("", "hugot-model-*")
	if err != nil {
		return "", fmt.Errorf("creating temp dir: %w", err)
	}

	err = fs.WalkDir(modelFS, "model_data", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Get relative path under model_data/
		relPath, err := filepath.Rel("model_data", path)
		if err != nil {
			return err
		}
		targetPath := filepath.Join(tmpDir, relPath)

		if d.IsDir() {
			return os.MkdirAll(targetPath, 0755)
		}

		data, err := modelFS.ReadFile(path)
		if err != nil {
			return fmt.Errorf("reading embedded file %s: %w", path, err)
		}

		return os.WriteFile(targetPath, data, 0644)
	})

	if err != nil {
		os.RemoveAll(tmpDir)
		return "", fmt.Errorf("extracting model: %w", err)
	}

	return tmpDir, nil
}
