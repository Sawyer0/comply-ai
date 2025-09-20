# Go (net/http) example

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

type ErrorBody struct {
	ErrorCode string `json:"error_code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
	Retryable bool   `json:"retryable"`
}

type Response struct {
	Taxonomy    []string               `json:"taxonomy"`
	Confidence  float64                `json:"confidence"`
	VersionInfo map[string]interface{} `json:"version_info"`
	Notes       string                 `json:"notes"`
}

func main() {
	baseURL := getenv("MAPPER_BASE_URL", "http://localhost:8000")
	apiKey := getenv("MAPPER_API_KEY", "YOUR_API_KEY")
	tenantID := getenv("MAPPER_TENANT_ID", "YOUR_TENANT_ID")

	payload := map[string]interface{}{
		"detector":   "orchestrated-multi",
		"output":     "toxic|hate|pii_detected",
		"tenant_id":  tenantID,
		"metadata": map[string]interface{}{
			"contributing_detectors": []string{"deberta-toxicity", "openai-moderation"},
			"aggregation_method":     "weighted_average",
			"coverage_achieved":      1.0,
			"provenance":             []map[string]interface{}{{"detector": "deberta-toxicity", "confidence": 0.93}},
		},
	}
	buf, _ := json.Marshal(payload)

	req, _ := http.NewRequest("POST", baseURL+"/map", bytes.NewReader(buf))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", apiKey)
	req.Header.Set("X-Tenant-ID", tenantID)
	req.Header.Set("Idempotency-Key", "example-req-123")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)

	if resp.StatusCode == 200 {
		var out Response
		_ = json.Unmarshal(b, &out)
		fmt.Println("taxonomy:", out.Taxonomy)
		fmt.Println("confidence:", out.Confidence)
		fmt.Println("version_info:", out.VersionInfo)
		return
	}
	var wrapper struct{ Detail ErrorBody `json:"detail"` }
	_ = json.Unmarshal(b, &wrapper)
	fmt.Println("error_code:", wrapper.Detail.ErrorCode)
	fmt.Println("retryable:", wrapper.Detail.Retryable)
	fmt.Println("message:", wrapper.Detail.Message)
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
