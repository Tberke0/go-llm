package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════
// OpenAI Provider
// ═══════════════════════════════════════════════════════════════════════════

const openAIBaseURL = "https://api.openai.com/v1"

// OpenAIProvider implements Provider for OpenAI
type OpenAIProvider struct {
	config     ProviderConfig
	httpClient *http.Client
}

// NewOpenAIProvider creates an OpenAI provider
func NewOpenAIProvider(config ProviderConfig) *OpenAIProvider {
	if config.BaseURL == "" {
		config.BaseURL = openAIBaseURL
	}
	if config.APIKey == "" {
		config.APIKey = os.Getenv("OPENAI_API_KEY")
	}
	client := http.DefaultClient
	if config.Timeout > 0 {
		client = &http.Client{Timeout: config.Timeout}
	}
	return &OpenAIProvider{config: config, httpClient: client}
}

func (p *OpenAIProvider) Name() string {
	return "openai"
}

func (p *OpenAIProvider) Capabilities() ProviderCapabilities {
	return ProviderCapabilities{
		Tools:      true,
		Vision:     true,
		Streaming:  true,
		JSON:       true,
		Thinking:   true, // o1 models support reasoning
		Embeddings: true,
		TTS:        true,
		STT:        true,

		// Responses API built-in tools
		WebSearch:       true,
		FileSearch:      true,
		CodeInterpreter: true,
		MCP:             true,
		ImageGeneration: true,
		ComputerUse:     true,
		Shell:           true,
		ApplyPatch:      true,
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Send
// ═══════════════════════════════════════════════════════════════════════════

func (p *OpenAIProvider) Send(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "OPENAI_API_KEY not set",
		}
	}

	// Use Responses API when built-in tools are present
	if len(req.BuiltinTools) > 0 {
		return p.sendResponses(ctx, req)
	}

	oaiReq := p.buildRequest(req)

	body, err := json.Marshal(oaiReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST %s\n", colorDim("→"), p.Name(), "/chat/completions")
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "request failed", Err: err}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to read response", Err: err}
	}

	return p.parseResponse(respBody)
}

// ═══════════════════════════════════════════════════════════════════════════
// SendStream
// ═══════════════════════════════════════════════════════════════════════════

func (p *OpenAIProvider) SendStream(ctx context.Context, req *ProviderRequest, callback StreamCallback) (*ProviderResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "OPENAI_API_KEY not set",
		}
	}

	oaiReq := p.buildRequest(req)
	oaiReq.Stream = true

	body, err := json.Marshal(oaiReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST %s (stream)\n", colorDim("→"), p.Name(), "/chat/completions")
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "request failed", Err: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &ProviderError{
			Provider: p.Name(),
			Code:     fmt.Sprintf("%d", resp.StatusCode),
			Message:  string(body),
		}
	}

	var fullContent strings.Builder
	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, &ProviderError{Provider: p.Name(), Message: "stream read error", Err: err}
		}

		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		data := bytes.TrimPrefix(line, []byte("data: "))
		if string(data) == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}

		if err := json.Unmarshal(data, &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			fullContent.WriteString(content)
			callback(content)
		}
	}

	completionTokens := len(fullContent.String()) / 4

	return &ProviderResponse{
		Content:          fullContent.String(),
		CompletionTokens: completionTokens,
		TotalTokens:      completionTokens,
	}, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

type openAIRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Stream         bool            `json:"stream,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     any             `json:"tool_choice,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	// OpenAI uses "reasoning_effort" for o1 models
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

func (p *OpenAIProvider) buildRequest(req *ProviderRequest) *openAIRequest {
	oaiReq := &openAIRequest{
		Model:    resolveModel(ProviderOpenAI, Model(req.Model)),
		Messages: req.Messages,
	}

	if req.Temperature != nil {
		oaiReq.Temperature = req.Temperature
	}

	// OpenAI o1 models use reasoning_effort: low, medium, high
	if req.Thinking != "" {
		switch req.Thinking {
		case ThinkingLow:
			oaiReq.ReasoningEffort = "low"
		case ThinkingMedium:
			oaiReq.ReasoningEffort = "medium"
		case ThinkingHigh:
			oaiReq.ReasoningEffort = "high"
		}
	}

	if len(req.Tools) > 0 {
		oaiReq.Tools = req.Tools
		oaiReq.ToolChoice = "auto"
	}

	if req.JSONMode {
		oaiReq.ResponseFormat = &ResponseFormat{Type: "json_object"}
	}

	return oaiReq
}

func (p *OpenAIProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.config.APIKey)

	for k, v := range p.config.Headers {
		req.Header.Set(k, v)
	}
}

func (p *OpenAIProvider) parseResponse(body []byte) (*ProviderResponse, error) {
	var result struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Role      string     `json:"role"`
				Content   string     `json:"content"`
				ToolCalls []ToolCall `json:"tool_calls,omitempty"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  fmt.Sprintf("parse error: %v\nBody: %s", err, string(body)),
		}
	}

	if result.Error != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Code:     result.Error.Code,
			Message:  result.Error.Message,
		}
	}

	if len(result.Choices) == 0 {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  "no response choices",
		}
	}

	choice := result.Choices[0]
	return &ProviderResponse{
		Content:          choice.Message.Content,
		ToolCalls:        choice.Message.ToolCalls,
		PromptTokens:     result.Usage.PromptTokens,
		CompletionTokens: result.Usage.CompletionTokens,
		TotalTokens:      result.Usage.TotalTokens,
		FinishReason:     choice.FinishReason,
	}, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Responses API (for built-in tools: web_search, file_search, etc.)
// ═══════════════════════════════════════════════════════════════════════════

// responsesRequest is the request format for /v1/responses
type responsesRequest struct {
	Model        string        `json:"model"`
	Input        any           `json:"input"` // string or []responsesInputItem
	Instructions string        `json:"instructions,omitempty"`
	Tools        []any         `json:"tools,omitempty"`
	ToolChoice   string        `json:"tool_choice,omitempty"`
	Reasoning    *reasoningCfg `json:"reasoning,omitempty"`
}

type reasoningCfg struct {
	Effort string `json:"effort,omitempty"` // "low", "medium", "high"
}

// responsesInputItem for multi-turn conversations
type responsesInputItem struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (p *OpenAIProvider) sendResponses(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error) {
	// Build input from messages
	var input any
	if len(req.Messages) == 1 && req.Messages[0].Role == "user" {
		// Simple single message - use string input
		if content, ok := req.Messages[0].Content.(string); ok {
			input = content
		}
	}
	if input == nil {
		// Convert messages to input items
		var items []responsesInputItem
		for _, msg := range req.Messages {
			content := ""
			if s, ok := msg.Content.(string); ok {
				content = s
			}
			items = append(items, responsesInputItem{
				Role:    msg.Role,
				Content: content,
			})
		}
		input = items
	}

	// Build tools array
	var tools []any
	for _, bt := range req.BuiltinTools {
		tools = append(tools, p.buildBuiltinTool(bt))
	}
	// Also include function tools if any
	for _, ft := range req.Tools {
		tools = append(tools, ft)
	}

	// Find system message for instructions
	var instructions string
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			if s, ok := msg.Content.(string); ok {
				instructions = s
			}
			break
		}
	}

	respReq := responsesRequest{
		Model:        resolveModel(ProviderOpenAI, Model(req.Model)),
		Input:        input,
		Instructions: instructions,
		Tools:        tools,
		ToolChoice:   "auto",
	}

	// Set reasoning effort if thinking is configured
	if req.Thinking != "" {
		respReq.Reasoning = &reasoningCfg{Effort: string(req.Thinking)}
	}

	body, err := json.Marshal(respReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal responses request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	if Debug {
		fmt.Printf("%s [%s] POST %s\n", colorDim("→"), p.Name(), "/responses")
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "request failed", Err: err}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to read response", Err: err}
	}

	return p.parseResponsesResponse(respBody)
}

// buildBuiltinTool converts BuiltinTool to the API format
func (p *OpenAIProvider) buildBuiltinTool(bt BuiltinTool) map[string]any {
	tool := map[string]any{
		"type": bt.Type,
	}

	switch bt.Type {
	case "web_search":
		if bt.UserLocation != nil {
			tool["user_location"] = bt.UserLocation
		}
		if bt.SearchFilter != nil {
			tool["filters"] = bt.SearchFilter
		}

	case "file_search":
		if len(bt.VectorStoreIDs) > 0 {
			tool["vector_store_ids"] = bt.VectorStoreIDs
		}
		if bt.MaxNumResults > 0 {
			tool["max_num_results"] = bt.MaxNumResults
		}
		if bt.FileFilter != nil {
			tool["filters"] = bt.FileFilter
		}

	case "code_interpreter":
		if bt.Container != nil {
			tool["container"] = bt.Container
		}

	case "mcp":
		if bt.ServerLabel != "" {
			tool["server_label"] = bt.ServerLabel
		}
		if bt.ServerURL != "" {
			tool["server_url"] = bt.ServerURL
		}
		if bt.ServerDescription != "" {
			tool["server_description"] = bt.ServerDescription
		}
		if bt.ConnectorID != "" {
			tool["connector_id"] = bt.ConnectorID
		}
		if bt.Authorization != "" {
			tool["authorization"] = bt.Authorization
		}
		if bt.RequireApproval != nil {
			tool["require_approval"] = bt.RequireApproval
		}
		if len(bt.AllowedTools) > 0 {
			tool["allowed_tools"] = bt.AllowedTools
		}

	case "image_generation":
		if bt.ImageSize != "" {
			tool["size"] = bt.ImageSize
		}
		if bt.ImageQuality != "" {
			tool["quality"] = bt.ImageQuality
		}
		if bt.ImageFormat != "" {
			tool["output_format"] = bt.ImageFormat
		}
		if bt.ImageCompression > 0 {
			tool["compression"] = bt.ImageCompression
		}
		if bt.ImageBackground != "" {
			tool["background"] = bt.ImageBackground
		}
		if bt.PartialImages > 0 {
			tool["partial_images"] = bt.PartialImages
		}

	case "computer_use_preview":
		if bt.DisplayWidth > 0 {
			tool["display_width"] = bt.DisplayWidth
		}
		if bt.DisplayHeight > 0 {
			tool["display_height"] = bt.DisplayHeight
		}
		if bt.Environment != "" {
			tool["environment"] = bt.Environment
		}

	case "shell":
		// No additional configuration needed

	case "apply_patch":
		// No additional configuration needed
	}

	return tool
}

// parseResponsesResponse parses the Responses API output
func (p *OpenAIProvider) parseResponsesResponse(body []byte) (*ProviderResponse, error) {
	var result struct {
		ID     string `json:"id"`
		Status string `json:"status"`
		Output []struct {
			ID      string `json:"id"`
			Type    string `json:"type"`
			Status  string `json:"status,omitempty"`
			CallID  string `json:"call_id,omitempty"`
			Role    string `json:"role,omitempty"`
			Content []struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				Annotations []struct {
					Type       string `json:"type"`
					URL        string `json:"url,omitempty"`
					Title      string `json:"title,omitempty"`
					FileID     string `json:"file_id,omitempty"`
					Filename   string `json:"filename,omitempty"`
					StartIndex int    `json:"start_index,omitempty"`
					EndIndex   int    `json:"end_index,omitempty"`
				} `json:"annotations,omitempty"`
			} `json:"content,omitempty"`
			// Tool call fields
			ServerLabel string `json:"server_label,omitempty"`
			Name        string `json:"name,omitempty"`
			Arguments   string `json:"arguments,omitempty"`
			OutputText  string `json:"output,omitempty"`
			Error       string `json:"error,omitempty"`

			// Image generation fields
			RevisedPrompt string `json:"revised_prompt,omitempty"`
			Result        string `json:"result,omitempty"` // base64 image

			// Shared action field - used by both computer_call and shell_call with different structures
			// We use json.RawMessage to handle the polymorphic nature
			Action json.RawMessage `json:"action,omitempty"`

			// Safety checks (computer use)
			PendingSafetyChecks []struct {
				ID      string `json:"id"`
				Code    string `json:"code"`
				Message string `json:"message"`
			} `json:"pending_safety_checks,omitempty"`

			// Apply patch fields
			Operation *struct {
				Type string `json:"type"`
				Path string `json:"path"`
				Diff string `json:"diff,omitempty"`
			} `json:"operation,omitempty"`
		} `json:"output"`
		OutputText string `json:"output_text,omitempty"` // Convenience field
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Message:  fmt.Sprintf("parse error: %v\nBody: %s", err, string(body)),
		}
	}

	if result.Error != nil {
		return nil, &ProviderError{
			Provider: p.Name(),
			Code:     result.Error.Code,
			Message:  result.Error.Message,
		}
	}

	// Extract text content and build ResponsesOutput
	var textContent string
	var citations []Citation
	var toolCalls []ResponsesToolCall

	for _, item := range result.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" || c.Type == "text" {
					textContent += c.Text
					// Extract citations
					for _, ann := range c.Annotations {
						citations = append(citations, Citation{
							Type:       ann.Type,
							URL:        ann.URL,
							Title:      ann.Title,
							FileID:     ann.FileID,
							Filename:   ann.Filename,
							StartIndex: ann.StartIndex,
							EndIndex:   ann.EndIndex,
						})
					}
				}
			}

		case "web_search_call", "file_search_call", "mcp_call", "code_interpreter_call":
			toolCalls = append(toolCalls, ResponsesToolCall{
				ID:          item.ID,
				Type:        item.Type,
				Status:      item.Status,
				CallID:      item.CallID,
				ServerLabel: item.ServerLabel,
				Name:        item.Name,
				Arguments:   item.Arguments,
				Output:      item.OutputText,
				Error:       item.Error,
			})

		case "image_generation_call":
			toolCalls = append(toolCalls, ResponsesToolCall{
				ID:            item.ID,
				Type:          item.Type,
				Status:        item.Status,
				CallID:        item.CallID,
				RevisedPrompt: item.RevisedPrompt,
				ImageResult:   item.Result,
			})

		case "computer_call":
			tc := ResponsesToolCall{
				ID:     item.ID,
				Type:   item.Type,
				Status: item.Status,
				CallID: item.CallID,
			}
			if len(item.Action) > 0 {
				var action ComputerAction
				if err := json.Unmarshal(item.Action, &action); err == nil {
					tc.Action = &action
				}
			}
			for _, sc := range item.PendingSafetyChecks {
				tc.PendingSafetyChecks = append(tc.PendingSafetyChecks, SafetyCheck{
					ID:      sc.ID,
					Code:    sc.Code,
					Message: sc.Message,
				})
			}
			toolCalls = append(toolCalls, tc)

		case "shell_call":
			tc := ResponsesToolCall{
				ID:     item.ID,
				Type:   item.Type,
				Status: item.Status,
				CallID: item.CallID,
			}
			if len(item.Action) > 0 {
				var action ShellAction
				if err := json.Unmarshal(item.Action, &action); err == nil {
					tc.ShellAction = &action
				}
			}
			toolCalls = append(toolCalls, tc)

		case "apply_patch_call":
			tc := ResponsesToolCall{
				ID:     item.ID,
				Type:   item.Type,
				Status: item.Status,
				CallID: item.CallID,
			}
			if item.Operation != nil {
				tc.PatchOperation = &PatchOperation{
					Type: item.Operation.Type,
					Path: item.Operation.Path,
					Diff: item.Operation.Diff,
				}
			}
			toolCalls = append(toolCalls, tc)
		}
	}

	// Use output_text convenience field if available
	if textContent == "" && result.OutputText != "" {
		textContent = result.OutputText
	}

	return &ProviderResponse{
		Content:          textContent,
		PromptTokens:     result.Usage.InputTokens,
		CompletionTokens: result.Usage.OutputTokens,
		TotalTokens:      result.Usage.TotalTokens,
		ResponsesOutput: &ResponsesOutput{
			Text:      textContent,
			Citations: citations,
			ToolCalls: toolCalls,
		},
	}, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Embeddings
// ═══════════════════════════════════════════════════════════════════════════

func (p *OpenAIProvider) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{Provider: p.Name(), Message: "OPENAI_API_KEY not set"}
	}

	oaiReq := struct {
		Model      string   `json:"model"`
		Input      []string `json:"input"`
		Dimensions int      `json:"dimensions,omitempty"`
	}{
		Model: req.Model,
		Input: req.Input,
	}
	if req.Dimensions > 0 {
		oaiReq.Dimensions = req.Dimensions
	}

	body, err := json.Marshal(oaiReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "request failed", Err: err}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to read response", Err: err}
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
		Model string `json:"model"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
		Error *struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "parse error", Err: err}
	}

	if result.Error != nil {
		return nil, &ProviderError{Provider: p.Name(), Code: result.Error.Code, Message: result.Error.Message}
	}

	embeddings := make([][]float64, len(result.Data))
	var dims int
	for _, d := range result.Data {
		embeddings[d.Index] = d.Embedding
		if dims == 0 {
			dims = len(d.Embedding)
		}
	}

	return &EmbeddingResponse{
		Embeddings:  embeddings,
		Model:       result.Model,
		TotalTokens: result.Usage.TotalTokens,
		Dimensions:  dims,
	}, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Text-to-Speech
// ═══════════════════════════════════════════════════════════════════════════

func (p *OpenAIProvider) TextToSpeech(ctx context.Context, req *TTSRequest) (*TTSResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{Provider: p.Name(), Message: "OPENAI_API_KEY not set"}
	}

	oaiReq := struct {
		Model          string  `json:"model"`
		Input          string  `json:"input"`
		Voice          string  `json:"voice"`
		ResponseFormat string  `json:"response_format,omitempty"`
		Speed          float64 `json:"speed,omitempty"`
	}{
		Model:          req.Model,
		Input:          req.Input,
		Voice:          req.Voice,
		ResponseFormat: req.Format,
		Speed:          req.Speed,
	}

	body, err := json.Marshal(oaiReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to marshal request", Err: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/audio/speech", bytes.NewReader(body))
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "request failed", Err: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		return nil, &ProviderError{
			Provider: p.Name(),
			Code:     fmt.Sprintf("%d", resp.StatusCode),
			Message:  string(errBody),
		}
	}

	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to read audio", Err: err}
	}

	return &TTSResponse{
		Audio:       audio,
		Format:      req.Format,
		ContentType: resp.Header.Get("Content-Type"),
	}, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// Speech-to-Text
// ═══════════════════════════════════════════════════════════════════════════

func (p *OpenAIProvider) SpeechToText(ctx context.Context, req *STTRequest) (*STTResponse, error) {
	if p.config.APIKey == "" {
		return nil, &ProviderError{Provider: p.Name(), Message: "OPENAI_API_KEY not set"}
	}

	// Create multipart form
	var buf bytes.Buffer
	writer := newMultipartWriter(&buf)

	// Add file
	filename := req.Filename
	if filename == "" {
		filename = "audio.mp3"
	}
	fw, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create form file", Err: err}
	}
	if _, err := fw.Write(req.Audio); err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to write audio", Err: err}
	}

	// Add model
	if err := writer.WriteField("model", req.Model); err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to write model", Err: err}
	}

	// Optional fields
	if req.Language != "" {
		writer.WriteField("language", req.Language)
	}
	if req.Prompt != "" {
		writer.WriteField("prompt", req.Prompt)
	}
	if req.Timestamps {
		writer.WriteField("timestamp_granularities[]", "word")
		writer.WriteField("response_format", "verbose_json")
	}

	writer.Close()

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/audio/transcriptions", &buf)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to create request", Err: err}
	}

	httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)
	httpReq.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "request failed", Err: err}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &ProviderError{Provider: p.Name(), Message: "failed to read response", Err: err}
	}

	if resp.StatusCode != http.StatusOK {
		return nil, &ProviderError{
			Provider: p.Name(),
			Code:     fmt.Sprintf("%d", resp.StatusCode),
			Message:  string(respBody),
		}
	}

	// Parse response
	var result struct {
		Text     string  `json:"text"`
		Language string  `json:"language,omitempty"`
		Duration float64 `json:"duration,omitempty"`
		Words    []struct {
			Word  string  `json:"word"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"words,omitempty"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		// Simple text response
		return &STTResponse{Text: string(respBody)}, nil
	}

	sttResp := &STTResponse{
		Text:     result.Text,
		Language: result.Language,
		Duration: result.Duration,
	}

	for _, w := range result.Words {
		sttResp.Words = append(sttResp.Words, WordTimestamp{
			Word:  w.Word,
			Start: w.Start,
			End:   w.End,
		})
	}

	return sttResp, nil
}
