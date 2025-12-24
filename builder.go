package ai

import (
	gocontext "context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Builder provides a fluent API for constructing AI requests
type Builder struct {
	model       Model
	system      string
	messages    []Message
	vars        Vars
	fileContext []string // file contents to inject (renamed from context)
	debug       bool
	maxRetries  int
	fallbacks   []Model
	jsonMode    bool
	temperature *float64
	thinking    ThinkingLevel

	// Tool calling (function tools)
	tools        []Tool
	toolHandlers map[string]ToolHandler

	// Built-in tools (Responses API: web_search, file_search, code_interpreter, mcp)
	builtinTools []BuiltinTool

	// Vision
	images []ImageInput

	// Documents (PDF)
	documents []DocumentInput

	// Schema enforcement
	schema any

	// Context for cancellation/timeout
	ctx gocontext.Context

	// Provider client (nil = use default)
	client *Client

	// Smart retry with backoff
	retryConfig *RetryConfig

	// Validation / Guardrails
	validators []Validator
}

// New creates a new builder for the specified model
func New(model Model) *Builder {
	return &Builder{
		model:       model,
		messages:    []Message{},
		vars:        Vars{},
		fileContext: []string{},
		maxRetries:  0,
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// System Prompt Methods
// ═══════════════════════════════════════════════════════════════════════════

// System sets the system prompt
func (b *Builder) System(prompt string) *Builder {
	b.system = prompt
	return b
}

// SystemFile loads system prompt from a file
func (b *Builder) SystemFile(path string) *Builder {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("%s Error loading prompt from %s: %v\n", colorRed("✗"), path, err)
		return b
	}
	b.system = string(data)
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Message Methods
// ═══════════════════════════════════════════════════════════════════════════

// User adds a user message
func (b *Builder) User(content string) *Builder {
	b.messages = append(b.messages, Message{Role: "user", Content: content})
	return b
}

// Assistant adds an assistant message (for context)
func (b *Builder) Assistant(content string) *Builder {
	b.messages = append(b.messages, Message{Role: "assistant", Content: content})
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Context Injection - Add files as context
// ═══════════════════════════════════════════════════════════════════════════

// Context adds a file's content to the context
func (b *Builder) Context(path string) *Builder {
	// Check for glob pattern
	if strings.Contains(path, "*") {
		matches, err := filepath.Glob(path)
		if err != nil {
			fmt.Printf("%s Error with glob pattern %s: %v\n", colorRed("✗"), path, err)
			return b
		}
		for _, match := range matches {
			b.addFileContext(match)
		}
	} else {
		b.addFileContext(path)
	}
	return b
}

func (b *Builder) addFileContext(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("%s Error loading context from %s: %v\n", colorRed("✗"), path, err)
		return
	}
	b.fileContext = append(b.fileContext, fmt.Sprintf("--- %s ---\n%s", path, string(data)))
}

// ContextString adds raw string as context
func (b *Builder) ContextString(name, content string) *Builder {
	b.fileContext = append(b.fileContext, fmt.Sprintf("--- %s ---\n%s", name, content))
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Template Variables
// ═══════════════════════════════════════════════════════════════════════════

// With adds template variables to replace {{key}} in prompts
func (b *Builder) With(vars Vars) *Builder {
	for k, v := range vars {
		b.vars[k] = v
	}
	return b
}

// Var adds a single template variable
func (b *Builder) Var(key, value string) *Builder {
	b.vars[key] = value
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Retry & Fallback
// ═══════════════════════════════════════════════════════════════════════════

// Retry sets max retry attempts on failure
func (b *Builder) Retry(times int) *Builder {
	b.maxRetries = times
	return b
}

// Fallback sets fallback models to try if primary fails
func (b *Builder) Fallback(models ...Model) *Builder {
	b.fallbacks = models
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON Mode
// ═══════════════════════════════════════════════════════════════════════════

// JSON enables JSON mode (instructs model to return JSON)
func (b *Builder) JSON() *Builder {
	b.jsonMode = true
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Temperature
// ═══════════════════════════════════════════════════════════════════════════

// Temperature sets the sampling temperature (0.0-2.0)
func (b *Builder) Temperature(temp float64) *Builder {
	b.temperature = &temp
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Thinking Level (Reasoning Effort)
// ═══════════════════════════════════════════════════════════════════════════

// Thinking sets the reasoning/thinking effort level (minimal, low, medium, high)
func (b *Builder) Thinking(level ThinkingLevel) *Builder {
	b.thinking = level
	return b
}

// ThinkMinimal sets thinking to minimal (Gemini Flash only - fastest)
func (b *Builder) ThinkMinimal() *Builder { return b.Thinking(ThinkingMinimal) }

// ThinkLow sets thinking to low effort
func (b *Builder) ThinkLow() *Builder { return b.Thinking(ThinkingLow) }

// ThinkMedium sets thinking to medium effort
func (b *Builder) ThinkMedium() *Builder { return b.Thinking(ThinkingMedium) }

// ThinkHigh sets thinking to high effort
func (b *Builder) ThinkHigh() *Builder { return b.Thinking(ThinkingHigh) }

// ═══════════════════════════════════════════════════════════════════════════
// Debug Mode
// ═══════════════════════════════════════════════════════════════════════════

// Debug enables debug output for this request
func (b *Builder) Debug() *Builder {
	b.debug = true
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Execution
// ═══════════════════════════════════════════════════════════════════════════

// buildMessages constructs the final message list
func (b *Builder) buildMessages() []Message {
	var msgs []Message

	// Build system message
	system := b.system
	if len(b.vars) > 0 {
		system = applyTemplate(system, b.vars)
	}

	// Add JSON instruction if enabled
	if b.jsonMode && system != "" {
		system += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanation."
	} else if b.jsonMode {
		system = "Respond with valid JSON only. No markdown, no explanation."
	}

	// Add context to system if present
	if len(b.fileContext) > 0 {
		contextStr := "\n\n# Context\n" + strings.Join(b.fileContext, "\n\n")
		system += contextStr
	}

	if system != "" {
		msgs = append(msgs, Message{Role: "system", Content: system})
	}

	// Add user/assistant messages with template vars applied
	for _, m := range b.messages {
		content := m.Content
		if str, ok := content.(string); ok && len(b.vars) > 0 {
			content = applyTemplate(str, b.vars)
		}
		msgs = append(msgs, Message{Role: m.Role, Content: content, ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID})
	}

	// If we have images or documents, convert the last user message to multimodal
	if (len(b.images) > 0 || len(b.documents) > 0) && len(msgs) > 0 {
		// Find last user message
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == "user" {
				// Convert to multimodal content
				var parts []ContentPart

				// Add text content first
				if text, ok := msgs[i].Content.(string); ok && text != "" {
					parts = append(parts, ContentPart{Type: "text", Text: text})
				}

				// Add images
				for _, img := range b.images {
					parts = append(parts, ContentPart{
						Type: "image_url",
						ImageURL: &ImageURL{
							URL:    img.URL,
							Detail: img.Detail,
						},
					})
				}

				// Add documents (PDFs)
				for _, doc := range b.documents {
					parts = append(parts, ContentPart{
						Type: "document",
						Document: &DocumentRef{
							Data:     doc.Data,
							URL:      doc.URL,
							MimeType: doc.MimeType,
							Name:     doc.Name,
						},
					})
				}

				msgs[i].Content = parts
				break
			}
		}
	}

	return msgs
}

// Send executes the request and returns the response
func (b *Builder) Send() (string, error) {
	meta := b.SendWithMeta()
	return meta.Content, meta.Error
}

// ResponseMeta contains response metadata
type ResponseMeta struct {
	Content          string
	Error            error
	Model            Model
	Tokens           int
	PromptTokens     int
	CompletionTokens int
	Latency          time.Duration
	Retries          int

	// Responses API output (populated when using built-in tools)
	// Contains citations, sources, and tool call details
	ResponsesOutput *ResponsesOutput
}

// SendWithMeta executes the request and returns response with metadata
func (b *Builder) SendWithMeta() *ResponseMeta {
	msgs := b.buildMessages()
	start := time.Now()

	// Enable debug for this request if set
	oldDebug := Debug
	if b.debug {
		Debug = true
	}
	defer func() { Debug = oldDebug }()

	// Get the client to use
	client := b.client
	if client == nil {
		client = getDefaultClient()
	}

	// Get context
	ctx := b.getContext()

	// Try primary model with fallbacks
	models := append([]Model{b.model}, b.fallbacks...)
	var lastErr error
	var totalRetries int

	for _, model := range models {
		// Build provider request
		req := &ProviderRequest{
			Model:        string(model),
			Messages:     msgs,
			Temperature:  b.temperature,
			Thinking:     b.thinking,
			Tools:        b.tools,
			BuiltinTools: b.builtinTools,
			JSONMode:     b.jsonMode,
		}

		// Check capability warnings
		if len(b.tools) > 0 {
			checkCapability(client.provider, "tools", client.provider.Capabilities().Tools)
		}
		if b.thinking != "" {
			checkCapability(client.provider, "thinking/reasoning", client.provider.Capabilities().Thinking)
		}
		// Check built-in tool capabilities
		for _, bt := range b.builtinTools {
			switch bt.Type {
			case "web_search":
				checkCapability(client.provider, "web_search", client.provider.Capabilities().WebSearch)
			case "file_search":
				checkCapability(client.provider, "file_search", client.provider.Capabilities().FileSearch)
			case "code_interpreter":
				checkCapability(client.provider, "code_interpreter", client.provider.Capabilities().CodeInterpreter)
			case "mcp":
				checkCapability(client.provider, "mcp", client.provider.Capabilities().MCP)
			case "image_generation":
				checkCapability(client.provider, "image_generation", client.provider.Capabilities().ImageGeneration)
			case "computer_use_preview":
				checkCapability(client.provider, "computer_use", client.provider.Capabilities().ComputerUse)
			case "shell":
				checkCapability(client.provider, "shell", client.provider.Capabilities().Shell)
			case "apply_patch":
				checkCapability(client.provider, "apply_patch", client.provider.Capabilities().ApplyPatch)
			}
		}

		if Debug {
			printDebugRequest(model, msgs)
		}

		// Use smart retry if configured
		var resp *ProviderResponse
		var err error

		if b.retryConfig != nil {
			// Smart retry with exponential backoff + jitter
			var retries int
			resp, err = WithRetry(ctx, b.retryConfig, func() (*ProviderResponse, error) {
				retries++
				if retries > 1 {
					totalRetries++
				}
				waitForRateLimit()
				return client.provider.Send(ctx, req)
			})
		} else if b.maxRetries > 0 {
			// Legacy retry (simple)
			for attempt := 0; attempt <= b.maxRetries; attempt++ {
				if attempt > 0 {
					totalRetries++
					time.Sleep(time.Duration(attempt*attempt) * 100 * time.Millisecond)
				}
				waitForRateLimit()
				resp, err = client.provider.Send(ctx, req)
				if err == nil {
					break
				}
			}
		} else {
			// No retry
			waitForRateLimit()
			resp, err = client.provider.Send(ctx, req)
		}

		if err == nil {
			// Validate response if validators configured (and apply any content filters)
			content := resp.Content
			if len(b.validators) > 0 {
				validated, validationErr := b.runValidators(content)
				if validationErr != nil {
					return &ResponseMeta{
						Error:   validationErr,
						Model:   model,
						Latency: time.Since(start),
						Retries: totalRetries,
					}
				}
				content = validated
			}

			meta := &ResponseMeta{
				Content:          content,
				Model:            model,
				Latency:          time.Since(start),
				Retries:          totalRetries,
				Tokens:           resp.TotalTokens,
				PromptTokens:     resp.PromptTokens,
				CompletionTokens: resp.CompletionTokens,
				ResponsesOutput:  resp.ResponsesOutput,
			}

			if Pretty {
				printPrettyResponse(model, content)
			}

			// Track stats
			trackRequest(meta)

			return meta
		}
		lastErr = err
	}

	return &ResponseMeta{Error: lastErr, Model: b.model, Latency: time.Since(start), Retries: totalRetries}
}

// Ask is an alias for User().Send() - quick question
func (b *Builder) Ask(prompt string) (string, error) {
	return b.User(prompt).Send()
}

// AskJSON sends a request and parses JSON response into target
func (b *Builder) AskJSON(prompt string, target any) error {
	resp, err := b.JSON().User(prompt).Send()
	if err != nil {
		return err
	}

	// Clean response (remove markdown code blocks if present)
	resp = strings.TrimPrefix(resp, "```json")
	resp = strings.TrimPrefix(resp, "```")
	resp = strings.TrimSuffix(resp, "```")
	resp = strings.TrimSpace(resp)

	return json.Unmarshal([]byte(resp), target)
}

// ═══════════════════════════════════════════════════════════════════════════
// Conversation Mode
// ═══════════════════════════════════════════════════════════════════════════

// Chat returns a conversation helper for interactive use
func (b *Builder) Chat() *Conversation {
	return &Conversation{
		builder: b,
		history: []Message{},
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Switching
// ═══════════════════════════════════════════════════════════════════════════

// Model changes the model for this builder
func (b *Builder) Model(model Model) *Builder {
	b.model = model
	return b
}

// UseModel changes model by string ID
func (b *Builder) UseModel(modelID string) *Builder {
	b.model = Model(modelID)
	return b
}

// GetModel returns the current model
func (b *Builder) GetModel() Model {
	return b.model
}

// GetSystem returns the current system prompt
func (b *Builder) GetSystem() string {
	return b.system
}

// ═══════════════════════════════════════════════════════════════════════════
// Provider Switching
// ═══════════════════════════════════════════════════════════════════════════

// WithClient sets a specific client/provider for this builder
func (b *Builder) WithClient(client *Client) *Builder {
	b.client = client
	return b
}

// Provider switches to a different provider
// Usage: ai.Claude().Provider(ai.ProviderAnthropic).Ask("...")
func (b *Builder) Provider(providerType ProviderType) *Builder {
	b.client = NewClient(providerType)
	return b
}

// GetClient returns the current client (nil means default)
func (b *Builder) GetClient() *Client {
	return b.client
}

// Clone creates a copy of this builder
func (b *Builder) Clone() *Builder {
	var tempCopy *float64
	if b.temperature != nil {
		v := *b.temperature
		tempCopy = &v
	}
	newB := &Builder{
		model:        b.model,
		system:       b.system,
		messages:     make([]Message, len(b.messages)),
		vars:         make(Vars),
		fileContext:  make([]string, len(b.fileContext)),
		debug:        b.debug,
		maxRetries:   b.maxRetries,
		fallbacks:    make([]Model, len(b.fallbacks)),
		jsonMode:     b.jsonMode,
		temperature:  tempCopy,
		thinking:     b.thinking,
		tools:        make([]Tool, len(b.tools)),
		builtinTools: make([]BuiltinTool, len(b.builtinTools)),
		images:       make([]ImageInput, len(b.images)),
		documents:    make([]DocumentInput, len(b.documents)),
		client:       b.client,
		ctx:          b.ctx,
		retryConfig:  b.retryConfig,
		validators:   make([]Validator, len(b.validators)),
	}
	copy(newB.messages, b.messages)
	copy(newB.fileContext, b.fileContext)
	copy(newB.fallbacks, b.fallbacks)
	copy(newB.tools, b.tools)
	copy(newB.builtinTools, b.builtinTools)
	copy(newB.images, b.images)
	copy(newB.documents, b.documents)
	copy(newB.validators, b.validators)
	maps.Copy(newB.vars, b.vars)
	if b.toolHandlers != nil {
		newB.toolHandlers = make(map[string]ToolHandler)
		for k, v := range b.toolHandlers {
			newB.toolHandlers[k] = v
		}
	}
	return newB
}
