package ai

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// Provider Interface
// ═══════════════════════════════════════════════════════════════════════════

// Provider represents an AI API provider (OpenAI, Anthropic, Ollama, etc.)
type Provider interface {
	// Name returns the provider identifier
	Name() string

	// Send makes a request and returns the response
	Send(ctx context.Context, req *ProviderRequest) (*ProviderResponse, error)

	// SendStream makes a streaming request
	SendStream(ctx context.Context, req *ProviderRequest, callback StreamCallback) (*ProviderResponse, error)

	// Capabilities returns what this provider supports
	Capabilities() ProviderCapabilities
}

// ProviderCapabilities describes what features a provider supports
type ProviderCapabilities struct {
	Tools      bool
	Vision     bool
	Streaming  bool
	JSON       bool // structured output / JSON mode
	Thinking   bool // reasoning/thinking effort
	PDF        bool // document/PDF input support
	Embeddings bool // embedding/vector support
	TTS        bool // text-to-speech
	STT        bool // speech-to-text (transcription)

	// OpenAI Responses API built-in tools
	WebSearch       bool // web search tool
	FileSearch      bool // vector store file search
	CodeInterpreter bool // sandboxed Python execution
	MCP             bool // remote MCP servers / connectors
	ImageGeneration bool // GPT Image model integration
	ComputerUse     bool // computer-use-preview CUA model
	Shell           bool // shell command execution (GPT-5.1+)
	ApplyPatch      bool // structured file editing (GPT-5.1+)
}

// ═══════════════════════════════════════════════════════════════════════════
// Extended Provider Interfaces (optional capabilities)
// ═══════════════════════════════════════════════════════════════════════════

// Embedder is implemented by providers that support embeddings
type Embedder interface {
	Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)
}

// AudioProvider is implemented by providers that support TTS/STT
type AudioProvider interface {
	// TextToSpeech converts text to audio
	TextToSpeech(ctx context.Context, req *TTSRequest) (*TTSResponse, error)
	// SpeechToText transcribes audio to text
	SpeechToText(ctx context.Context, req *STTRequest) (*STTResponse, error)
}

// ═══════════════════════════════════════════════════════════════════════════
// Provider Types
// ═══════════════════════════════════════════════════════════════════════════

// ProviderType identifies a provider
type ProviderType string

const (
	ProviderOpenRouter ProviderType = "openrouter"
	ProviderOpenAI     ProviderType = "openai"
	ProviderAnthropic  ProviderType = "anthropic"
	ProviderGoogle     ProviderType = "google"
	ProviderOllama     ProviderType = "ollama"
	ProviderAzure      ProviderType = "azure"
)

// ═══════════════════════════════════════════════════════════════════════════
// Provider Configuration
// ═══════════════════════════════════════════════════════════════════════════

// ProviderConfig holds provider-specific configuration
type ProviderConfig struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
	Timeout time.Duration
}

// ═══════════════════════════════════════════════════════════════════════════
// Provider Request/Response (provider-agnostic format)
// ═══════════════════════════════════════════════════════════════════════════

// ProviderRequest is the unified request format for all providers
type ProviderRequest struct {
	Model        string
	Messages     []Message
	Temperature  *float64
	Thinking     ThinkingLevel
	Tools        []Tool        // Function calling tools
	BuiltinTools []BuiltinTool // Responses API built-in tools (web_search, file_search, etc.)
	JSONMode     bool
	Stream       bool
}

// ProviderResponse is the unified response format from all providers
type ProviderResponse struct {
	Content          string
	ToolCalls        []ToolCall
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	FinishReason     string

	// Responses API output (populated when using built-in tools)
	ResponsesOutput *ResponsesOutput
}

// ═══════════════════════════════════════════════════════════════════════════
// Provider Errors
// ═══════════════════════════════════════════════════════════════════════════

// ProviderError wraps errors with provider context
type ProviderError struct {
	Provider string
	Code     string
	Message  string
	Err      error
}

func (e *ProviderError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: [%s] %s", e.Provider, e.Code, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Provider, e.Message)
}

func (e *ProviderError) Unwrap() error {
	return e.Err
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Mapping
// ═══════════════════════════════════════════════════════════════════════════

// modelMappings maps our Model constants to provider-specific model IDs
var modelMappings = map[ProviderType]map[Model]string{
	ProviderOpenRouter: {
		// NOTE: Do not include alias constants as separate keys (e.g. ModelGPT5 == ModelGPT52),
		// or Go will treat them as duplicate map keys.
		ModelGPT52:                    "openai/gpt-5.2",
		ModelGPT52Pro:                 "openai/gpt-5.2-pro",
		ModelGPT51:                    "openai/gpt-5.1",
		ModelGPT5Base:                 "openai/gpt-5",
		ModelGPT5Pro:                  "openai/gpt-5-pro",
		ModelGPT5Mini:                 "openai/gpt-5-mini",
		ModelGPT5Nano:                 "openai/gpt-5-nano",
		ModelGPT51Codex:               "openai/gpt-5.1-codex",
		ModelGPT51CodexMax:            "openai/gpt-5.1-codex-max",
		ModelGPT5CodexBase:            "openai/gpt-5-codex",
		ModelGPT51CodexMini:           "openai/gpt-5.1-codex-mini",
		ModelCodexMiniLatest:          "openai/codex-mini-latest",
		ModelGPT5SearchAPI:            "openai/gpt-5-search-api",
		ModelComputerUsePreview:       "openai/computer-use-preview",
		ModelGPT52ChatLatest:          "openai/gpt-5.2-chat-latest",
		ModelGPT51ChatLatest:          "openai/gpt-5.1-chat-latest",
		ModelGPT5ChatLatest:           "openai/gpt-5-chat-latest",
		ModelChatGPT4oLatest:          "openai/chatgpt-4o-latest",
		ModelGPT41:                    "openai/gpt-4.1",
		ModelGPT41Mini:                "openai/gpt-4.1-mini",
		ModelGPT41Nano:                "openai/gpt-4.1-nano",
		ModelGPT4o:                    "openai/gpt-4o",
		ModelGPT4o20240513:            "openai/gpt-4o-2024-05-13",
		ModelGPT4oMini:                "openai/gpt-4o-mini",
		ModelO1:                       "openai/o1",
		ModelO1Mini:                   "openai/o1-mini",
		ModelO1Pro:                    "openai/o1-pro",
		ModelO1Preview:                "openai/o1-preview",
		ModelO3:                       "openai/o3",
		ModelO3Mini:                   "openai/o3-mini",
		ModelO3Pro:                    "openai/o3-pro",
		ModelO3DeepResearch:           "openai/o3-deep-research",
		ModelO4Mini:                   "openai/o4-mini",
		ModelO4MiniDeepResearch:       "openai/o4-mini-deep-research",
		ModelGPTRealtime:              "openai/gpt-realtime",
		ModelGPTRealtimeMini:          "openai/gpt-realtime-mini",
		ModelGPT4oRealtimePreview:     "openai/gpt-4o-realtime-preview",
		ModelGPT4oMiniRealtimePreview: "openai/gpt-4o-mini-realtime-preview",
		ModelGPTAudio:                 "openai/gpt-audio",
		ModelGPTAudioMini:             "openai/gpt-audio-mini",
		ModelGPT4oAudioPreview:        "openai/gpt-4o-audio-preview",
		ModelGPT4oMiniAudioPreview:    "openai/gpt-4o-mini-audio-preview",
		ModelGPT4oMiniSearchPreview:   "openai/gpt-4o-mini-search-preview",
		ModelGPT4oSearchPreview:       "openai/gpt-4o-search-preview",
		ModelGPT4oMiniTTS:             "openai/gpt-4o-mini-tts",
		ModelGPT4oTranscribe:          "openai/gpt-4o-transcribe",
		ModelGPT4oTranscribeDiarize:   "openai/gpt-4o-transcribe-diarize",
		ModelGPT4oMiniTranscribe:      "openai/gpt-4o-mini-transcribe",
		ModelGPTImage15:               "openai/gpt-image-1.5",
		ModelChatGPTImageLatest:       "openai/chatgpt-image-latest",
		ModelGPTImage1:                "openai/gpt-image-1",
		ModelGPTImage1Mini:            "openai/gpt-image-1-mini",
		ModelSora2:                    "openai/sora-2",
		ModelSora2Pro:                 "openai/sora-2-pro",
		ModelClaudeOpus:               "anthropic/claude-opus-4.5",
		ModelClaudeSonnet:             "anthropic/claude-sonnet-4.5",
		ModelClaudeHaiku:              "anthropic/claude-haiku-4.5",
		ModelClaudeOpus41:             "anthropic/claude-opus-4.1",
		ModelClaudeOpus4:              "anthropic/claude-opus-4",
		ModelClaudeSonnet4:            "anthropic/claude-sonnet-4",
		ModelClaudeSonnet37:           "anthropic/claude-3.7-sonnet",
		ModelClaudeHaiku35:            "anthropic/claude-3.5-haiku",
		ModelClaudeHaiku3:             "anthropic/claude-3-haiku",
		ModelClaudeOpus3:              "anthropic/claude-3-opus",
		ModelClaudeSonnet3:            "anthropic/claude-3-sonnet",
		ModelGemini3Pro:               "google/gemini-3-pro-preview",
		ModelGemini3Flash:             "google/gemini-3-flash-preview",
		ModelGemini25Pro:              "google/gemini-2.5-pro",
		ModelGemini25Flash:            "google/gemini-2.5-flash",
		ModelGemini25FlashLite:        "google/gemini-2.5-flash-lite",
		ModelGemini2Flash:             "google/gemini-2.0-flash-001",
		ModelGemini2FlashLite:         "google/gemini-2.0-flash-lite-001",
		ModelGrok41Fast:               "x-ai/grok-4.1-fast",
		ModelGrok3:                    "x-ai/grok-3",
		ModelGrok3Mini:                "x-ai/grok-3-mini",
		ModelQwen3Next:                "qwen/qwen3-next",
		ModelQwen3:                    "qwen/qwen-3-235b",
		ModelLlama4:                   "meta-llama/llama-4-maverick",
		ModelMistralLarge:             "mistralai/mistral-large",
	},
	ProviderOpenAI: {
		// NOTE: Do not include alias constants as separate keys (e.g. ModelGPT5 == ModelGPT52),
		// or Go will treat them as duplicate map keys.
		ModelGPT52:                    "gpt-5.2",
		ModelGPT52Pro:                 "gpt-5.2-pro",
		ModelGPT51:                    "gpt-5.1",
		ModelGPT5Base:                 "gpt-5",
		ModelGPT5Pro:                  "gpt-5-pro",
		ModelGPT5Mini:                 "gpt-5-mini",
		ModelGPT5Nano:                 "gpt-5-nano",
		ModelGPT51Codex:               "gpt-5.1-codex",
		ModelGPT51CodexMax:            "gpt-5.1-codex-max",
		ModelGPT5CodexBase:            "gpt-5-codex",
		ModelGPT51CodexMini:           "gpt-5.1-codex-mini",
		ModelCodexMiniLatest:          "codex-mini-latest",
		ModelGPT5SearchAPI:            "gpt-5-search-api",
		ModelComputerUsePreview:       "computer-use-preview",
		ModelGPT52ChatLatest:          "gpt-5.2-chat-latest",
		ModelGPT51ChatLatest:          "gpt-5.1-chat-latest",
		ModelGPT5ChatLatest:           "gpt-5-chat-latest",
		ModelChatGPT4oLatest:          "chatgpt-4o-latest",
		ModelGPT41:                    "gpt-4.1",
		ModelGPT41Mini:                "gpt-4.1-mini",
		ModelGPT41Nano:                "gpt-4.1-nano",
		ModelGPT4o:                    "gpt-4o",
		ModelGPT4o20240513:            "gpt-4o-2024-05-13",
		ModelGPT4oMini:                "gpt-4o-mini",
		ModelO1:                       "o1",
		ModelO1Mini:                   "o1-mini",
		ModelO1Pro:                    "o1-pro",
		ModelO1Preview:                "o1-preview",
		ModelO3:                       "o3",
		ModelO3Mini:                   "o3-mini",
		ModelO3Pro:                    "o3-pro",
		ModelO3DeepResearch:           "o3-deep-research",
		ModelO4Mini:                   "o4-mini",
		ModelO4MiniDeepResearch:       "o4-mini-deep-research",
		ModelGPTRealtime:              "gpt-realtime",
		ModelGPTRealtimeMini:          "gpt-realtime-mini",
		ModelGPT4oRealtimePreview:     "gpt-4o-realtime-preview",
		ModelGPT4oMiniRealtimePreview: "gpt-4o-mini-realtime-preview",
		ModelGPTAudio:                 "gpt-audio",
		ModelGPTAudioMini:             "gpt-audio-mini",
		ModelGPT4oAudioPreview:        "gpt-4o-audio-preview",
		ModelGPT4oMiniAudioPreview:    "gpt-4o-mini-audio-preview",
		ModelGPT4oMiniSearchPreview:   "gpt-4o-mini-search-preview",
		ModelGPT4oSearchPreview:       "gpt-4o-search-preview",
		ModelGPT4oMiniTTS:             "gpt-4o-mini-tts",
		ModelGPT4oTranscribe:          "gpt-4o-transcribe",
		ModelGPT4oTranscribeDiarize:   "gpt-4o-transcribe-diarize",
		ModelGPT4oMiniTranscribe:      "gpt-4o-mini-transcribe",
		ModelGPTImage15:               "gpt-image-1.5",
		ModelChatGPTImageLatest:       "chatgpt-image-latest",
		ModelGPTImage1:                "gpt-image-1",
		ModelGPTImage1Mini:            "gpt-image-1-mini",
	},
	ProviderAnthropic: {
		// Prefer stable snapshots (not floating aliases) for deterministic behavior.
		// Latest (Claude 4.5)
		ModelClaudeSonnet: "claude-sonnet-4-5-20250929",
		ModelClaudeHaiku:  "claude-haiku-4-5-20251001",
		ModelClaudeOpus:   "claude-opus-4-5-20251101",

		// Legacy / still available
		ModelClaudeOpus41:  "claude-opus-4-1-20250805",
		ModelClaudeSonnet4: "claude-sonnet-4-20250514",
		ModelClaudeOpus4:   "claude-opus-4-20250514",

		// Deprecated / legacy snapshots (kept for compatibility)
		ModelClaudeSonnet37: "claude-3-7-sonnet-20250219",
		ModelClaudeHaiku35:  "claude-3-5-haiku-20241022",
		ModelClaudeHaiku3:   "claude-3-haiku-20240307",
		ModelClaudeOpus3:    "claude-3-opus-20240229",
		ModelClaudeSonnet3:  "claude-3-sonnet-20240229",
	},
	ProviderGoogle: {
		// Gemini 3 (preview)
		ModelGemini3Pro:   "gemini-3-pro-preview",
		ModelGemini3Flash: "gemini-3-flash-preview",

		// Gemini 2.5 (stable)
		ModelGemini25Pro:       "gemini-2.5-pro",
		ModelGemini25Flash:     "gemini-2.5-flash",
		ModelGemini25FlashLite: "gemini-2.5-flash-lite",

		// Gemini 2.0 (stable)
		ModelGemini2Flash:     "gemini-2.0-flash",
		ModelGemini2FlashLite: "gemini-2.0-flash-lite",
	},
	// Azure OpenAI uses OpenAI model IDs, but typically routes by deployment in the URL.
	// We keep Azure's model mapping identical to OpenAI for convenience/consistency.
	ProviderAzure: {},
	// Ollama: uses raw model names, no mapping needed
}

func init() {
	// Keep Azure mappings aligned with OpenAI mappings (avoid duplicate lists).
	// This also implicitly supports any future OpenAI model constants.
	if m, ok := modelMappings[ProviderOpenAI]; ok {
		modelMappings[ProviderAzure] = m
	}
}

// resolveModel converts our Model to provider-specific model ID
func resolveModel(providerType ProviderType, model Model) string {
	if mapping, ok := modelMappings[providerType]; ok {
		if resolved, ok := mapping[model]; ok {
			return resolved
		}
	}

	raw := string(model)

	// Normalize common "vendor/model" IDs into provider-native model IDs.
	// This lets callers pass OpenRouter-style IDs (e.g. "openai/gpt-5.2") to the native providers,
	// and pass native OpenAI IDs (e.g. "gpt-5.2") to OpenRouter.
	switch providerType {
	case ProviderOpenAI, ProviderAzure:
		// OpenAI expects raw model IDs like "gpt-5.2", not "openai/gpt-5.2".
		if strings.HasPrefix(raw, "openai/") {
			return strings.TrimPrefix(raw, "openai/")
		}
	case ProviderOpenRouter:
		// OpenRouter generally expects "vendor/model". If the caller passes a bare OpenAI model ID,
		// assume OpenAI and prefix it. (We intentionally do not try to infer other vendors.)
		if !strings.Contains(raw, "/") && looksLikeOpenAIModelID(raw) {
			return "openai/" + raw
		}
	case ProviderAnthropic:
		// Anthropic native API expects raw model IDs like "claude-sonnet-4-5-YYYYMMDD" (or aliases).
		// Users may pass OpenRouter-style Claude IDs (e.g. "anthropic/claude-sonnet-4.5"), so normalize.
		return normalizeAnthropicModelID(raw)
	case ProviderGoogle:
		// Google Gemini native API expects raw model IDs like "gemini-2.5-pro".
		if strings.HasPrefix(raw, "google/") {
			return strings.TrimPrefix(raw, "google/")
		}
	}

	// Fallback: use raw model string (for Ollama, custom models)
	return raw
}

func normalizeAnthropicModelID(raw string) string {
	// Allow OpenRouter-style prefixes.
	if after, ok := strings.CutPrefix(raw, "anthropic/"); ok {
		raw = after
	}

	// OpenRouter sometimes uses suffixes like ":thinking" for specific routing/behavior.
	// Anthropic native API does not use those suffixes; thinking is controlled via request params.
	if i := strings.IndexByte(raw, ':'); i >= 0 {
		raw = raw[:i]
	}

	// If the caller already passed a valid snapshot (contains YYYYMMDD) or legacy IDs, keep it.
	// We only map the common OpenRouter-style slugs (with dots) to stable Anthropic IDs (with dashes).
	switch raw {
	// Current (Claude 4.5)
	case "claude-sonnet-4.5":
		return "claude-sonnet-4-5-20250929"
	case "claude-haiku-4.5":
		return "claude-haiku-4-5-20251001"
	case "claude-opus-4.5":
		return "claude-opus-4-5-20251101"

	// Claude 4 / 4.1
	case "claude-opus-4.1":
		return "claude-opus-4-1-20250805"
	case "claude-opus-4":
		return "claude-opus-4-20250514"
	case "claude-sonnet-4":
		return "claude-sonnet-4-20250514"

	// Claude 3.x (legacy/deprecated)
	case "claude-3.7-sonnet":
		return "claude-3-7-sonnet-20250219"
	case "claude-3.5-haiku":
		return "claude-3-5-haiku-20241022"
	case "claude-3-haiku":
		return "claude-3-haiku-20240307"
	case "claude-3-opus":
		return "claude-3-opus-20240229"
	case "claude-3-sonnet":
		return "claude-3-sonnet-20240229"
	}

	// For any remaining IDs:
	// - If it was an OpenRouter-style dotted version we don't explicitly map, convert dots to dashes
	//   to be closer to Anthropic naming (e.g. "claude-3.7-sonnet" -> "claude-3-7-sonnet").
	// This is best-effort; some names require full snapshot IDs.
	raw = strings.ReplaceAll(raw, ".1", "-1")
	raw = strings.ReplaceAll(raw, ".5", "-5")
	raw = strings.ReplaceAll(raw, ".7", "-7")
	raw = strings.ReplaceAll(raw, ".0", "-0")
	raw = strings.ReplaceAll(raw, ".2", "-2")
	raw = strings.ReplaceAll(raw, ".3", "-3")
	raw = strings.ReplaceAll(raw, ".4", "-4")
	raw = strings.ReplaceAll(raw, ".6", "-6")
	raw = strings.ReplaceAll(raw, ".8", "-8")
	raw = strings.ReplaceAll(raw, ".9", "-9")
	return raw
}

func looksLikeOpenAIModelID(id string) bool {
	// Common OpenAI families as of Dec 2025 (per OpenAI Models docs):
	// - gpt-* (text/multimodal, audio/realtime, open-weight, image models)
	// - chatgpt-* (ChatGPT "latest" aliases)
	// - o* (reasoning family, e.g. o1, o3, o4-mini, etc.)
	// - sora-* (video generation family)
	switch {
	case strings.HasPrefix(id, "gpt-"):
		return true
	case strings.HasPrefix(id, "chatgpt-"):
		return true
	case strings.HasPrefix(id, "sora-"):
		return true
	case strings.HasPrefix(id, "whisper-"):
		return true
	case strings.HasPrefix(id, "tts-"):
		return true
	case strings.HasPrefix(id, "text-embedding-"):
		return true
	}

	// o-series: "o" followed by a digit (o1, o3, o4-mini, o3-pro, etc.)
	if len(id) >= 2 && id[0] == 'o' && id[1] >= '0' && id[1] <= '9' {
		return true
	}

	return false
}

// ═══════════════════════════════════════════════════════════════════════════
// Feature Check Helpers
// ═══════════════════════════════════════════════════════════════════════════

// checkCapability logs a warning if using unsupported feature
func checkCapability(provider Provider, feature string, supported bool) {
	if !supported && Debug {
		fmt.Printf("%s Warning: %s does not support %s\n",
			colorYellow("⚠"), provider.Name(), feature)
	}
}
