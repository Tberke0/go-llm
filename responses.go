package ai

// ═══════════════════════════════════════════════════════════════════════════
// OpenAI Responses API - Built-in Tools
// ═══════════════════════════════════════════════════════════════════════════
//
// These tools use OpenAI's Responses API (/v1/responses) which provides
// built-in capabilities like web search, file search, code execution, and MCP.
//
// Usage:
//   resp, _ := ai.GPT5().WebSearch().User("Latest AI news?").Send()
//   resp, _ := ai.GPT5().FileSearch("vs_abc123").User("What's our policy?").Send()
//   resp, _ := ai.GPT5().CodeInterpreter().User("Calculate factorial of 50").Send()
//   resp, _ := ai.GPT5().MCP("dice", "https://example.com/mcp").User("Roll 2d6").Send()
//
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Built-in Tool Types
// ═══════════════════════════════════════════════════════════════════════════

// BuiltinTool represents a built-in tool for the Responses API
type BuiltinTool struct {
	Type string `json:"type"` // "web_search", "file_search", "code_interpreter", "mcp", "image_generation", "computer_use_preview", "shell", "apply_patch"

	// Web Search options
	UserLocation *UserLocation `json:"user_location,omitempty"`
	SearchFilter *SearchFilter `json:"search_filters,omitempty"` // domain filters for web search

	// File Search options
	VectorStoreIDs []string `json:"vector_store_ids,omitempty"`
	MaxNumResults  int      `json:"max_num_results,omitempty"`
	FileFilter     any      `json:"file_filters,omitempty"` // attribute filters for file search

	// Code Interpreter options
	Container any `json:"container,omitempty"` // string ID or ContainerConfig

	// MCP options
	ServerLabel       string   `json:"server_label,omitempty"`
	ServerURL         string   `json:"server_url,omitempty"`
	ServerDescription string   `json:"server_description,omitempty"`
	ConnectorID       string   `json:"connector_id,omitempty"`
	Authorization     string   `json:"authorization,omitempty"`
	RequireApproval   any      `json:"require_approval,omitempty"` // "always", "never", or ApprovalConfig
	AllowedTools      []string `json:"allowed_tools,omitempty"`

	// Image Generation options
	ImageSize        string `json:"size,omitempty"`           // e.g., "1024x1024", "1024x1536", "auto"
	ImageQuality     string `json:"quality,omitempty"`        // "low", "medium", "high", "auto"
	ImageFormat      string `json:"output_format,omitempty"`  // "png", "jpeg", "webp"
	ImageCompression int    `json:"compression,omitempty"`    // 0-100 for JPEG/WebP
	ImageBackground  string `json:"background,omitempty"`     // "transparent", "opaque", "auto"
	PartialImages    int    `json:"partial_images,omitempty"` // 1-3 for streaming

	// Computer Use options
	DisplayWidth  int    `json:"display_width,omitempty"`
	DisplayHeight int    `json:"display_height,omitempty"`
	Environment   string `json:"environment,omitempty"` // "browser", "mac", "windows", "ubuntu"

	// Shell options (no additional fields needed - configuration is in the request)

	// Apply Patch options (no additional fields needed - model knows how to emit patches)
}

// UserLocation for web search geo-targeting
type UserLocation struct {
	Type     string `json:"type"`               // "approximate"
	Country  string `json:"country,omitempty"`  // ISO country code (e.g., "US")
	City     string `json:"city,omitempty"`     // City name
	Region   string `json:"region,omitempty"`   // State/region
	Timezone string `json:"timezone,omitempty"` // IANA timezone
}

// SearchFilter for web search domain filtering
type SearchFilter struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
}

// ContainerConfig for code interpreter
type ContainerConfig struct {
	Type        string   `json:"type"`                   // "auto"
	MemoryLimit string   `json:"memory_limit,omitempty"` // "1g", "4g", "16g", "64g"
	FileIDs     []string `json:"file_ids,omitempty"`
}

// ApprovalConfig for MCP tool approval
type ApprovalConfig struct {
	Never  *ToolNameFilter `json:"never,omitempty"`
	Always *ToolNameFilter `json:"always,omitempty"`
}

// ToolNameFilter for MCP allowed/approval tools
type ToolNameFilter struct {
	ToolNames []string `json:"tool_names,omitempty"`
}

// ═══════════════════════════════════════════════════════════════════════════
// Option Types for Builder Methods
// ═══════════════════════════════════════════════════════════════════════════

// WebSearchOptions configures web search behavior
type WebSearchOptions struct {
	// Location for geo-targeted results
	Country  string // ISO country code (e.g., "US", "GB")
	City     string // City name
	Region   string // State/region
	Timezone string // IANA timezone (e.g., "America/New_York")

	// Domain filtering
	AllowedDomains []string // Limit search to these domains
}

// FileSearchOptions configures file search behavior
type FileSearchOptions struct {
	VectorStoreIDs []string // Vector store IDs to search
	MaxNumResults  int      // Max results (default 10, max 50)
	Filters        any      // Attribute filters for metadata
}

// CodeInterpreterOptions configures code interpreter
type CodeInterpreterOptions struct {
	ContainerID string   // Existing container ID (optional)
	MemoryLimit string   // "1g", "4g", "16g", "64g" (default "1g")
	FileIDs     []string // Files to make available
}

// MCPOptions configures MCP server connection
type MCPOptions struct {
	Label           string   // Unique label for this server
	URL             string   // Server URL (for remote MCP)
	ConnectorID     string   // Connector ID (for built-in connectors)
	Description     string   // Description for the model
	Authorization   string   // OAuth token or API key
	RequireApproval string   // "always", "never", or use ApprovalConfig
	AllowedTools    []string // Limit to specific tools
}

// ImageGenerationOptions configures image generation
type ImageGenerationOptions struct {
	Size          string // Image dimensions: "1024x1024", "1024x1536", "auto"
	Quality       string // Rendering quality: "low", "medium", "high", "auto"
	Format        string // Output format: "png", "jpeg", "webp"
	Compression   int    // Compression level 0-100 (for JPEG/WebP)
	Background    string // "transparent", "opaque", "auto"
	PartialImages int    // Number of partial images for streaming (1-3)
}

// ComputerUseOptions configures computer use (CUA) tool
type ComputerUseOptions struct {
	DisplayWidth  int    // Screen width in pixels (e.g., 1024)
	DisplayHeight int    // Screen height in pixels (e.g., 768)
	Environment   string // "browser", "mac", "windows", "ubuntu"
}

// ShellOptions configures the shell tool
// Note: The shell tool itself requires no configuration - your code handles execution
type ShellOptions struct {
	// No options needed - execution is handled by your integration
}

// ApplyPatchOptions configures the apply_patch tool
// Note: The tool requires no configuration - your code handles patch application
type ApplyPatchOptions struct {
	// No options needed - patch application is handled by your integration
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Methods
// ═══════════════════════════════════════════════════════════════════════════

// WebSearch enables web search for this request
func (b *Builder) WebSearch() *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type: "web_search",
	})
	return b
}

// WebSearchWith enables web search with custom options
func (b *Builder) WebSearchWith(opts WebSearchOptions) *Builder {
	tool := BuiltinTool{Type: "web_search"}

	// Set location if any location field is provided
	if opts.Country != "" || opts.City != "" || opts.Region != "" || opts.Timezone != "" {
		tool.UserLocation = &UserLocation{
			Type:     "approximate",
			Country:  opts.Country,
			City:     opts.City,
			Region:   opts.Region,
			Timezone: opts.Timezone,
		}
	}

	// Set domain filter
	if len(opts.AllowedDomains) > 0 {
		tool.SearchFilter = &SearchFilter{
			AllowedDomains: opts.AllowedDomains,
		}
	}

	b.builtinTools = append(b.builtinTools, tool)
	return b
}

// FileSearch enables file search with the specified vector stores
func (b *Builder) FileSearch(vectorStoreIDs ...string) *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type:           "file_search",
		VectorStoreIDs: vectorStoreIDs,
	})
	return b
}

// FileSearchWith enables file search with custom options
func (b *Builder) FileSearchWith(opts FileSearchOptions) *Builder {
	tool := BuiltinTool{
		Type:           "file_search",
		VectorStoreIDs: opts.VectorStoreIDs,
		MaxNumResults:  opts.MaxNumResults,
		FileFilter:     opts.Filters,
	}
	b.builtinTools = append(b.builtinTools, tool)
	return b
}

// CodeInterpreter enables Python code execution
func (b *Builder) CodeInterpreter() *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type: "code_interpreter",
		Container: ContainerConfig{
			Type: "auto",
		},
	})
	return b
}

// CodeInterpreterWith enables code interpreter with custom options
func (b *Builder) CodeInterpreterWith(opts CodeInterpreterOptions) *Builder {
	tool := BuiltinTool{Type: "code_interpreter"}

	if opts.ContainerID != "" {
		tool.Container = opts.ContainerID
	} else {
		tool.Container = ContainerConfig{
			Type:        "auto",
			MemoryLimit: opts.MemoryLimit,
			FileIDs:     opts.FileIDs,
		}
	}

	b.builtinTools = append(b.builtinTools, tool)
	return b
}

// MCP connects to a remote MCP server
func (b *Builder) MCP(label, url string) *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type:            "mcp",
		ServerLabel:     label,
		ServerURL:       url,
		RequireApproval: "never",
	})
	return b
}

// MCPWith connects to an MCP server with custom options
func (b *Builder) MCPWith(opts MCPOptions) *Builder {
	tool := BuiltinTool{
		Type:              "mcp",
		ServerLabel:       opts.Label,
		ServerURL:         opts.URL,
		ConnectorID:       opts.ConnectorID,
		ServerDescription: opts.Description,
		Authorization:     opts.Authorization,
		AllowedTools:      opts.AllowedTools,
	}

	if opts.RequireApproval != "" {
		tool.RequireApproval = opts.RequireApproval
	} else {
		tool.RequireApproval = "never" // sensible default
	}

	b.builtinTools = append(b.builtinTools, tool)
	return b
}

// MCPConnector connects to a built-in OpenAI connector (Dropbox, Gmail, etc.)
func (b *Builder) MCPConnector(label, connectorID, authToken string) *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type:            "mcp",
		ServerLabel:     label,
		ConnectorID:     connectorID,
		Authorization:   authToken,
		RequireApproval: "never",
	})
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Image Generation
// ═══════════════════════════════════════════════════════════════════════════

// ImageGeneration enables image generation for this request.
// The model can generate images using GPT Image models.
//
// Usage:
//
//	resp, _ := ai.GPT5().
//	    ImageGeneration().
//	    User("Generate an image of a cat hugging an otter").
//	    Send()
func (b *Builder) ImageGeneration() *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type: "image_generation",
	})
	return b
}

// ImageGenerationWith enables image generation with custom options.
//
// Usage:
//
//	resp, _ := ai.GPT5().
//	    ImageGenerationWith(ai.ImageGenerationOptions{
//	        Size:    "1024x1536",
//	        Quality: "high",
//	        Format:  "png",
//	    }).
//	    User("Generate a portrait of a sunset").
//	    Send()
func (b *Builder) ImageGenerationWith(opts ImageGenerationOptions) *Builder {
	tool := BuiltinTool{Type: "image_generation"}

	if opts.Size != "" {
		tool.ImageSize = opts.Size
	}
	if opts.Quality != "" {
		tool.ImageQuality = opts.Quality
	}
	if opts.Format != "" {
		tool.ImageFormat = opts.Format
	}
	if opts.Compression > 0 {
		tool.ImageCompression = opts.Compression
	}
	if opts.Background != "" {
		tool.ImageBackground = opts.Background
	}
	if opts.PartialImages > 0 {
		tool.PartialImages = opts.PartialImages
	}

	b.builtinTools = append(b.builtinTools, tool)
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Computer Use (CUA)
// ═══════════════════════════════════════════════════════════════════════════

// ComputerUse enables the computer use tool for building computer-using agents.
// This allows the model to control a computer interface by sending actions
// like click, type, scroll, etc.
//
// Your code must execute the actions and return screenshots to the model.
// See ComputerUseAction and ComputerUseOutput types for the action loop.
//
// Usage:
//
//	resp, _ := ai.ComputerUsePreview().
//	    ComputerUse(1024, 768, "browser").
//	    User("Book a flight to Paris").
//	    Send()
func (b *Builder) ComputerUse(displayWidth, displayHeight int, environment string) *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type:          "computer_use_preview",
		DisplayWidth:  displayWidth,
		DisplayHeight: displayHeight,
		Environment:   environment,
	})
	return b
}

// ComputerUseWith enables computer use with options struct.
func (b *Builder) ComputerUseWith(opts ComputerUseOptions) *Builder {
	return b.ComputerUse(opts.DisplayWidth, opts.DisplayHeight, opts.Environment)
}

// ═══════════════════════════════════════════════════════════════════════════
// Shell Tool
// ═══════════════════════════════════════════════════════════════════════════

// Shell enables the shell tool for executing shell commands.
// The model will emit shell_call items with commands to execute.
// Your code must execute commands and return outputs via ShellCallOutput.
//
// Works with GPT-5.1 and newer models.
//
// Usage:
//
//	resp, _ := ai.GPT51().
//	    Shell().
//	    User("Find the largest PDF in ~/Documents").
//	    Send()
func (b *Builder) Shell() *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type: "shell",
	})
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Apply Patch Tool
// ═══════════════════════════════════════════════════════════════════════════

// ApplyPatch enables the apply_patch tool for structured file editing.
// The model will emit apply_patch_call items with file operations.
// Your code must apply patches and return results via ApplyPatchCallOutput.
//
// Supports: create_file, update_file, delete_file operations with V4A diffs.
// Works with GPT-5.1 and newer models.
//
// Usage:
//
//	resp, _ := ai.GPT51().
//	    ApplyPatch().
//	    User("Rename the function from fib to fibonacci across all files").
//	    Send()
func (b *Builder) ApplyPatch() *Builder {
	b.builtinTools = append(b.builtinTools, BuiltinTool{
		Type: "apply_patch",
	})
	return b
}

// ═══════════════════════════════════════════════════════════════════════════
// Connector IDs (for MCPConnector)
// ═══════════════════════════════════════════════════════════════════════════

const (
	ConnectorDropbox         = "connector_dropbox"
	ConnectorGmail           = "connector_gmail"
	ConnectorGoogleCalendar  = "connector_googlecalendar"
	ConnectorGoogleDrive     = "connector_googledrive"
	ConnectorMicrosoftTeams  = "connector_microsoftteams"
	ConnectorOutlookCalendar = "connector_outlookcalendar"
	ConnectorOutlookEmail    = "connector_outlookemail"
	ConnectorSharePoint      = "connector_sharepoint"
)

// ═══════════════════════════════════════════════════════════════════════════
// Response Types (Responses API Output)
// ═══════════════════════════════════════════════════════════════════════════

// ResponsesOutput represents the parsed output from Responses API
type ResponsesOutput struct {
	// Text content from the model
	Text string

	// Citations from web/file search
	Citations []Citation

	// Sources from web search (all URLs consulted)
	Sources []Source

	// Tool calls made (web_search_call, file_search_call, mcp_call, etc.)
	ToolCalls []ResponsesToolCall

	// Raw output items (for advanced use)
	OutputItems []any
}

// Citation represents a URL or file citation in the response
type Citation struct {
	Type       string `json:"type"` // "url_citation" or "file_citation"
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	FileID     string `json:"file_id,omitempty"`
	Filename   string `json:"filename,omitempty"`
	StartIndex int    `json:"start_index,omitempty"`
	EndIndex   int    `json:"end_index,omitempty"`
}

// Source represents a web source consulted
type Source struct {
	URL   string `json:"url"`
	Title string `json:"title,omitempty"`
}

// ResponsesToolCall represents a tool call from Responses API
type ResponsesToolCall struct {
	ID          string `json:"id"`
	Type        string `json:"type"` // "web_search_call", "file_search_call", "mcp_call", "code_interpreter_call", "image_generation_call", "computer_call", "shell_call", "apply_patch_call"
	Status      string `json:"status"`
	CallID      string `json:"call_id,omitempty"`      // for responding to tool calls
	ServerLabel string `json:"server_label,omitempty"` // for MCP
	Name        string `json:"name,omitempty"`         // tool name
	Arguments   string `json:"arguments,omitempty"`
	Output      string `json:"output,omitempty"`
	Error       string `json:"error,omitempty"`

	// Image Generation specific fields
	RevisedPrompt string `json:"revised_prompt,omitempty"` // optimized prompt used
	ImageResult   string `json:"result,omitempty"`         // base64-encoded image

	// Computer Use specific fields
	Action *ComputerAction `json:"action,omitempty"` // action to execute

	// Shell specific fields
	ShellAction *ShellAction `json:"shell_action,omitempty"` // shell commands to execute

	// Apply Patch specific fields
	PatchOperation *PatchOperation `json:"operation,omitempty"` // file operation

	// Safety checks (Computer Use)
	PendingSafetyChecks []SafetyCheck `json:"pending_safety_checks,omitempty"`
}

// ═══════════════════════════════════════════════════════════════════════════
// Computer Use Types
// ═══════════════════════════════════════════════════════════════════════════

// ComputerAction represents an action the model wants to perform
type ComputerAction struct {
	Type string `json:"type"` // "click", "double_click", "scroll", "keypress", "type", "wait", "screenshot"

	// Click/scroll coordinates
	X int `json:"x,omitempty"`
	Y int `json:"y,omitempty"`

	// Click button
	Button string `json:"button,omitempty"` // "left", "right", "middle"

	// Scroll amounts
	ScrollX int `json:"scroll_x,omitempty"`
	ScrollY int `json:"scroll_y,omitempty"`

	// Keypress keys
	Keys []string `json:"keys,omitempty"`

	// Type text
	Text string `json:"text,omitempty"`
}

// SafetyCheck represents a pending safety check from Computer Use
type SafetyCheck struct {
	ID      string `json:"id"`
	Code    string `json:"code"` // "malicious_instructions", "irrelevant_domain", "sensitive_domain"
	Message string `json:"message"`
}

// ComputerCallOutput is the input you send back after executing a computer action
type ComputerCallOutput struct {
	CallID                   string        `json:"call_id"`
	Output                   ImageInput    `json:"output"` // screenshot
	CurrentURL               string        `json:"current_url,omitempty"`
	AcknowledgedSafetyChecks []SafetyCheck `json:"acknowledged_safety_checks,omitempty"`
}

// ═══════════════════════════════════════════════════════════════════════════
// Shell Types
// ═══════════════════════════════════════════════════════════════════════════

// ShellAction represents shell commands the model wants to execute
type ShellAction struct {
	Commands        []string `json:"commands"`                    // commands to run (can be concurrent)
	TimeoutMs       int      `json:"timeout_ms,omitempty"`        // suggested timeout
	MaxOutputLength int      `json:"max_output_length,omitempty"` // for truncation
}

// ShellCallOutput is the input you send back after executing shell commands
type ShellCallOutput struct {
	CallID          string               `json:"call_id"`
	MaxOutputLength int                  `json:"max_output_length,omitempty"` // pass back if provided
	Output          []ShellCommandResult `json:"output"`
}

// ShellCommandResult represents the result of a single shell command
type ShellCommandResult struct {
	Stdout  string       `json:"stdout"`
	Stderr  string       `json:"stderr"`
	Outcome ShellOutcome `json:"outcome"`
}

// ShellOutcome represents how a shell command completed
type ShellOutcome struct {
	Type     string `json:"type"` // "exit" or "timeout"
	ExitCode int    `json:"exit_code,omitempty"`
}

// ═══════════════════════════════════════════════════════════════════════════
// Apply Patch Types
// ═══════════════════════════════════════════════════════════════════════════

// PatchOperation represents a file operation from apply_patch
type PatchOperation struct {
	Type string `json:"type"`           // "create_file", "update_file", "delete_file"
	Path string `json:"path"`           // file path
	Diff string `json:"diff,omitempty"` // V4A diff (for create/update)
}

// ApplyPatchCallOutput is the input you send back after applying a patch
type ApplyPatchCallOutput struct {
	CallID string `json:"call_id"`
	Status string `json:"status"`           // "completed" or "failed"
	Output string `json:"output,omitempty"` // success message or error details
}

// HasBuiltinTools returns true if any built-in tools are configured
func (b *Builder) HasBuiltinTools() bool {
	return len(b.builtinTools) > 0
}
