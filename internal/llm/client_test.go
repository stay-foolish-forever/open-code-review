package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestNormalizeOpenAIBaseURL verifies URL normalization for the OpenAI SDK.
// The SDK appends /chat/completions automatically, so we strip that suffix.
func TestNormalizeOpenAIBaseURL(t *testing.T) {
	tests := []struct {
		name     string
		inputURL string
		wantURL  string
	}{
		{
			name:     "base URL without trailing slash",
			inputURL: "https://api.example.com/v1",
			wantURL:  "https://api.example.com/v1",
		},
		{
			name:     "base URL with trailing slash",
			inputURL: "https://api.example.com/v1/",
			wantURL:  "https://api.example.com/v1",
		},
		{
			name:     "full URL with chat/completions suffix stripped",
			inputURL: "https://api.example.com/v1/chat/completions",
			wantURL:  "https://api.example.com/v1",
		},
		{
			name:     "full URL with trailing slash stripped",
			inputURL: "https://api.example.com/v1/chat/completions/",
			wantURL:  "https://api.example.com/v1",
		},
		{
			name:     "bare host unchanged",
			inputURL: "https://api.example.com",
			wantURL:  "https://api.example.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeOpenAIBaseURL(tt.inputURL)
			if got != tt.wantURL {
				t.Errorf("normalizeOpenAIBaseURL(%q) = %q, want %q", tt.inputURL, got, tt.wantURL)
			}
		})
	}
}

// TestNormalizeAnthropicBaseURL verifies URL normalization for the Anthropic SDK.
// The SDK appends /v1/messages automatically, so we strip that suffix.
func TestNormalizeAnthropicBaseURL(t *testing.T) {
	tests := []struct {
		name     string
		inputURL string
		wantURL  string
	}{
		{
			name:     "bare host",
			inputURL: "https://api.anthropic.com",
			wantURL:  "https://api.anthropic.com",
		},
		{
			name:     "bare host with trailing slash",
			inputURL: "https://api.anthropic.com/",
			wantURL:  "https://api.anthropic.com",
		},
		{
			name:     "full URL with /v1/messages suffix stripped",
			inputURL: "https://api.anthropic.com/v1/messages",
			wantURL:  "https://api.anthropic.com",
		},
		{
			name:     "full URL with trailing slash stripped",
			inputURL: "https://api.anthropic.com/v1/messages/",
			wantURL:  "https://api.anthropic.com",
		},
		{
			name:     "custom proxy base URL unchanged",
			inputURL: "https://proxy.example.com/anthropic",
			wantURL:  "https://proxy.example.com/anthropic",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeAnthropicBaseURL(tt.inputURL)
			if got != tt.wantURL {
				t.Errorf("normalizeAnthropicBaseURL(%q) = %q, want %q", tt.inputURL, got, tt.wantURL)
			}
		})
	}
}

// TestNewOpenAIClient_CreatesSuccessfully verifies that the client is created without panics.
func TestNewOpenAIClient_CreatesSuccessfully(t *testing.T) {
	client := NewOpenAIClient(ClientConfig{
		URL:    "https://api.example.com/v1",
		APIKey: "test-key",
		Model:  "gpt-4",
	})
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

// TestNewAnthropicClient_CreatesSuccessfully verifies that the client is created without panics.
func TestNewAnthropicClient_CreatesSuccessfully(t *testing.T) {
	client := NewAnthropicClient(ClientConfig{
		URL:    "https://api.anthropic.com",
		APIKey: "test-key",
		Model:  "claude-sonnet-4-20250514",
	})
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

// TestBuildParams_MaxTokensBranching verifies that buildParams uses MaxTokens or
// MaxCompletionTokens based on the UseMaxCompletionTokens config flag.
func TestBuildParams_MaxTokensBranching(t *testing.T) {
	tests := []struct {
		name                   string
		useMaxCompletionTokens bool
		maxTokens              int
		wantMaxTokensSet       bool
		wantMaxCompletionSet   bool
	}{
		{
			name:                   "default uses max_tokens",
			useMaxCompletionTokens: false,
			maxTokens:              4096,
			wantMaxTokensSet:       true,
			wantMaxCompletionSet:   false,
		},
		{
			name:                   "enabled uses max_completion_tokens",
			useMaxCompletionTokens: true,
			maxTokens:              4096,
			wantMaxTokensSet:       false,
			wantMaxCompletionSet:   true,
		},
		{
			name:                   "zero max_tokens sets neither",
			useMaxCompletionTokens: true,
			maxTokens:              0,
			wantMaxTokensSet:       false,
			wantMaxCompletionSet:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewOpenAIClient(ClientConfig{
				URL:                    "https://api.example.com/v1",
				APIKey:                 "test-key",
				Model:                  "gpt-4",
				UseMaxCompletionTokens: tt.useMaxCompletionTokens,
			})

			params := client.buildParams("gpt-4", ChatRequest{
				MaxTokens: tt.maxTokens,
			})

			maxTokensSet := params.MaxTokens.Valid()
			maxCompletionSet := params.MaxCompletionTokens.Valid()

			if maxTokensSet != tt.wantMaxTokensSet {
				t.Errorf("MaxTokens present = %v, want %v", maxTokensSet, tt.wantMaxTokensSet)
			}
			if maxCompletionSet != tt.wantMaxCompletionSet {
				t.Errorf("MaxCompletionTokens present = %v, want %v", maxCompletionSet, tt.wantMaxCompletionSet)
			}
		})
	}
}

// --- Tests for convertMessagesToOpenAI ---

func TestConvertMessagesToOpenAI_BasicRoles(t *testing.T) {
	messages := []Message{
		NewTextMessage("system", "You are a helpful assistant"),
		NewTextMessage("user", "Hello"),
		NewTextMessage("assistant", "Hi there"),
	}

	result := convertMessagesToOpenAI(messages)
	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(result))
	}

	// Verify roles by checking which union variant is set
	if result[0].OfSystem == nil {
		t.Error("expected system message at index 0")
	}
	if result[1].OfUser == nil {
		t.Error("expected user message at index 1")
	}
	if result[2].OfAssistant == nil {
		t.Error("expected assistant message at index 2")
	}
}

func TestConvertMessagesToOpenAI_ToolMessage(t *testing.T) {
	messages := []Message{
		NewToolResultMessage("call-123", "tool result content"),
	}

	result := convertMessagesToOpenAI(messages)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfTool == nil {
		t.Error("expected tool message")
	}
	if result[0].OfTool.ToolCallID != "call-123" {
		t.Errorf("expected tool_call_id %q, got %q", "call-123", result[0].OfTool.ToolCallID)
	}
}

func TestConvertMessagesToOpenAI_AssistantWithToolCalls(t *testing.T) {
	messages := []Message{
		NewToolCallMessage("thinking...", []ToolCall{
			{
				ID:   "call-1",
				Type: "function",
				Function: FunctionCall{
					Name:      "get_weather",
					Arguments: `{"city":"Tokyo"}`,
				},
			},
		}),
	}

	result := convertMessagesToOpenAI(messages)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfAssistant == nil {
		t.Fatal("expected assistant message")
	}
	if len(result[0].OfAssistant.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result[0].OfAssistant.ToolCalls))
	}
	tc := result[0].OfAssistant.ToolCalls[0]
	if tc.OfFunction == nil {
		t.Fatal("expected function tool call")
	}
	if tc.OfFunction.ID != "call-1" {
		t.Errorf("expected ID %q, got %q", "call-1", tc.OfFunction.ID)
	}
	if tc.OfFunction.Function.Name != "get_weather" {
		t.Errorf("expected function name %q, got %q", "get_weather", tc.OfFunction.Function.Name)
	}
}

func TestConvertMessagesToOpenAI_EmptyMessages(t *testing.T) {
	result := convertMessagesToOpenAI(nil)
	if len(result) != 0 {
		t.Fatalf("expected 0 messages, got %d", len(result))
	}
}

// --- Tests for convertMessagesToAnthropic ---

func TestConvertMessagesToAnthropic_SystemExtraction(t *testing.T) {
	messages := []Message{
		NewTextMessage("system", "Be concise"),
		NewTextMessage("user", "Hello"),
	}

	system, result := convertMessagesToAnthropic(messages)
	if system != "Be concise" {
		t.Errorf("expected system %q, got %q", "Be concise", system)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 message (user only), got %d", len(result))
	}
}

func TestConvertMessagesToAnthropic_NoSystem(t *testing.T) {
	messages := []Message{
		NewTextMessage("user", "Hello"),
		NewTextMessage("assistant", "Hi"),
	}

	system, result := convertMessagesToAnthropic(messages)
	if system != "" {
		t.Errorf("expected empty system, got %q", system)
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}
}

func TestConvertMessagesToAnthropic_ToolResultBatching(t *testing.T) {
	// Multiple consecutive tool results should be batched into a single user message
	messages := []Message{
		NewTextMessage("user", "Do tasks"),
		NewToolCallMessage("", []ToolCall{
			{ID: "call-1", Type: "function", Function: FunctionCall{Name: "fn1", Arguments: "{}"}},
			{ID: "call-2", Type: "function", Function: FunctionCall{Name: "fn2", Arguments: "{}"}},
		}),
		NewToolResultMessage("call-1", "result 1"),
		NewToolResultMessage("call-2", "result 2"),
		NewTextMessage("assistant", "Done"),
	}

	system, result := convertMessagesToAnthropic(messages)
	if system != "" {
		t.Errorf("unexpected system: %q", system)
	}
	// Expected: user, assistant(tool_calls), user(batched tool results), assistant
	if len(result) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(result))
	}
}

func TestConvertMessagesToAnthropic_AssistantWithToolCalls(t *testing.T) {
	messages := []Message{
		NewToolCallMessage("thinking", []ToolCall{
			{ID: "call-1", Type: "function", Function: FunctionCall{Name: "search", Arguments: `{"q":"test"}`}},
		}),
	}

	_, result := convertMessagesToAnthropic(messages)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	// The assistant message should have both text block and tool_use block
	msg := result[0]
	if msg.Role != "assistant" {
		t.Errorf("expected role assistant, got %q", msg.Role)
	}
}

// --- Tests for utility functions ---

func TestStripThinkTags(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"no tags", "hello world", "hello world"},
		{"with think tags", "<think>reasoning</think>answer", "reasoninganswer"},
		{"only open tag", "<think>partial", "partial"},
		{"only close tag", "partial</think>", "partial"},
		{"empty string", "", ""},
		{"nested content", "<think>step1</think>result<think>step2</think>final", "step1resultstep2final"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripThinkTags(tt.input)
			if got != tt.want {
				t.Errorf("stripThinkTags(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestExtractErrorMessage(t *testing.T) {
	tests := []struct {
		name string
		body []byte
		want string
	}{
		{
			name: "OpenAI error format",
			body: []byte(`{"error":{"message":"Rate limit exceeded","type":"rate_limit_error"}}`),
			want: "Rate limit exceeded",
		},
		{
			name: "Anthropic error format",
			body: []byte(`{"type":"error","error":{"message":"Invalid API key","type":"authentication_error"}}`),
			want: "Invalid API key",
		},
		{
			name: "empty body",
			body: []byte{},
			want: "(empty body)",
		},
		{
			name: "nil body",
			body: nil,
			want: "(empty body)",
		},
		{
			name: "unrecognized JSON",
			body: []byte(`{"status":"error","detail":"something went wrong"}`),
			want: `{"status":"error","detail":"something went wrong"}`,
		},
		{
			name: "truncates long body",
			body: []byte(strings.Repeat("x", 600)),
			want: strings.Repeat("x", 512) + "... (truncated)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractErrorMessage(tt.body)
			if got != tt.want {
				t.Errorf("extractErrorMessage() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestExtractText(t *testing.T) {
	tests := []struct {
		name    string
		message Message
		want    string
	}{
		{
			name:    "string content",
			message: Message{Role: "user", Content: "hello"},
			want:    "hello",
		},
		{
			name:    "content block array",
			message: Message{Role: "user", Content: []ContentBlock{{Type: "text", Text: "part1"}, {Type: "text", Text: "part2"}}},
			want:    "part1part2",
		},
		{
			name:    "nil content",
			message: Message{Role: "user", Content: nil},
			want:    "",
		},
		{
			name:    "empty string",
			message: Message{Role: "user", Content: ""},
			want:    "",
		},
		{
			name: "nested content blocks",
			message: Message{Role: "user", Content: []ContentBlock{
				{Type: "tool_result", ToolUseID: "id1", Content: []ContentBlock{{Type: "text", Text: "nested"}}},
			}},
			want: "nested",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.message.ExtractText()
			if got != tt.want {
				t.Errorf("ExtractText() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestEncodingForModel(t *testing.T) {
	tests := []struct {
		model string
		want  string
	}{
		{"o1-preview", "o200k_base"},
		{"o3-mini", "o200k_base"},
		{"o4-mini", "o200k_base"},
		{"gpt-4", "cl100k_base"},
		{"gpt-4o", "cl100k_base"},
		{"claude-opus-4-6", "cl100k_base"},
		{"", "cl100k_base"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			got := encodingForModel(tt.model)
			if got != tt.want {
				t.Errorf("encodingForModel(%q) = %q, want %q", tt.model, got, tt.want)
			}
		})
	}
}

func TestChatResponseContent(t *testing.T) {
	strPtr := func(s string) *string { return &s }

	tests := []struct {
		name string
		resp ChatResponse
		want string
	}{
		{
			name: "normal content",
			resp: ChatResponse{Choices: []Choice{{Message: ResponseMessage{Content: strPtr("hello")}}}},
			want: "hello",
		},
		{
			name: "content with think tags",
			resp: ChatResponse{Choices: []Choice{{Message: ResponseMessage{Content: strPtr("<think>reasoning</think>answer")}}}},
			want: "reasoninganswer",
		},
		{
			name: "fallback to reasoning content",
			resp: ChatResponse{Choices: []Choice{{Message: ResponseMessage{Content: strPtr(""), ReasoningContent: "reasoning"}}}},
			want: "reasoning",
		},
		{
			name: "nil content fallback to reasoning",
			resp: ChatResponse{Choices: []Choice{{Message: ResponseMessage{Content: nil, ReasoningContent: "fallback"}}}},
			want: "fallback",
		},
		{
			name: "empty choices",
			resp: ChatResponse{Choices: []Choice{}},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.resp.Content()
			if got != tt.want {
				t.Errorf("Content() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNewTextMessage(t *testing.T) {
	msg := NewTextMessage("user", "hello")
	if msg.Role != "user" {
		t.Errorf("Role = %q, want %q", msg.Role, "user")
	}
	if msg.Content != "hello" {
		t.Errorf("Content = %v, want %q", msg.Content, "hello")
	}
}

func TestNewToolCallMessage(t *testing.T) {
	calls := []ToolCall{{ID: "1", Type: "function", Function: FunctionCall{Name: "fn", Arguments: "{}"}}}
	msg := NewToolCallMessage("text", calls)
	if msg.Role != "assistant" {
		t.Errorf("Role = %q, want %q", msg.Role, "assistant")
	}
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(msg.ToolCalls))
	}
	// Verify it's a copy, not the same slice
	calls[0].ID = "modified"
	if msg.ToolCalls[0].ID == "modified" {
		t.Error("ToolCalls should be a copy, not a reference")
	}
}

func TestNewToolResultMessage(t *testing.T) {
	msg := NewToolResultMessage("call-1", "result")
	if msg.Role != "tool" {
		t.Errorf("Role = %q, want %q", msg.Role, "tool")
	}
	if msg.ToolCallID != "call-1" {
		t.Errorf("ToolCallID = %q, want %q", msg.ToolCallID, "call-1")
	}
	if msg.Content != "result" {
		t.Errorf("Content = %v, want %q", msg.Content, "result")
	}
}

// Ensure json import is used (for potential future tests using json assertions)
var _ = json.Marshal
