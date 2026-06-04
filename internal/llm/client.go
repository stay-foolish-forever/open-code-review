// Package llm provides LLM client interfaces supporting multiple protocols.
// Supported protocols: Anthropic Messages API, OpenAI Chat Completions API.
// Implementations use the official SDKs: github.com/openai/openai-go and github.com/anthropics/anthropic-sdk-go.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
	tiktoken "github.com/pkoukk/tiktoken-go"

	"github.com/open-code-review/open-code-review/internal/stdout"
)

const maxRetries = 10 // Maximum number of retry attempts with exponential backoff.

var AppVersion = "dev"

func userAgent(provider string) string {
	ua := "open-code-review/" + AppVersion
	if provider != "" {
		ua += " | " + provider
	}
	return ua
}

// LLMClient is the unified interface for all LLM protocol implementations.
type LLMClient interface {
	Completions(req ChatRequest) (*ChatResponse, error)
	CompletionsWithCtx(ctx context.Context, req ChatRequest) (*ChatResponse, error)
	StreamCompletion(ctx context.Context, req ChatRequest, cb func(chunk []byte) error) error
}

// --- Shared data types ---

// Message represents a single message in a chat conversation.
// Content can be either plain string (for system/user/assistant/tool messages)
// or an array of content blocks (used by Claude for multi-part content).
// ToolCallID is used by OpenAI-format APIs to identify which tool call this result responds to.
type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`                // string or []ContentBlock
	ToolCallID string     `json:"tool_call_id,omitempty"` // OpenAI tool call identifier
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`   // assistant tool invocations
}

// ContentBlock represents a single block within a multi-part message content.
// Used by Claude's Messages API for tool results and multimodal content.
type ContentBlock struct {
	Type      string         `json:"type"`                  // "text" or "tool_result"
	Text      string         `json:"text,omitempty"`        // for type="text"
	ToolUseID string         `json:"tool_use_id,omitempty"` // for type="tool_result"
	Content   []ContentBlock `json:"content,omitempty"`     // nested text blocks inside tool_result
}

// NewTextMessage creates a message with simple string content.
func NewTextMessage(role, content string) Message {
	return Message{Role: role, Content: content}
}

// NewToolCallMessage creates an assistant message with text content and tool invocations.
func NewToolCallMessage(content string, toolCalls []ToolCall) Message {
	var tc []ToolCall
	if len(toolCalls) > 0 {
		tc = make([]ToolCall, len(toolCalls))
		copy(tc, toolCalls)
	}
	return Message{Role: "assistant", Content: content, ToolCalls: tc}
}

// NewToolResultMessage creates a tool-role message with the given result.
// Uses the OpenAI Chat Completions format: role="tool" with tool_call_id and plain string content.
func NewToolResultMessage(toolCallID, result string) Message {
	return Message{
		Role:       "tool",
		Content:    result,
		ToolCallID: toolCallID,
	}
}

// ExtractText returns the concatenated text content from a Message's Content field.
// Handles both plain string and content block array formats.
func (m *Message) ExtractText() string {
	switch v := m.Content.(type) {
	case string:
		return v
	case []ContentBlock:
		var sb strings.Builder
		for _, block := range v {
			sb.WriteString(extractBlockText(block))
		}
		return sb.String()
	default:
		return ""
	}
}

func extractBlockText(block ContentBlock) string {
	if block.Text != "" {
		return block.Text
	}
	var sb strings.Builder
	for _, nested := range block.Content {
		sb.WriteString(extractBlockText(nested))
	}
	return sb.String()
}

// Choice holds a single choice from the response.
type Choice struct {
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// ToolCall represents a function call requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the name and arguments of a tool call.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON-encoded string
}

// ResponseMessage extends Message with optional reasoning content.
type ResponseMessage struct {
	Role             string     `json:"role"`
	Content          *string    `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

// ChatResponse is the parsed result of a completion request.
type ChatResponse struct {
	ID      string      `json:"-"`
	Model   string      `json:"-"`
	Choices []Choice    `json:"-"`
	Headers http.Header `json:"-"` // Raw response headers (may contain session IDs, etc.)
	Usage   *UsageInfo  `json:"-"` // Token usage extracted from API response
}

// Content extracts the text content from the first choice, falling back to reasoning content.
func (r *ChatResponse) Content() string {
	if len(r.Choices) == 0 {
		return ""
	}
	msg := r.Choices[0].Message
	if msg.Content != nil && *msg.Content != "" {
		cleaned := stripThinkTags(*msg.Content)
		return strings.TrimSpace(cleaned)
	}
	return msg.ReasoningContent
}

// ToolCalls extracts tool calls from the first choice.
func (r *ChatResponse) ToolCalls() []ToolCall {
	if len(r.Choices) == 0 {
		return nil
	}
	return r.Choices[0].Message.ToolCalls
}

// ToolDef defines a tool/function available to the model.
type ToolDef struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

// FunctionDef specifies the metadata for a tool definition.
type FunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// ClientConfig holds configuration for connecting to an LLM service.
type ClientConfig struct {
	URL                    string         // Full API endpoint URL
	APIKey                 string         // Bearer token / API key
	Model                  string         // Default model override
	Timeout                time.Duration  // Request timeout
	ExtraBody              map[string]any // Vendor-specific fields merged into every request body
	UseMaxCompletionTokens bool           // use max_completion_tokens instead of max_tokens
}

// --- Factory ---

// NewLLMClient creates the appropriate client based on the resolved endpoint protocol.
// protocol: "anthropic" -> AnthropicClient, anything else -> OpenAIClient.
func NewLLMClient(ep ResolvedEndpoint) LLMClient {
	cfg := ClientConfig{
		URL:                    ep.URL,
		APIKey:                 ep.Token,
		Model:                  ep.Model,
		ExtraBody:              ep.ExtraBody,
		UseMaxCompletionTokens: ep.UseMaxCompletionTokens,
	}
	if ep.Protocol == "anthropic" {
		return NewAnthropicClient(cfg)
	}
	return NewOpenAIClient(cfg)
}

// --- Token counting with tiktoken ---

// modelTokenizerCache caches initialized tiktoken encoders keyed by encoding name.
type modelTokenizerCache struct {
	mu    sync.RWMutex
	cache map[string]*tiktoken.Tiktoken
}

func newModelTokenizerCache() *modelTokenizerCache {
	return &modelTokenizerCache{cache: make(map[string]*tiktoken.Tiktoken)}
}

func (c *modelTokenizerCache) getOrLoad(encName string) (*tiktoken.Tiktoken, error) {
	c.mu.RLock()
	if tke, ok := c.cache[encName]; ok {
		c.mu.RUnlock()
		return tke, nil
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()

	if tke, ok := c.cache[encName]; ok {
		return tke, nil
	}
	enc, err := tiktoken.GetEncoding(encName)
	if err != nil {
		return nil, fmt.Errorf("get tiktoken encoding %q: %w", encName, err)
	}
	c.cache[encName] = enc
	return enc, nil
}

var defaultTokenizer = newModelTokenizerCache()

func countTokensWithEncoding(text string, encName string) int {
	tke, err := defaultTokenizer.getOrLoad(encName)
	if err != nil {
		return len([]byte(text)) / 4
	}
	return len(tke.Encode(text, nil, nil))
}

func CountTokens(text string) int {
	return CountTokensForModel(text, "")
}

func CountTokensForModel(text string, modelName string) int {
	if text == "" {
		return 0
	}
	encName := encodingForModel(modelName)
	return countTokensWithEncoding(text, encName)
}

func encodingForModel(modelName string) string {
	lower := strings.ToLower(modelName)
	switch {
	case strings.Contains(lower, "o1") || strings.Contains(lower, "o3") || strings.Contains(lower, "o4"):
		return "o200k_base"
	default:
		return "cl100k_base"
	}
}

// ChatRequest represents the payload for a chat completion call.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Tools       []ToolDef `json:"tools,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

// --- OpenAIClient ---

// OpenAIClient sends requests to an OpenAI-compatible chat completion API via the official SDK.
type OpenAIClient struct {
	cfg    ClientConfig
	client openai.Client
}

// NewOpenAIClient creates a new OpenAI-compatible LLM client using the official SDK.
func NewOpenAIClient(cfg ClientConfig) *OpenAIClient {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 5 * time.Minute
	}

	// Normalize URL: strip /chat/completions suffix since SDK appends it automatically.
	baseURL := normalizeOpenAIBaseURL(cfg.URL)

	opts := []openaioption.RequestOption{
		openaioption.WithAPIKey(cfg.APIKey),
		openaioption.WithBaseURL(baseURL),
		openaioption.WithHeader("User-Agent", userAgent("")),
		openaioption.WithMaxRetries(maxRetries),
		openaioption.WithRequestTimeout(cfg.Timeout),
	}

	client := openai.NewClient(opts...)
	return &OpenAIClient{cfg: cfg, client: client}
}

// NewClient is kept as an alias for backward compatibility during transition.
func NewClient(cfg ClientConfig) *OpenAIClient {
	return NewOpenAIClient(cfg)
}

// Completions sends a chat completion request and returns the parsed response.
func (c *OpenAIClient) Completions(req ChatRequest) (*ChatResponse, error) {
	return c.CompletionsWithCtx(context.Background(), req)
}

// CompletionsWithCtx sends a chat completion request with context support for cancellation and timeout.
func (c *OpenAIClient) CompletionsWithCtx(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = c.cfg.Model
	}

	params := c.buildParams(model, req)
	reqOpts := c.buildExtraBodyOptions()

	// Capture raw HTTP response for headers
	var httpResp *http.Response
	reqOpts = append(reqOpts, openaioption.WithResponseInto(&httpResp))

	completion, err := c.client.Chat.Completions.New(ctx, params, reqOpts...)
	if err != nil {
		return nil, fmt.Errorf("API error: %w", err)
	}

	return c.convertResponse(completion, httpResp), nil
}

// GeneralRequest sends a simple chat request without or with optional tool calls.
func (c *OpenAIClient) GeneralRequest(messages []Message, model string, tools []ToolDef) (*ChatResponse, error) {
	return c.GeneralRequestWithCtx(context.Background(), messages, model, tools)
}

// GeneralRequestWithCtx sends a simple chat request with context support.
func (c *OpenAIClient) GeneralRequestWithCtx(ctx context.Context, messages []Message, model string, tools []ToolDef) (*ChatResponse, error) {
	return c.CompletionsWithCtx(ctx, ChatRequest{
		Model:    model,
		Messages: messages,
		Tools:    tools,
	})
}

// StreamCompletion initiates a streaming chat completion. The callback is invoked per chunk.
func (c *OpenAIClient) StreamCompletion(ctx context.Context, req ChatRequest, cb func(chunk []byte) error) error {
	model := req.Model
	if model == "" {
		model = c.cfg.Model
	}

	params := c.buildParams(model, req)
	// Enable usage reporting in stream so the final chunk contains token stats.
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}
	reqOpts := c.buildExtraBodyOptions()

	stream := c.client.Chat.Completions.NewStreaming(ctx, params, reqOpts...)
	defer stream.Close()

	for stream.Next() {
		chunk := stream.Current()
		// Use RawJSON to preserve the original API response format for downstream consumers.
		raw := chunk.RawJSON()
		if raw == "" {
			continue
		}
		if err := cb([]byte(raw)); err != nil {
			return err
		}
	}
	return stream.Err()
}

// buildParams converts internal ChatRequest to OpenAI SDK params.
func (c *OpenAIClient) buildParams(model string, req ChatRequest) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Model:    model,
		Messages: convertMessagesToOpenAI(req.Messages),
	}

	if len(req.Tools) > 0 {
		params.Tools = convertToolsToOpenAI(req.Tools)
	}

	if req.MaxTokens > 0 {
		if c.cfg.UseMaxCompletionTokens {
			params.MaxCompletionTokens = openai.Int(int64(req.MaxTokens))
		} else {
			params.MaxTokens = openai.Int(int64(req.MaxTokens))
		}
	}

	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}

	return params
}

// buildExtraBodyOptions creates request options for ExtraBody fields.
func (c *OpenAIClient) buildExtraBodyOptions() []openaioption.RequestOption {
	if len(c.cfg.ExtraBody) == 0 {
		return nil
	}
	opts := make([]openaioption.RequestOption, 0, len(c.cfg.ExtraBody))
	for k, v := range c.cfg.ExtraBody {
		opts = append(opts, openaioption.WithJSONSet(k, v))
	}
	return opts
}

// convertResponse maps the SDK ChatCompletion to our internal ChatResponse.
func (c *OpenAIClient) convertResponse(comp *openai.ChatCompletion, httpResp *http.Response) *ChatResponse {
	if comp == nil {
		return &ChatResponse{}
	}
	choices := make([]Choice, 0, len(comp.Choices))
	for _, ch := range comp.Choices {
		var content *string
		if ch.Message.Content != "" {
			s := ch.Message.Content
			content = &s
		}

		toolCalls := make([]ToolCall, 0, len(ch.Message.ToolCalls))
		for _, tc := range ch.Message.ToolCalls {
			toolCalls = append(toolCalls, ToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}

		choices = append(choices, Choice{
			Message: ResponseMessage{
				Role:      string(ch.Message.Role),
				Content:   content,
				ToolCalls: toolCalls,
			},
			FinishReason: string(ch.FinishReason),
		})
	}

	var usage *UsageInfo
	if comp.Usage.TotalTokens > 0 || comp.Usage.PromptTokens > 0 || comp.Usage.CompletionTokens > 0 {
		cachedTokens := comp.Usage.PromptTokensDetails.CachedTokens
		usage = &UsageInfo{
			TotalTokens:      comp.Usage.TotalTokens,
			PromptTokens:     comp.Usage.PromptTokens - cachedTokens,
			CompletionTokens: comp.Usage.CompletionTokens,
			CacheReadTokens:  cachedTokens,
		}
		if usage.TotalTokens == 0 {
			usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens + usage.CacheReadTokens
		}
	}

	var headers http.Header
	if httpResp != nil {
		headers = httpResp.Header
	}

	return &ChatResponse{
		ID:      comp.ID,
		Model:   comp.Model,
		Choices: choices,
		Headers: headers,
		Usage:   usage,
	}
}

// convertMessagesToOpenAI maps internal Message slice to OpenAI SDK message params.
func convertMessagesToOpenAI(messages []Message) []openai.ChatCompletionMessageParamUnion {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			result = append(result, openai.SystemMessage(msg.ExtractText()))
		case "user":
			result = append(result, openai.UserMessage(msg.ExtractText()))
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(msg.ToolCalls))
				for _, tc := range msg.ToolCalls {
					toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
							ID: tc.ID,
							Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
								Name:      tc.Function.Name,
								Arguments: tc.Function.Arguments,
							},
						},
					})
				}
				text := msg.ExtractText()
				param := openai.ChatCompletionAssistantMessageParam{
					ToolCalls: toolCalls,
				}
				if text != "" {
					param.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: openai.String(text),
					}
				}
				result = append(result, openai.ChatCompletionMessageParamUnion{
					OfAssistant: &param,
				})
			} else {
				result = append(result, openai.AssistantMessage(msg.ExtractText()))
			}
		case "tool":
			result = append(result, openai.ToolMessage(msg.ExtractText(), msg.ToolCallID))
		}
	}
	return result
}

// convertToolsToOpenAI maps internal ToolDef slice to OpenAI SDK tool params.
func convertToolsToOpenAI(tools []ToolDef) []openai.ChatCompletionToolUnionParam {
	result := make([]openai.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		result = append(result, openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        t.Function.Name,
			Description: openai.String(t.Function.Description),
			Parameters:  openai.FunctionParameters(t.Function.Parameters),
		}))
	}
	return result
}

// normalizeOpenAIBaseURL strips the /chat/completions suffix since the SDK appends it.
func normalizeOpenAIBaseURL(rawURL string) string {
	u := strings.TrimRight(rawURL, "/")
	u = strings.TrimSuffix(u, "/chat/completions")
	return u
}

// --- AnthropicClient ---

// AnthropicClient implements the Anthropic Messages API via the official SDK.
type AnthropicClient struct {
	cfg    ClientConfig
	client anthropic.Client
}

// NewAnthropicClient creates a new Anthropic Messages API client using the official SDK.
func NewAnthropicClient(cfg ClientConfig) *AnthropicClient {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 5 * time.Minute
	}

	// Normalize URL: strip /v1/messages suffix since SDK appends it automatically.
	baseURL := normalizeAnthropicBaseURL(cfg.URL)

	opts := []anthropicoption.RequestOption{
		anthropicoption.WithAPIKey(cfg.APIKey),
		anthropicoption.WithBaseURL(baseURL),
		anthropicoption.WithHeader("User-Agent", userAgent("claude")),
		anthropicoption.WithMaxRetries(maxRetries),
		anthropicoption.WithRequestTimeout(cfg.Timeout),
	}

	client := anthropic.NewClient(opts...)
	return &AnthropicClient{cfg: cfg, client: client}
}

// Completions sends a chat completion request and returns the parsed response.
func (c *AnthropicClient) Completions(req ChatRequest) (*ChatResponse, error) {
	return c.CompletionsWithCtx(context.Background(), req)
}

// CompletionsWithCtx sends a chat completion request with context support.
func (c *AnthropicClient) CompletionsWithCtx(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = c.cfg.Model
	}

	params := c.buildParams(model, req)
	reqOpts := c.buildExtraBodyOptions()

	// Capture raw HTTP response for headers
	var httpResp *http.Response
	reqOpts = append(reqOpts, anthropicoption.WithResponseInto(&httpResp))

	message, err := c.client.Messages.New(ctx, params, reqOpts...)
	if err != nil {
		return nil, fmt.Errorf("API error: %w", err)
	}

	return c.convertResponse(message, httpResp), nil
}

// StreamCompletion initiates a streaming chat completion using SSE.
func (c *AnthropicClient) StreamCompletion(ctx context.Context, req ChatRequest, cb func(chunk []byte) error) error {
	model := req.Model
	if model == "" {
		model = c.cfg.Model
	}

	params := c.buildParams(model, req)
	reqOpts := c.buildExtraBodyOptions()

	stream := c.client.Messages.NewStreaming(ctx, params, reqOpts...)
	defer stream.Close()

	for stream.Next() {
		event := stream.Current()
		// Use RawJSON to preserve the original API response format for downstream consumers.
		raw := event.RawJSON()
		if raw == "" {
			continue
		}
		if err := cb([]byte(raw)); err != nil {
			return err
		}
	}
	return stream.Err()
}

// buildParams converts internal ChatRequest to Anthropic SDK params.
func (c *AnthropicClient) buildParams(model string, req ChatRequest) anthropic.MessageNewParams {
	system, messages := convertMessagesToAnthropic(req.Messages)

	maxTokens := int64(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 8192
	}

	params := anthropic.MessageNewParams{
		Model:     model,
		MaxTokens: maxTokens,
		Messages:  messages,
	}

	if system != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: system},
		}
	}

	if len(req.Tools) > 0 {
		params.Tools = convertToolsToAnthropic(req.Tools)
	}

	if req.Temperature != nil {
		params.Temperature = anthropic.Float(*req.Temperature)
	}

	return params
}

// buildExtraBodyOptions creates request options for ExtraBody fields.
func (c *AnthropicClient) buildExtraBodyOptions() []anthropicoption.RequestOption {
	if len(c.cfg.ExtraBody) == 0 {
		return nil
	}
	opts := make([]anthropicoption.RequestOption, 0, len(c.cfg.ExtraBody))
	for k, v := range c.cfg.ExtraBody {
		opts = append(opts, anthropicoption.WithJSONSet(k, v))
	}
	return opts
}

// convertResponse maps Anthropic SDK Message to our internal ChatResponse.
func (c *AnthropicClient) convertResponse(msg *anthropic.Message, httpResp *http.Response) *ChatResponse {
	var textParts []string
	var toolCalls []ToolCall

	for _, block := range msg.Content {
		switch variant := block.AsAny().(type) {
		case anthropic.TextBlock:
			textParts = append(textParts, variant.Text)
		case anthropic.ToolUseBlock:
			argsJSON, _ := json.Marshal(variant.Input)
			toolCalls = append(toolCalls, ToolCall{
				ID:   variant.ID,
				Type: "function",
				Function: FunctionCall{
					Name:      variant.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	var contentStr *string
	if len(textParts) > 0 {
		s := strings.Join(textParts, "\n")
		contentStr = &s
	}

	finishReason := string(msg.StopReason)
	if finishReason == "" {
		finishReason = "stop"
	}

	var usage *UsageInfo
	if msg.Usage.InputTokens > 0 || msg.Usage.OutputTokens > 0 || msg.Usage.CacheReadInputTokens > 0 || msg.Usage.CacheCreationInputTokens > 0 {
		usage = &UsageInfo{
			PromptTokens:     msg.Usage.InputTokens,
			CompletionTokens: msg.Usage.OutputTokens,
			CacheReadTokens:  msg.Usage.CacheReadInputTokens,
			CacheWriteTokens: msg.Usage.CacheCreationInputTokens,
		}
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens + usage.CacheReadTokens + usage.CacheWriteTokens
	}

	var headers http.Header
	if httpResp != nil {
		headers = httpResp.Header
	}

	return &ChatResponse{
		ID:    msg.ID,
		Model: string(msg.Model),
		Choices: []Choice{{
			Message: ResponseMessage{
				Role:      string(msg.Role),
				Content:   contentStr,
				ToolCalls: toolCalls,
			},
			FinishReason: finishReason,
		}},
		Headers: headers,
		Usage:   usage,
	}
}

// convertMessagesToAnthropic separates system message and converts remaining messages.
func convertMessagesToAnthropic(messages []Message) (string, []anthropic.MessageParam) {
	var systemMsg string
	var result []anthropic.MessageParam
	var pendingToolResults []Message

	flushToolResults := func() {
		if len(pendingToolResults) == 0 {
			return
		}
		var blocks []anthropic.ContentBlockParamUnion
		for _, tr := range pendingToolResults {
			blocks = append(blocks, anthropic.NewToolResultBlock(
				tr.ToolCallID,
				tr.ExtractText(),
				false,
			))
		}
		result = append(result, anthropic.NewUserMessage(blocks...))
		pendingToolResults = nil
	}

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			if s, ok := msg.Content.(string); ok {
				systemMsg = s
			}
			flushToolResults()
		case "tool":
			pendingToolResults = append(pendingToolResults, msg)
		case "assistant":
			flushToolResults()
			var blocks []anthropic.ContentBlockParamUnion
			text := msg.ExtractText()
			if text != "" {
				blocks = append(blocks, anthropic.NewTextBlock(text))
			}
			for _, tc := range msg.ToolCalls {
				argsMap := map[string]any{}
				if tc.Function.Arguments != "" {
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &argsMap); err != nil {
						fmt.Fprintf(stdout.Writer(), "[llm] WARNING: failed to parse tool call arguments JSON for %q: %v\n", tc.ID, err)
					}
				}
				inputJSON, _ := json.Marshal(argsMap)
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    tc.ID,
						Name:  tc.Function.Name,
						Input: json.RawMessage(inputJSON),
					},
				})
			}
			if len(blocks) > 0 {
				result = append(result, anthropic.NewAssistantMessage(blocks...))
			}
		default:
			// user or other roles
			flushToolResults()
			result = append(result, anthropic.NewUserMessage(anthropic.NewTextBlock(msg.ExtractText())))
		}
	}
	flushToolResults()

	return systemMsg, result
}

// convertToolsToAnthropic maps internal ToolDef slice to Anthropic SDK tool params.
func convertToolsToAnthropic(tools []ToolDef) []anthropic.ToolUnionParam {
	result := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		schema := anthropic.ToolInputSchemaParam{
			Type:       "object",
			Properties: t.Function.Parameters["properties"],
		}
		// Preserve required field constraints from the original schema.
		if req, ok := t.Function.Parameters["required"]; ok {
			if reqSlice, ok := req.([]any); ok {
				required := make([]string, 0, len(reqSlice))
				for _, r := range reqSlice {
					if s, ok := r.(string); ok {
						required = append(required, s)
					}
				}
				schema.Required = required
			}
		}
		result = append(result, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Function.Name,
				Description: anthropic.String(t.Function.Description),
				InputSchema: schema,
			},
		})
	}
	return result
}

// normalizeAnthropicBaseURL strips the /v1/messages suffix since the SDK appends it.
func normalizeAnthropicBaseURL(rawURL string) string {
	u := strings.TrimRight(rawURL, "/")
	u = strings.TrimSuffix(u, "/v1/messages")
	return u
}

// --- Utility functions ---

// stripThinkTags removes reasoning wrapper tags from content.
func stripThinkTags(s string) string {
	openBytes := []byte{0x3c, 't', 'h', 'i', 'n', 'k', 0x3e}
	closeBytes := []byte{0x3c, 0x2f, 't', 'h', 'i', 'n', 'k', 0x3e}
	s = strings.ReplaceAll(s, string(openBytes), "")
	s = strings.ReplaceAll(s, string(closeBytes), "")
	return s
}

// extractErrorMessage attempts to pull a human-readable error message from
// a JSON API error response body. Falls back to truncating the raw body if
// the structure is not recognised or decoding fails.
func extractErrorMessage(body []byte) string {
	type openAIError struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	type anthropicError struct {
		Type  string `json:"type"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if len(body) == 0 {
		return "(empty body)"
	}

	var oe openAIError
	if err := json.Unmarshal(body, &oe); err == nil && oe.Error.Message != "" {
		return oe.Error.Message
	}
	var ae anthropicError
	if err := json.Unmarshal(body, &ae); err == nil && ae.Error.Message != "" {
		return ae.Error.Message
	}

	bodyText := string(body)
	if len(bodyText) > 512 {
		bodyText = bodyText[:512] + "... (truncated)"
	}
	return bodyText
}
