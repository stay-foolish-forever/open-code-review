package main

import (
	"testing"
)

func boolVal(b bool) *bool { return &b }

func TestSetConfigValue_StringKeys(t *testing.T) {
	tests := []struct {
		key     string
		value   string
		checkFn func(*Config) string
	}{
		{"llm.url", "https://api.example.com/v1", func(c *Config) string { return c.Llm.URL }},
		{"llm.URL", "https://api.example.com/v2", func(c *Config) string { return c.Llm.URL }},
		{"llm.auth_token", "sk-token-123", func(c *Config) string { return c.Llm.AuthToken }},
		{"llm.AuthToken", "sk-token-456", func(c *Config) string { return c.Llm.AuthToken }},
		{"llm.model", "gpt-4", func(c *Config) string { return c.Llm.Model }},
		{"llm.Model", "claude-opus-4-6", func(c *Config) string { return c.Llm.Model }},
		{"language", "English", func(c *Config) string { return c.Language }},
		{"Language", "Chinese", func(c *Config) string { return c.Language }},
		{"telemetry.exporter", "otlp", func(c *Config) string { return c.Telemetry.Exporter }},
		{"telemetry.Exporter", "console", func(c *Config) string { return c.Telemetry.Exporter }},
		{"telemetry.otlp_endpoint", "localhost:4317", func(c *Config) string { return c.Telemetry.OTLPEndpoint }},
		{"telemetry.OTLPEndpoint", "collector:4317", func(c *Config) string { return c.Telemetry.OTLPEndpoint }},
	}

	for _, tt := range tests {
		t.Run(tt.key+"="+tt.value, func(t *testing.T) {
			cfg := &Config{}
			if err := setConfigValue(cfg, tt.key, tt.value); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			got := tt.checkFn(cfg)
			if got != tt.value {
				t.Errorf("got %q, want %q", got, tt.value)
			}
		})
	}
}

func TestSetConfigValue_BoolKeys(t *testing.T) {
	tests := []struct {
		key     string
		value   string
		want    bool
		checkFn func(*Config) bool
	}{
		{"llm.use_anthropic", "true", true, func(c *Config) bool { return *c.Llm.UseAnthropic }},
		{"llm.UseAnthropic", "false", false, func(c *Config) bool { return *c.Llm.UseAnthropic }},
		{"llm.use_max_completion_tokens", "true", true, func(c *Config) bool { return *c.Llm.UseMaxCompletionTokens }},
		{"llm.UseMaxCompletionTokens", "false", false, func(c *Config) bool { return *c.Llm.UseMaxCompletionTokens }},
		{"telemetry.enabled", "true", true, func(c *Config) bool { return c.Telemetry.Enabled }},
		{"telemetry.Enabled", "false", false, func(c *Config) bool { return c.Telemetry.Enabled }},
		{"telemetry.content_logging", "true", true, func(c *Config) bool { return c.Telemetry.ContentLog }},
		{"telemetry.ContentLog", "false", false, func(c *Config) bool { return c.Telemetry.ContentLog }},
	}

	for _, tt := range tests {
		t.Run(tt.key+"="+tt.value, func(t *testing.T) {
			cfg := &Config{}
			if err := setConfigValue(cfg, tt.key, tt.value); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			got := tt.checkFn(cfg)
			if got != tt.want {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSetConfigValue_BoolKeys_InvalidValue(t *testing.T) {
	keys := []string{
		"llm.use_anthropic",
		"llm.use_max_completion_tokens",
		"telemetry.enabled",
		"telemetry.content_logging",
	}

	for _, key := range keys {
		t.Run(key, func(t *testing.T) {
			cfg := &Config{}
			err := setConfigValue(cfg, key, "not-a-bool")
			if err == nil {
				t.Fatal("expected error for invalid boolean value")
			}
		})
	}
}

func TestSetConfigValue_ExtraBody(t *testing.T) {
	tests := []struct {
		name    string
		key     string
		value   string
		wantErr bool
	}{
		{"valid JSON", "llm.extra_body", `{"thinking":{"type":"disabled"}}`, false},
		{"alias key", "llm.ExtraBody", `{"key":"value"}`, false},
		{"invalid JSON", "llm.extra_body", `not json`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{}
			err := setConfigValue(cfg, tt.key, tt.value)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error for invalid JSON")
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if cfg.Llm.ExtraBody == nil {
					t.Fatal("expected non-nil ExtraBody")
				}
			}
		})
	}
}

func TestSetConfigValue_UnknownKey(t *testing.T) {
	cfg := &Config{}
	err := setConfigValue(cfg, "unknown.key", "value")
	if err == nil {
		t.Fatal("expected error for unknown key")
	}
}

func TestSetConfigValue_TelemetryInitialized(t *testing.T) {
	cfg := &Config{}
	if cfg.Telemetry != nil {
		t.Fatal("telemetry should be nil initially")
	}

	if err := setConfigValue(cfg, "telemetry.enabled", "true"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.Telemetry == nil {
		t.Fatal("telemetry should be initialized after setting telemetry key")
	}
}
