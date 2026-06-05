# Multi-Provider Support

[中文版本](#多模型服务商支持)

## Overview

This branch implements multi-provider support for OpenCodeReview, enabling seamless integration with various LLM service providers beyond the currently supported protocols (Anthropic Messages API and OpenAI Chat Completions API).

## Motivation

OpenCodeReview currently supports two main LLM API protocols:
- Anthropic Messages API
- OpenAI Chat Completions API

While this covers major providers like Anthropic Claude, OpenAI, and Azure OpenAI, there's a growing ecosystem of LLM providers with their own APIs or OpenAI-compatible endpoints. This feature aims to:

1. **Support more LLM providers** - Enable integration with providers like:
   - Google Gemini
   - Alibaba Qwen (通义千问)
   - Baidu ERNIE (文心一言)
   - Zhipu AI GLM
   - DeepSeek
   - Other OpenAI-compatible endpoints

2. **Simplify configuration** - Provide a unified configuration interface for different providers

3. **Enable provider-specific optimizations** - Allow provider-specific features like:
   - Custom authentication methods
   - Provider-specific parameters
   - Optimized request/response handling
   - Cost tracking per provider

---

# 多模型服务商支持

## 概述

本分支为 OpenCodeReview 实现多模型服务商支持功能，使其能够无缝接入除当前支持的协议（Anthropic Messages API 和 OpenAI Chat Completions API）之外的各类 LLM 服务提供商。

## 背景

OpenCodeReview 目前支持两种主要的 LLM API 协议：
- Anthropic Messages API
- OpenAI Chat Completions API

虽然这已覆盖主要服务商如 Anthropic Claude、OpenAI 和 Azure OpenAI，但 LLM 服务商生态正在快速发展，许多服务商提供自有 API 或 OpenAI 兼容端点。本功能旨在：

1. **支持更多 LLM 服务商** - 支持接入：
   - Google Gemini
   - 通义千问
   - 百度文心一言
   - 智谱 AI GLM
   - DeepSeek
   - 其他 OpenAI 兼容端点

2. **简化配置** - 提供统一的多服务商配置接口

3. **支持服务商特性优化** - 允许针对不同服务商：
   - 自定义认证方式
   - 服务商特定参数
   - 优化的请求/响应处理
   - 按服务商的成本追踪
