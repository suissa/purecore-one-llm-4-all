# Changelog

## [Release] v1.1.0

### What's Changed

- [x] Interface unificada `UnifiedLLMParams`: obrigatórios (`messages`, `model`) e opcionais com defaults (`max_tokens`, `temperature`, `top_p`, `stream`, `response_format`, etc.)
- [x] Constantes `LLM_DEFAULTS` para valores padrão
- [x] Fluent/Builder: `sendPrompt(prompt, opts).getText()` e `sendPrompt(prompt, opts).getJSONResponse()`
- [x] Adaptador `runLLM` para Groq, OpenRouter, Anthropic e Gemini (modo não-streaming)
- [x] Tipos: `LLMMessage`, `LLMMessageRole`, `LLMProvider`, `UnifiedLLMTextResult`, `PromptFluent`, `SendPromptOptions`

### New Contributors

- *N/A*

---

## [Release] v1.0.0

### What's Changed

- [x] Instalação dos SDKs de LLM: `groq-sdk`, `@openrouter/sdk`, `@anthropic-ai/sdk`, `@google/genai`
- [x] Criação de `.gitignore` com `node_modules` e `dist`
- [x] Relatório em `reports/25-01-2025_instalacao-llm-sdks.md` com libs, exemplos e links

### New Contributors

- *N/A*
