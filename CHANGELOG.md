# Changelog

## [Release] v2.0.1

### What's Changed

- ✅ **TDD & BDD**: Reestruturação completa dos testes em `tests/unit` e `tests/integration`.
- ✅ **Cobertura de Testes**: Adicionada cobertura unitária para todos os clientes nativos (OpenAI, Anthropic, Gemini).
- ✅ **Cenários BDD**: Implementados testes de integração baseados em cenários de uso real com `sendPrompt`.

## [Release] v2.0.0

### What's Changed

- [x] **Arquitetura Multi-Provider Nativa**: Removidos SDKs externos (`groq-sdk`, `@anthropic-ai/sdk`, etc.) em favor de implementações locais em `/packages`.
- [x] **Novo Pacote `reqify`**: Cliente HTTP nativo com suporte a retentativas (retry) e streaming utilizando `fetch`.
- [x] **Suporte a Novos Provedores**: Adicionados OpenAI, DeepSeek, Mistral e Perplexity via interface unificada.
- [x] **Tipagem Semântica Nominal**: Implementação de Branded Types para `ApiKey`, `ModelId` e `PromptContent` visando maior segurança de tipos.
- [x] **Suporte a Streaming**: Adicionado método `.getStream()` na interface fluente com parsers de SSE otimizados.
- [x] **Suite de Testes Abrangente**: Implementados testes unitários e de integração utilizando `Vitest` e `MSW`.
- [x] **Documentação OpenAPI**: Gerado arquivo `openapi.json` e `swagger.html` para visualização técnica da interface.

### New Providers Added

- OpenAI (Direct)
- DeepSeek
- Mistral
- Perplexity

---

## [Release] v1.1.0

### What's Changed

- [x] Interface unificada `UnifiedLLMParams`: obrigatórios (`messages`, `model`) e opcionais com defaults (`max_tokens`, `temperature`, `top_p`, `stream`, `response_format`, etc.)
- [x] Constantes `LLM_DEFAULTS` para valores padrão
- [x] Fluent/Builder: `sendPrompt(prompt, opts).getText()` e `sendPrompt(prompt, opts).getJSONResponse()`
- [x] Adaptador `runLLM` para Groq, OpenRouter, Anthropic e Gemini (modo não-streaming)
- [x] Tipos: `LLMMessage`, `LLMMessageRole`, `LLMProvider`, `UnifiedLLMTextResult`, `PromptFluent`, `SendPromptOptions`

---

## [Release] v1.0.0

### What's Changed

- [x] Instalação dos SDKs de LLM: `groq-sdk`, `@openrouter/sdk`, `@anthropic-ai/sdk`, `@google/genai`
- [x] Criação de `.gitignore` com `node_modules` e `dist`
- [x] Relatório em `reports/25-01-2025_instalacao-llm-sdks.md` com libs, exemplos e links
