# Relatório: Interface unificada e Fluent para LLMs

**Data:** 25-01-2025  
**Objetivo:** Criar uma interface única com propriedades necessárias/opcionais e unificar as funções em `sendPrompt().getJSONResponse()` e `sendPrompt().getText()` (padrão encadeável).

---

## O que foi feito

1. **Interface unificada `UnifiedLLMParams`**
   - Obrigatórios: `messages` (string ou `LLMMessage[]`), `model`
   - Opcionais com defaults: `max_tokens`, `temperature`, `top_p`, `stream`, `stop`, `system`, `response_format`, `seed`, `provider`, `apiKey`

2. **Constantes `LLM_DEFAULTS`**
   - `max_tokens: 1024`, `temperature: 0.7`, `top_p: 1`, `stream: false`, `response_format: 'text'`

3. **Fluent / Builder**
   - `sendPrompt(prompt: string, options)` e `sendPrompt(messages: LLMMessage[], options)` retornam um objeto com:
     - `getText(): Promise<string>`
     - `getJSONResponse<T>(): Promise<T>` (usa `response_format: json_object` no provider)

4. **Adaptador `runLLM`**
   - Traduz `UnifiedLLMParams` para Groq, OpenRouter, Anthropic e Gemini
   - Modo não-streaming; extrai texto (e JSON quando `output === 'json'`)

5. **Tipos exportados**
   - `LLMMessage`, `LLMMessageRole`, `LLMProvider`, `UnifiedLLMTextResult`, `PromptFluent`, `SendPromptOptions`, `LLMDefaults`

---

## Por que foi feito

- Unificar parâmetros entre Groq, OpenRouter, Anthropic e Gemini em uma única interface.
- Oferecer uma API encadeável (`sendPrompt().getText()` / `getJSONResponse()`) para uso consistente, independente do provedor.

---

## Nome do padrão encadeável

Quando funções podem chamar outra de forma encadeada (ex.: `sendPrompt().getJSONResponse()`), os nomes usados são:

1. **Fluent Interface (API fluente)**  
   - Métodos retornam o próprio objeto ou um “próximo passo” para encadear.  
   - Ref.: Martin Fowler, [FluentInterface](https://martinfowler.com/bliki/FluentInterface.html).

2. **Builder Pattern**  
   - Construção em etapas; um método final (“executor”) dispara a ação (aqui: `getText`, `getJSONResponse`).  
   - Ref.: GoF, *Design Patterns*.

Neste projeto: **Builder** para montar a requisição (prompt + opts) e **Fluent** no retorno de `sendPrompt`, que expõe `.getText()` e `.getJSONResponse()`.

---

## Estrutura de arquivos

```
src/
  types/llm.types.ts      # UnifiedLLMParams, LLMMessage, LLMProvider, etc.
  constants/llm-defaults.ts
  adapters/run-llm.ts     # runLLM(params, 'text'|'json') → Groq, OpenRouter, Anthropic, Gemini
  fluent/send-prompt.ts   # sendPrompt(...) → { getText, getJSONResponse }
  index.ts                # reexporta tudo
```

---

## Exemplo de uso

```ts
import { sendPrompt } from 'one-llm-4-all';

const txt = await sendPrompt('O que é um LLM?', {
  model: 'llama-3.1-8b-instant',
  provider: 'groq',
  apiKey: process.env.GROQ_API_KEY,
}).getText();

const json = await sendPrompt('Retorne {"x":1}', {
  model: 'llama-3.1-8b-instant',
  provider: 'groq',
  apiKey: process.env.GROQ_API_KEY,
}).getJSONResponse<{ x: number }>();
```

---

## Fontes

- Martin Fowler, [FluentInterface](https://martinfowler.com/bliki/FluentInterface.html)
- GoF, *Design Patterns* (Builder)
- Groq: [console.groq.com/docs](https://console.groq.com/docs)
- OpenRouter: [openrouter.ai/docs](https://openrouter.ai/docs), [@openrouter/sdk](https://www.npmjs.com/package/@openrouter/sdk)
- Anthropic: [docs.anthropic.com](https://docs.anthropic.com)
- Google Gemini: [ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs), [@google/genai](https://www.npmjs.com/package/@google/genai)
