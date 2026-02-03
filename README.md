# one-llm-4-all

Interface unificada e **Fluent** para **OpenAI**, Groq, OpenRouter, Claude (Anthropic) e Gemini.

- [CHANGELOG](./CHANGELOG.md)

## Uso

```ts
import { sendPrompt } from 'one-llm-4-all';

// Texto
const txt = await sendPrompt('Explique LLMs em uma frase.', {
  model: 'llama-3.1-8b-instant',
  provider: 'groq',
  apiKey: process.env.GROQ_API_KEY,
}).getText();

// JSON (response_format json_object no provider)
const obj = await sendPrompt('Retorne apenas um JSON: { "ok": true }', {
  model: 'llama-3.1-8b-instant',
  provider: 'groq',
  apiKey: process.env.GROQ_API_KEY,
}).getJSONResponse();
```

## Interface unificada

`UnifiedLLMParams`: propriedades que valem para qualquer LLM.

| Obrigatórias | Opcionais (com default) |
|--------------|-------------------------|
| `messages` (string ou `LLMMessage[]`) | `max_tokens` = 1024 |
| `model` | `temperature` = 0.7 |
| | `top_p` = 1 |
| | `stream` = false |
| | `stop`, `system`, `response_format`, `seed` |
| | `provider`, `apiKey` |

## Padrão encadeável

`sendPrompt(...).getText()` e `.getJSONResponse()` usam:

- **Fluent Interface** (API fluente): métodos que retornam o “próximo passo” para encadear.
- **Builder Pattern**: montagem em etapas; `.getText()` e `.getJSONResponse()` são os métodos que executam a requisição.

## Provedores

- `openai` (**OpenAI**)
- `groq` (Groq)
- `openrouter` (OpenRouter)
- `anthropic` (Claude)
- `gemini` (Google Gemini)

API keys: `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (ou `apiKey` nas opções).

## Testes de Rotação Real

Recentemente adicionamos um sistema de testes de integração real que rotaciona modelos a cada 1 minuto. Isso é ideal para validar a estabilidade de múltiplos endpoints.

### Como foi feito
O script `scripts/real_rotation_test.ts` utiliza a interface fluente da biblioteca para percorrer o `free_models.json`, enviando um prompt de teste para cada modelo com um intervalo de 60 segundos.

### Como funciona
1. Carrega os modelos suportados.
2. Itera sequencialmente.
3. Registra logs detalhados em `rotation_results.log`.

### Como testar
```bash
bun run test:real-rotation
```

---

Para saber mais sobre as mudanças recentes, veja o [CHANGELOG.md](./CHANGELOG.md).
