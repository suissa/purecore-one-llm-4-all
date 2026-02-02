# Relatório: Instalação dos SDKs de LLM (Groq, OpenRouter, Claude, Gemini)

**Data:** 25-01-2025  
**Objetivo:** Documentar as bibliotecas oficiais e instalar localmente no projeto `one-llm-4-all`.

---

## O que foi feito

- Pesquisa das bibliotecas oficiais/recomendadas para **Groq**, **OpenRouter**, **Claude (Anthropic)** e **Gemini (Google)**.
- Instalação das 4 dependências no projeto com **bun**.
- Criação de `.gitignore` com `node_modules` e `dist` (projeto não tinha).

---

## Por que foi feito

- Unificar em um único projeto os SDKs necessários para consumir múltiplos provedores de LLM.
- Usar sempre as libs oficiais ou recomendadas para cada provedor.

---

## Bibliotecas instaladas

| Provedor     | Pacote             | Versão  | Descrição breve                                         |
|-------------|--------------------|---------|----------------------------------------------------------|
| **Groq**    | `groq-sdk`         | 0.37.0  | SDK oficial Groq, compatível com API de chat completions |
| **OpenRouter** | `@openrouter/sdk` | 0.4.0   | SDK TypeScript oficial, acesso a 300+ modelos           |
| **Claude**  | `@anthropic-ai/sdk` | 0.71.2 | SDK oficial Anthropic para Claude                       |
| **Gemini**  | `@google/genai`    | 1.38.0  | SDK **novo** do Google; `@google/generative-ai` está deprecated |

---

## Exemplos mínimos de uso

### Groq

```js
import Groq from 'groq-sdk';

const client = new Groq({ apiKey: process.env.GROQ_API_KEY });
const chat = await client.chat.completions.create({
  messages: [{ role: 'user', content: 'Olá' }],
  model: 'llama-3.1-8b-instant',
});
```

- Docs: https://console.groq.com/docs  
- npm: https://www.npmjs.com/package/groq-sdk  
- GitHub: https://github.com/groq/groq-typescript  

### OpenRouter

```js
import OpenRouter from '@openrouter/sdk';

const client = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
const res = await client.chat.send({
  model: 'openai/gpt-4o-mini',
  messages: [{ role: 'user', content: 'Olá' }],
});
```

- Docs: https://openrouter.ai/docs  
- SDK TypeScript: https://openrouter.ai/docs/sdks/typescript  
- npm: https://www.npmjs.com/package/@openrouter/sdk  

### Claude (Anthropic)

```js
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const msg = await client.messages.create({
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Olá' }],
  model: 'claude-sonnet-4-5-20250929',
});
```

- Docs: https://docs.anthropic.com  
- Client SDKs: https://docs.anthropic.com/en/api/client-sdks  
- npm: https://www.npmjs.com/package/@anthropic-ai/sdk  

### Gemini (Google)

```js
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const res = await ai.models.generateContent({
  model: 'gemini-2.0-flash',
  contents: 'Olá',
});
```

- Docs: https://ai.google.dev/gemini-api/docs  
- Libs: https://ai.google.dev/gemini-api/docs/libraries  
- npm: https://www.npmjs.com/package/@google/genai  
- Migração do antigo: https://ai.google.dev/gemini-api/docs/migrate  

**Nota:** O pacote `@google/generative-ai` está em manutenção limitada e EOL em 31/08/2025; use `@google/genai`.

---

## Variáveis de ambiente sugeridas

- `GROQ_API_KEY` — [Groq Console](https://console.groq.com)
- `OPENROUTER_API_KEY` — [OpenRouter](https://openrouter.ai)
- `ANTHROPIC_API_KEY` — [Anthropic](https://console.anthropic.com)
- `GEMINI_API_KEY` ou `GOOGLE_GENAI_API_KEY` — [Google AI](https://aistudio.google.com/apikey)

---

## Comando de instalação utilizado

```bash
bun add groq-sdk @openrouter/sdk @anthropic-ai/sdk @google/genai
```

---

## Fontes

- Groq: https://www.npmjs.com/package/groq-sdk , https://console.groq.com/docs/libraries  
- OpenRouter: https://openrouter.ai/docs , https://openrouter.ai/docs/sdks/typescript  
- Anthropic: https://docs.anthropic.com/en/api/client-sdks , https://www.npmjs.com/package/@anthropic-ai/sdk  
- Google Gemini: https://ai.google.dev/gemini-api/docs/libraries , https://ai.google.dev/gemini-api/docs/migrate , https://www.npmjs.com/package/@google/genai  
