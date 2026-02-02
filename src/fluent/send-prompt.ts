/**
 * Fluent Interface / Builder para enviar prompt e obter texto ou JSON.
 *
 * Nomes do padrão (métodos encadeáveis que retornam o próximo passo):
 * - Fluent Interface (API fluente): métodos retornam o próprio objeto ou um "próximo passo".
 * - Builder Pattern: construção em etapas; método final (getText, getJSONResponse) executa.
 *
 * Uso:
 *   const txt = await sendPrompt('Olá', { model, provider, apiKey }).getText();
 *   const obj = await sendPrompt('Retorne { "x": 1 }', { model, provider, apiKey }).getJSONResponse();
 */

import { LLM_DEFAULTS } from '../constants/llm-defaults.js';
import { runLLM } from '../adapters/run-llm.js';
import type { UnifiedLLMParams, LLMMessage, LLMProvider } from '../types/llm.types.js';

export type SendPromptOptions = Partial<Omit<UnifiedLLMParams, 'messages'>> & {
  model: string;
  provider?: LLMProvider;
  apiKey?: string;
};

/**
 * Constrói params completos a partir de prompt (string) ou options.messages.
 */
function buildParams(promptOrMessages: string | LLMMessage[], opts: SendPromptOptions): UnifiedLLMParams {
  const messages: LLMMessage[] =
    typeof promptOrMessages === 'string' ? [{ role: 'user', content: promptOrMessages }] : promptOrMessages;
  return {
    ...LLM_DEFAULTS,
    ...opts,
    messages,
    model: opts.model,
    provider: opts.provider ?? 'groq',
    apiKey: opts.apiKey,
  } as UnifiedLLMParams;
}

/**
 * Objeto fluente retornado por sendPrompt. Encadeia .getText() ou .getJSONResponse().
 */
export interface PromptFluent {
  /** Retorna o texto puro da resposta. */
  getText(): Promise<string>;
  /** Retorna o corpo parseado como JSON. Usa response_format json_object no provider. */
  getJSONResponse<T = unknown>(): Promise<T>;
}

function createFluent(params: UnifiedLLMParams): PromptFluent {
  return {
    async getText() {
      const r = await runLLM(params, 'text');
      return r.text;
    },
    async getJSONResponse<T>() {
      const r = await runLLM(params, 'json');
      return (r.json ?? {}) as T;
    },
  };
}

/**
 * Envia um prompt e retorna um objeto fluente com .getText() e .getJSONResponse().
 *
 * @example
 * const txt = await sendPrompt('Explique LLMs', { model: 'llama-3.1-8b-instant', provider: 'groq' }).getText();
 * const obj = await sendPrompt('Dados em JSON: {"a":1}', { model: 'llama-3.1-8b-instant', provider: 'groq' }).getJSONResponse();
 */
export function sendPrompt(prompt: string, options: SendPromptOptions): PromptFluent;

/**
 * Envia uma lista de mensagens (multi-turn) e retorna um objeto fluente.
 */
export function sendPrompt(messages: LLMMessage[], options: SendPromptOptions): PromptFluent;

export function sendPrompt(
  promptOrMessages: string | LLMMessage[],
  options: SendPromptOptions
): PromptFluent {
  const params = buildParams(promptOrMessages, options);
  return createFluent(params);
}
