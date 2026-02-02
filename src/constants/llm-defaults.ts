/**
 * Valores default para parâmetros opcionais da interface unificada.
 * Baseado em: Groq, OpenRouter, Anthropic, Gemini.
 *
 * @see https://console.groq.com/docs
 * @see https://openrouter.ai/docs
 * @see https://docs.anthropic.com
 * @see https://ai.google.dev/gemini-api/docs
 */

import type { UnifiedLLMParams } from '../types/llm.types.js';

/** Defaults para opcionais: usados quando o usuário não informa. */
export const LLM_DEFAULTS: Pick<
  UnifiedLLMParams,
  'max_tokens' | 'temperature' | 'top_p' | 'stream' | 'response_format'
> = {
  max_tokens: 1024,
  temperature: 0.7,
  top_p: 1,
  stream: false,
  response_format: 'text',
} as const;

export type LLMDefaults = typeof LLM_DEFAULTS;
