/**
 * Tipos unificados para qualquer provedor de LLM.
 * Propriedades mapeadas a partir de Groq, OpenRouter, Anthropic e Gemini.
 */

/** Role de mensagem suportada em todos os LLMs (formato conversacional). */
export type LLMMessageRole = 'system' | 'user' | 'assistant';

/** Uma mensagem no formato unificado (compatível com OpenAI/OpenRouter/Anthropic). */
export interface LLMMessage {
  role: LLMMessageRole;
  content: string;
}

/** Provedores suportados. */
export type LLMProvider = 'groq' | 'openrouter' | 'anthropic' | 'gemini';

/**
 * Interface unificada: propriedades NECESSÁRIAS em qualquer LLM
 * e opcionais com valores default aplicados no cliente.
 *
 * Nomes sugeridos para o padrão de encadeamento (sendPrompt().getJSONResponse()):
 * - **Fluent Interface** (API fluente) — métodos que retornam o próprio objeto ou um "próximo passo" para encadear.
 * - **Builder Pattern** — construção em etapas; ao final, um método "executor" (getText, getJSONResponse) dispara a ação.
 *
 * Aqui usamos os dois: Builder para montar a requisição + Fluent para encadear .getText() / .getJSONResponse().
 */
export interface UnifiedLLMParams {
  // --- OBRIGATÓRIOS (todos os LLMs) ---
  /** Conteúdo do prompt (string) ou lista de mensagens para multi-turn. */
  messages: LLMMessage[] | string;
  /** ID do modelo (ex: llama-3.1-8b-instant, claude-sonnet-4-5-20250929, gemini-2.0-flash). */
  model: string;

  // --- OPCIONAIS (com defaults em LLM_DEFAULTS) ---
  /** Máximo de tokens na resposta. Anthropic exige um valor; outros usam default. */
  max_tokens?: number;
  /** Aleatoriedade (0 = determinístico, 2 = mais criativo). */
  temperature?: number;
  /** Nucleus sampling (0–1). Alternativo a temperature; não usar os dois juntos. */
  top_p?: number;
  /** Se true, retorna stream. Default false. */
  stream?: boolean;
  /** Sequências que interrompem a geração. */
  stop?: string | string[];
  /** Instrução de sistema (contexto/ persona). */
  system?: string;
  /** Formato desejado: 'text' | 'json_object' | 'json_schema'. Para getJSONResponse usamos json_object. */
  response_format?: 'text' | 'json_object' | 'json_schema';
  /** Para json_schema: definição do schema (formato do provedor). */
  response_schema?: Record<string, unknown>;
  /** Seed para reprodutibilidade. */
  seed?: number;

  // --- IDENTIFICAÇÃO DO PROVEDOR E AUTH ---
  /** Qual SDK usar. */
  provider?: LLMProvider;
  /** Chave de API (ou via env: GROQ_API_KEY, OPENROUTER_API_KEY, etc.). */
  apiKey?: string;
}

/** Resposta unificada em modo texto. */
export interface UnifiedLLMTextResult {
  text: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
  model?: string;
  finish_reason?: string;
}
