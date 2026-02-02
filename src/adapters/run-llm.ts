/**
 * Adaptador: recebe UnifiedLLMParams e chama o SDK correto.
 * Extrai o texto da resposta de cada provedor.
 */

import Groq from 'groq-sdk';
import { OpenRouter } from '@openrouter/sdk';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenAI } from '@google/genai';
import { LLM_DEFAULTS } from '../constants/llm-defaults.js';
import type {
  UnifiedLLMParams,
  UnifiedLLMTextResult,
  LLMMessage,
  LLMProvider,
} from '../types/llm.types.js';

function toMessages(m: UnifiedLLMParams['messages']): LLMMessage[] {
  if (typeof m === 'string') return [{ role: 'user', content: m }];
  return m;
}

const ENV_KEYS: Record<LLMProvider, string> = {
  groq: 'GROQ_API_KEY',
  openrouter: 'OPENROUTER_API_KEY',
  anthropic: 'ANTHROPIC_API_KEY',
  gemini: 'GEMINI_API_KEY',
};

function getApiKey(provider: LLMProvider, apiKey?: string): string {
  const k = apiKey ?? process.env[ENV_KEYS[provider]] ?? process.env.GOOGLE_GENAI_API_KEY;
  if (!k) throw new Error(`Missing API key for ${provider}. Set apiKey or ${ENV_KEYS[provider]}.`);
  return k;
}

export async function runLLM(
  params: UnifiedLLMParams,
  output: 'text' | 'json'
): Promise<UnifiedLLMTextResult & { json?: unknown }> {
  const messages = toMessages(params.messages);
  const provider = (params.provider ?? 'groq') as LLMProvider;
  const apiKey = getApiKey(provider, params.apiKey);
  const D = LLM_DEFAULTS;

  if (provider === 'groq') {
    const client = new Groq({ apiKey });
    const res = await client.chat.completions.create({
      model: params.model,
      messages: messages.map((x) => ({ role: x.role as 'system' | 'user' | 'assistant', content: x.content })),
      max_tokens: params.max_tokens ?? D.max_tokens,
      temperature: params.temperature ?? D.temperature,
      top_p: params.top_p ?? D.top_p,
      stream: false,
      stop: params.stop,
      response_format: output === 'json' ? { type: 'json_object' } : undefined,
    });
    const done = res as { choices?: { message?: { content?: string }; finish_reason?: string }[]; usage?: { prompt_tokens?: number; completion_tokens?: number }; model?: string };
    const text = (done.choices?.[0]?.message?.content as string) ?? '';
    const usage = done.usage;
    const out: UnifiedLLMTextResult & { json?: unknown } = { text, usage, model: done.model, finish_reason: done.choices?.[0]?.finish_reason };
    if (output === 'json') {
      try {
        out.json = JSON.parse(text);
      } catch {
        out.json = { raw: text };
      }
    }
    return out;
  }

  if (provider === 'openrouter') {
    const client = new OpenRouter({ apiKey });
    const res = await client.chat.send({
      model: params.model,
      messages: messages.map((x) => ({ role: x.role, content: x.content })),
      maxTokens: params.max_tokens ?? D.max_tokens,
      temperature: params.temperature ?? D.temperature,
      topP: params.top_p ?? D.top_p,
      stream: false,
      stop: params.stop,
      responseFormat: output === 'json' ? { type: 'json_object' } : undefined,
    });
    const raw = (res as { choices?: { message?: { content?: string | Array<{ text?: string }> } }[] }).choices?.[0]?.message?.content;
    const text = typeof raw === 'string' ? raw : (Array.isArray(raw) ? raw.map((x) => (typeof x === 'object' && x && 'text' in x ? (x as { text: string }).text : String(x))).join('') : '');
    const usage = (res as { usage?: { prompt_tokens?: number; completion_tokens?: number } }).usage;
    const out: UnifiedLLMTextResult & { json?: unknown } = { text, usage, model: (res as { model?: string }).model, finish_reason: (res as { choices?: { finishReason?: string }[] }).choices?.[0]?.finishReason ?? undefined };
    if (output === 'json') {
      try {
        out.json = JSON.parse(text);
      } catch {
        out.json = { raw: text };
      }
    }
    return out;
  }

  if (provider === 'anthropic') {
    const client = new Anthropic({ apiKey });
    const max = params.max_tokens ?? D.max_tokens ?? 1024;
    const sys = params.system ?? messages.find((x) => x.role === 'system')?.content;
    const res = await client.messages.create({
      model: params.model,
      max_tokens: max,
      messages: messages.filter((x) => x.role !== 'system').map((x) => ({ role: x.role as 'user' | 'assistant', content: x.content })),
      system: sys,
      temperature: params.temperature ?? D.temperature,
      stream: false,
      stop_sequences: Array.isArray(params.stop) ? params.stop : params.stop ? [params.stop] : undefined,
    });
    const msg = res as { content?: { type?: string; text?: string }[]; usage?: { input_tokens?: number; output_tokens?: number }; model?: string; stop_reason?: string };
    const text = (msg.content?.find((c: { type?: string }) => c.type === 'text') as { text?: string } | undefined)?.text ?? '';
    const usage = msg.usage;
    const out: UnifiedLLMTextResult & { json?: unknown } = {
      text,
      usage: usage ? { prompt_tokens: usage.input_tokens, completion_tokens: usage.output_tokens, total_tokens: (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0) } : undefined,
      model: msg.model,
      finish_reason: msg.stop_reason ?? undefined,
    };
    if (output === 'json') {
      try {
        out.json = JSON.parse(text);
      } catch {
        out.json = { raw: text };
      }
    }
    return out;
  }

  if (provider === 'gemini') {
    const ai = new GoogleGenAI({ apiKey });
    const userOnly = messages.filter((m) => m.role === 'user');
    const contents: string | { role: string; parts: { text: string }[] }[] =
      userOnly.length === 1 && messages.length === 1
        ? userOnly[0]!.content
        : messages
            .filter((m) => m.role !== 'system')
            .map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] }));
    const res = await ai.models.generateContent({
      model: params.model,
      contents,
      config: {
        temperature: params.temperature ?? D.temperature,
        topP: params.top_p ?? D.top_p,
        maxOutputTokens: params.max_tokens ?? D.max_tokens,
        stopSequences: Array.isArray(params.stop) ? params.stop : params.stop ? [params.stop] : undefined,
        systemInstruction: params.system ?? messages.find((m) => m.role === 'system')?.content,
        responseMimeType: output === 'json' ? 'application/json' : undefined,
      },
    });
    const text: string = (res as { text?: string }).text ?? '';
    const um = (res as { usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number } }).usageMetadata;
    const out: UnifiedLLMTextResult & { json?: unknown } = {
      text,
      usage: um ? { prompt_tokens: um.promptTokenCount, completion_tokens: um.candidatesTokenCount, total_tokens: (um.promptTokenCount ?? 0) + (um.candidatesTokenCount ?? 0) } : undefined,
      model: (res as { modelVersion?: string }).modelVersion ?? params.model,
    };
    if (output === 'json') {
      try {
        out.json = JSON.parse(text);
      } catch {
        out.json = { raw: text };
      }
    }
    return out;
  }

  throw new Error(`Unsupported provider: ${provider}`);
}
