import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { OpenAIClient } from '../packages/openai/index.js';
import { sendPrompt } from '../src/index.js';
import { parseSSE } from '../src/utils/stream-parser.js';

const encoder = new TextEncoder();

const server = setupServer(
  http.post('https://api.openai.com/v1/chat/completions', async ({ request }) => {
    const body: any = await request.json();
    if (body.stream) {
      const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder();
          await new Promise(r => setTimeout(r, 10));
          controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"Stream "}}]}\n\n'));
          await new Promise(r => setTimeout(r, 10));
          controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"works!"}}]}\n\n'));
          await new Promise(r => setTimeout(r, 10));
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        },
      });
      return new HttpResponse(stream, {
        headers: { 'Content-Type': 'text/event-stream' },
      });
    }
    return HttpResponse.json({
      choices: [{ message: { content: 'Hello! I am OpenAI.' } }]
    });
  }),

  http.post('https://api.groq.com/openai/v1/chat/completions', async () => {
    return HttpResponse.json({
      choices: [{ message: { content: 'Hello from Groq!' } }]
    });
  }),

  http.post('https://api.anthropic.com/v1/messages', async () => {
    return HttpResponse.json({
      id: 'msg_123',
      content: [{ type: 'text', text: 'Hello from Anthropic!' }],
      model: 'claude-3',
      usage: { input_tokens: 10, output_tokens: 20 }
    });
  }),

  http.post('https://generativelanguage.googleapis.com/v1beta/models/:modelId', async () => {
    return HttpResponse.json({
      candidates: [{ content: { parts: [{ text: 'Hello from Gemini!' }] } }],
      usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 10 }
    });
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('OpenAI & Compatible Providers', () => {
  it('should call OpenAI API correctly', async () => {
    const client = new OpenAIClient({ apiKey: 'test-key' });
    const res = await client.createChatCompletion({
      model: 'gpt-4',
      messages: [{ role: 'user', content: 'Hi' }]
    });
    expect(res.choices[0].message.content).toBe('Hello! I am OpenAI.');
  });

  it('should work via sendPrompt fluent interface (OpenAI)', async () => {
    process.env.OPENAI_API_KEY = 'test-key';
    const text = await sendPrompt('Hi', { model: 'gpt-4', provider: 'openai' }).getText();
    expect(text).toBe('Hello! I am OpenAI.');
  });

  it('should work via sendPrompt fluent interface (Groq)', async () => {
    process.env.GROQ_API_KEY = 'test-key';
    const text = await sendPrompt('Hi', { model: 'llama3-8b', provider: 'groq' }).getText();
    expect(text).toBe('Hello from Groq!');
  });

  it('should work via sendPrompt fluent interface (Anthropic)', async () => {
    process.env.ANTHROPIC_API_KEY = 'test-key';
    const text = await sendPrompt('Hi', { model: 'claude-3', provider: 'anthropic' }).getText();
    expect(text).toBe('Hello from Anthropic!');
  });

  it('should work via sendPrompt fluent interface (Gemini)', async () => {
    process.env.GEMINI_API_KEY = 'test-key';
    const text = await sendPrompt('Hi', { model: 'gemini-pro', provider: 'gemini' }).getText();
    expect(text).toBe('Hello from Gemini!');
  });

  describe('Stream Parser Unit Test', () => {
    it('should parse SSE chunks correctly', async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"A"}}]}\n\ndata: {"choices":[{"delta":{"content":"B"}}]}\n\ndata: [DONE]\n\n'));
          controller.close();
        }
      });
      const gen = parseSSE(stream);
      let text = '';
      for await (const chunk of gen) {
        text += chunk;
      }
      expect(text).toBe('AB');
    });
  });

  it('should support streaming via getStream()', async () => {
    process.env.OPENAI_API_KEY = 'test-key';
    const stream = await sendPrompt('Hi', { model: 'gpt-4', provider: 'openai' }).getStream();
    let text = '';
    for await (const chunk of stream) {
      text += chunk;
    }
    expect(text).toBe('Stream works!');
  });
});
