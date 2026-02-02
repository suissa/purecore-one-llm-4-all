import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { OpenAIClient } from '../../packages/openai/index.js';
import { handlers } from '../mocks/handlers.js';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Unit: OpenAIClient', () => {
  it('should initialize correctly', () => {
    const client = new OpenAIClient({ apiKey: 'test-key' });
    expect(client).toBeDefined();
  });

  it('should create chat completion (text)', async () => {
    const client = new OpenAIClient({ apiKey: 'test-key' });
    const res = await client.createChatCompletion({
      model: 'gpt-4',
      messages: [{ role: 'user', content: 'Hi' }]
    });

    expect(res.choices[0].message.content).toBe('Hello! I am OpenAI.');
    expect(res.model).toBe('gpt-4');
  });

  it('should create chat completion (stream)', async () => {
    const client = new OpenAIClient({ apiKey: 'test-key' });
    const stream = await client.createChatCompletion({
      model: 'gpt-4',
      messages: [{ role: 'user', content: 'Hi' }],
      stream: true
    });

    expect(stream).toBeDefined();
    expect(stream).toBeInstanceOf(ReadableStream);
  });
});
