import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { AnthropicClient } from '../../packages/anthropic/index.js';
import { handlers } from '../mocks/handlers.js';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Unit: AnthropicClient', () => {
  it('should initialize correctly', () => {
    const client = new AnthropicClient({ apiKey: 'test-key' });
    expect(client).toBeDefined();
  });

  it('should create message', async () => {
    const client = new AnthropicClient({ apiKey: 'test-key' });
    const res = await client.createMessage({
      model: 'claude-3',
      messages: [{ role: 'user', content: 'Hi' }],
      max_tokens: 100
    });

    expect(res.content[0].text).toBe('Hello from Anthropic!');
    expect(res.model).toBe('claude-3');
    expect(res.stop_reason).toBe('end_turn');
  });
});
