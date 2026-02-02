import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { GoogleGenAIClient } from '../../packages/google-genai/index.js';
import { handlers } from '../mocks/handlers.js';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Unit: GoogleGenAIClient', () => {
  it('should initialize correctly', () => {
    const client = new GoogleGenAIClient({ apiKey: 'test-key' });
    expect(client).toBeDefined();
  });

  it('should generate content', async () => {
    const client = new GoogleGenAIClient({ apiKey: 'test-key' });
    const res = await client.generateContent({
      model: 'gemini-pro',
      contents: [{ role: 'user', parts: [{ text: 'Hi' }] }]
    });

    expect(res.candidates[0].content.parts[0].text).toBe('Hello from Gemini!');
    expect(res.usageMetadata).toBeDefined();
  });
});
