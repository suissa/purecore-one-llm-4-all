import { reqify } from '../reqify/index.js';

export interface AnthropicClientOptions {
  apiKey: string;
  baseURL?: string;
  version?: string;
}

export interface MessagesParams {
  model: string;
  messages: Array<{ role: string; content: string }>;
  system?: string;
  temperature?: number;
  max_tokens: number;
  stop_sequences?: string[];
  stream?: boolean;
}

export class AnthropicClient {
  private apiKey: string;
  private baseURL: string;
  private version: string;

  constructor(options: AnthropicClientOptions) {
    this.apiKey = options.apiKey;
    this.baseURL = options.baseURL || 'https://api.anthropic.com/v1';
    this.version = options.version || '2023-06-01';
  }

  async createMessage(params: MessagesParams) {
    const response = await reqify.post(
      `${this.baseURL}/messages`,
      params,
      {
        headers: {
          'x-api-key': this.apiKey,
          'anthropic-version': this.version,
        },
      }
    );
    return response.data;
  }
}
