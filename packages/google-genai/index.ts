import { reqify } from '../reqify/index.js';

export interface GoogleGenAIClientOptions {
  apiKey: string;
}

export interface GenerateContentParams {
  model: string;
  contents: any;
  generationConfig?: {
    temperature?: number;
    topP?: number;
    maxOutputTokens?: number;
    stopSequences?: string[];
    responseMimeType?: string;
  };
  systemInstruction?: {
    parts: Array<{ text: string }>;
  };
}

export class GoogleGenAIClient {
  private apiKey: string;
  private baseURL: string = 'https://generativelanguage.googleapis.com/v1beta/models';

  constructor(options: GoogleGenAIClientOptions) {
    this.apiKey = options.apiKey;
  }

  async generateContent(params: GenerateContentParams) {
    const { model, ...rest } = params;
    const response = await reqify.post(
      `${this.baseURL}/${model}:generateContent?key=${this.apiKey}`,
      rest
    );
    return response.data;
  }
}
