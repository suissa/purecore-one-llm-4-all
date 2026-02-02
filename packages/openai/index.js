import { reqify } from '../reqify/index.js';
export class OpenAIClient {
    apiKey;
    baseURL;
    constructor(options) {
        this.apiKey = options.apiKey;
        this.baseURL = options.baseURL || 'https://api.openai.com/v1';
    }
    async createChatCompletion(params) {
        if (params.stream) {
            return this.createChatCompletionStream(params);
        }
        const response = await reqify.post(`${this.baseURL}/chat/completions`, params, {
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
            },
        });
        return response.data;
    }
    async createChatCompletionStream(params) {
        const response = await reqify.postStream(`${this.baseURL}/chat/completions`, params, {
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
            },
        });
        return response.stream;
    }
}
