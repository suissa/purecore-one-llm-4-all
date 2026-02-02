import { reqify } from '../reqify/index.js';
export class AnthropicClient {
    apiKey;
    baseURL;
    version;
    constructor(options) {
        this.apiKey = options.apiKey;
        this.baseURL = options.baseURL || 'https://api.anthropic.com/v1';
        this.version = options.version || '2023-06-01';
    }
    async createMessage(params) {
        const response = await reqify.post(`${this.baseURL}/messages`, params, {
            headers: {
                'x-api-key': this.apiKey,
                'anthropic-version': this.version,
            },
        });
        return response.data;
    }
}
