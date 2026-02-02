import { reqify } from '../reqify/index.js';
export class GoogleGenAIClient {
    apiKey;
    baseURL = 'https://generativelanguage.googleapis.com/v1beta/models';
    constructor(options) {
        this.apiKey = options.apiKey;
    }
    async generateContent(params) {
        const { model, ...rest } = params;
        const response = await reqify.post(`${this.baseURL}/${model}:generateContent?key=${this.apiKey}`, rest);
        return response.data;
    }
}
