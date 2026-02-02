export interface OpenAIClientOptions {
    apiKey: string;
    baseURL?: string;
}
export interface ChatCompletionParams {
    model: string;
    messages: Array<{
        role: string;
        content: string;
    }>;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    stop?: string | string[];
    response_format?: {
        type: 'json_object' | 'text';
    };
    stream?: boolean;
}
export declare class OpenAIClient {
    private apiKey;
    private baseURL;
    constructor(options: OpenAIClientOptions);
    createChatCompletion(params: ChatCompletionParams): Promise<any>;
    private createChatCompletionStream;
}
//# sourceMappingURL=index.d.ts.map