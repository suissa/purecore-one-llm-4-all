export interface AnthropicClientOptions {
    apiKey: string;
    baseURL?: string;
    version?: string;
}
export interface MessagesParams {
    model: string;
    messages: Array<{
        role: string;
        content: string;
    }>;
    system?: string;
    temperature?: number;
    max_tokens: number;
    stop_sequences?: string[];
    stream?: boolean;
}
export declare class AnthropicClient {
    private apiKey;
    private baseURL;
    private version;
    constructor(options: AnthropicClientOptions);
    createMessage(params: MessagesParams): Promise<any>;
}
//# sourceMappingURL=index.d.ts.map