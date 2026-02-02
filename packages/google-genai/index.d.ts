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
        parts: Array<{
            text: string;
        }>;
    };
}
export declare class GoogleGenAIClient {
    private apiKey;
    private baseURL;
    constructor(options: GoogleGenAIClientOptions);
    generateContent(params: GenerateContentParams): Promise<any>;
}
//# sourceMappingURL=index.d.ts.map