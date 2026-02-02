export interface ReqifyOptions {
    method?: string;
    headers?: Record<string, string>;
    body?: any;
    retries?: number;
    retryDelay?: number;
}
export interface ReqifyResponse<T = any> {
    data: T;
    status: number;
    headers: Headers;
}
export interface ReqifyStreamResponse {
    stream: ReadableStream;
    status: number;
    headers: Headers;
}
export declare const reqify: {
    get: <T = any>(url: string, options?: ReqifyOptions) => Promise<ReqifyResponse<T>>;
    post: <T = any>(url: string, body: any, options?: ReqifyOptions) => Promise<ReqifyResponse<T>>;
    postStream: (url: string, body: any, options?: ReqifyOptions) => Promise<ReqifyStreamResponse>;
    put: <T = any>(url: string, body: any, options?: ReqifyOptions) => Promise<ReqifyResponse<T>>;
    delete: <T = any>(url: string, options?: ReqifyOptions) => Promise<ReqifyResponse<T>>;
};
//# sourceMappingURL=index.d.ts.map