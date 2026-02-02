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

export const reqify = {
  get: <T = any>(url: string, options: ReqifyOptions = {}) => 
    request<T>(url, { ...options, method: 'GET' }),
  
  post: <T = any>(url: string, body: any, options: ReqifyOptions = {}) => 
    request<T>(url, { ...options, method: 'POST', body }),

  postStream: (url: string, body: any, options: ReqifyOptions = {}) => 
    requestStream(url, { ...options, method: 'POST', body }),
  
  put: <T = any>(url: string, body: any, options: ReqifyOptions = {}) => 
    request<T>(url, { ...options, method: 'PUT', body }),
  
  delete: <T = any>(url: string, options: ReqifyOptions = {}) => 
    request<T>(url, { ...options, method: 'DELETE' }),
};

async function request<T>(url: string, options: ReqifyOptions): Promise<ReqifyResponse<T>> {
  const { retries = 0, retryDelay = 1000, ...fetchOptions } = options;
  let lastError: any;

  for (let i = 0; i <= retries; i++) {
    try {
      const headers = new Headers(fetchOptions.headers);
      if (fetchOptions.body && !headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/json');
      }

      const response = await fetch(url, {
        method: fetchOptions.method,
        headers,
        body: fetchOptions.body ? JSON.stringify(fetchOptions.body) : undefined,
      });

      let data: any;
      const contentType = response.headers.get('Content-Type');
      if (contentType?.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text();
      }

      if (!response.ok) {
        throw {
          message: `Request failed with status ${response.status}`,
          status: response.status,
          data,
        };
      }

      return {
        data: data as T,
        status: response.status,
        headers: response.headers,
      };
    } catch (error) {
      lastError = error;
      if (i < retries) {
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      }
    }
  }

  throw lastError;
}

async function requestStream(url: string, options: ReqifyOptions): Promise<ReqifyStreamResponse> {
  const headers = new Headers(options.headers);
  if (options.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(url, {
    method: options.method,
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });

  if (!response.ok) {
    const data = await response.text();
    throw {
      message: `Request failed with status ${response.status}`,
      status: response.status,
      data,
    };
  }

  if (!response.body) {
    throw new Error('Response body is empty');
  }

  return {
    stream: response.body,
    status: response.status,
    headers: response.headers,
  };
}
