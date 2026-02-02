export const reqify = {
    get: (url, options = {}) => request(url, { ...options, method: 'GET' }),
    post: (url, body, options = {}) => request(url, { ...options, method: 'POST', body }),
    postStream: (url, body, options = {}) => requestStream(url, { ...options, method: 'POST', body }),
    put: (url, body, options = {}) => request(url, { ...options, method: 'PUT', body }),
    delete: (url, options = {}) => request(url, { ...options, method: 'DELETE' }),
};
async function request(url, options) {
    const { retries = 0, retryDelay = 1000, ...fetchOptions } = options;
    let lastError;
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
            let data;
            const contentType = response.headers.get('Content-Type');
            if (contentType?.includes('application/json')) {
                data = await response.json();
            }
            else {
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
                data: data,
                status: response.status,
                headers: response.headers,
            };
        }
        catch (error) {
            lastError = error;
            if (i < retries) {
                await new Promise(resolve => setTimeout(resolve, retryDelay));
            }
        }
    }
    throw lastError;
}
async function requestStream(url, options) {
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
