/**
 * Utilitário para transformar ReadableStream de diferentes provedores em um stream unificado de texto.
 */

export async function* parseSSE(stream: ReadableStream): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        // Processa o que restou no buffer
        if (buffer) {
          yield* processLines(buffer);
        }
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n');
      buffer = parts.pop() || '';

      for (const line of parts) {
        yield* processLine(line);
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function* processLines(chunk: string): Generator<string> {
  const lines = chunk.split('\n');
  for (const line of lines) {
    yield* processLine(line);
  }
}

function* processLine(line: string): Generator<string> {
  const trimmed = line.trim();
  if (!trimmed || !trimmed.startsWith('data:')) return;
  
  const data = trimmed.slice(5).trim();
  if (data === '[DONE]') return;

  try {
    const json = JSON.parse(data);
    const content = json.choices?.[0]?.delta?.content || json.choices?.[0]?.text || '';
    if (content) yield content;
  } catch (e) {
    // Ignora
  }
}

export async function* parseAnthropicStream(stream: ReadableStream): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n');
      buffer = parts.pop() || '';

      for (const line of parts) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;
        
        const data = trimmed.slice(5).trim();
        try {
          const json = JSON.parse(data);
          if (json.type === 'content_block_delta' && json.delta?.text) {
            yield json.delta.text;
          }
        } catch (e) {}
      }
    }
  } finally {
    reader.releaseLock();
  }
}
