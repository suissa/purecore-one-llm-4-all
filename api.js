#!/usr/bin/env node

import fastify from 'fastify';
import { sendPrompt } from './src/index.ts';
import { readFile } from 'fs/promises';
import { join } from 'path';

const server = fastify({ logger: true });

import { getTokensData, trackTokens, isModelExhausted, markModelExhausted, getTokensUsedToday } from './token_manager.js';

let freeModels = JSON.parse(await readFile(join(process.cwd(), 'free_models.json'), 'utf8'));

server.get('/api/v1/tokens/', async (request, reply) => {
  try {
    const data = getTokensData();
    reply.send(data);
  } catch (error) {
    server.log.error(error);
    reply.code(500).send({ error: 'Failed to retrieve token data' });
  }
});

server.post('/', async (request, reply) => {
  const { message } = request.body || {};
  
  if (!message) {
    return reply.code(400).send({ error: 'Message is required' });
  }

  const start = performance.now();
  const estimatedPromptTokens = Math.ceil(message.length / 4);
  const apiKey = process.env.OPENROUTER_KEY;
  
  // Refresh models list to get latest tokens_limit
  freeModels = JSON.parse(await readFile(join(process.cwd(), 'free_models.json'), 'utf8'));

  for (const modelDef of freeModels) {
    const model = modelDef.id;
    
    if (isModelExhausted(model, estimatedPromptTokens)) {
      server.log.info(`Model ${model} is exhausted for today. Skipping.`);
      continue;
    }

    try {
      const prompt = sendPrompt(message, {
        model,
        provider: 'openrouter',
        apiKey: apiKey
      });

      const result = await prompt.getFullResult();
      const replyContent = result.text;
      const end = performance.now();
      const responseTime = (end - start).toFixed(2);

      const usage = result.usage || {};
      const promptTokens = usage.prompt_tokens || estimatedPromptTokens;
      const completionTokens = usage.completion_tokens || Math.ceil(replyContent.length / 4);

      trackTokens(model, apiKey || 'unknown', promptTokens, completionTokens);

      reply.send({
        reply: replyContent,
        responseTime: `${responseTime}ms`,
        model
      });
      return; // Request handled successfully
    } catch (error) {
      console.error(`Error with model ${model}:`, error.message);
      // Check if it's a rate limit or token exhaustion error (429 or similar)
      if (error.status === 429 || error.status === 402 || (error.message && error.message.toLowerCase().includes('limit'))) {
        const usedToday = getTokensUsedToday(model);
        server.log.warn(`Model ${model} hit limits. Marking as exhausted with tokens_limit: ${usedToday}`);
        markModelExhausted(model, usedToday);
      }
      // Continue to the next model in the loop
    }
  }

  reply.code(500).send({ error: 'All models failed or are exhausted' });
});

server.listen({ port: 3000 }, (err, address) => {
  if (err) {
    server.log.error(err);
    process.exit(1);
  }
  server.log.info(`API listening on ${address}`);
});