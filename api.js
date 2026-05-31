#!/usr/bin/env node

import fastify from 'fastify';
import { sendPrompt } from './src/index.js';
import { readFile } from 'fs/promises';
import { join } from 'path';

const server = fastify({ logger: true });

const freeModels = JSON.parse(await readFile(join(process.cwd(), 'free_models.json'), 'utf8'));

server.post('/', async (request, reply) => {
  const { message } = request.body as { message: string };
  
  const start = performance.now();
  
  try {
    const model = freeModels[0].id;
    const prompt = sendPrompt(message, {
      model,
      provider: 'openrouter',
      apiKey: process.env.OPENROUTER_KEY
    });
    
    const replyContent = await prompt.getText();
    const end = performance.now();
    const responseTime = (end - start).toFixed(2);
    
    reply.send({ 
      reply: replyContent,
      responseTime: `${responseTime}ms`,
      model
    });
  } catch (error) {
    console.error('Error:', error);
    reply.code(500).send({ error: 'Failed to process message' });
  }
});

server.listen({ port: 3000 }, (err, address) => {
  if (err) {
    server.log.error(err);
    process.exit(1);
  }
  server.log.info(`API listening on ${address}`);
});