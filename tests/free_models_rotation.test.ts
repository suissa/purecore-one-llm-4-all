
import { describe, it, expect } from 'vitest';
import { runRotation } from '../scripts/run_free_rotation';
import { sendPrompt } from '../src/index';
import fs from 'fs';
import path from 'path';

// Load env vars if strictly needed, though Bun usually handles it.
// We assume OPENROUTER_API_KEY is present in .env

describe('Free Models Rotation Integration', () => {
  it('should successfully query a subset of free models using real API', async () => {
    // 1. Check if API Key exists
    if (!process.env.OPENROUTER_API_KEY) {
      console.warn('⚠️ OPENROUTER_API_KEY not found in env. Skipping real integration test.');
      return; 
    }

    // 2. Load models list
    const freeModelsPath = path.join(process.cwd(), 'free_models.json');
    if (!fs.existsSync(freeModelsPath)) {
      throw new Error('free_models.json not found. Please run "bun run fetch:free-models" first.');
    }
    const allFreeModels = JSON.parse(fs.readFileSync(freeModelsPath, 'utf-8'));

    // 3. Select specific reliable models for testing
    const testModels = allFreeModels
      .filter((m: any) => 
        (m.id.includes('mistral') || m.id.includes('google/gemma')) && 
        !m.id.includes('free') // Some free models behave oddly, but we want the ones from the free list that are known brands
      )
      .slice(0, 1); // Just 1 model is enough to prove integration
    
    // Fallback if filter too aggressive
    if (testModels.length === 0) {
       testModels.push(allFreeModels.find((m:any) => m.id !== 'openrouter/free'));
    } 

    console.log(`🧪 Testing with models: ${testModels.map((m: any) => m.name).join(', ')}`);

    // 4. Run rotation with real sendPrompt and a mocked delay to speed up passing (since we only have 2 requests, rate limit might be fine with less delay or we accept the 6s)
    // We will use a smaller delay for the test to be faster, assuming quota allows burst of 2.
    // OpenRouter free tier is mostly 5s between requests.
    const customDelay = async (ms: number) => new Promise(r => setTimeout(r, 6000)); 

    try {
      await runRotation(
        'Answer with a single word: "Pong".', 
        testModels, 
        sendPrompt, // Use REAL sendPrompt
        customDelay
      );
    } catch (error) {
      console.error('TEST FALHOU COM ERRO:', error);
      throw error;
    }

  }, 120000); // 2 minutes timeout
});
