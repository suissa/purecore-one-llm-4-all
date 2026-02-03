
import { sendPrompt } from '../src/index'; // Importing directly from source
import fs from 'fs';
import path from 'path';

export async function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export async function runRotation(
  prompt: string, 
  models?: any[], 
  sendPromptFn: any = sendPrompt, 
  delayFn: any = delay
) {
  let freeModels = models;
  
  if (!freeModels) {
      const freeModelsPath = path.join(process.cwd(), 'free_models.json');
      if (!fs.existsSync(freeModelsPath)) {
        console.error('Error: free_models.json not found. Run fetch_free_models.ts first.');
        throw new Error('free_models.json not found');
      }
      freeModels = JSON.parse(fs.readFileSync(freeModelsPath, 'utf-8'));
  }

  if (!freeModels) return;

  console.log(`Starting rotation for ${freeModels.length} models...`);
  console.log(`Prompt: "${prompt}"\n`);

  // Rate limit: 1 request every 5 seconds. We use 6000ms to be safe.
  const DELAY_MS = 6000; 

  for (const model of freeModels) {
    if (model.id === 'openrouter/free') continue; // Skip the router itself to test specific models

    console.log(`[${new Date().toLocaleTimeString()}] Sending to ${model.name} (${model.id})...`);
    
    try {
      // Use injected sendPromptFn
      const chain = sendPromptFn(prompt, {
        provider: 'openrouter',
        model: model.id,
      });
      
      // Handle both mocked return (which might not have getText) and real one
      const response = chain.getText ? await chain.getText() : await chain;

      console.log(`✅ Response from ${model.name}:`);
      console.log(`${response.substring(0, 100)}...`); // Show preview
      console.log('---------------------------------------------------');
    } catch (error) {
      console.error(`❌ Error with ${model.name}:`, error instanceof Error ? error.message : error);
      console.log('---------------------------------------------------');
    }

    console.log(`Waiting ${DELAY_MS/1000}s to request rate limits...`);
    await delayFn(DELAY_MS);
  }

  console.log('Rotation completed.');
}

// Execute if running directly
// @ts-ignore - Bun specific
if (import.meta.main) {
  const userPrompt = process.argv[2] || 'Explain the importance of open source in one sentence.';
  runRotation(userPrompt).catch(console.error);
}
