
import { sendPrompt } from '../src/index';
import fs from 'fs';
import path from 'path';

/**
 * Script para realizar testes reais rotacionando modelos a cada 1 minuto.
 * Objetivo: Validar a estabilidade e integração com diversos provedores via OpenRouter.
 */

export async function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function logResult(message: string) {
  const timestamp = new Date().toISOString();
  const formattedMessage = `[${timestamp}] ${message}\n`;
  console.log(formattedMessage.trim());
  fs.appendFileSync('rotation_results.log', formattedMessage);
}

export async function runRealRotation(prompt: string) {
  const freeModelsPath = path.join(process.cwd(), 'free_models.json');
  
  if (!fs.existsSync(freeModelsPath)) {
    console.error('Error: free_models.json not found. Run fetch_free_models.ts first.');
    throw new Error('free_models.json not found');
  }

  const freeModels = JSON.parse(fs.readFileSync(freeModelsPath, 'utf-8'));
  
  // Filter out the generic router
  const targetModels = freeModels.filter((m: any) => m.id !== 'openrouter/free');

  await logResult(`🚀 Iniciando rotação real para ${targetModels.length} modelos.`);
  await logResult(`📝 Prompt: "${prompt}"`);
  await logResult(`⏱️ Intervalo: 60 segundos por modelo.\n`);

  for (const model of targetModels) {
    await logResult(`Try model: ${model.name} (${model.id})...`);
    
    try {
      const startTime = Date.now();
      const chain = sendPrompt(prompt, {
        provider: 'openrouter',
        model: model.id,
      });
      
      const response = await chain.getText();
      const duration = ((Date.now() - startTime) / 1000).toFixed(2);

      await logResult(`✅ Sucesso (${duration}s): ${model.name}`);
      await logResult(`📄 Resposta: ${response.substring(0, 150).replace(/\n/g, ' ')}...`);
    } catch (error) {
       const errorMessage = error instanceof Error ? error.message : String(error);
       await logResult(`❌ Erro em ${model.name}: ${errorMessage}`);
    }

    await logResult(`---------------------------------------------------`);
    await logResult(`Waiting 60s for next model...`);
    await delay(60000);
  }

  await logResult('✅ Rotação concluída com sucesso.');
}

// Execução direta via Bun
if (import.meta.main) {
  const userPrompt = process.argv[2] || 'Olá, responda apenas com "OK" se você estiver funcionando.';
  runRealRotation(userPrompt).catch(async (err) => {
    await logResult(`💥 Erro fatal no script de rotação: ${err.message}`);
  });
}
