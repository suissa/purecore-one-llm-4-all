
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

async function logResult(message: string, onlyTerminal = false) {
  const timestamp = new Date().toISOString();
  const formattedMessage = `[${timestamp}] ${message}\n`;

  // Imprime no terminal sempre
  process.stdout.write(message + "\n");

  // Salva no log apenas se não for "onlyTerminal" (como countdowns)
  if (!onlyTerminal) {
    fs.appendFileSync("rotation_results.log", formattedMessage);
  }
}

async function countdown(seconds: number) {
  for (let i = seconds; i > 0; i--) {
    process.stdout.write(`\r⏳ Aguardando próxima requisição: ${i}s... `);
    await delay(1000);
  }
  process.stdout.write("\r\x1b[K"); // Limpa a linha
}

export async function runRealRotation(prompt: string) {
  const freeModelsPath = path.join(process.cwd(), "free_models.json");

  if (!fs.existsSync(freeModelsPath)) {
    console.error(
      "Error: free_models.json not found. Run fetch_free_models.ts first.",
    );
    throw new Error("free_models.json not found");
  }

  const freeModels = JSON.parse(fs.readFileSync(freeModelsPath, "utf-8"));

  // Filter out the generic router
  const targetModels = freeModels.filter(
    (m: any) => m.id !== "openrouter/free",
  );

  await logResult(
    `🚀 Iniciando rotação real para ${targetModels.length} modelos.`,
  );
  await logResult(`📝 Prompt: "${prompt}"`);
  await logResult(`⏱️ Intervalo: 60 segundos por modelo.`);
  await logResult(`---------------------------------------------------`);

  let current = 1;
  for (const model of targetModels) {
    await logResult(
      `[${current}/${targetModels.length}] 🤖 Tentando: ${model.name}`,
    );
    await logResult(`🆔 ID: ${model.id}`);

    try {
      const startTime = Date.now();
      const chain = sendPrompt(prompt, {
        provider: "openrouter",
        model: model.id,
      });

      const response = await chain.getText();
      const duration = ((Date.now() - startTime) / 1000).toFixed(2);

      await logResult(`✅ Sucesso (${duration}s)`);
      await logResult(
        `📄 Resposta: ${response.substring(0, 200).replace(/\n/g, " ")}...`,
      );
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      await logResult(`❌ Erro: ${errorMessage}`);
    }

    await logResult(`---------------------------------------------------`);

    if (current < targetModels.length) {
      await countdown(60);
    }
    current++;
  }

  await logResult("✨ Rotação concluída com sucesso.");
}

// Execução direta via Bun
if (import.meta.main) {
  const userPrompt = process.argv[2] || 'Olá, responda apenas com "OK" se você estiver funcionando.';
  runRealRotation(userPrompt).catch(async (err) => {
    await logResult(`💥 Erro fatal no script de rotação: ${err.message}`);
  });
}
