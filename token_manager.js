import fs from 'fs';
import path from 'path';

const TOKENS_FILE = path.join(process.cwd(), 'tokens_data.json');

export function getTokensData() {
  if (fs.existsSync(TOKENS_FILE)) {
    try {
      return JSON.parse(fs.readFileSync(TOKENS_FILE, 'utf8'));
    } catch (e) {
      console.error("Error reading tokens_data.json:", e);
      return {};
    }
  }
  return {};
}

function saveTokensData(data) {
  fs.writeFileSync(TOKENS_FILE, JSON.stringify(data, null, 2));
}

export function trackTokens(model, apiKey, promptTokens, completionTokens) {
  const dateStr = new Date().toISOString().split('T')[0];
  const data = getTokensData();

  const maskedApiKey = apiKey && apiKey.length > 8 ? apiKey.slice(0, 8) + '***' : 'unknown';

  if (!data[dateStr]) data[dateStr] = {};
  if (!data[dateStr][model]) data[dateStr][model] = {};
  if (!data[dateStr][model][maskedApiKey]) {
    data[dateStr][model][maskedApiKey] = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };
  }

  data[dateStr][model][maskedApiKey].promptTokens += (promptTokens || 0);
  data[dateStr][model][maskedApiKey].completionTokens += (completionTokens || 0);
  data[dateStr][model][maskedApiKey].totalTokens += ((promptTokens || 0) + (completionTokens || 0));

  saveTokensData(data);
}

export function getTokensUsedToday(model) {
  const dateStr = new Date().toISOString().split('T')[0];
  const data = getTokensData();

  if (!data[dateStr] || !data[dateStr][model]) {
    return 0;
  }

  let total = 0;
  for (const apiKey in data[dateStr][model]) {
    total += data[dateStr][model][apiKey].totalTokens;
  }
  return total;
}

const FREE_MODELS_FILE = path.join(process.cwd(), 'free_models.json');

export function markModelExhausted(modelId, tokensUsed) {
  if (fs.existsSync(FREE_MODELS_FILE)) {
    try {
      const freeModels = JSON.parse(fs.readFileSync(FREE_MODELS_FILE, 'utf8'));
      const modelIndex = freeModels.findIndex(m => m.id === modelId);
      if (modelIndex !== -1) {
        freeModels[modelIndex].tokens_limit = tokensUsed;
        fs.writeFileSync(FREE_MODELS_FILE, JSON.stringify(freeModels, null, 2));
      }
    } catch (e) {
      console.error("Error updating free_models.json for exhaustion:", e);
    }
  }
}

export function isModelExhausted(modelId, estimatedTokens) {
  if (fs.existsSync(FREE_MODELS_FILE)) {
    try {
      const freeModels = JSON.parse(fs.readFileSync(FREE_MODELS_FILE, 'utf8'));
      const model = freeModels.find(m => m.id === modelId);
      if (model && model.tokens_limit !== undefined) {
        const tokensUsed = getTokensUsedToday(modelId);
        if (tokensUsed + estimatedTokens > model.tokens_limit) {
          return true;
        }
      }
    } catch (e) {
      console.error("Error reading free_models.json for exhaustion check:", e);
    }
  }
  return false;
}
