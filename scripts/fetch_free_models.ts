
import fs from 'fs';
import path from 'path';

async function fetchFreeModels() {
  try {
    console.log('Fetching models from OpenRouter...');
    const response = await fetch('https://openrouter.ai/api/v1/models');
    const data = await response.json();

    const freeModels = data.data.filter((model: any) => {
      const prompt = parseFloat(model.pricing.prompt);
      const completion = parseFloat(model.pricing.completion);
      return prompt === 0 && completion === 0;
    }).map((model: any) => ({
      id: model.id,
      name: model.name,
      context_length: model.context_length,
    }));

    console.log(`Found ${freeModels.length} free models.`);
    
    // Save to a JSON file
    const outputPath = path.join(process.cwd(), 'free_models.json');
    fs.writeFileSync(outputPath, JSON.stringify(freeModels, null, 2));
    console.log(`Saved list to ${outputPath}`);

    return freeModels;
  } catch (error) {
    console.error('Error fetching models:', error);
    return [];
  }
}

fetchFreeModels();
