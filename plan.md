1. **Add `getFullResult` to `PromptFluent`:**
   - Update `src/fluent/send-prompt.ts` to include `getFullResult` which returns the full result (including usage statistics).
   - Update `src/types/llm.types.ts` if needed (it already has `UnifiedLLMTextResult` which includes `usage`).
2. **Create `token-manager.js`:**
   - Implement functions to track tokens per day, model, and API key.
   - Store tracking data in a JSON file (e.g., `tokens_data.json`).
   - Implement functions to read/update `free_models.json` for managing model exhaustion and `tokens_limit`.
3. **Update `api.js`:**
   - Implement the route `GET /api/v1/tokens/` to return the token data.
   - Update the POST `/` route to loop through `freeModels`.
   - Estimate prompt tokens (e.g., `message.length / 4`).
   - Check if `tokens_used_today + estimated_prompt_tokens > model.tokens_limit` (if limit is set). If it exceeds, skip to the next model.
   - If a request fails with a rate limit/token error (usually 429), catch it, update `tokens_limit` in `free_models.json` for that model based on `tokens_used_today`, and retry with the next model.
   - After a successful request, update the token usage for the model, API key, and day.
4. **Pre-commit and Test:**
   - Call `pre_commit_instructions` to ensure testing and review.
   - Run tests if available and verify the routes work.
