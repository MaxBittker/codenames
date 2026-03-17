import type { Model } from "@mariozechner/pi-ai";

export function createQwenModel(baseUrl?: string, modelId?: string): Model<"openai-completions"> {
  const id = modelId ?? process.env.LLM_MODEL ?? "qwen/qwen3.5-35b-a3b";
  return {
    id,
    name: id,
    api: "openai-completions",
    provider: "openrouter",
    baseUrl: baseUrl ?? process.env.LLM_BASE_URL ?? "https://openrouter.ai/api/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 0.1625, output: 1.3, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 262144,
    maxTokens: 4096,
    compat: {
      supportsDeveloperRole: false,
      supportsStore: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    },
  };
}

export function createModelFromEnv(): Model<"openai-completions"> {
  return createQwenModel(process.env.LLM_BASE_URL, process.env.LLM_MODEL);
}
