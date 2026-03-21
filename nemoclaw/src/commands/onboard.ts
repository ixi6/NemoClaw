// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync, execSync } from "node:child_process";
import type { PluginLogger, NemoClawConfig } from "../index.js";
import {
  describeOnboardEndpoint,
  describeOnboardProvider,
  loadOnboardConfig,
  saveOnboardConfig,
  type EndpointType,
  type NemoClawOnboardConfig,
} from "../onboard/config.js";
import { promptInput, promptConfirm, promptSelect } from "../onboard/prompt.js";
import { validateApiKey, maskApiKey } from "../onboard/validate.js";

export interface OnboardOptions {
  apiKey?: string;
  provider?: string;
  endpoint?: string;
  ncpPartner?: string;
  endpointUrl?: string;
  model?: string;
  logger: PluginLogger;
  pluginConfig: NemoClawConfig;
}

const ENDPOINT_TYPES: EndpointType[] = ["build", "openai", "anthropic", "gemini", "ncp", "nim-local", "vllm", "ollama", "custom"];
const SUPPORTED_ENDPOINT_TYPES: EndpointType[] = ["build", "openai", "anthropic", "gemini", "ncp", "ollama", "custom"];

function isExperimentalEnabled(): boolean {
  return process.env.NEMOCLAW_EXPERIMENTAL === "1";
}

const BUILD_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1";
const OPENAI_ENDPOINT_URL = "https://api.openai.com/v1";
const ANTHROPIC_ENDPOINT_URL = "https://api.anthropic.com";
const GEMINI_ENDPOINT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/";
const HOST_GATEWAY_URL = "http://host.openshell.internal";

const DEFAULT_MODELS = [
  { id: "nvidia/nemotron-3-super-120b-a12b", label: "Nemotron 3 Super 120B" },
  { id: "moonshotai/kimi-k2.5", label: "Kimi K2.5" },
  { id: "z-ai/glm5", label: "GLM-5" },
  { id: "minimaxai/minimax-m2.5", label: "MiniMax M2.5" },
  { id: "qwen/qwen3.5-397b-a17b", label: "Qwen3.5 397B A17B" },
  { id: "openai/gpt-oss-120b", label: "GPT-OSS 120B" },
];
const DEFAULT_OLLAMA_MODEL = "nemotron-3-nano:30b";
const OPENAI_DEFAULT_MODEL_CANDIDATES = ["gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1"];
const ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5";
const GEMINI_DEFAULT_MODEL_CANDIDATES = ["gemini-3-flash-preview", "gemini-3-pro", "gemini-2.5-pro", "gemini-2.5-flash"];
const CUSTOM_COMPATIBLE_PROVIDER_NAME = "compatible-endpoint";

function resolveProfile(endpointType: EndpointType): string {
  switch (endpointType) {
    case "build":
      return "default";
    case "openai":
      return "openai";
    case "anthropic":
      return "anthropic";
    case "gemini":
      return "gemini";
    case "ncp":
    case "custom":
      return "ncp";
    case "nim-local":
      return "nim-local";
    case "vllm":
      return "vllm";
    case "ollama":
      return "ollama";
  }
}

function resolveProviderName(endpointType: EndpointType): string {
  switch (endpointType) {
    case "build":
      return "nvidia-prod";
    case "openai":
      return "openai-api";
    case "anthropic":
      return "anthropic-prod";
    case "gemini":
      return "gemini-api";
    case "ncp":
      return "nvidia-ncp";
    case "custom":
      return CUSTOM_COMPATIBLE_PROVIDER_NAME;
    case "nim-local":
      return "nim-local";
    case "vllm":
      return "vllm-local";
    case "ollama":
      return "ollama-local";
  }
}

function resolveCredentialEnv(endpointType: EndpointType): string {
  switch (endpointType) {
    case "openai":
      return "OPENAI_API_KEY";
    case "anthropic":
      return "ANTHROPIC_API_KEY";
    case "gemini":
      return "GEMINI_API_KEY";
    case "build":
    case "ncp":
    case "custom":
      return "NVIDIA_API_KEY";
    case "nim-local":
      return "NIM_API_KEY";
    case "vllm":
    case "ollama":
      return "OPENAI_API_KEY";
  }
}

function isNonInteractive(opts: OnboardOptions): boolean {
  const provider = opts.provider ?? opts.endpoint;
  if (!provider || !opts.model) return false;
  const ep = provider as EndpointType;
  if (endpointRequiresApiKey(ep) && !opts.apiKey) return false;
  if ((ep === "anthropic" || ep === "gemini" || ep === "ncp" || ep === "nim-local" || ep === "custom") && !opts.endpointUrl) {
    return false;
  }
  if (ep === "ncp" && !opts.ncpPartner) return false;
  return true;
}

function endpointRequiresApiKey(endpointType: EndpointType): boolean {
  return (
    endpointType === "openai" ||
    endpointType === "anthropic" ||
    endpointType === "gemini" ||
    endpointType === "build" ||
    endpointType === "ncp" ||
    endpointType === "nim-local" ||
    endpointType === "custom"
  );
}

function defaultCredentialForEndpoint(endpointType: EndpointType): string {
  switch (endpointType) {
    case "vllm":
      return "dummy";
    case "ollama":
      return "ollama";
    default:
      return "";
  }
}

function defaultEndpointUrl(endpointType: EndpointType): string | null {
  switch (endpointType) {
    case "build":
      return BUILD_ENDPOINT_URL;
    case "openai":
      return OPENAI_ENDPOINT_URL;
    case "anthropic":
      return ANTHROPIC_ENDPOINT_URL;
    case "gemini":
      return GEMINI_ENDPOINT_URL;
    case "vllm":
      return `${HOST_GATEWAY_URL}:8000/v1`;
    case "ollama":
      return `${HOST_GATEWAY_URL}:11434/v1`;
    default:
      return null;
  }
}

function apiKeyLabel(endpointType: EndpointType, credentialEnv: string): string {
  if (endpointType === "build" || endpointType === "ncp") {
    return `NVIDIA API key (${credentialEnv})`;
  }
  if (endpointType === "openai") {
    return `OpenAI API key (${credentialEnv})`;
  }
  if (endpointType === "anthropic") {
    return `Anthropic API key (${credentialEnv})`;
  }
  if (endpointType === "gemini") {
    return `Gemini API key (${credentialEnv})`;
  }
  return `API key (${credentialEnv})`;
}

function getApiKeyHelp(endpointType: EndpointType): string | null {
  switch (endpointType) {
    case "build":
    case "ncp":
      return "Get an API key from: https://build.nvidia.com/settings/api-keys";
    case "openai":
      return "Get an API key from: https://platform.openai.com/api-keys";
    case "anthropic":
      return "Get an API key from: https://console.anthropic.com/settings/keys";
    case "gemini":
      return "Get an API key from: https://aistudio.google.com/app/apikey";
    default:
      return null;
  }
}

function providerTypeForEndpoint(endpointType: EndpointType): string {
  switch (endpointType) {
    case "build":
      return "nvidia";
    case "anthropic":
      return "anthropic";
    default:
      return "openai";
  }
}

function buildProviderCommandArgs(
  action: "create" | "update",
  endpointType: EndpointType,
  providerName: string,
  credentialEnv: string,
  endpointUrl: string,
): string[] {
  if (endpointType === "build") {
    return action === "create"
      ? [
          "provider",
          "create",
          "--name",
          providerName,
          "--type",
          providerTypeForEndpoint(endpointType),
          "--credential",
          credentialEnv,
        ]
      : [
          "provider",
          "update",
          providerName,
          "--type",
          providerTypeForEndpoint(endpointType),
          "--credential",
          credentialEnv,
        ];
  }

  if (endpointType === "anthropic") {
    return action === "create"
      ? [
          "provider",
          "create",
          "--name",
          providerName,
          "--type",
          providerTypeForEndpoint(endpointType),
          "--credential",
          credentialEnv,
        ]
      : [
          "provider",
          "update",
          providerName,
          "--type",
          providerTypeForEndpoint(endpointType),
          "--credential",
          credentialEnv,
        ];
  }

  return action === "create"
    ? [
        "provider",
        "create",
        "--name",
        providerName,
        "--type",
        providerTypeForEndpoint(endpointType),
        "--credential",
        credentialEnv,
        "--config",
        `OPENAI_BASE_URL=${endpointUrl}`,
      ]
    : [
        "provider",
        "update",
        providerName,
        "--credential",
        credentialEnv,
        "--config",
        `OPENAI_BASE_URL=${endpointUrl}`,
      ];
}

function pickRecommendedModel(models: string[], candidates: string[]): string | null {
  for (const candidate of candidates) {
    if (models.includes(candidate)) {
      return candidate;
    }
  }
  return models[0] ?? null;
}

function shouldSkipInferenceVerification(endpointType: EndpointType): boolean {
  return endpointType === "openai" || endpointType === "gemini";
}

function detectOllama(): { installed: boolean; running: boolean } {
  const installed = testCommand("command -v ollama >/dev/null 2>&1");
  const running = testCommand("curl -sf http://localhost:11434/api/tags >/dev/null 2>&1");
  return { installed, running };
}

function parseOllamaList(output: string): string[] {
  return output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !/^NAME\s+/i.test(line))
    .map((line) => line.split(/\s{2,}/)[0])
    .filter(Boolean);
}

function getOllamaModelOptions(): string[] {
  try {
    const output = execSync("ollama list", { encoding: "utf-8", shell: "/bin/bash" });
    const parsed = parseOllamaList(output);
    if (parsed.length > 0) {
      return parsed;
    }
  } catch {}
  return [DEFAULT_OLLAMA_MODEL];
}

function getDefaultOllamaModel(): string {
  const models = getOllamaModelOptions();
  return models.includes(DEFAULT_OLLAMA_MODEL) ? DEFAULT_OLLAMA_MODEL : models[0];
}

function testCommand(command: string): boolean {
  try {
    execSync(command, { encoding: "utf-8", stdio: "ignore", shell: "/bin/bash" });
    return true;
  } catch {
    return false;
  }
}

function showConfig(config: NemoClawOnboardConfig, logger: PluginLogger): void {
  logger.info(`  Endpoint:    ${describeOnboardEndpoint(config)}`);
  logger.info(`  Provider:    ${describeOnboardProvider(config)}`);
  if (config.ncpPartner) {
    logger.info(`  NVIDIA Cloud Partner: ${config.ncpPartner}`);
  }
  logger.info(`  Model:       ${config.model}`);
  logger.info(`  Credential:  $${config.credentialEnv}`);
  logger.info(`  Profile:     ${config.profile}`);
  logger.info(`  Onboarded:   ${config.onboardedAt}`);
}

async function promptEndpoint(
  ollama: { installed: boolean; running: boolean },
): Promise<EndpointType> {
  const options = [
    {
      label: "NVIDIA hosted",
      value: "build",
      hint: "recommended — zero infra, free credits",
    },
    {
      label: "OpenAI",
      value: "openai",
      hint: "uses OpenAI defaults; base URL override available",
    },
    {
      label: "Anthropic",
      value: "anthropic",
      hint: "native Claude API via OpenShell",
    },
    {
      label: "Google Gemini",
      value: "gemini",
      hint: "official OpenAI-compatible Gemini API",
    },
  ];

  options.push({
    label: "Local Ollama",
    value: "ollama",
    hint: ollama.running
      ? "detected on localhost:11434"
      : ollama.installed
        ? "installed locally"
        : "localhost:11434",
  });

  if (isExperimentalEnabled()) {
    options.push(
      {
        label: "Local NVIDIA NIM [experimental]",
        value: "nim-local",
        hint: "experimental — your own NIM container deployment",
      },
      {
        label: "Local vLLM [experimental]",
        value: "vllm",
        hint: "experimental — local development",
      },
    );
  }

  options.push(
    {
      label: "NVIDIA Cloud Partner",
      value: "ncp",
      hint: "dedicated capacity, SLA-backed",
    },
    {
      label: "Other compatible endpoint",
      value: "custom",
      hint: "advanced — OpenAI-compatible providers like OpenRouter or hosted vLLM",
    },
  );

  return (await promptSelect("Where should NemoClaw get its model?", options)) as EndpointType;
}

function execOpenShell(args: string[], env: NodeJS.ProcessEnv = process.env): string {
  return execFileSync("openshell", args, {
    encoding: "utf-8",
    stdio: ["pipe", "pipe", "pipe"],
    env,
  });
}

export async function cliOnboard(opts: OnboardOptions): Promise<void> {
  const { logger } = opts;
  const nonInteractive = isNonInteractive(opts);

  logger.info("NemoClaw Onboarding");
  logger.info("-------------------");

  // Step 0: Check existing config
  const existing = loadOnboardConfig();
  if (existing) {
    logger.info("");
    logger.info("Existing configuration found:");
    showConfig(existing, logger);
    logger.info("");

    if (!nonInteractive) {
      const reconfigure = await promptConfirm("Reconfigure?", false);
      if (!reconfigure) {
        logger.info("Keeping existing configuration.");
        return;
      }
    }
  }

  // Step 1: Endpoint Selection
  let endpointType: EndpointType;
  const selectedProvider = opts.provider ?? opts.endpoint;
  if (selectedProvider) {
    if (!ENDPOINT_TYPES.includes(selectedProvider as EndpointType)) {
      logger.error(
        `Invalid endpoint type: ${selectedProvider}. Must be one of: ${ENDPOINT_TYPES.join(", ")}`,
      );
      return;
    }
    const ep = selectedProvider as EndpointType;
    if (!SUPPORTED_ENDPOINT_TYPES.includes(ep)) {
      logger.warn(
        `Note: '${ep}' is experimental and may not work reliably.`,
      );
    }
    endpointType = ep;
  } else {
    const ollama = detectOllama();
    if (ollama.running) {
      logger.info("Detected local inference option: Ollama.");
      logger.info("Select it explicitly if you want to use it.");
    }
    endpointType = await promptEndpoint(ollama);
  }

  // Step 2: Endpoint URL resolution
  let endpointUrl: string;
  let ncpPartner: string | null = null;

  switch (endpointType) {
    case "build":
    case "openai":
    case "anthropic":
    case "gemini":
      endpointUrl = opts.endpointUrl ?? defaultEndpointUrl(endpointType) ?? "";
      break;
    case "ncp":
      ncpPartner = opts.ncpPartner ?? (await promptInput("NVIDIA Cloud Partner name"));
      endpointUrl =
        opts.endpointUrl ??
        (await promptInput("NVIDIA Cloud Partner endpoint URL (e.g., https://partner.api.nvidia.com/v1)"));
      break;
    case "nim-local":
      endpointUrl =
        opts.endpointUrl ??
        (await promptInput("NIM endpoint URL", "http://nim-service.local:8000/v1"));
      break;
    case "vllm":
      endpointUrl = opts.endpointUrl ?? defaultEndpointUrl(endpointType) ?? "";
      break;
    case "ollama":
      endpointUrl = opts.endpointUrl ?? defaultEndpointUrl(endpointType) ?? "";
      break;
    case "custom":
      endpointUrl =
        opts.endpointUrl ??
        (await promptInput("OpenAI-compatible base URL (e.g., https://openrouter.ai/api/v1)"));
      break;
  }

  if (!endpointUrl) {
    logger.error("No endpoint URL provided. Aborting.");
    return;
  }

  const credentialEnv = resolveCredentialEnv(endpointType);
  const requiresApiKey = endpointRequiresApiKey(endpointType);

  // Step 3: Credential
  let apiKey = defaultCredentialForEndpoint(endpointType);
  if (requiresApiKey) {
    const keyLabel = apiKeyLabel(endpointType, credentialEnv);
    if (opts.apiKey) {
      apiKey = opts.apiKey;
    } else {
      const envKey = process.env[credentialEnv];
      if (envKey) {
        logger.info(`Detected ${credentialEnv} in environment (${maskApiKey(envKey)})`);
        const useEnv = nonInteractive ? true : await promptConfirm("Use this key?");
        apiKey = useEnv ? envKey : await promptInput(`Enter your ${keyLabel}`, { secret: true });
      } else {
        const help = getApiKeyHelp(endpointType);
        if (help) {
          logger.info(help);
        }
        apiKey = await promptInput(`Enter your ${keyLabel}`, { secret: true });
      }
    }
  } else {
    logger.info(
      `No API key required for ${endpointType}. Using local credential value '${apiKey}'.`,
    );
  }

  if (!apiKey) {
    logger.error("No API key provided. Aborting.");
    return;
  }

  // Step 4: Validate API Key
  // For local endpoints (vllm, ollama, nim-local), validation is best-effort since the
  // service may not be running yet during onboarding.
  const isLocalEndpoint =
    endpointType === "vllm" || endpointType === "ollama" || endpointType === "nim-local";
  logger.info("");
  logger.info(`Validating ${requiresApiKey ? "credential" : "endpoint"} against ${endpointUrl}...`);
  const validation = await validateApiKey(apiKey, endpointUrl, endpointType);

  if (!validation.valid) {
    if (isLocalEndpoint) {
      logger.warn(
        `Could not reach ${endpointUrl} (${validation.error ?? "unknown error"}). Continuing anyway — the service may not be running yet.`,
      );
    } else {
      logger.error(`API key validation failed: ${validation.error ?? "unknown error"}`);
      const help = getApiKeyHelp(endpointType);
      if (help) {
        logger.info(help);
      }
      return;
    }
  } else {
    logger.info(
      `${requiresApiKey ? "Credential" : "Endpoint"} valid. ${String(validation.models.length)} model(s) available.`,
    );
  }

  // Step 5: Model Selection
  let model: string;
  if (opts.model) {
    model = opts.model;
  } else {
    const discoveredModelOptions =
      endpointType === "ollama"
        ? getOllamaModelOptions().map((id) => ({ label: id, value: id }))
        : validation.models.map((id) => ({ label: id, value: id }));
    const curatedCloudOptions =
      endpointType === "build" || endpointType === "ncp"
        ? DEFAULT_MODELS.filter((option) => validation.models.includes(option.id)).map((option) => ({
            label: `${option.label} (${option.id})`,
            value: option.id,
          }))
        : [];
    const curatedOpenAiOptions =
      endpointType === "openai"
        ? OPENAI_DEFAULT_MODEL_CANDIDATES.filter((id) => validation.models.includes(id)).map((id) => ({
            label: id,
            value: id,
          }))
        : [];
    const curatedGeminiOptions =
      endpointType === "gemini"
        ? GEMINI_DEFAULT_MODEL_CANDIDATES.filter((id) => validation.models.includes(id)).map((id) => ({
            label: id,
            value: id,
          }))
        : [];
    const curatedAnthropicOptions =
      endpointType === "anthropic"
        ? validation.models.map((id) => ({
            label: id === ANTHROPIC_DEFAULT_MODEL ? `${id} (recommended)` : id,
            value: id,
          }))
        : [];
    const defaultIndex =
      endpointType === "ollama"
        ? Math.max(
            0,
            discoveredModelOptions.findIndex((option) => option.value === getDefaultOllamaModel()),
          )
        : endpointType === "openai"
          ? Math.max(
              0,
              curatedOpenAiOptions.findIndex(
                (option) =>
                  option.value === pickRecommendedModel(validation.models, OPENAI_DEFAULT_MODEL_CANDIDATES),
              ),
            )
        : endpointType === "gemini"
          ? Math.max(
              0,
              curatedGeminiOptions.findIndex(
                (option) =>
                  option.value === pickRecommendedModel(validation.models, GEMINI_DEFAULT_MODEL_CANDIDATES),
              ),
            )
        : endpointType === "anthropic"
          ? Math.max(
              0,
              curatedAnthropicOptions.findIndex((option) => option.value === ANTHROPIC_DEFAULT_MODEL),
            )
        : 0;
    const modelOptions =
      curatedCloudOptions.length > 0
        ? curatedCloudOptions
        : curatedOpenAiOptions.length > 0
          ? curatedOpenAiOptions
        : curatedGeminiOptions.length > 0
          ? curatedGeminiOptions
        : curatedAnthropicOptions.length > 0
          ? curatedAnthropicOptions
        : discoveredModelOptions.length > 0
          ? discoveredModelOptions
          : DEFAULT_MODELS.map((m) => ({ label: `${m.label} (${m.id})`, value: m.id }));

    model = await promptSelect("Select your primary model:", modelOptions, defaultIndex);
  }

  // Step 6: Resolve profile
  const profile = resolveProfile(endpointType);
  const providerName = resolveProviderName(endpointType);
  const summaryConfig: NemoClawOnboardConfig = {
    endpointType,
    endpointUrl,
    ncpPartner,
    model,
    profile,
    credentialEnv,
    provider: providerName,
    providerLabel: undefined,
    onboardedAt: "",
  };
  summaryConfig.providerLabel = describeOnboardProvider(summaryConfig);

  // Step 7: Confirmation
  logger.info("");
  logger.info("Configuration summary:");
  logger.info(`  Endpoint:    ${describeOnboardEndpoint(summaryConfig)}`);
  logger.info(`  Provider:    ${summaryConfig.providerLabel}`);
  if (ncpPartner) {
    logger.info(`  NVIDIA Cloud Partner: ${ncpPartner}`);
  }
  logger.info(`  Model:       ${model}`);
  logger.info(
    `  API Key:     ${requiresApiKey ? maskApiKey(apiKey) : "not required (local provider)"}`,
  );
  if (endpointType === "openai" && endpointUrl !== OPENAI_ENDPOINT_URL) {
    logger.info("  Base URL:    custom override");
  }
  logger.info(`  Credential:  $${credentialEnv}`);
  logger.info(`  Profile:     ${profile}`);
  logger.info(`  Provider:    ${providerName}`);
  logger.info("");

  if (!nonInteractive) {
    const proceed = await promptConfirm("Apply this configuration?");
    if (!proceed) {
      logger.info("Onboarding cancelled.");
      return;
    }
  }

  // Step 8: Apply
  logger.info("");
  logger.info("Applying configuration...");

  const providerEnv = { ...process.env, [credentialEnv]: apiKey };

  // 7a: Create/update provider
  try {
    execOpenShell(
      buildProviderCommandArgs("create", endpointType, providerName, credentialEnv, endpointUrl),
      providerEnv,
    );
    logger.info(`Created provider: ${providerName}`);
  } catch (err) {
    const stderr =
      err instanceof Error && "stderr" in err ? String((err as { stderr: unknown }).stderr) : "";
    if (stderr.includes("AlreadyExists") || stderr.includes("already exists")) {
      try {
        execOpenShell(
          buildProviderCommandArgs("update", endpointType, providerName, credentialEnv, endpointUrl),
          providerEnv,
        );
        logger.info(`Updated provider: ${providerName}`);
      } catch (updateErr) {
        const updateStderr =
          updateErr instanceof Error && "stderr" in updateErr
            ? String((updateErr as { stderr: unknown }).stderr)
            : "";
        logger.error(`Failed to update provider: ${updateStderr || String(updateErr)}`);
        return;
      }
    } else {
      logger.error(`Failed to create provider: ${stderr || String(err)}`);
      return;
    }
  }

  // 7b: Set inference route
  try {
    const inferenceArgs = ["inference", "set"];
    if (shouldSkipInferenceVerification(endpointType)) {
      inferenceArgs.push("--no-verify");
      logger.warn(
        "Skipping OpenShell inference verification for OpenAI models for now due to current GPT-5-family verifier incompatibility.",
      );
    }
    inferenceArgs.push("--provider", providerName, "--model", model);
    execOpenShell(inferenceArgs, providerEnv);
    logger.info(`Inference route set: ${providerName} -> ${model}`);
  } catch (err) {
    const stderr =
      err instanceof Error && "stderr" in err ? String((err as { stderr: unknown }).stderr) : "";
    logger.error(`Failed to set inference route: ${stderr || String(err)}`);
    return;
  }

  // 7c: Save config
  saveOnboardConfig({
    endpointType,
    endpointUrl,
    ncpPartner,
    model,
    profile,
    credentialEnv,
    provider: providerName,
    providerLabel: summaryConfig.providerLabel,
    onboardedAt: new Date().toISOString(),
  });

  // Step 9: Success
  logger.info("");
  logger.info("Onboarding complete!");
  logger.info("");
  logger.info(`  Endpoint:   ${describeOnboardEndpoint(summaryConfig)}`);
  logger.info(`  Provider:   ${summaryConfig.providerLabel}`);
  logger.info(`  Model:      ${model}`);
  logger.info(`  Credential: $${credentialEnv}`);
  logger.info("");
  logger.info("Next steps:");
  logger.info("  openclaw nemoclaw launch     # Bootstrap sandbox");
  logger.info("  openclaw nemoclaw status     # Check configuration");
}
