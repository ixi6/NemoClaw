// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PluginLogger, NemoClawConfig } from "../index.js";

vi.mock("node:child_process", () => ({
  execFileSync: vi.fn(),
  execSync: vi.fn(),
}));

vi.mock("../onboard/config.js", () => ({
  describeOnboardEndpoint: vi.fn((config: { endpointType: string; endpointUrl: string }) => `${config.endpointType} (${config.endpointUrl})`),
  describeOnboardProvider: vi.fn((config: { endpointType: string }) => {
    if (config.endpointType === "openai") return "OpenAI";
    if (config.endpointType === "anthropic") return "Anthropic";
    if (config.endpointType === "gemini") return "Google Gemini";
    return "NVIDIA hosted";
  }),
  loadOnboardConfig: vi.fn(() => null),
  saveOnboardConfig: vi.fn(),
}));

vi.mock("../onboard/prompt.js", () => ({
  promptInput: vi.fn(),
  promptConfirm: vi.fn(),
  promptSelect: vi.fn(),
}));

vi.mock("../onboard/validate.js", () => ({
  validateApiKey: vi.fn(),
  maskApiKey: vi.fn((value: string) => `****${value.slice(-4)}`),
}));

const { execFileSync, execSync } = await import("node:child_process");
const { loadOnboardConfig, saveOnboardConfig } = await import("../onboard/config.js");
const { promptInput, promptConfirm, promptSelect } = await import("../onboard/prompt.js");
const { validateApiKey } = await import("../onboard/validate.js");
const { cliOnboard } = await import("./onboard.js");

const pluginConfig: NemoClawConfig = {
  blueprintVersion: "latest",
  blueprintRegistry: "ghcr.io/nvidia/nemoclaw-blueprint",
  sandboxName: "openclaw",
  inferenceProvider: "nvidia",
};

function captureLogger(): { lines: string[]; logger: PluginLogger } {
  const lines: string[] = [];
  return {
    lines,
    logger: {
      info: (message: string) => lines.push(`INFO:${message}`),
      warn: (message: string) => lines.push(`WARN:${message}`),
      error: (message: string) => lines.push(`ERROR:${message}`),
      debug: () => {},
    },
  };
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(loadOnboardConfig).mockReturnValue(null);
  vi.mocked(promptConfirm).mockResolvedValue(true);
  vi.mocked(execSync).mockImplementation(() => {
    throw new Error("not installed");
  });
  vi.mocked(execFileSync).mockReturnValue("");
  vi.mocked(validateApiKey).mockResolvedValue({
    valid: true,
    models: ["gpt-5.2", "gpt-4.1"],
    error: null,
  });
});

describe("cliOnboard", () => {
  it("uses the native NVIDIA provider type for the NVIDIA hosted flow", async () => {
    const { logger } = captureLogger();

    await cliOnboard({
      provider: "build",
      apiKey: "nvapi-test-secret",
      model: "nvidia/nemotron-3-super-120b-a12b",
      logger,
      pluginConfig,
    });

    expect(execFileSync).toHaveBeenNthCalledWith(
      1,
      "openshell",
      [
        "provider",
        "create",
        "--name",
        "nvidia-prod",
        "--type",
        "nvidia",
        "--credential",
        "NVIDIA_API_KEY",
      ],
      expect.objectContaining({
        env: expect.objectContaining({
          NVIDIA_API_KEY: "nvapi-test-secret",
        }),
      }),
    );
    expect(execFileSync).toHaveBeenNthCalledWith(
      2,
      "openshell",
      ["inference", "set", "--provider", "nvidia-prod", "--model", "nvidia/nemotron-3-super-120b-a12b"],
      expect.objectContaining({
        env: expect.objectContaining({
          NVIDIA_API_KEY: "nvapi-test-secret",
        }),
      }),
    );
    expect(saveOnboardConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        endpointType: "build",
        provider: "nvidia-prod",
        credentialEnv: "NVIDIA_API_KEY",
      }),
    );
  });

  it("applies OpenAI onboarding with a bare credential env name instead of embedding the key", async () => {
    const { logger } = captureLogger();

    await cliOnboard({
      provider: "openai",
      apiKey: "sk-test-secret",
      model: "gpt-5.2",
      logger,
      pluginConfig,
    });

    expect(execFileSync).toHaveBeenCalledTimes(2);
    expect(execFileSync).toHaveBeenNthCalledWith(
      1,
      "openshell",
      [
        "provider",
        "create",
        "--name",
        "openai-api",
        "--type",
        "openai",
        "--credential",
        "OPENAI_API_KEY",
        "--config",
        "OPENAI_BASE_URL=https://api.openai.com/v1",
      ],
      expect.objectContaining({
        encoding: "utf-8",
        env: expect.objectContaining({
          OPENAI_API_KEY: "sk-test-secret",
        }),
      }),
    );
    expect(execFileSync).toHaveBeenNthCalledWith(
      2,
      "openshell",
      ["inference", "set", "--no-verify", "--provider", "openai-api", "--model", "gpt-5.2"],
      expect.objectContaining({
        env: expect.objectContaining({
          OPENAI_API_KEY: "sk-test-secret",
        }),
      }),
    );
    expect(saveOnboardConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        endpointType: "openai",
        endpointUrl: "https://api.openai.com/v1",
        credentialEnv: "OPENAI_API_KEY",
        model: "gpt-5.2",
        provider: "openai-api",
      }),
    );
  });

  it("applies Anthropic onboarding with the native provider type and recommended model input", async () => {
    const { logger } = captureLogger();
    vi.mocked(validateApiKey).mockResolvedValueOnce({
      valid: true,
      models: ["claude-opus-4-6", "claude-sonnet-4-5"],
      error: null,
    });

    await cliOnboard({
      provider: "anthropic",
      apiKey: "sk-ant-test-secret",
      endpointUrl: "https://api.anthropic.com",
      model: "claude-sonnet-4-5",
      logger,
      pluginConfig,
    });

    expect(validateApiKey).toHaveBeenCalledWith(
      "sk-ant-test-secret",
      "https://api.anthropic.com",
      "anthropic",
    );
    expect(execFileSync).toHaveBeenCalledTimes(2);
    expect(execFileSync).toHaveBeenNthCalledWith(
      1,
      "openshell",
      [
        "provider",
        "create",
        "--name",
        "anthropic-prod",
        "--type",
        "anthropic",
        "--credential",
        "ANTHROPIC_API_KEY",
      ],
      expect.objectContaining({
        encoding: "utf-8",
        env: expect.objectContaining({
          ANTHROPIC_API_KEY: "sk-ant-test-secret",
        }),
      }),
    );
    expect(execFileSync).toHaveBeenNthCalledWith(
      2,
      "openshell",
      ["inference", "set", "--provider", "anthropic-prod", "--model", "claude-sonnet-4-5"],
      expect.objectContaining({
        env: expect.objectContaining({
          ANTHROPIC_API_KEY: "sk-ant-test-secret",
        }),
      }),
    );
    expect(saveOnboardConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        endpointType: "anthropic",
        endpointUrl: "https://api.anthropic.com",
        credentialEnv: "ANTHROPIC_API_KEY",
        model: "claude-sonnet-4-5",
        provider: "anthropic-prod",
      }),
    );
  });

  it("applies Gemini onboarding with the official OpenAI-compatible endpoint", async () => {
    const { logger } = captureLogger();
    vi.mocked(validateApiKey).mockResolvedValueOnce({
      valid: true,
      models: ["gemini-2.5-flash", "gemini-3-flash-preview"],
      error: null,
    });

    await cliOnboard({
      provider: "gemini",
      apiKey: "AIza-test-secret",
      endpointUrl: "https://generativelanguage.googleapis.com/v1beta/openai/",
      model: "gemini-3-flash-preview",
      logger,
      pluginConfig,
    });

    expect(validateApiKey).toHaveBeenCalledWith(
      "AIza-test-secret",
      "https://generativelanguage.googleapis.com/v1beta/openai/",
      "gemini",
    );
    expect(execFileSync).toHaveBeenCalledTimes(2);
    expect(execFileSync).toHaveBeenNthCalledWith(
      1,
      "openshell",
      [
        "provider",
        "create",
        "--name",
        "gemini-api",
        "--type",
        "openai",
        "--credential",
        "GEMINI_API_KEY",
        "--config",
        "OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/",
      ],
      expect.objectContaining({
        env: expect.objectContaining({
          GEMINI_API_KEY: "AIza-test-secret",
        }),
      }),
    );
    expect(execFileSync).toHaveBeenNthCalledWith(
      2,
      "openshell",
      ["inference", "set", "--no-verify", "--provider", "gemini-api", "--model", "gemini-3-flash-preview"],
      expect.objectContaining({
        env: expect.objectContaining({
          GEMINI_API_KEY: "AIza-test-secret",
        }),
      }),
    );
    expect(saveOnboardConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        endpointType: "gemini",
        endpointUrl: "https://generativelanguage.googleapis.com/v1beta/openai/",
        credentialEnv: "GEMINI_API_KEY",
        model: "gemini-3-flash-preview",
        provider: "gemini-api",
      }),
    );
  });

  it("uses a provider-first interactive flow for OpenAI without prompting for the default base URL", async () => {
    const { logger } = captureLogger();
    vi.mocked(promptSelect)
      .mockResolvedValueOnce("openai")
      .mockResolvedValueOnce("gpt-5.2");
    vi.mocked(promptInput).mockResolvedValue("sk-live-secret");

    await cliOnboard({
      logger,
      pluginConfig,
    });

    expect(promptSelect).toHaveBeenNthCalledWith(
      1,
      "Where should NemoClaw get its model?",
      expect.arrayContaining([
        expect.objectContaining({ label: "NVIDIA hosted", value: "build" }),
        expect.objectContaining({ label: "OpenAI", value: "openai" }),
        expect.objectContaining({ label: "Google Gemini", value: "gemini" }),
        expect.objectContaining({ label: "Other compatible endpoint", value: "custom" }),
      ]),
    );
    expect(promptInput).toHaveBeenCalledTimes(1);
    expect(promptInput).toHaveBeenCalledWith(
      expect.stringContaining("OpenAI API key"),
      expect.objectContaining({ secret: true }),
    );
    expect(validateApiKey).toHaveBeenCalledWith("sk-live-secret", "https://api.openai.com/v1", "openai");
  });

  it("offers Anthropic in the provider menu and defaults its model prompt", async () => {
    const { logger } = captureLogger();
    vi.mocked(promptSelect).mockResolvedValueOnce("anthropic");
    vi.mocked(promptInput).mockResolvedValueOnce("sk-ant-live-secret");
    vi.mocked(validateApiKey).mockResolvedValueOnce({
      valid: true,
      models: ["claude-opus-4-6", "claude-sonnet-4-5"],
      error: null,
    });
    vi.mocked(promptSelect).mockResolvedValueOnce("claude-sonnet-4-5");

    await cliOnboard({
      logger,
      pluginConfig,
    });

    expect(promptSelect).toHaveBeenNthCalledWith(
      1,
      "Where should NemoClaw get its model?",
      expect.arrayContaining([
        expect.objectContaining({ label: "Anthropic", value: "anthropic" }),
      ]),
    );
    expect(promptInput).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining("Anthropic API key"),
      expect.objectContaining({ secret: true }),
    );
    expect(validateApiKey).toHaveBeenCalledWith(
      "sk-ant-live-secret",
      "https://api.anthropic.com",
      "anthropic",
    );
    expect(promptSelect).toHaveBeenNthCalledWith(
      2,
      "Select your primary model:",
      [
        { label: "claude-opus-4-6", value: "claude-opus-4-6" },
        { label: "claude-sonnet-4-5 (recommended)", value: "claude-sonnet-4-5" },
      ],
      1,
    );
  });

  it("offers Gemini in the provider menu and prefers Gemini default models", async () => {
    const { logger } = captureLogger();
    vi.mocked(promptSelect)
      .mockResolvedValueOnce("gemini")
      .mockResolvedValueOnce("gemini-3-flash-preview");
    vi.mocked(promptInput).mockResolvedValueOnce("AIza-live-secret");
    vi.mocked(validateApiKey).mockResolvedValueOnce({
      valid: true,
      models: ["gemini-2.5-flash", "gemini-3-flash-preview"],
      error: null,
    });

    await cliOnboard({
      logger,
      pluginConfig,
    });

    expect(promptInput).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining("Gemini API key"),
      expect.objectContaining({ secret: true }),
    );
    expect(validateApiKey).toHaveBeenCalledWith(
      "AIza-live-secret",
      "https://generativelanguage.googleapis.com/v1beta/openai/",
      "gemini",
    );
    expect(promptSelect).toHaveBeenNthCalledWith(
      2,
      "Select your primary model:",
      [
        { label: "gemini-3-flash-preview", value: "gemini-3-flash-preview" },
        { label: "gemini-2.5-flash", value: "gemini-2.5-flash" },
      ],
      0,
    );
  });
});
