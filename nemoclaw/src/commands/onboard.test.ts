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
  describeOnboardProvider: vi.fn((config: { endpointType: string }) =>
    config.endpointType === "openai" ? "OpenAI" : "NVIDIA hosted",
  ),
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
        expect.objectContaining({ label: "Other compatible endpoint", value: "custom" }),
      ]),
    );
    expect(promptInput).toHaveBeenCalledTimes(1);
    expect(promptInput).toHaveBeenCalledWith(
      expect.stringContaining("OpenAI API key"),
      expect.objectContaining({ secret: true }),
    );
    expect(validateApiKey).toHaveBeenCalledWith("sk-live-secret", "https://api.openai.com/v1");
  });
});
