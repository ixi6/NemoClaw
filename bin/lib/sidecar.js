// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sidecar dispatcher — unified interface for Docker sidecar inference providers.
// Maps provider keys ("ollama-k3s", "lmstudio-k3s") to their container modules.

const ollamaContainer = require("./ollama-container");
const lmstudioContainer = require("./lmstudio-container");

const SIDECARS = {
  "ollama-k3s": {
    start: ollamaContainer.startOllamaContainer,
    waitForHealth: ollamaContainer.waitForOllamaHealth,
    pullModel: ollamaContainer.pullModel,
    hasModel: ollamaContainer.hasModel,
    loadModel: () => {},  // Ollama auto-loads on first inference
    validateModel: ollamaContainer.validateModel,
    warmupModel: ollamaContainer.warmupModel,
    stop: ollamaContainer.stopOllamaContainer,
    isRunning: ollamaContainer.isOllamaContainerRunning,
    getBaseUrl: ollamaContainer.getSidecarBaseUrl,
    getProviderName: () => "ollama-k3s",
    getCredential: () => "ollama",
    containerName: ollamaContainer.containerName,
    label: "Ollama",
    models: require("./ollama-models.json"),
    // argv array for background spawn (no shell needed)
    // Ollama model IDs are the same for pull and API (e.g., "qwen3:0.6b")
    getApiModelId: (model) => model,
    getPullArgs: (containerName, model) => ["docker", "exec", containerName, "ollama", "pull", model],
  },
  "lmstudio-k3s": {
    start: lmstudioContainer.startLmstudioContainer,
    waitForHealth: lmstudioContainer.waitForHealth,
    pullModel: lmstudioContainer.pullModel,
    hasModel: lmstudioContainer.hasModel,
    loadModel: lmstudioContainer.loadModel,
    validateModel: lmstudioContainer.validateModel,
    warmupModel: lmstudioContainer.warmupModel,
    stop: lmstudioContainer.stopLmstudioContainer,
    isRunning: lmstudioContainer.isRunning,
    getBaseUrl: lmstudioContainer.getBaseUrl,
    getProviderName: () => "lmstudio-k3s",
    getCredential: () => "lm-studio",
    containerName: lmstudioContainer.containerName,
    label: "LM Studio",
    models: require("./lmstudio-models.json"),
    // LM Studio uses "name@quant" for download but API model ID is just "name"
    getApiModelId: (model) => model.split("@")[0],
    getPullArgs: (containerName, model) => ["docker", "exec", containerName, "lms", "get", model, "--yes"],
  },
};

function getSidecar(providerKey) {
  return SIDECARS[providerKey] || null;
}

function isSidecarProvider(providerKey) {
  return providerKey in SIDECARS;
}

function getSidecarModelTiers(providerKey) {
  const sidecar = SIDECARS[providerKey];
  return sidecar ? sidecar.models : [];
}

function getRecommendedSidecarModel(providerKey, vramMB) {
  const tiers = getSidecarModelTiers(providerKey);
  return tiers.find((t) => vramMB >= t.minMB) || tiers[tiers.length - 1] || null;
}

module.exports = {
  SIDECARS,
  getSidecar,
  getRecommendedSidecarModel,
  getSidecarModelTiers,
  isSidecarProvider,
};
