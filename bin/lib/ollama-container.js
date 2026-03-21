// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Ollama Docker sidecar — runs ollama/ollama as a Docker container sharing the
// OpenShell gateway's network namespace.  This avoids all host-networking
// issues (eth0 IP, 0.0.0.0 binding, host.docker.internal) because the
// container shares localhost with the gateway and its k3s cluster.

const { run, runCapture } = require("./runner");

const OLLAMA_IMAGE = "ollama/ollama";
const CONTAINER_NAME_PREFIX = "nemoclaw-ollama";
const GATEWAY_CONTAINER_PREFIX = "openshell-cluster-nemoclaw";

function containerName(sandboxName) {
  return `${CONTAINER_NAME_PREFIX}-${sandboxName || "default"}`;
}

/**
 * Find the running OpenShell gateway container name.
 */
function findGatewayContainer() {
  const output = runCapture(
    `docker ps --filter "name=${GATEWAY_CONTAINER_PREFIX}" --format '{{.Names}}' 2>/dev/null`,
    { ignoreError: true }
  );
  if (!output) return null;
  // Take the first match
  return output.split("\n").map((l) => l.trim()).filter(Boolean)[0] || null;
}

/**
 * Start the Ollama sidecar container sharing the gateway's network namespace.
 * Uses --gpus all for GPU acceleration on WSL2/Linux.
 */
function startOllamaContainer(sandboxName) {
  const name = containerName(sandboxName);
  const gateway = findGatewayContainer();
  if (!gateway) {
    console.error("  Cannot find OpenShell gateway container. Is the gateway running?");
    process.exit(1);
  }

  // Remove any stale container with the same name
  run(`docker rm -f ${name} 2>/dev/null || true`, { ignoreError: true });

  // Detect whether --gpus all is supported — use host nvidia-smi first (instant),
  // fall back to container probe only if needed (avoids pulling a 4GB image).
  const hasGpu = !!runCapture("nvidia-smi -L 2>/dev/null", { ignoreError: true });
  const gpuFlag = hasGpu ? "--gpus all" : "";

  run(
    `docker run -d ${gpuFlag} --network container:${gateway} ` +
    `-v nemoclaw-ollama-models:/root/.ollama ` +
    `--name ${name} ${OLLAMA_IMAGE}`,
    { ignoreError: false }
  );

  return name;
}

/**
 * Wait for the Ollama sidecar to become healthy (respond on port 11434).
 * Since it shares the gateway's network, we check via docker exec.
 */
function waitForOllamaHealth(sandboxName, timeout = 60) {
  const name = containerName(sandboxName);
  const start = Date.now();

  while ((Date.now() - start) / 1000 < timeout) {
    // Use `ollama list` as health check — the ollama/ollama image has no curl/wget.
    const result = runCapture(
      `docker exec ${name} ollama list 2>/dev/null`,
      { ignoreError: true }
    );
    if (result !== undefined && result !== null && result !== "") return true;
    require("child_process").spawnSync("sleep", ["2"]);
  }
  return false;
}

/**
 * Pull a model inside the Ollama sidecar container.
 */
function pullModel(sandboxName, model) {
  const name = containerName(sandboxName);
  run(`docker exec ${name} ollama pull ${model}`, { ignoreError: false });
}

/**
 * Check if a model is already available in the sidecar.
 */
function hasModel(sandboxName, model) {
  const name = containerName(sandboxName);
  const output = runCapture(
    `docker exec ${name} ollama list 2>/dev/null`,
    { ignoreError: true }
  );
  return !!output && output.includes(model);
}

/**
 * Prime/warmup a model inside the sidecar to keep it loaded in VRAM.
 */
function warmupModel(sandboxName, model, keepAlive = "15m") {
  const name = containerName(sandboxName);
  // Use `ollama run` to send a short prompt — keeps the model loaded in VRAM.
  // The ollama/ollama image has no curl/wget, so we use the native CLI.
  run(
    `docker exec ${name} ollama run ${model} "hello" --keepalive ${keepAlive} > /dev/null 2>&1`,
    { ignoreError: true }
  );
}

/**
 * Validate that the model responds to a probe inside the sidecar.
 */
function validateModel(sandboxName, model, timeoutSeconds = 120) {
  const name = containerName(sandboxName);
  // Use `ollama run` with a short prompt as the probe — the image has no curl/wget.
  // Timeout via the container exec; ollama run streams output so any response = healthy.
  const output = runCapture(
    `timeout ${timeoutSeconds} docker exec ${name} ollama run ${model} "hello" --keepalive 15m 2>&1`,
    { ignoreError: true }
  );
  if (!output) {
    return {
      ok: false,
      message:
        `Ollama model '${model}' did not answer the probe in time. ` +
        "It may still be loading, too large for the GPU, or otherwise unhealthy.",
    };
  }
  // Check for error messages in output
  if (output.includes("Error:") || output.includes("error:")) {
    const errorLine = output.split("\n").find((l) => /[Ee]rror/.test(l)) || output.slice(0, 200);
    return { ok: false, message: `Ollama model '${model}' probe failed: ${errorLine.trim()}` };
  }
  return { ok: true };
}

/**
 * Stop and remove the Ollama sidecar container.
 */
function stopOllamaContainer(sandboxName) {
  const name = containerName(sandboxName);
  run(`docker stop ${name} 2>/dev/null || true`, { ignoreError: true });
  run(`docker rm ${name} 2>/dev/null || true`, { ignoreError: true });
}

/**
 * Check if the Ollama sidecar is running.
 */
function isOllamaContainerRunning(sandboxName) {
  const name = containerName(sandboxName);
  const state = runCapture(
    `docker inspect --format '{{.State.Status}}' ${name} 2>/dev/null`,
    { ignoreError: true }
  );
  return state === "running";
}

/**
 * Get the base URL for the sidecar provider.
 * Since the sidecar shares the gateway's network namespace, the gateway
 * (and its k3s pods) can reach Ollama at localhost:11434.
 */
function getSidecarBaseUrl() {
  return "http://127.0.0.1:11434/v1";
}

module.exports = {
  CONTAINER_NAME_PREFIX,
  OLLAMA_IMAGE,
  containerName,
  findGatewayContainer,
  getSidecarBaseUrl,
  hasModel,
  isOllamaContainerRunning,
  pullModel,
  startOllamaContainer,
  stopOllamaContainer,
  validateModel,
  waitForOllamaHealth,
  warmupModel,
};
