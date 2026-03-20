#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Locks openclaw.json so the sandboxed agent cannot modify gateway
# security settings (auth token, CORS origins, etc.).
#
# This script runs as root via a narrow sudoers entry.  The sandbox
# user can only invoke this exact path — no arguments, no variations.
# Ref: https://github.com/NVIDIA/NemoClaw/issues/514

set -eu

CONFIG="${HOME:-/sandbox}/.openclaw/openclaw.json"

if [ ! -f "$CONFIG" ]; then
  echo "[lock-gateway-config] config not found: $CONFIG" >&2
  exit 1
fi

chown root:root "$CONFIG"
chmod 444 "$CONFIG"

# Revoke our own sudoers entry — this privilege is single-use.
rm -f /etc/sudoers.d/lock-gateway-config
