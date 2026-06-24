# vLLM NIXL Connector Import

This directory contains an unmodified copy of the vLLM NIXL KV connector from:

- Repository: `vllm-project/vllm`
- Release branch: `releases/v0.24.0`
- Source path: `vllm/distributed/kv_transfer/kv_connector/v1/nixl`
- Commit: `6c427dd40141870b9076c9a9f128eec3a7ce86bc`
- Commit date: `2026-06-23 13:43:53 +0800`
- Commit subject: `[BugFix] Omit empty tool_calls from OpenAI chat responses (#44105)`

The Python files retain their upstream SPDX headers:

- `SPDX-License-Identifier: Apache-2.0`
- `SPDX-FileCopyrightText: Copyright contributors to the vLLM project`

## License Notes

vLLM is licensed under Apache License 2.0. PegaFlow is also licensed under
Apache License 2.0, so copying and modifying this connector is generally
compatible with PegaFlow's open-source distribution and internal production
use, provided the Apache 2.0 obligations are preserved.

Practical obligations for follow-up modifications:

- Keep the upstream copyright and SPDX notices.
- Keep a copy of the Apache 2.0 license in the distribution.
- Mark modified files clearly when PegaFlow changes the imported connector.
- Preserve any upstream NOTICE content if vLLM adds one in the imported scope.
- Track the exact upstream commit so future rebases and security fixes are
  auditable.

Main risks to manage:

- If this code is substantially modified, the result is a derivative work of
  the vLLM connector and must continue to carry the required attribution and
  license notices.
- Apache 2.0 includes patent-license termination language. Company policy
  should review this if there is active patent litigation involving the work.
- Runtime dependencies such as NIXL, UCX, CUDA, and vLLM may have their own
  licenses and deployment obligations; this README only covers the imported
  connector source files.
- Internal production use is allowed by Apache 2.0, but distribution of
  modified source or binaries must satisfy the redistribution requirements.
