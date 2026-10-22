# Supervisor interfaces

## Astra

`astra_agent_v4.py` defines the tool schema, RGB crop, agent state and validation.
`AstraClient` supports an explicitly supplied `OPENAI_API_KEY` and
`OPENAI_BASE_URL`. Model access depends on the configured account.

The current live runner uses `BridgeClient`, a loopback HTTP client.
`astra_bridge_v6.py` serializes each episode's history and bounds total requests
and reported token usage. Its existing subscription transport expects a
separately authorized credential record at
`~/.config/agent-inhand-siwc/credentials.json`. This repository neither creates
that authorization nor distributes credentials. Having a ChatGPT subscription
alone is not a guarantee that this transport is available.

With an existing authorized configuration, create an experiment-specific bridge
key outside the repository and start the service:

```bash
python -c 'import secrets,pathlib; p=pathlib.Path("/tmp/rotation-bridge.key"); p.write_text(secrets.token_urlsafe(32)); p.chmod(0o600)'
python scripts/agentic/astra_bridge_v6.py \
  --key-file /tmp/rotation-bridge.key --audit-dir outputs/api_audit \
  --max-calls 36 --max-tokens 150000 --max-sessions 6 --workers 2
```

In the simulator shell:

```bash
export ASTRA_BRIDGE_URL=http://127.0.0.1:18761
export ASTRA_BRIDGE_KEY_FILE=/tmp/rotation-bridge.key
python scripts/run_astra.py --record-video --output outputs/astra_live
```

The example is bounded to six episodes with at most six requests each. The
reported-token threshold can be exceeded by requests already in flight.
A new experiment should use a new audit directory and an explicitly chosen
budget. Failed requests with unknown usage remain recorded.

The transmitted data is the instruction, rendered RGB/crop, sensor-derived
feedback and prior tool results. The evaluator keeps ground-truth layout and
pose separate. Inference is event-triggered and may occur repeatedly, not only
at episode start. Astra returns structured tools rather than a learned latent.

## Learned VLM latent

The Qwen path consists of `latent_goal_v4.py`, `latent_supervisor_v4.py` and
`latent_service_v7.py`. The service requires a checkpoint directory containing:

```text
vlm_checkpoint/
  goal_head.pt
  lora/adapter_config.json
  lora/adapter_model.safetensors
```

`goal_head.pt` records the base-model location and latent-head dimensions. Set
that model location for the target host. The optional VLM environment also needs
`peft`, `bitsandbytes`, `accelerate` and a compatible CUDA PyTorch installation.
Weights are supplied separately from the small student release.

```bash
python scripts/agentic/latent_service_v7.py --checkpoint /path/to/vlm_checkpoint
python scripts/run_vlm.py --latent-url http://127.0.0.1:18772 \
  --record-video --output outputs/vlm_eval
```

The predicted 768-dimensional goal tensor is passed into the local student.
Scheduling in this baseline is periodic. `vlm_server.py` additionally preserves
the RGB/text JSON service used by the rule-based semantic supervisor.

The VLM evaluator terminates on correct goal reach. The current Astra evaluator
requires a correct autonomous stop. Compare methods only after matching those
protocols.
