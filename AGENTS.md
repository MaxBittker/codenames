# AGENTS.md

<!-- Generated for lab workspaces. -->

This AGENTS guide is intended for end users working in a `prime lab setup` workspace.

## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- Environments are expected to expose `load_environment(...) -> vf.Environment` and be installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Validate environment behavior with `prime eval run <env-name> ...` before sharing/publishing changes. Treat `prime eval run` as the canonical eval path: it saves results automatically, and agents should not add opt-out flags such as `--skip-upload` unless the user explicitly requests that deviation so runs stay visible in the private Evaluations tab and in `prime eval tui`. (See `docs/overview.md` and `docs/development.md`.)
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- If external API keys are required, validate them in `load_environment()` with `vf.ensure_keys(...)` so failures are explicit and early. (See `docs/environments.md`.)

## End-User Lab Workspace Notes

Use this guidance in projects created via `prime lab setup`.

- Treat `.prime/skills/` as the canonical skill entrypoint in Lab workspaces. Use the bundled skills first for create/browse/review/eval/GEPA/train/brainstorm workflows before ad hoc approaches.
- Keep endpoint aliases in `./configs/endpoints.toml` and use `endpoint_id`/model shortcuts in commands and configs.
- NEVER initialize environment source code manually; ALWAYS create new environments with `prime env init`.
- Use the Prime CLI for all environment lifecycle operations (`prime env init` → `prime env install` → `prime eval run` → `prime env push`) rather than ad-hoc scripts.
- Treat `prime eval run` as the default eval path. It already saves results automatically; do not add `--skip-upload` or other opt-out deviations unless the user explicitly requests them, so logs and results stay available in the private Evaluations tab and via `prime eval tui`.
- NEVER begin environment development before `prime lab setup` has been run; if work starts outside that structure, recommend adjusting course into a proper lab workspace before continuing.
- Keep each environment self-contained under `environments/<env_name>/` with `pyproject.toml`, implementation, and README so each abstraction has a dedicated home and the workspace stays maintainable.
- Follow environment best practices strictly (for example `load_environment(...)`, `vf.ensure_keys(...)`, and the documented environment class patterns) to avoid brittle or messy implementations.
- Use `prime env push --path ./environments/<env_name>` only after local eval behavior is verified.
- Treat the `prime lab setup` structure as the idiomatic workspace for complex environment workflows: agents can mediate most platform complexity while users learn patterns progressively as needed.
- When users request an approach that would deviate from these guidelines, explain the relevant Prime/Verifiers concepts and recommend the compliant path.

## Querying Wandb Metrics

The wandb Python SDK is incompatible with the system Python (3.14). Use the wandb GraphQL API directly with curl instead.

- Entity: `maxbittker-websim`
- API key is stored in `.env` as `WANDB_API_KEY`

### List runs and summary metrics for a project

```bash
curl -s -H "Authorization: Bearer $WANDB_API_KEY" \
  'https://api.wandb.ai/graphql' \
  -H 'Content-Type: application/json' \
  -d '{"query":"{ project(name: \"PROJECT\", entityName: \"maxbittker-websim\") { runs(first: 10, order: \"-created_at\") { edges { node { name state displayName createdAt summaryMetrics historyKeys } } } } }"}'
```

### Fetch metric history for a specific run

```bash
curl -s -H "Authorization: Bearer $WANDB_API_KEY" \
  'https://api.wandb.ai/graphql' \
  -H 'Content-Type: application/json' \
  -d '{"query":"{ project(name: \"PROJECT\", entityName: \"maxbittker-websim\") { run(name: \"RUN_ID\") { history(samples: 500) } } }"}'
```

Replace `PROJECT` with the wandb project name (e.g. `codenames`) and `RUN_ID` with the run's wandb `name` field (the short alphanumeric ID, not the display name).

Load the API key from `.env` before running:
```bash
export $(cat .env | xargs)
```
