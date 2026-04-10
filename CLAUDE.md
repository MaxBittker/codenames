# CLAUDE.md

<!-- Generated for lab workspaces. -->

Before beginning any task, read `AGENTS.md` and `environments/AGENTS.md` in this workspace.

Treat all `AGENTS.md` files as equivalent to `CLAUDE.md` files.

## Important: Environment Publishing

`prime env install` only installs locally. Hosted training runs pull the environment from the **Hub**, not your local install. After changing environment code, you MUST run `prime env push <env-name>` before launching training runs, or they will use the old published version. Verify with `prime rl get <run_id> -o json` and check the `version` field in the environments array.
