---
title: Agent skills
subtitle: "Example agent skills"
---

# Example OpenCode skills

This directory ships two **example skills** for working with `stressnet`.
They are documentation, not active configuration: nothing here is auto-loaded by
OpenCode, and no script reads these files. Copy the skill you want into a skill
directory (see below) to make it available to the agent.

These skills contain **no site-specific values**. Dataset and model locations
are always resolved through `site_config.yaml` / `stressnet.config`; the concrete
values for a machine live in the gitignored `HELP.md`.

## What is a skill?

A skill is a single `SKILL.md` file inside a directory named after the skill:

```
skills/
  stressnet-usage/
    SKILL.md
  stressnet-datasets/
    SKILL.md
```

`SKILL.md` starts with YAML **frontmatter** between `---` fences, followed by
Markdown instructions for the agent:

```markdown
---
name: my-skill
description: >
  One or more sentences describing what the skill does and when to use it,
  including the words a user is likely to say (the triggers).
---

# My skill

Instructions for the agent...
```

Frontmatter fields:

- `name` (required) — the skill identifier; keep it the same as the directory name.
- `description` (required) — what the skill does **and when to load it**. List the
  user-facing triggers (train, fine-tune, run an experiment, ...) so the agent can
  match a request to the skill.
- `tools` (optional) — restrict the tools the skill may use, e.g. a YAML list such
  as `tools: [bash]`.

## Installing a skill

Pick a scope and copy the skill's directory next to other skills there:

- **Project scope** — only this checkout:

  ```bash
  mkdir -p .opencode/skills
  cp -r docs/skills/stressnet-usage .opencode/skills/
  ```

- **User scope** — every project on this machine:

  ```bash
  mkdir -p ~/.config/opencode/skills
  cp -r docs/skills/stressnet-usage ~/.config/opencode/skills/
  ```

After copying, the skill is picked up by name/description; restart the agent
session if it was already running. Remove the copied directory to uninstall.

## The skills

| Skill | Use it for |
| --- | --- |
| [`stressnet-usage`](stressnet-usage/SKILL.md) | Training, fine-tuning (including from an existing checkpoint), running experiments, agreeing a model/run naming convention, and reporting on experiment results. |
| [`stressnet-datasets`](stressnet-datasets/SKILL.md) | Loading and validating a dataset (or helping the user point the project at / prepare one), including the expected directory layout, TIFF naming and label spreadsheets. |

Both skills assume the repository is set up as a [uv](https://docs.astral.sh/uv/)
project and that commands are run from the repository root with `uv run python ...`.
