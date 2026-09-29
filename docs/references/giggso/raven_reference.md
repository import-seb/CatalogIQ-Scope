# Raven — Setup and Usage Reference

Repository: https://github.com/giggsoinc/raven

## What Raven Is

Raven is a local AI engineering control layer for AI coding environments. It provides routing to domain specialists, structured planning/debugging workflows, local guards, project configuration, audit information, and Git pre-commit checks.

Main components include:

- **Andie** — architecture, design, and new-work decisions
- **Andie-Jr** — debugging existing code
- **Specialists** — domain-specific engineering guidance
- **Guards** — checks for secrets, CVEs, stack rules, style, architecture, and selected database issues
- **Project manifest** — local/project configuration used by Raven

## Standard Claude Code Setup

### 1. Clone Raven

```bash
git clone https://github.com/giggsoinc/raven.git
```

### 2. Install the plugin

```bash
claude plugin install ./raven/plugin
```

Restart Claude Code afterward.

Raven is installed from the cloned/downloaded plugin directory rather than by marketplace name.

### 3. Machine setup

Run once per developer machine:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/giggsoinc/raven/main/install.sh)
```

### 4. Project setup

Inside the project repository:

```bash
cd your-project
raven-setup
```

This configures Raven for that project and installs the local project hooks.

## Important Files

### `.raven/manifest.json`

Project configuration describing items such as:

- work mode
- language/stack
- database
- cloud provider
- solo/team/enterprise mode
- guard behavior

For existing repositories, Raven may detect the broad project type but still ask you to enter some stack information manually.

### `.raven/manifest.secrets.json`

Optional local notification credentials.

Do not commit this file.

### Git pre-commit hook

Raven can install a local Git pre-commit hook that runs checks before a commit is accepted.

Conceptually:

```text
git add .
   ↓
git commit
   ↓
Raven checks staged changes
   ↓
PASS → commit proceeds
BLOCK → fix reported issue
```

Git hooks are local to each clone, so every teammate should perform the project setup on their own machine.

## Team Setup

For a shared repository, each developer generally needs:

```text
Raven plugin installed
        ↓
machine setup completed
        ↓
local clone of project
        ↓
raven-setup
```

Project-level configuration such as `.raven/manifest.json` can be shared in the repository. Local credentials should remain local.

## Everyday Usage

Once Raven is installed and the project is configured, normal requests can be routed automatically.

Examples:

```text
Why is this API failing?
```

This may route to Andie-Jr.

```text
Should we change our database design?
```

This may route to Andie.

Simple read-only questions may bypass the full planning workflow.

## Useful Commands

Examples documented by Raven include:

```text
/andie
/andie-jr
/raven-init
/raven-debug
/run-costs
```

The exact command set can change by version.

## Guided / Education Mode

Raven includes an education mode. In guided mode, read-only work can proceed while write operations can require approval.

The setting is stored under:

```text
.raven/educate.json
```

This allows Raven to explain, review, or investigate before modifying files.

## Guard Categories

Raven documents local guards for areas including:

### Secrets
- API keys
- tokens
- SSH material
- bearer tokens

### Dependency vulnerabilities
- CVE checks for newly introduced libraries

### Stack validation
- checks against approved project dependencies

### Style
- selected style/type-hint/documentation rules

### Architecture
- checks against architecture documentation

### Database
- selected SQL and migration checks

Whether a finding warns or blocks depends on configuration and Raven version.

## Reviewing Existing Work

Raven can be used on code written manually or by any other tool.

Examples:

```text
Review this implementation for security and architecture issues.
```

```text
Why is this existing pipeline failing?
```

```text
Check this commit for dependency and secret problems.
```

## Other Hosts

The Raven repository also contains support files for:

- Claude Desktop
- Codex
- Cursor
- Grok
- Windsurf
- VS Code / Copilot
- Replit
- Gemini CLI
- AntiGravity

For several non-Claude-Code hosts, Raven documents:

```bash
bash install-host.sh /path/to/project
```

See `plugin/HOSTS.md` in the Raven repository for host-specific details.

## Verification

After setup:

1. Restart the supported coding environment.
2. Confirm `.raven/manifest.json` exists.
3. Confirm `raven-setup` completed.
4. Confirm the local pre-commit hook exists.
5. Start a new session in the project.
6. Use a simple Raven command or prompt to confirm routing.

## Updating Raven

If installed from a Git clone:

```bash
cd path/to/raven
git pull
claude plugin install ./plugin
```

Restart the coding session afterward.

## Troubleshooting

### Skills do not appear
Restart the coding environment and open a new session.

### Guards do not run
Run:

```bash
raven-setup
```

inside the project and confirm hooks/configuration were created.

### Stack information is incomplete
Re-run setup and provide the requested language/database/cloud values manually.

### Marketplace-style install fails
Install from the local plugin directory:

```bash
claude plugin install ./raven/plugin
```

## Current Documentation

- README: https://github.com/giggsoinc/raven/blob/main/README.md
- Plugin install guide: https://github.com/giggsoinc/raven/blob/main/claude_plugin_readme.md
- Host guide: https://github.com/giggsoinc/raven/blob/main/plugin/HOSTS.md
- Releases: https://github.com/giggsoinc/raven/releases

## Quick Reference

```text
Clone Raven
    ↓
Install plugin
    ↓
Run machine setup once
    ↓
Run raven-setup in each project clone
    ↓
Review project manifest
    ↓
Use normal coding workflow
    ↓
Raven routing + local guards
    ↓
Pre-commit checks before commits
```
