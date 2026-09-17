# Claude Tooling and MCP — Team Notes

**Author:** Maria Garcia
**Last updated:** September 16, 2026
**Scope:** Claude API, MCP, Agent Skills, Claude Code. What each one is and where it fits our build.

## One scope, not four options

These four are not four choices we pick between. They stack:

| Piece | Where it sits |
|---|---|
| Claude API | Runs inside our product. Core deliverable. |
| MCP | How our product reaches outside data and tools. |
| Agent Skills | Reusable instructions layered on top. |
| Claude Code | Developer tool. Helps us build. Not part of the product. |

The Building with the Claude API course already covers MCP and Claude Code, so we start there instead of learning four things from zero.

## Claude API

Three uses for us.

**RAG.** Claude reads unstructured text (earnings releases, filings, news) and writes our weekly brief. The part that matters is a real evaluation set, not a demo that looks good once.

**Prompt evaluation.** Build test datasets and grade the answers, some model graded and some code graded. This is how we prove it works.

**Tool use and caching.** Claude calls our own functions. Caching lowers cost and latency when we send the same context repeatedly.

**What the API is not:** our forecaster. Demand forecasting is a time series problem. The API handles ingestion, retrieval, and explanation.

## MCP (Model Context Protocol)

Open standard that connects AI apps to outside data and tools. Three primitives:

- **Tools** — model controlled
- **Resources** — app controlled
- **Prompts** — user controlled

Three scenarios for us:

1. Raven or Aryx already expose an MCP server → we just connect, no integration code.
2. Giggso expects us to build an MCP server → new work, changes our deliverables.
3. Neither → skip it. Our public sources are files and REST, and plain HTTP is simpler.

**Compliance note:** the MCP connector is a beta feature and is not covered by Zero Data Retention. If Giggso has data handling rules, raise this early, not at the end of the semester.

## Agent Skills

A folder of instructions Claude loads when they are relevant.

Supporting, not core. Revisit when we are producing the same formatted output more than twice, or when Team A and Team B need their briefs to look the same. Adding it now is scope creep.

## Claude Code

Anthropic's agentic coding tool. Headless mode is a real option for scheduled jobs, but a plain scheduled Python job is easier to defend and easier to hand over. It does not go on the product architecture diagram.

## Open questions for Ravi (week 2)

1. Do Raven or Aryx expose an MCP server we can connect to?
2. Does Giggso expect Team B to build an MCP server exposing Market Pulse data?
3. Is a written market brief part of the deliverable, or only the model and its outputs?
4. Are there data handling requirements that rule out beta features without Zero Data Retention?

All four change our scope, so we want them answered early.
