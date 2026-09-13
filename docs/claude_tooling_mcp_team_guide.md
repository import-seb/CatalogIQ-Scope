# Claude Tooling and MCP — Simple Team Guide

What **Market Pulse Team B** needs to know.

```
Owner: Maria Garcia Sehara (AI Engineering Specialist)
Team:  Market Pulse, Team B — US retail consumer demand, public data
Scope: Claude API, Claude Code, Agent Skills, Model Context Protocol
```

I screened all four topics against our slice of Market Pulse. One is clearly in scope, one depends on an answer we need from Ravi, and two are supporting rather than core.

```
Claude API    -> In scope. The only one of the four that runs inside the product.
MCP           -> Blocked. Two questions for Ravi before we decide.
Agent Skills  -> Revisit later. Useful once we have repeated work, not before.
Claude Code   -> Mostly team tooling. One narrow production path worth knowing about.
```

---

## 0. One framing to keep straight

Before the Week 2 scope conversation: **these are not four competing options.**

```
Agent Skills
"Reusable instructions that sit on top of either surface"
        |
Claude API
"Runs inside the product"
        |
MCP
"How the product reaches data"
        |
Claude Code
"A tool we use to build the product"
```

Also worth knowing for planning: the four learning resources overlap heavily. The **Building with the Claude API** course already contains a full MCP module and a Claude Code module.

Anyone picking this topic up should start there rather than treating the four as four separate workloads.

---

## 1. Claude API

### What is it

The programmatic way to send text to Claude and get a response back. It handles single requests, multi-turn conversations, streaming, structured output, tool use, and batch jobs where many requests go out at once.

### Why is it useful

It is the only one of the four we would actually run in production. Everything else is either a wrapper around it or a way of working with it.

Beyond basic calls, three capabilities matter for a market intelligence product:

```
Retrieval Augmented Generation (RAG)
    chunking, embeddings, BM25 lexical search, contextual retrieval
    -> how a system answers questions over a corpus
       it was never trained on

Prompt evaluation
    test dataset generation, model-based grading, code-based grading
    -> how you prove an AI component works instead of asserting it

Tool use and prompt caching
    tool use -> Claude calls our own functions
    caching  -> cuts cost and latency when the same context
                is sent repeatedly
```

### Does Market Pulse need it

Yes, if the product reads any unstructured text or produces any written output. Both are likely for our slice.

### Where would it fit

**Reading unstructured public text.** Retail demand signal does not only live in numbers. Earnings call transcripts, retailer press releases, and news coverage carry signal too.

Example:

```
Target / Costco earnings release
        |
        v
same-store sales commentary
        |
        v
labeled field that joins our numeric data
```

**Retrieval layer.** If we build a corpus of filings and news, RAG is how an analyst queries it. This is the single most relevant piece of the API for Market Pulse and it should be on our architecture diagram.

**Generating the brief.** If the deliverable includes a weekly market summary, the API writes that narrative from the model output.

**Validation.** Prompt evaluation belongs in the same conversation as our leakage and validation work. Any AI component in the pipeline needs an eval set, not a demo.

### Important limit

```
The API is not the forecaster.
```

Demand forecasting is a time series and econometrics problem. If our architecture shows Claude predicting retail demand, that is a weak point and it will be found.

The defensible framing is:

```
quantitative core
    + AI layer around it for ingestion, retrieval, and explanation
```

---

## 2. Model Context Protocol (MCP)

### What is it

An open standard for how an AI application connects to outside data and tools. Instead of writing a custom integration for every source, a source exposes an MCP server and any MCP client can talk to it.

There are three primitives:

```
Tools      -> model-controlled. Claude decides when to call them.
Resources  -> app-controlled. Read-only data exposed to the application.
Prompts    -> user-controlled. Pre-written instructions for common workflows.
```

### Why is it useful

Two separate things sit under this name and it is worth not confusing them.

```
Consuming a server
    Anthropic's MCP connector lets you attach a remote MCP server
    directly in a Messages API request without building your own client.
    You can allowlist or denylist individual tools, and MCP servers
    can be included in batch requests.

Building a server
    Using the Python SDK, you define tools with decorators instead of
    hand-writing JSON schemas, and test with the MCP Inspector.
    This is what you do when you want to expose your own data to Claude.
```

### Does Market Pulse need it

This is the real open question and it splits two ways. I cannot close it without Ravi.

```
If Giggso already exposes Raven or Aryx through an MCP server
    -> Yes, clearly. We connect and skip writing integration code.

If Giggso wants Team B to BUILD an MCP server so Market Pulse data
is reachable by their other systems
    -> That is a real deliverable and it changes our scope.
       Worth asking directly, because it is the kind of thing
       a sponsor assumes and never states.

If neither
    -> MCP is overhead for us. Our public data sources are files
       and REST endpoints. Plain HTTP requests are simpler than
       standing up a protocol layer for a single consumer.
```

### Where would it fit

```
Case 1 -> data access layer
Case 2 -> a separate deliverable
```

### Compliance note

```
The MCP connector is a beta feature
and is not covered by Zero Data Retention arrangements.
```

Flag this early if Giggso has data handling requirements.

---

## 3. Agent Skills

### What is it

A folder of instructions, scripts, and resources that Claude loads when a task calls for it. Each one is a `SKILL.md` file with frontmatter giving a name and a description.

Claude sees only the short description up front and loads the full instructions when they become relevant, which keeps the context window efficient. This is called **progressive disclosure**.

Skills exist on two surfaces, which is easy to conflate:

```
In Claude Code
    they sit alongside CLAUDE.md, hooks, and subagents as ways to
    configure behavior, and can be shared with a team by committing
    them to a repository or packaging them as a plugin.

Through the Claude API
    they run in the code execution environment.
    Pre-built skills exist for Excel, PowerPoint, Word, and PDF.
```

### Why is it useful

It stops five people from writing five versions of the same prompt. If the whole team needs a market brief in one house format, that format becomes a skill instead of tribal knowledge.

### Does Market Pulse need it

Not to make the product work. A skill is a way to make repeated work consistent, and right now we do not yet have repeated work. Adding it in September would be scope creep.

### Revisit when

```
we are producing the same formatted output more than twice
    OR
Team A and Team B need matching brief formats
```

At that point the pre-built Excel or PowerPoint skill is likely a one-hour win on the December deliverable.

---

## 4. Claude Code

### What is it

Anthropic's agentic coding tool. It reads and edits a codebase from the terminal, an IDE, or the desktop app.

Beyond interactive use it supports:

```
plan mode
permission modes
hooks that enforce rules a run cannot skip
scheduled routines
headless mode for running inside your own pipeline
a GitHub Action for pull request review
```

### Why is it useful

Mainly team velocity while we build the pipeline. But the headless and GitHub Actions side is worth naming rather than dismissing, because it blurs the line between tooling and system.

### Does Market Pulse need it

Not as a component of the forecasting product. Individual choice whether to use it while building.

There is one narrow case where it becomes more than tooling:

```
if we want scheduled or automated jobs in our own pipeline
    -> headless mode is a real option
```

I would not build our data pipeline on it for a semester project, since a scheduled Python job is simpler to defend and easier to hand over, but the team should know the option exists rather than assume it does not.

### Where would it fit

```
Development workflow, and optionally CI on the repository.
NOT on the product architecture diagram.
```

---

## 5. Scope recommendations

```
Claude API
    -> in scope for Team B.
       Prioritise RAG and prompt evaluation, since those touch
       our retrieval and validation work directly.

MCP
    -> blocked on Ravi. Revisit in Week 2 with his answer.

Agent Skills
    -> revisit later. Trigger is repeated formatted output.

Claude Code
    -> out of scope as a product component.
       Optional as development tooling.
```

---

## 6. Open questions for Ravi (Week 2)

```
1. Do Raven or Aryx expose an MCP server we can connect to?

2. Does Giggso expect Team B to build an MCP server
   exposing Market Pulse data?

3. Is a written market brief part of the expected deliverable,
   or only the model and its outputs?
   -> This decides how much of the Claude API we actually use.

4. Are there data handling requirements that would rule out
   beta features not covered by Zero Data Retention?
```

---

## Quick cheat sheet

```
Claude API
= the only one of the four that runs inside the product
  -> RAG, prompt evaluation, tool use, prompt caching

MCP
= open standard for connecting an AI app to outside data and tools
  -> tools (model-controlled), resources (app-controlled),
     prompts (user-controlled)

Agent Skills
= reusable instructions in a folder, loaded when relevant
  -> progressive disclosure

Claude Code
= agentic coding tool for the terminal, IDE, or desktop
  -> development workflow, not a product component

RAG
= answering questions over a corpus the model never trained on

Prompt evaluation
= proving an AI component works instead of asserting it
```

## The most important distinction

If you remember only one thing:

```
Claude Code   -> we use it to BUILD the product
Claude API    -> it RUNS INSIDE the product
MCP           -> how the product REACHES data
Agent Skills  -> reusable instructions ON TOP of either surface
```

And the one that decides our architecture:

```
The API is not the forecaster.
Quantitative core, AI layer around it.
```
