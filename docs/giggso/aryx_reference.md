# Aryx — Setup and Usage Reference

Repository: https://github.com/giggsoinc/aryx

## What Aryx Is

Aryx is a self-hosted context and knowledge-graph system intended to connect data from multiple sources, discover entities and relationships, resolve duplicate identities, preserve provenance, and expose the resulting context to users, applications, and AI agents.

General flow:

```text
Data
  ↓
Discover entities and relationships
  ↓
Resolve duplicate/related entities
  ↓
Build context / knowledge graph
  ↓
Query through UI, API, Ask, or MCP
```

Supported source types documented by Aryx include:

- PostgreSQL
- MySQL / MariaDB
- Oracle
- CSV
- JSON
- PDF
- DOCX
- PowerPoint
- images
- REST-style sources

## Core Concepts

### Workspace
An isolated Aryx project containing its own data, model, graph, brief, corrections, and query context.

### Entity
A real-world object such as a customer, company, product, asset, ticket, or order.

### Relationship
A connection between entities.

Example:

```text
Customer → opened → Ticket
Customer → purchased → Product
```

### Ontology / Model
The structured definition of entity types, attributes, relationships, and selected rules.

Aryx can propose this model from data instead of requiring the entire ontology to be designed manually first.

### Entity Resolution
The process of deciding when different records refer to the same real-world entity.

### Provenance
The connection from a graph fact/entity back to the source record that produced it.

## Architecture

The current public Aryx stack uses:

- **PostgreSQL** — main system of record
- **FalkorDB** — graph projection
- **FastAPI** — backend/API
- **Next.js** — web UI
- **Ollama or cloud LLM providers** — discovery and Q&A
- **MCP** — interface for compatible AI tools/agents

Default local surfaces include:

| Component | Address |
|---|---|
| Web UI | `http://localhost:3000` |
| API | `http://localhost:8088` |
| API docs | `http://localhost:8088/docs` |
| MCP SSE | `http://localhost:8765/sse` |
| PostgreSQL host port | `55432` |
| FalkorDB | `6379` |

## Recommended Setup: Docker Compose

### Requirements

- Git
- Docker
- Docker Compose v2
- several GB of free disk space
- about 8 GB+ RAM recommended if using local Ollama models

### 1. Clone Aryx

```bash
git clone https://github.com/giggsoinc/aryx.git
cd aryx
```

### 2. Create the environment file

```bash
cp .env.example .env
```

Windows PowerShell equivalent:

```powershell
Copy-Item .env.example .env
```

Review `.env` before use.

Important categories include:

- PostgreSQL credentials
- Aryx database DSN
- FalkorDB URL
- LLM provider
- model names
- API key
- API authentication mode

Do not commit `.env`.

### 3. Start Aryx

```bash
docker compose pull
docker compose up -d
```

The first run may take longer while images and local models download.

## Verify Installation

Check services:

```bash
docker compose ps
```

Check API health:

```bash
curl -s http://localhost:8088/health
```

Open:

```text
http://localhost:3000
```

API documentation:

```text
http://localhost:8088/docs
```

## LLM Configuration

Aryx can use:

- Ollama
- Claude / Anthropic
- OpenAI-compatible providers
- Gemini
- Grok / xAI

Configuration is available under **Settings** in the web UI.

Local Ollama does not require a cloud API key.

Keys entered through the UI are held in API process memory and are cleared when the API container restarts unless configured through environment variables.

## First-Time Workflow

### 1. Create a workspace

Open the web UI and create a workspace for the use case.

### 2. Load data

Use the setup wizard to:

- connect a database
- upload files
- or define types only

### 3. Smart review

Aryx proposes:

- what the data appears to contain
- project/domain brief
- entity types
- attributes
- relationships
- graph plan
- follow-up questions

Review and correct the proposal before building.

### 4. Build

Aryx ingests the data, resolves entities, stores the source information, and creates the graph projection.

## Main UI Areas

### Home
Create, open, reset, and manage workspaces.

### Brief
Stores domain/use-case context such as:
- aim
- objectives
- scope
- roles
- proof questions

### Data
Inspect entities and source provenance through:
- Tree
- Table
- Graph views

### Model
Inspect and edit:
- entity types
- attributes
- relationships
- survivorship behavior
- selected rules

### Ask
Natural-language Q&A over the workspace graph with grounded citations/provenance where available.

### Lab
Compare results with ontology/context enabled versus disabled.

### Observe
Inspect jobs, workspace health, and processing/storage state.

### MCP
View connection information for compatible AI tools and agents.

Default endpoint:

```text
http://localhost:8765/sse
```

## Typical Aryx Workflow

```text
Create workspace
      ↓
Load data
      ↓
Aryx samples data
      ↓
Review brief + proposed model
      ↓
Approve / correct
      ↓
Build
      ↓
Inspect entities + provenance
      ↓
Review model
      ↓
Resolve uncertain merges if needed
      ↓
Use Ask, API, or MCP
```

## Correcting Data

Aryx includes human-in-the-loop correction.

Users can review and correct items such as:

- wrong entity type
- incorrect merge
- missing relationship
- other entity/data issues

Corrections can be retained as standing rules for later ingests.

## Reviewing Entity Merges

When entity resolution is uncertain, Aryx can place candidate pairs in a review queue.

A user can:

```text
Approve & merge
```

or:

```text
Reject
```

## API Usage

Interactive API documentation is available after startup at:

```text
http://localhost:8088/docs
```

Use the live OpenAPI documentation for the exact endpoints supported by the installed version.

## MCP Usage

Aryx exposes an MCP server at:

```text
http://localhost:8765/sse
```

The web UI includes an MCP page at:

```text
http://localhost:3000/mcp
```

That page documents connection information, tools, parameters, and examples.

## Team Usage

### Local-per-developer
Each teammate runs their own Docker stack.

Pros:
- isolated experimentation
- no shared server required

Consideration:
- workspaces/data remain separate unless deliberately synchronized

### Shared deployment
Run Aryx on one shared server/VM and let teammates access the web UI/API.

For network-exposed use, apply the security guidance below.

## Security Notes

For a network-accessible deployment:

- set `ARYX_API_AUTH=required`
- change default database passwords
- do not expose PostgreSQL/FalkorDB publicly
- use TLS/reverse proxying
- protect LLM/API credentials
- configure MCP authentication as appropriate

Never commit:

```text
.env
API keys
database passwords
provider credentials
```

## Updating Aryx

```bash
git pull origin main
docker compose pull
docker compose up -d
```

For source-built deployments, rebuild affected services as needed.

Example clean rebuild:

```bash
docker compose build --no-cache api web
docker compose up -d --force-recreate api web
```

## Stopping Aryx

```bash
docker compose down
```

Avoid deleting volumes unless intentionally resetting stored data.

## Troubleshooting

### Check all services

```bash
docker compose ps
```

### API logs

```bash
docker compose logs -f api
```

### Web logs

```bash
docker compose logs -f web
```

### Ollama/model logs

```bash
docker compose logs ollama ollama-init
docker compose exec ollama ollama list
```

### Ask/discovery errors

Check the UI settings for:
- provider
- endpoint
- model
- API key

For Ollama inside Docker, the endpoint is normally:

```text
http://ollama:11434
```

rather than `localhost`.

## License

The public Aryx repository uses the Business Source License 1.1.

Review before production/commercial deployment:

- https://github.com/giggsoinc/aryx/blob/main/LICENSE
- https://github.com/giggsoinc/aryx/blob/main/docs/LICENSING.md

## Current Documentation

- README: https://github.com/giggsoinc/aryx/blob/main/README.md
- Installation: https://github.com/giggsoinc/aryx/blob/main/docs/INSTALL.md
- User guide: https://github.com/giggsoinc/aryx/blob/main/docs/USER_GUIDE.md
- Architecture: https://github.com/giggsoinc/aryx/blob/main/docs/ARCHITECTURE.md
- MCP quick start: https://github.com/giggsoinc/aryx/blob/main/docs/guides/MCP_QUICKSTART.md
- Ingestion guide: https://github.com/giggsoinc/aryx/blob/main/docs/INGESTION_GUIDE.md

## Quick Reference

```text
Clone Aryx
    ↓
copy .env.example → .env
    ↓
docker compose pull
    ↓
docker compose up -d
    ↓
open localhost:3000
    ↓
create workspace
    ↓
load data
    ↓
review proposed model
    ↓
build graph
    ↓
inspect Data / Model
    ↓
query through Ask, API, or MCP
```
