# Taxonomy, Ontology, and Agentic AI — Study Notes

## Sources

1. https://www.youtube.com/watch?v=oOYu1FJi2Y4&list=PLdFtObdjKwAjC2YoxJqmhya4jaO8zVj-g&index=7
2. https://www.youtube.com/watch?v=Sir59K8ZDPU

> **Note on Video 1:** The first link is part of the **AI-Taxonomy & Ontology** playlist, but its exact title/transcript could not be reliably retrieved. The first section below therefore covers the core taxonomy/ontology concepts needed to understand the playlist topic rather than pretending to be a transcript-specific summary.

---

# 1. Taxonomy and Ontology Foundations

## Main Idea

Both **taxonomies** and **ontologies** organize knowledge, but they do different jobs.

- A **taxonomy** mainly tells us **how things are classified**.
- An **ontology** tells us **what things are, how they relate, and what rules apply to them**.

A taxonomy is usually simpler. An ontology is more expressive and can support machine reasoning.

---

## Taxonomy

A **taxonomy** is a hierarchical classification system.

Think of it like folders inside folders.

Example:

```text
Vehicle
├── Car
│   ├── Sedan
│   └── SUV
└── Truck
    ├── Pickup
    └── Semi
```

The main relationship is usually:

```text
"is a type of"
```

For example:

```text
SUV is a type of Car
Car is a type of Vehicle
```

### What taxonomies are useful for

- organizing information
- creating categories
- improving search
- tagging documents or products
- keeping terminology consistent

### Limitation

A taxonomy usually does **not** describe many different kinds of relationships.

It can tell us:

```text
Dog -> Mammal -> Animal
```

But it does not naturally express things such as:

```text
Dog hasOwner Person
Dog eats Food
Dog treatedBy Veterinarian
```

That is where an ontology becomes useful.

---

# 2. Ontology

An **ontology** is a formal model of a domain.

It defines:

- the important **things**
- their **types**
- their **properties**
- their **relationships**
- sometimes the **rules or constraints** governing them

A simple way to think about it:

> A taxonomy organizes categories.  
> An ontology models meaning.

---

## Basic Parts of an Ontology

### Classes

General categories of things.

Examples:

```text
Person
Order
Product
Payment
Supplier
```

### Instances

Specific members of a class.

Example:

```text
Person: Alice
Product: Laptop123
Supplier: AcmeCorp
```

### Properties

Characteristics of an entity.

Example:

```text
Product
- price
- weight
- serialNumber
```

### Relationships

Connections between entities.

Example:

```text
Customer places Order
Order contains Product
Supplier supplies Product
```

This structure begins to look like a graph.

---

# 3. Taxonomy vs. Ontology

| Feature | Taxonomy | Ontology |
|---|---|---|
| Main goal | Classification | Model meaning |
| Typical structure | Tree/hierarchy | Graph |
| Main relationship | "is-a" | Many relationship types |
| Rules/constraints | Usually limited | Can be explicit |
| Machine reasoning | Limited | Stronger |
| Complexity | Lower | Higher |

### Easy example

A taxonomy might say:

```text
Material
├── Metal
│   ├── Aluminum
│   └── Copper
└── Polymer
```

An ontology could additionally say:

```text
Part contains Material
Supplier supplies Part
Material hasCriticalityLevel CriticalityLevel
Copper isConductive true
```

Now the computer understands more than just category membership.

---

# 4. Ontology vs. Knowledge Graph

These ideas are closely related but are not exactly the same.

### Ontology

Defines the **model**.

Example:

```text
Supplier supplies Part
Part contains Material
```

### Knowledge Graph

Stores actual entities and relationships that follow the model.

Example:

```text
Supplier_123 supplies Part_ABC
Part_ABC contains Nickel
```

A useful analogy:

> **Ontology = grammar/rules**  
> **Knowledge graph = sentences/data created using those rules**

---

# 5. Why Ontologies Matter for AI

Large language models are very good at recognizing patterns and generating plausible answers.

However, they are **probabilistic**.

That means they usually produce what appears statistically reasonable rather than proving that every answer follows a strict logical rule.

This becomes more dangerous when an AI system can take actions.

For example, an AI agent might:

- issue a refund
- update a database
- change an order status
- send money
- call an API
- modify a business record

A plausible answer is no longer enough.

The action must also be **valid**.

---

# 6. Video 2 — Why Agentic Systems Need Ontologies

**Speaker:** Frank Coyle, UC Berkeley  
**Main argument:** AI agents need a logical layer outside the language model to prevent invalid actions.

---

## The Core Problem

An LLM may generate something that looks perfectly reasonable but is still wrong.

Examples discussed around the talk include situations such as:

```text
Refund the same order twice
Send a payout to the wrong type of person
Create an invalid order status
```

The model may generate valid-looking text or JSON while violating the real rules of the system.

The problem is not necessarily bad syntax.

The problem is **bad meaning**.

---

# 7. Probabilistic AI vs. Symbolic Rules

The talk argues for combining two approaches.

## Probabilistic reasoning

This is what modern LLMs are good at.

They are useful for:

- interpreting language
- generating plans
- choosing possible actions
- recognizing patterns
- dealing with ambiguity

But their answers are not guaranteed.

---

## Symbolic reasoning

Symbolic systems use explicit logic and rules.

Example:

```text
An order may only be refunded once.
```

or:

```text
PaymentStatus must be one of:
- pending
- completed
- failed
```

These rules are deterministic.

The system can check:

```text
Does this action violate the rule?
```

The answer can be yes or no.

---

# 8. Neurosymbolic AI

Combining neural AI with symbolic logic is often called:

**neurosymbolic AI**

The idea is roughly:

```text
LLM = flexible reasoning
Ontology = logical boundaries
```

A useful summary is:

> **Probabilistic reasoning inside, logic outside.**

The LLM can propose an action.

The ontology can decide whether the proposed action makes sense within the domain.

---

# 9. Agent Loop

An AI agent usually operates in a loop.

Simplified:

```text
1. Observe the situation
2. Reason about what to do
3. Select a tool/action
4. Execute the tool
5. Observe the result
6. Repeat
```

This is powerful because the agent can perform multi-step tasks.

It is also risky because one incorrect step can affect later steps.

Example:

```text
Wrong assumption
    ↓
Wrong tool call
    ↓
Wrong database update
    ↓
Later reasoning uses the bad data
```

Errors can compound.

---

# 10. Where the Ontology Fits

Instead of allowing the model to directly perform an action:

```text
LLM
  ↓
Tool
  ↓
Database
```

the system can insert validation:

```text
LLM proposes action
        ↓
Type/schema validation
        ↓
Ontology/domain validation
        ↓
Tool executes
        ↓
Result validation
        ↓
Database
```

This creates a safety boundary between the AI's suggestion and the real system.

---

# 11. Pydantic vs. Ontology Validation

The talk distinguishes between checking **structure** and checking **meaning**.

## Pydantic / schema validation

Can check things like:

```text
amount is a number
customer_id is a string
status is an allowed data type
```

This answers:

> Is the data shaped correctly?

---

## Ontology validation

Can check things like:

```text
Is this person actually a customer?
Has this order already been refunded?
Is this relationship allowed?
Is this status meaningful for this type of object?
```

This answers:

> Does this action make sense in the domain?

Both layers are useful.

---

# 12. RDF, RDFS, and OWL

The talk mentions technologies from the Semantic Web.

You do not need to memorize all of their details yet.

## RDF

**Resource Description Framework**

Represents facts using triples:

```text
Subject -> Predicate -> Object
```

Example:

```text
Alice -> purchased -> Laptop
```

or:

```text
Supplier123 -> supplies -> Part456
```

---

## RDFS

**RDF Schema**

Adds basic semantic structure.

For example:

```text
Supplier is a type of Organization
```

or:

```text
supplies connects Supplier to Part
```

---

## OWL

**Web Ontology Language**

Allows richer logical rules and relationships.

It can represent things such as:

- class relationships
- restrictions
- equivalence
- transitive relationships
- functional properties
- logical constraints

Example:

```text
If A isPartOf B
and B isPartOf C

then A may also be considered part of C
```

depending on how the relationship is defined.

---

# 13. Reusing Existing Ontologies and Taxonomies

An important practical point is:

> Do not model everything from scratch if a good standard already exists.

Many industries already have:

- taxonomies
- controlled vocabularies
- schemas
- domain ontologies
- classification standards

Existing models can save work and improve interoperability.

A reasonable workflow is:

```text
1. Define the domain/problem
2. Search for existing standards
3. Reuse useful concepts
4. Add project-specific concepts only when necessary
```

---

# 14. Top-Down vs. Bottom-Up Ontology Building

There are two common approaches.

## Top-down

Start with the major concepts and rules.

Example:

```text
Asset
Material
Supplier
Organization
Location
```

Then create more specific concepts.

Advantage:

- organized
- conceptually clean

Risk:

- may become too theoretical
- may not match actual data

---

## Bottom-up

Start with the real data.

Example:

```text
supplier_id
part_number
material_name
manufacturer
country
```

Then determine what entities and relationships these fields represent.

Advantage:

- closely connected to reality

Risk:

- may simply reproduce messy database structure

---

## Better approach

Usually combine both.

```text
Domain knowledge
      +
Real data
      ↓
Practical ontology
```

---

# 15. Example: Building a Small Ontology

Suppose we are analyzing industrial parts.

Possible entities:

```text
Part
Material
Supplier
Manufacturer
Country
```

Relationships:

```text
Supplier supplies Part
Manufacturer manufactures Part
Part contains Material
Supplier locatedIn Country
```

Actual data could then produce:

```text
CAGE_1234 supplies NIIN_5678
NIIN_5678 contains Nickel
CAGE_1234 locatedIn UnitedStates
```

Now a system can ask more meaningful questions such as:

```text
Which parts contain Nickel?

Which suppliers provide parts containing Nickel?

Which critical materials depend on suppliers in one country?
```

This is much richer than simply storing separate database tables.

---

# 16. Important Distinction: Database Schema vs. Ontology

A database schema tells the computer:

> How is the data stored?

An ontology tells the computer:

> What does the data mean?

Example database:

```text
PART_TABLE
- part_id
- supplier_id
- material_code
```

An ontology might explain:

```text
part_id identifies a Part
supplier_id identifies a Supplier
Supplier supplies Part
material_code identifies a Material
Part contains Material
```

These layers can work together.

You do **not** necessarily replace SQL or PostgreSQL with an ontology.

The ontology can sit above the database as a semantic layer.

---

# 17. Why Prompt Engineering Alone Is Not Enough

You could tell an AI:

```text
Never refund an order twice.
Always send payments to customers.
Only use approved status values.
```

But those instructions are still natural language.

An LLM interprets them probabilistically.

For important rules, it is safer to make them machine-checkable.

Instead of:

```text
"Please don't refund twice."
```

use a rule that the program can actually verify before execution.

That is one of the strongest arguments in the talk.

---

# 18. What You Should Remember

If you only remember a few things, remember these:

1. **Taxonomy = classification hierarchy.**

2. **Ontology = formal model of entities, relationships, properties, and rules.**

3. **Knowledge graph = actual connected data that can use an ontology as its semantic model.**

4. **LLMs are probabilistic.** They can generate something plausible that is still invalid.

5. **AI agents are riskier than ordinary chatbots** because they can take actions.

6. **Ontology can act as a logical guardrail** between an LLM and a real-world action.

7. **Schema validation checks structure; ontology validation checks meaning.**

8. **RDF represents facts as triples. RDFS and OWL add semantic structure and reasoning capabilities.**

9. **Do not build an ontology from scratch unless necessary.** Reuse existing standards when possible.

10. The general architecture is:

```text
AI proposes
    ↓
Rules validate
    ↓
System acts
```

---

# One-Sentence Summary

> Ontologies give AI systems an explicit model of what exists, how things relate, and what rules must be followed, allowing flexible LLM reasoning to operate inside deterministic domain guardrails.
