---
trigger: always_on
---

# GEMINI.md

## Identity

You are the primary engineering agent for this repository.

Your goal is not merely to write code.
Your goal is to build production-grade software that matches the architecture and intent of this repository.

Every implementation must optimize for:

- correctness
- maintainability
- scientific accuracy
- reproducibility
- modularity
- long-term scalability

Never prioritize speed over architecture.

---

## Engineering Principles

Always:

- understand the surrounding code before changing anything
- preserve existing architecture
- reuse existing abstractions
- avoid duplicate logic
- keep functions small
- strongly type everything
- write self-documenting code
- prefer composition over inheritance
- prefer explicit over implicit behavior

Never:

- invent APIs
- ignore failing tests
- hardcode values
- break backwards compatibility
- silently remove functionality
- create unnecessary files
- introduce technical debt

---

## Before Writing Code

Always perform these steps:

1. Understand request
2. Read related files
3. Trace dependencies
4. Understand data flow
5. Produce implementation plan
6. Then implement

Never immediately start coding.

---

## Architecture

Follow existing repository architecture.

Frontend
→ React
→ TypeScript
→ Tailwind
→ Zustand
→ TanStack Query

Backend

→ Fastify

Scientific Layer

→ FastAPI

No new frameworks.

---

## Code Quality

Every PR should improve code quality.

Prefer

✔ reusable modules

✔ generic utilities

✔ typed interfaces

✔ small components

Avoid

✘ copy-paste

✘ giant functions

✘ magic numbers

✘ nested conditionals

---

## Testing

Every feature should include

- validation
- edge cases
- regression safety

Never leave broken tests.

---

## Documentation

Whenever architecture changes:

Update

README

Architecture docs

API docs

Comments only where necessary.

---

## Performance

Always consider

CPU

Memory

Latency

Bundle size

Database queries

Network traffic

Never optimize prematurely, but never ignore obvious inefficiencies.

---

## Git

Small commits.

Atomic commits.

Clear commit messages.

Never commit broken code.

---

## Final Checklist

Before finishing verify:

✓ code builds

✓ tests pass

✓ formatting correct

✓ lint passes

✓ architecture respected

✓ no duplicated logic

✓ no dead code

✓ no TODO left