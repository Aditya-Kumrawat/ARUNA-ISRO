---
description: Structured workflow for Gemini in Antigravity that enforces repository-first development, planning before coding, iterative implementation, validation, testing, and production-quality engineering practices.
---

# WORKFLOW.md

## Primary Workflow

For every task follow this process.

---

### Phase 1

Understand

- Read request carefully.
- Determine objective.
- Ask clarifying questions if required.

Never assume.

---

### Phase 2

Investigate

Read all relevant files.

Understand:

- architecture
- dependencies
- data flow
- existing patterns

Do not modify code yet.

---

### Phase 3

Plan

Produce a concise implementation plan.

Consider:

- affected modules
- edge cases
- performance
- testing

---

### Phase 4

Implement

Implement incrementally.

After every logical change:

- verify build
- verify types
- verify imports

Never create massive edits.

---

### Phase 5

Validate

Run:

tests

lint

typecheck

Review implementation.

Search for:

dead code

duplication

bugs

regressions

---

### Phase 6

Reflect

Ask:

Can this be simpler?

Can this reuse existing code?

Does this match repository conventions?

Would another engineer understand this in six months?

If not,

refactor before finishing.

---

## Editing Rules

Modify existing code whenever possible.

Avoid creating new files unless necessary.

Never rewrite working code without reason.

---

## Error Handling

Always:

fail gracefully

log meaningful errors

return actionable messages

avoid silent failures

---

## Security

Never expose secrets.

Never log credentials.

Validate every external input.

Escape user-controlled output.

---

## Scientific Projects

For scientific repositories:

Never change equations unless requested.

Never alter thresholds without configuration.

Keep implementations reproducible.

Preserve numerical stability.

Document assumptions.

---

## Completion Criteria

Task is complete only if:

✓ feature implemented

✓ validated

✓ tested

✓ documented

✓ consistent with architecture

✓ production ready