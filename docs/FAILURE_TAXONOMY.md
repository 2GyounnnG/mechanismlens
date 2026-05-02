# Failure Taxonomy

## Semantic Grounding Failure

The model assigns the wrong object identity, entity label, role, or task grounding.

## Causal Validity Failure

The model predicts intervention effects on the wrong objects, in the wrong direction, or with
implausible locality.

## Physics Feasibility Failure

The rollout violates simple physical constraints such as boundaries, object penetration,
momentum consistency, or required physical support.

## Cross-Layer Mechanism Mismatch

Semantic labels, causal structure, and physical behavior contradict each other. For example,
an object labeled `static` moves, or an `obstacle` lacks a radius.

## Decision/Planner Exploitation Risk

A planner may exploit uncertainty or model errors to choose actions that look good inside the
model but are unsafe or infeasible in the target world.
