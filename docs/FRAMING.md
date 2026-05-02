# Research Framing

Modern world models are increasingly semantic, causal, and physical. They do not merely predict
pixels or coordinates; they often represent objects, relations, actions, interventions, and
constraints. That integration is powerful, but it also creates new failure modes.

Existing evaluation often checks prediction accuracy, task return, or isolated physics and
causal benchmarks. Those are useful signals, but an integrated world model can fail through
cross-layer mismatch:

- A semantic label is wrong while the physics simulation remains locally plausible.
- A causal edge is plausible in a graph but implies a physically impossible transition.
- A physics rollout is valid but grounded on the wrong object or task entity.

MechanismLens treats these cases as audit targets. The aim is not to prove full safety or train
a better model. The aim is to make assumptions explicit enough that a user can ask: did the
semantic, causal, physical, and decision layers agree on the same imagined future?

## Mechanized Hallucination

We use **mechanized hallucination** to describe a coherent but wrong imagined future supported
by mutually reinforcing semantic, causal, and physical layers. The rollout may look internally
consistent: objects move smoothly, causal explanations appear available, and labels sound
reasonable. Yet the imagined future is wrong because one or more layer interfaces are
misgrounded.

MechanismLens is positioned as a contract-based audit toolkit for this setting. A
`DomainContract` encodes the measurements that matter in a domain, and `AuditSuite` turns those
measurements into findings, metrics, and reports.
