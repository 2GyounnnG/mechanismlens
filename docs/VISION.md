# Vision

MechanismLens studies mechanism-alignment failures in world models. The motivating claim is
that a world model can be predictively plausible while still internally mismatching semantic,
causal, physical, and decision layers.

The package is deliberately small. It provides typed trajectory data structures and a minimal
audit pipeline that flags obvious rollout failures without training or loading a model.

MechanismLens is framework-general but contract-specific: the core audit runner does not assume
a model architecture, while `DomainContract` plugins define what counts as a meaningful failure
inside a particular world or simulation domain.
