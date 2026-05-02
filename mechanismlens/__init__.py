"""MechanismLens: lightweight audits for mechanism-alignment failures."""

from .audit_suite import AuditSuite
from .contracts import DomainContract, GenericTrajectoryContract, ToyRigidBodyContract
from .schema import AuditInput, AuditReport, Finding, ObjectState, Trajectory

__version__ = "0.5.0"

__all__ = [
    "AuditInput",
    "AuditReport",
    "AuditSuite",
    "DomainContract",
    "Finding",
    "GenericTrajectoryContract",
    "ObjectState",
    "ToyRigidBodyContract",
    "Trajectory",
    "__version__",
]
