"""Command-line interface for MechanismLens."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _run_rollout_demo() -> None:
    from mechanismlens.examples.toy_rollout_demo import main

    main()


def _run_counterfactual_demo() -> None:
    from mechanismlens.examples.toy_counterfactual_demo import main

    main()


def _run_decision_demo() -> None:
    from mechanismlens.examples.toy_decision_risk_demo import main

    main()


def _run_toy_benchmark() -> None:
    from mechanismlens.examples.run_toy_benchmark import main

    main()


def _print_version() -> None:
    try:
        from mechanismlens import __version__
    except ImportError:
        print("MechanismLens")
    else:
        print(__version__ or "MechanismLens")


def build_parser() -> argparse.ArgumentParser:
    """Build the MechanismLens CLI parser."""

    parser = argparse.ArgumentParser(
        prog="mechanismlens",
        description="MechanismLens command-line audit utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print the MechanismLens version.")
    version_parser.set_defaults(func=lambda _args: _print_version())

    demo_parser = subparsers.add_parser("demo", help="Run bundled toy demos.")
    demo_subparsers = demo_parser.add_subparsers(dest="demo_name", required=True)
    rollout_parser = demo_subparsers.add_parser("rollout", help="Run the toy rollout audit.")
    rollout_parser.set_defaults(func=lambda _args: _run_rollout_demo())
    counterfactual_parser = demo_subparsers.add_parser(
        "counterfactual",
        help="Run the toy counterfactual audit.",
    )
    counterfactual_parser.set_defaults(func=lambda _args: _run_counterfactual_demo())
    decision_parser = demo_subparsers.add_parser("decision", help="Run the toy decision-risk audit.")
    decision_parser.set_defaults(func=lambda _args: _run_decision_demo())

    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark suites.")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_name", required=True)
    toy_parser = benchmark_subparsers.add_parser("toy", help="Run the bundled toy benchmark.")
    toy_parser.set_defaults(func=lambda _args: _run_toy_benchmark())

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MechanismLens CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
