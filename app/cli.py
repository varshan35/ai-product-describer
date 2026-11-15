"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-product-describer", description="Small business owners use a CLI tool to generate creative product descriptions for their inventory by training a custom GPT on examples of existing product texts.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("AI Product Describer is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
