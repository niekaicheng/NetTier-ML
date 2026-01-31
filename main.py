import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Network Intrusion Detection System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--stage", type=str, choices=["1", "2", "all"], default="all", help="Stage to train")

    # Evaluate Command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("--file", type=str, help="Path to test file")

    args = parser.parse_args()

    if args.command == "train":
        print(f"Starting training for Stage {args.stage}...")
        # TODO: Import and call training logic
    elif args.command == "evaluate":
        print(f"Evaluating on {args.file}...")
        # TODO: Import and call evaluation logic
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
