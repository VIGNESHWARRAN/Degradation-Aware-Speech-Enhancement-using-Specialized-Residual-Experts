"""
Script to run comparison experiments: baseline single model, shared residual stack, and proposed expert model.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run enhancement experiments")
    parser.add_argument("--config", type=str, help="Path to experiment config file")
    args = parser.parse_args()

    # load configuration, set up models, datasets, trainers
    print("Starting experiments with config", args.config)


if __name__ == "__main__":
    main()
