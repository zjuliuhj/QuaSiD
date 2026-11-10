from evaluation.entrypoints import build_task_parser, run_checkpoint_evaluation


def parse_args():
    """Parse the standalone condition-guided-generation evaluation CLI."""
    return build_task_parser(
        mode="cond",
        description="Run PaCoDi conditional checkpoint evaluation.",
        default_num_samples=5,
    ).parse_args()


def main():
    """Evaluate a conditional checkpoint and write metrics/figures under eval/."""
    run_checkpoint_evaluation("cond", parse_args())


if __name__ == "__main__":
    main()
