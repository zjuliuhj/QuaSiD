from evaluation.entrypoints import build_task_parser, run_checkpoint_evaluation


def parse_args():
    """Parse the standalone unconditional-generation evaluation CLI."""
    return build_task_parser(
        mode="uncond",
        description="Run QuaSiD unconditional checkpoint evaluation.",
        default_num_samples=5,
    ).parse_args()


def main():
    """Evaluate an unconditional checkpoint and write metrics/figures under eval/."""
    run_checkpoint_evaluation("uncond", parse_args())


if __name__ == "__main__":
    main()
