from evaluation.entrypoints import build_task_parser, run_checkpoint_evaluation


def parse_args():
    """Parse the standalone probabilistic forecasting evaluation CLI."""
    return build_task_parser(
        mode="forecast",
        description="Run QuaSiD forecasting checkpoint evaluation.",
        default_num_samples=20,
    ).parse_args()


def main():
    """Evaluate a forecasting checkpoint and write horizon metrics/figures under eval/."""
    run_checkpoint_evaluation("forecast", parse_args())


if __name__ == "__main__":
    main()
