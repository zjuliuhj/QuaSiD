"""Run the TensorBoard reader with compatibility shims for the AI Studio environment."""

import functools
import inspect

import numpy as np
from google.protobuf import json_format


def main() -> None:
    # TensorBoard 2.14 still references NumPy scalar aliases removed in NumPy 2.
    np.__dict__.setdefault("string_", np.bytes_)
    np.__dict__.setdefault("unicode_", np.str_)

    message_to_json = json_format.MessageToJson
    if "including_default_value_fields" not in inspect.signature(message_to_json).parameters:
        @functools.wraps(message_to_json)
        def compatible_json(message, *args, including_default_value_fields: bool = False, **kwargs):
            # HParams passes the legacy protobuf flag when serializing Time Series metadata.
            kwargs.setdefault("always_print_fields_with_no_presence", including_default_value_fields)
            return message_to_json(message, *args, **kwargs)

        json_format.MessageToJson = compatible_json

    from tensorboard import main as tensorboard_main

    tensorboard_main.run_main()


if __name__ == "__main__":
    main()
