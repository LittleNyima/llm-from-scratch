import itertools
import sys
from typing import TextIO

import rich
from rich.console import Console
from rich.pretty import pprint

from llmstarter.tokenization import Encoding


def visualize_encoding(
    encoding: Encoding,
    key: str = "tokens",
    colorful: bool = True,
    stream: TextIO = sys.stdout,
):
    assert key in ["tokens", "ids"], f"key must be either 'tokens' or 'ids', got {key}"
    visualized_object = encoding.tokens if key == "tokens" else encoding.ids
    if not colorful or key == "ids":
        pprint(visualized_object, console=Console(file=stream))
        return

    colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    color_cycle = itertools.cycle(colors)
    for token, bg_color in zip(visualized_object, color_cycle):
        rich.print(f"[black on bright_{bg_color}]{token}[/]", end="")
