"""
Miscellaneous Utility Functions
"""
import click
import warnings
from torch.utils.data import Dataset
def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))

def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs))
