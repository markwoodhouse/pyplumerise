import click

from ..notebooks import launch_jupyter_example

@click.command()
@click.argument("n", type=int)
def run_example(n):
    """Run a PlumeRise jupyter notebook example.

    n is the example number; choices are 1"""

    click.echo(f"Running PlumeRise Example {n}")

    launch_jupyter_example(n)