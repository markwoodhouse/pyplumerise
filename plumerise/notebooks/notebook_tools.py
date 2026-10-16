import subprocess
from importlib import resources


def run_notebook(file):
    subprocess.Popen(["jupyter", "notebook", file])


def launch_jupyter_example(n: int):
    ref = resources.files('plumerise.notebooks') / f"pyplumerise_example{n}.ipynb"
    with resources.as_file(ref) as path:
        print(f"Running {path}")
        run_notebook(path)