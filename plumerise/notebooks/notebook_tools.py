import subprocess

from pkg_resources import resource_filename


def run_notebook(file):
    subprocess.Popen(["jupyter notebook " + file], shell=True)


def launch_jupyter_example(n: int):
    file = resource_filename(__name__, f"/pyplumerise_example{n}.ipynb")
    run_notebook(file)
