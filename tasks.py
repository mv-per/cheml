import os

from invoke import task

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@task
def clean(c):
    patterns = ["build"]
    patterns.append("**/*.whl")
    patterns.append("**/*.tar.gz")
    for pattern in patterns:
        c.run("rm -rf {}".format(pattern))


@task
def build(c, dev=True):
    with c.cd(BASE_DIR):
        c.run(f"pip install {'-e' if dev else ''} . ")


@task
def test(c, force_regen=False):
    with c.cd(BASE_DIR):
        if force_regen:
            c.run("pytest src/ -v --force-regen")
        else:
            c.run("pytest src/ -v")


@task
def type_check(c):
    with c.cd(BASE_DIR):
        c.run("mypy src/")


@task
def lint(c):
    c.run("pre-commit run --all-files")


@task
def start_app(c):
    """
    Starts the application using Uvicorn.

    Args:
        c: The context instance for running shell commands.
    """
    app_dir = os.path.join(BASE_DIR, "src/cheml")
    with c.cd(BASE_DIR):
        c.run(f"uvicorn {app_dir}/main.py:app --host 0.0.0.0 --port 8000")


@task
def build_api_image(c):
    with c.cd(BASE_DIR):
        c.run("sudo docker build -f docker/Dockerfile -t cheml-api .")


@task
def run_api_image(c, port=5000):
    with c.cd(BASE_DIR):
        c.run(f"sudo docker run -p 8000:{port} cheml-api")


@task
def start_mlflow(
    c, host: str = "localhost", port: int = 5000, store: bool = True
) -> None:
    cmd = f"mlflow server --host {host} --port {port}"
    if store:
        cmd += " --backend-store-uri sqlite:///local.db"
    c.run(cmd)


@task
def stop_mlflow(c, port: int = 5000) -> None:
    c.run(f"fuser -k {port}/tcp")
