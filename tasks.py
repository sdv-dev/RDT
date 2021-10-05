import os
import re
import shutil
import stat
from pathlib import Path

from invoke import task


@task
def pytest(c):
    c.run(
        'pytest ./tests/unit ./tests/performance/tests ./tests/datasets/tests '
        '--cov=rdt --cov-report=xml'
    )


@task
def integration(c):
    c.run('pytest ./tests/integration')


@task
def performance(c):
    c.run('pytest -v ./tests/performance/test_performance.py')


@task
def install_minimum(c):
    with open('setup.py', 'r') as setup_py:
        lines = setup_py.read().splitlines()

    versions = []
    started = False
    for line in lines:
        if started:
            if line == ']':
                started = False
                continue

            line = line.strip()
            line = re.sub(r',?<=?[\d.]*,?', '', line)
            line = re.sub(r'>=?', '==', line)
            line = re.sub(r"""['",]""", '', line)
            versions.append(line)

        elif line.startswith('install_requires = [') or \
            line.startswith('copulas_requires = ['):
            started = True

    c.run(f'python -m pip install {" ".join(versions)}')


@task
def minimum(c):
    install_minimum(c)
    c.run('python -m pip check')
    c.run('python -m pytest ./tests/unit ./tests/integration ./tests/performance/tests')


@task
def readme(c):
    test_path = Path('tests/readme_test')
    if test_path.exists() and test_path.is_dir():
        shutil.rmtree(test_path)

    cwd = os.getcwd()
    os.makedirs(test_path, exist_ok=True)
    shutil.copy('README.md', test_path / 'README.md')
    os.chdir(test_path)
    c.run('rundoc run --single-session python3 -t python3 README.md')
    os.chdir(cwd)
    shutil.rmtree(test_path)


@task
def lint(c):
    c.run('flake8 rdt')
    c.run('flake8 tests --ignore=D')
    c.run('isort -c --recursive rdt tests')
    c.run('pylint rdt tests/performance --rcfile=setup.cfg')


def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)


@task
def rmdir(c, path):
    try:
        shutil.rmtree(path, onerror=remove_readonly)
    except PermissionError:
        pass
