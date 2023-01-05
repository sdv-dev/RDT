import inspect
import operator
import os
import pkg_resources
import platform
import re
import shutil
import stat
from pathlib import Path

from invoke import task


COMPARISONS = {
    '>=': operator.ge,
    '>': operator.gt,
    '<': operator.lt,
    '<=': operator.le
}


if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


@task
def check_dependencies(c):
    c.run('python -m pip check')


@task
def unit(c):
    c.run(
        'python -m pytest ./tests/unit ./tests/performance/tests ./tests/datasets/tests '
        '--cov=rdt --cov-report=xml --cov-fail-under=100'
    )


@task
def integration(c):
    c.run('python -m pytest ./tests/integration')


@task
def performance(c):
    c.run('python -m pytest -v ./tests/performance/test_performance.py')


@task
def quality(c):
    c.run('pytest -v ./tests/quality/test_quality.py')


def _validate_python_version(line):
    is_valid = True
    for python_version_match in re.finditer(r"python_version(<=?|>=?|==)\'(\d\.?)+\'", line):
        python_version = python_version_match.group(0)
        comparison = re.search(r'(>=?|<=?|==)', python_version).group(0)
        version_number = python_version.split(comparison)[-1].replace("'", "")
        comparison_function = COMPARISONS[comparison]
        is_valid = is_valid and comparison_function(
            pkg_resources.parse_version(platform.python_version()),
            pkg_resources.parse_version(version_number),
        )

    return is_valid


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
            try:
                if _validate_python_version(line):
                    requirement = re.match(r'[^>]*', line).group(0)
                    requirement = re.sub(r"""['",]""", '', requirement)
                    version = re.search(r'>=?[^(,|#)]*', line).group(0)
                    if version:
                        version = re.sub(r'>=?', '==', version)
                        version = re.sub(r"""['",]""", '', version)
                        requirement += version
                    versions.append(requirement)
            except:
                # Skipping because Copulas is a link
                pass

        elif line.startswith('install_requires = [') or \
            line.startswith('copulas_requires = ['):
            started = True


    versions.append(
        'copulas @ git+https://github.com/sdv-dev/copulas.git@fix-pandas-versions'
    )
    c.run(f'python -m pip install {" ".join(versions)}')


@task
def minimum(c):
    install_minimum(c)
    check_dependencies(c)
    unit(c)
    integration(c)


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
    check_dependencies(c)
    c.run('flake8 rdt')
    c.run('pydocstyle rdt')
    c.run('flake8 tests --ignore=D')
    c.run('pydocstyle tests')
    c.run('isort -c --recursive rdt tests')
    c.run('pylint rdt tests/performance --rcfile=setup.cfg')
    c.run('pytest tests/code_style.py -v --disable-warnings --no-header')


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
