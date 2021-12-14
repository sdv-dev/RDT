#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script for the addons packages."""

import json
import os
import shutil
import sys
from copy import deepcopy
from glob import glob
from tempfile import TemporaryDirectory

from setuptools import find_namespace_packages, setup

import rdt

with open('README.md', encoding='utf-8') as readme_file:
    README = readme_file.read()

RDT_VERSION = rdt.__version__
ADDONS_PATH = os.path.dirname(os.path.realpath(__file__))


def _build_setup(addon_json):

    with open(addon_json, 'r', encoding='utf-8') as f:
        addon = json.load(f)

    addon_name = addon.get('name')
    addon_module = addon.get('transformers')[0].split('.')
    addon_module = '.'.join(addon_module[:-2])

    install_requires = [f'rdt>={RDT_VERSION}']
    install_requires.extend(addon.get('requirements', []))

    # this does something
    setup(
        author='MIT Data To AI Lab',
        author_email='dailabmit@gmail.com',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        description='Reversible Data Transforms',
        include_package_data=False,
        install_requires=install_requires,
        keywords=['rdt', addon_name],
        license='MIT license',
        long_description=README,
        long_description_content_type='text/markdown',
        name=addon_name,
        packages=find_namespace_packages(include=[addon_module]),
        python_requires='>=3.6,<3.10',
        url='https://github.com/sdv-dev/RDT',
        version=RDT_VERSION,
        zip_safe=False,
    )


def _run():
    path = sys.argv[0]
    base_path = os.path.realpath(path).replace(path, '')

    # clear build if exists
    build_path = os.path.join(base_path, 'build')
    if os.path.exists(build_path):
        shutil.rmtree(build_path)

    families = deepcopy(sys.argv[1:])
    all_families = [family for family in os.listdir('.') if os.path.isdir(family)]

    families = list(set(families).intersection(set(all_families)))
    for addon in glob(f'{ADDONS_PATH}/*/*.json'):
        with TemporaryDirectory() as temp_dir:
            build_command = [
                path, 'bdist_wheel', '--keep-temp', '--dist-dir', 'dist', '--bdist-dir', temp_dir,
                'sdist', '--keep-temp', '--dist-dir', 'dist', 'egg_info', '--egg-base', temp_dir
            ]

            base_name = os.path.basename(os.path.dirname(addon))
            sys.argv = deepcopy(build_command)

            if not families:
                _build_setup(addon)
            else:
                if os.path.basename(os.path.dirname(addon)) in families:
                    _build_setup(addon)

            remove_addon_build = os.path.join(
                base_path, 'build', 'lib', 'rdt', 'transformers', 'addons', base_name
            )

            # delete only the processed addon folder
            shutil.rmtree(remove_addon_build)


_run()
