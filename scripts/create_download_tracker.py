"""Script that creates the rdt-download-tracker package."""

import argparse
import logging
import os

import boto3
import build
import tomli
import tomli_w
from botocore.exceptions import ClientError

import rdt

LOGGER = logging.getLogger(__name__)
PROJECT_PATH = './scripts/rdt-download-tracker'
BUCKET = os.getenv('DOWNLOAD_TRACKER_BUCKET', '')
S3_PACKAGE_PATH = 'simple/rdt-download-tracker/'
INDEX_FILE_NAME = 'index.html'


def _set_version(version):
    toml_path = os.path.join(PROJECT_PATH, 'pyproject.toml')
    with open(toml_path, 'rb') as f:
        pyproject = tomli.load(f)
        pyproject['project']['version'] = version

    with open(toml_path, 'wb') as f:
        tomli_w.dump(pyproject, f)


def build_package():
    """Builds the wheel and sdist for 'rdt-download-tracker'."""
    _set_version(rdt.__version__)
    build.ProjectBuilder(PROJECT_PATH).build('wheel', 'dist')
    build.ProjectBuilder(PROJECT_PATH).build('sdist', 'dist')


def _load_local_index_file():
    local_index_file_path = os.path.join(PROJECT_PATH, INDEX_FILE_NAME)
    with open(local_index_file_path, 'rb') as local_index_file:
        file = local_index_file.read().decode('utf-8')
    return file


def _update_index_html(files, s3_client, dryrun=False):
    index_file_path = os.path.join(S3_PACKAGE_PATH, INDEX_FILE_NAME)
    if not dryrun:
        try:
            response = s3_client.get_object(Bucket=BUCKET, Key=index_file_path)
            current_index_file = response.get('Body').read().decode('utf-8')
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                LOGGER.info('Index file does not exist yet. Using local one instead.')
                current_index_file = _load_local_index_file()
    else:
        current_index_file = _load_local_index_file()

    insertion_point = current_index_file.find('</body>')
    current_text = current_index_file[:insertion_point]
    text_list = [current_text]
    for file in files:
        download_link = f'https://{BUCKET}.s3.us-east-1.amazonaws.com/{S3_PACKAGE_PATH}{file}'
        new_link = f"<a href='{download_link}'>{file}</a>"
        if new_link not in current_text:
            text_list.append(new_link)
            text_list.append('<br>')

    text_list.append(current_index_file[insertion_point:])
    new_index = '\n'.join(text_list)
    if dryrun:
        print('New index file:')  # noqa: T201 `print` found
        print(new_index)  # noqa: T201 `print` found
    else:
        s3_client.put_object(Bucket=BUCKET, Key=index_file_path, Body=new_index)


def upload_package(dryrun=False):
    """Uploads the built package to the S3 bucket.

    Args:
        dryrun (bool):
            If true, skip the actual uploading and just print out which files would be uploaded.
    """
    s3_client = boto3.client('s3')
    files = os.listdir('dist')
    for file_name in files:
        dest = os.path.join(S3_PACKAGE_PATH, file_name)
        if dryrun:
            print(f'Uploading {file_name} as {dest} to bucket {BUCKET}')  # noqa: T201 `print` found
        else:
            s3_client.upload_file(os.path.join('dist', file_name), BUCKET, dest)

    _update_index_html(files, s3_client, dryrun)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dryrun', action='store_true', help='Skip uploading built files.')
    args = parser.parse_args()
    build_package()
    upload_package(args.dryrun)
