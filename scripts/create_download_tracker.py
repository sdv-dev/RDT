"""Script that creates the rdt-download-tracker package."""

import argparse
import hashlib
import logging
import os
from html.parser import HTMLParser

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


class PackageIndexHTMLParser(HTMLParser):
    """Class to parse package index html files."""

    def __init__(self):
        """Initialize html parser for a package index html.

        This class stores parameters to track the different links and packages stored in the index.
        """
        super().__init__()
        self.package_to_href = {}
        self._current_href = None
        self._in_a_tag = False

    def handle_starttag(self, tag, attrs):
        """Get current href if the tag is an 'a' tag."""
        if tag == 'a':
            self._in_a_tag = True
            attrs_dict = dict(attrs)
            href = attrs_dict.get('href')
            if href:
                self._current_href = href

    def handle_endtag(self, tag):
        """Reset tag information if the tag is an 'a' tag."""
        if tag == 'a' and self._in_a_tag:
            self._in_a_tag = False
            self._current_href = None

    def handle_data(self, data):
        """Record href and package name if in an 'a' tag."""
        if self._in_a_tag:
            self.package_to_href[data] = self._current_href


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


def _get_index_file(s3_client, index_file_path, dryrun=False):
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

    return current_index_file


def _update_index_html(current_index_file, files, s3_client, index_file_path, dryrun=False):
    insertion_point = current_index_file.find('</body>')
    current_text = current_index_file[:insertion_point]
    text_list = [current_text]
    for file, hash in files.items():
        download_link = f'https://{BUCKET}.s3.us-east-1.amazonaws.com/{S3_PACKAGE_PATH}{file}'
        new_link = f"<a href='{download_link}#sha256={hash}'>{file}</a>"
        if new_link not in current_text:
            text_list.append(new_link)
            text_list.append('<br>')

    text_list.append(current_index_file[insertion_point:])
    new_index = '\n'.join(text_list)
    if dryrun:
        print('New index file:')  # noqa: T201 `print` found
        print(new_index)  # noqa: T201 `print` found
    else:
        s3_client.put_object(
            Bucket=BUCKET,
            Key=index_file_path,
            Body=new_index,
            ContentType='text/html',
            CacheControl='no-cache',
        )


def _get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)

    return h.hexdigest()


def _get_links(index_file):
    parser = PackageIndexHTMLParser()
    parser.feed(index_file)
    return parser.package_to_href


def upload_package(dryrun=False):
    """Uploads the built package to the S3 bucket.

    Args:
        dryrun (bool):
            If true, skip the actual uploading and just print out which files would be uploaded.
    """
    s3_client = boto3.client('s3')
    files = os.listdir('dist')
    files_to_hashes = {}
    index_file_path = os.path.join(S3_PACKAGE_PATH, INDEX_FILE_NAME)
    current_index_file = _get_index_file(s3_client, index_file_path, dryrun)
    links = _get_links(current_index_file)
    for file_name in files:
        dest = os.path.join(S3_PACKAGE_PATH, file_name)
        if dryrun:
            print(f'Uploading {file_name} as {dest} to bucket {BUCKET}')  # noqa: T201 `print` found
        else:
            if file_name not in links:
                filepath = os.path.join('dist', file_name)
                file_hash = _get_file_hash(filepath)
                s3_client.upload_file(
                    filepath, BUCKET, dest, ExtraArgs={'ChecksumAlgorithm': 'SHA256'}
                )
                files_to_hashes[file_name] = file_hash
            else:
                raise RuntimeError(f'The file {file_name} is already in this package index.')

    _update_index_html(current_index_file, files_to_hashes, s3_client, index_file_path, dryrun)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dryrun', action='store_true', help='Skip uploading built files.')
    args = parser.parse_args()
    build_package()
    upload_package(args.dryrun)
