import argparse
import os
import tarfile

from rdt.hyper_transformer import HyperTransformer


def download_airbnb_dataset(bucket, dirname):

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    sessions_KEY = 'sdv-demo/sessions_demo.csv'
    users_KEY = 'sdv-demo/users_demo.csv'
    meta_KEY = 'sdv-demo/Airbnb_demo_meta.json'

    # bucket.download_file(sessions_KEY, os.path.join(dirname, 'sessions_demo.csv'))
    # bucket.download_file(users_KEY, os.path.join(dirname, 'users_demo.csv'))
    bucket.download_file(meta_KEY, os.path.join(dirname, 'Airbnb_demo_meta.json'))



def get_bucket(bucket_name):
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    return s3.Bucket(bucket_name)


def run_airbnb_demo(data_dir):
    """HyperTransfomer will transform back and forth data airbnb data."""

    # Setup
    meta_file = os.path.join(data_dir, 'Airbnb_demo_meta.json')
    transformer_list = ['NumberTransformer', 'DTTransformer', 'CatTransformer']
    ht = HyperTransformer(meta_file)

    # Run
    transformed = ht.fit_transform(transformer_list=transformer_list)
    result = ht.reverse_transform(tables=transformed)

    # Check
    assert result.keys() == ht.table_dict.keys()

    for name, table in result.items():
        assert not result[name].isnull().all().all()


if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    airbnb_dir = os.path.join(data_dir, 'airbnb')

    if not os.path.exists(airbnb_dir):
        tar_name = airbnb_dir + '.tar.gz'
        with tarfile.open(tar_name, mode='r:gz') as tf:
            tf.extractall(data_dir)

    run_airbnb_demo(airbnb_dir)
