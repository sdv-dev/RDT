from unittest import TestCase, skipIf

from rdt.hyper_transformer import HyperTransformer

# SKIPPED TESTS
TESTS_WITH_DATA = True


@skipIf(TESTS_WITH_DATA, 'demo_downloader should have been run.')
class TestHyperTransformer(TestCase):

    def test_data_airbnb(self):
        """HyperTransfomer will transform back and forth data airbnb data."""
        # Setup
        meta_file = 'demo/Airbnb_demo_meta.json'
        transformer_list = ['NumberTransformer', 'DTTransformer', 'CatTransformer']
        ht = HyperTransformer(meta_file)

        # Run
        transformed = ht.fit_transform(transformer_list=transformer_list)
        result = ht.reverse_transform(tables=transformed)

        # Check
        assert result.keys() == ht.table_dict.keys()

        for name, table in result.items():
            assert not result[name].isnull().all().all()
