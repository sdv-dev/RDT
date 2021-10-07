from rdt import get_demo


def test_get_demo():
    demo = get_demo()

    assert list(demo.columns) == ['0_int', '1_float', '2_str', '3_datetime']
    assert len(demo) == 10
    assert list(demo.isna().sum(axis=0)) == [2, 2, 2, 2]
