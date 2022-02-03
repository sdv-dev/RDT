import numpy as np
import pandas as pd

from rdt import get_demo


def test_get_demo():
    demo = get_demo()

    assert list(demo.columns) == [
        'last_login', 'email_optin', 'credit_card', 'age', 'dollars_spent'
    ]
    assert len(demo) == 5
    assert list(demo.isna().sum(axis=0)) == [1, 1, 1, 0, 1]


def test_get_demo_many_rows():
    demo = get_demo(10)

    login_dates = pd.Series([
        '2021-06-26', '2021-02-10', 'NaT', '2020-09-26', '2020-12-22', '2019-11-27',
        '2002-05-10', '2014-10-04', '2014-03-19', '2015-09-13'
    ], dtype='datetime64[ns]')
    email_optin = [False, False, False, True, np.nan, np.nan, False, True, False, False]
    credit_card = [
        'VISA', 'VISA', 'AMEX', np.nan, 'DISCOVER', 'AMEX', 'AMEX', 'DISCOVER', 'DISCOVER', 'VISA'
    ]
    age = [29, 18, 21, 45, 32, 50, 93, 75, 39, 66]
    dollars_spent = [99.99, np.nan, 2.50, 25.00, 19.99, 52.48, 39.99, 4.67, np.nan, 23.28]

    expected = pd.DataFrame({
        'last_login': login_dates,
        'email_optin': email_optin,
        'credit_card': credit_card,
        'age': age,
        'dollars_spent': dollars_spent
    })

    pd.testing.assert_frame_equal(demo, expected)
