from rdt import get_demo


def test_get_demo():
    demo = get_demo()

    assert list(demo.columns) == [
        'last_login', 'email_optin', 'credit_card', 'age', 'dollars_spent']
    assert len(demo) == 5
    assert list(demo.isna().sum(axis=0)) == [1, 1, 1, 0, 1]
