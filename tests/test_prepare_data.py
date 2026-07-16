import pandas as pd

from pipeline.prepare_data import (
    compute_sentiment_proportions,
    compute_split_sentiment_proportions,
)


def test_compute_sentiment_proportions_ignores_missing_in_denominator():
    df = pd.DataFrame({
        'Content Quality': [-1, 0, 1, 1, None],
        'UI/UX': [None, None, -1, 1, None],
    })

    result = compute_sentiment_proportions(df)

    content = result['Content Quality']
    assert content['n_labeled'] == 4
    assert content['n_missing'] == 1
    assert content['sentiments'] == {
        'Negatif': {'count': 1, 'percentage': 25.0},
        'Netral': {'count': 1, 'percentage': 25.0},
        'Positif': {'count': 2, 'percentage': 50.0},
    }

    ui_ux = result['UI/UX']
    assert ui_ux['n_labeled'] == 2
    assert ui_ux['n_missing'] == 3
    assert ui_ux['sentiments']['Netral'] == {
        'count': 0,
        'percentage': 0.0,
    }


def test_compute_sentiment_proportions_skips_absent_aspects():
    result = compute_sentiment_proportions(pd.DataFrame({'Komentar': ['bagus']}))

    assert result == {}


def test_compute_split_sentiment_proportions_separates_each_split():
    train = pd.DataFrame({'Content Quality': [-1, -1, 1]})
    val = pd.DataFrame({'Content Quality': [0, 1]})
    test = pd.DataFrame({'Content Quality': [1, None]})

    result = compute_split_sentiment_proportions(train, val, test)

    assert set(result) == {'train', 'val', 'test'}
    assert result['train']['Content Quality']['sentiments']['Negatif'] == {
        'count': 2,
        'percentage': 66.67,
    }
    assert result['val']['Content Quality']['sentiments']['Netral'] == {
        'count': 1,
        'percentage': 50.0,
    }
    assert result['test']['Content Quality']['n_labeled'] == 1
    assert result['test']['Content Quality']['n_missing'] == 1
