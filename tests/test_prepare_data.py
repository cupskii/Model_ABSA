import pandas as pd
from unittest.mock import patch

from pipeline.prepare_data import (
    compute_sentiment_proportions,
    compute_split_sentiment_proportions,
)
from preprocessing.preprocessing_functions import FINAL_ASPECTS


def test_compute_sentiment_proportions_includes_none_in_denominator():
    df = pd.DataFrame({
        'Content Quality': [-1, 0, 1, 1, None],
        'UI/UX': [None, None, -1, 1, None],
    })

    result = compute_sentiment_proportions(df)

    content = result['Content Quality']
    assert content['n_total'] == 5
    assert content['n_labeled'] == 4
    assert content['n_missing'] == 1
    assert content['sentiments'] == {
        'Negatif': {'count': 1, 'percentage': 20.0},
        'Netral': {'count': 1, 'percentage': 20.0},
        'Positif': {'count': 2, 'percentage': 40.0},
        'None': {'count': 1, 'percentage': 20.0},
    }

    ui_ux = result['UI/UX']
    assert ui_ux['n_total'] == 5
    assert ui_ux['n_labeled'] == 2
    assert ui_ux['n_missing'] == 3
    assert ui_ux['sentiments']['Netral'] == {
        'count': 0,
        'percentage': 0.0,
    }
    assert ui_ux['sentiments']['None'] == {
        'count': 3,
        'percentage': 60.0,
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
    assert result['test']['Content Quality']['sentiments']['Positif'] == {
        'count': 1,
        'percentage': 50.0,
    }
    assert result['test']['Content Quality']['sentiments']['None'] == {
        'count': 1,
        'percentage': 50.0,
    }


def test_prepare_data_preserves_preassigned_splits(tmp_path):
    from pipeline.prepare_data import prepare_data

    rows = 20
    df = pd.DataFrame({
        'Komentar': [f'ulasan {i}' for i in range(rows)],
        '__split': ['train'] * 14 + ['val'] * 3 + ['test'] * 3,
    })
    for aspect in FINAL_ASPECTS:
        df[aspect] = [(-1, 0, 1, None)[i % 4] for i in range(rows)]
    path = tmp_path / 'versioned.csv'
    df.to_csv(path, index=False)
    config = {
        'data': {
            'path': str(path),
            'text_column': 'Komentar',
            'split': {'train_ratio': .70, 'val_ratio': .15, 'random_state': 42},
        },
        'preprocessing': {
            'remove_emoji': False,
            'lowercase': False,
            'remove_url_mention': False,
            'compress_repeated_chars': False,
            'remove_special_chars': False,
            'normalize_slang': False,
            'remove_stopwords': False,
        },
    }

    with patch('pipeline.prepare_data.stratified_split') as split_mock:
        result = prepare_data(config)

    split_mock.assert_not_called()
    assert len(result['df_train']) == 14
    assert len(result['df_val']) == 3
    assert len(result['df_test']) == 3
