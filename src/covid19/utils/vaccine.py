from pathlib import Path

import pandas as pd


def metadata_for_vaccine(vaccine_name):
    metadata_file = Path('data') / 'vaccine' / f'{vaccine_name}' / 'metadata.csv'

    assert metadata_file.exists(), metadata_file.absolute()

    df_meta = pd.read_csv(metadata_file, index_col='name')

    assert df_meta is not None
    assert df_meta.shape[0] > 0

    return df_meta
