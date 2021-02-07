import pandas as pd
from functools import lru_cache
from pathlib import Path


country_dir = Path('data') / 'locations'

@lru_cache(maxsize=None)
def country_minmax_doses_administered_for_vaccine(vaccine, df_country, alpha3=None, country_name=None):
    assert alpha3 is not None or country_name is not None
    if not country_name:
        country_name = country_alpha3_to_name(alpha3)

    doses_received_file = country_dir / country_name / 'doses_received.csv'

    if doses_received_file.exists():
        pass




@lru_cache(maxsize=None)
def country_startdate(alpha3=None, country_name=None):
    assert alpha3 is not None or country_name is not None
    if not country_name:
        country_name = country_alpha3_to_name(alpha3)

    df_meta = country_metadata(country_name=country_name)

    if 'date_first_dose_administered' in df_meta.index:
        return df_meta.loc['date_first_dose_administered']['value']

    return None


@lru_cache(maxsize=None)
def country_vaccine_regimen(alpha3=None, country_name=None):
    assert alpha3 is not None or country_name is not None
    if not country_name:
        country_name = country_alpha3_to_name(alpha3)

    regimen_file = country_dir / country_name / 'vaccine_regimen.csv'

    if not regimen_file.exists():
        return None

    df_regimen = pd.read_csv(regimen_file, index_col='date')

    assert df_regimen is not None
    assert df_regimen.shape[0] > 0

    df_regimen.index = pd.to_datetime(df_regimen.index)
    df_regimen.sort_index(inplace=True)

    return df_regimen


@lru_cache(maxsize=None)
def country_metadata(alpha3=None, country_name=None):
    assert alpha3 is not None or country_name is not None
    if not country_name:
        country_name = country_name_to_alpha3(country_name)


    metadata_file = country_dir / country_name / 'metadata.csv'

    assert metadata_file.exists(), f'{metadata_file.absolute()}'

    df_country = pd.read_csv(metadata_file, index_col='name')

    assert df_country is not None
    assert df_country.shape[0] > 0

    return df_country


@lru_cache(maxsize=None)
def country_name_to_alpha3(name):
    for country in country_dir.glob('*/metadata.csv'):
        if country.parent.name != name:
            continue

        df_country = pd.read_csv(country, index_col='name')

        assert 'iso_alpha3' in df_country.index

        return df_country.loc['iso_alpha3']['value']


@lru_cache(maxsize=None)
def country_alpha3_to_name(alpha3):
    for country in country_dir.glob('*/metadata.csv'):
        df_country = pd.read_csv(country, index_col='name')

        assert 'iso_alpha3' in df_country.index

        if df_country.loc['iso_alpha3']['value'].upper() == alpha3.upper():
            return country.parent.name
