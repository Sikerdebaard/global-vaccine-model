import pandas as pd
from functools import lru_cache
import urllib.parse


@lru_cache(maxsize=None)
def locations():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/locations.csv',
        index_col='iso_code')

    assert df is not None
    assert df.shape[0] > 0

    return df


@lru_cache(maxsize=None)
def vaccinations():
    df = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv')

    assert df is not None
    assert df.shape[0] > 0

    return df


@lru_cache(maxsize=None)
def country_vaccines_in_use(*args, **kwargs):
    df_country_data = country_data(*args, **kwargs)

    vaccines = set()
    for vaccstr in df_country_data['vaccine'].unique():
        for vaccine in vaccstr.split(','):
            vaccines.add(vaccine.strip())

    return vaccines


@lru_cache(maxsize=None)
def country_data(alpha3=None, name=None):
    assert alpha3 is not None or name is not None

    if name is None:
        df_locations = locations()
        name = df_locations.loc[alpha3]['location']

    url = f'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/{urllib.parse.quote(name)}.csv'
    print(url)
    df = pd.read_csv(url, index_col='date')

    assert df is not None
    assert df.shape[0] > 0

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    return df


@lru_cache(maxsize=None)
def vaccinations_for_country(alpha3=None, name=None):
    assert alpha3 is not None or name is not None

    if alpha3 is None:
        df_locations = locations()
        alpha3 = df_locations[df_locations['location'] == name].index.values[0]

    df_owid = vaccinations()

    df_country = df_owid[df_owid['iso_code'] == alpha3].copy()

    assert df_country.shape[0] > 0

    df_country.set_index('date', inplace=True)
    df_country.index = pd.to_datetime(df_country.index)
    df_country.sort_index(inplace=True)

    return df_country


@lru_cache(maxsize=None)
def vaccines():
    df = locations()

    assert 'vaccines' in df.columns

    vaccinelist = set()

    for vaccines in df['vaccines'].unique():
        for vaccine in vaccines.split(','):
            vaccinelist.add(vaccine.strip())

    assert len(vaccinelist) > 0

    return list(vaccinelist)


@lru_cache(maxsize=None)
def vaccine_mappings():
    df = pd.read_csv('data/vaccine/owid_mappings.csv', index_col='owid')

    assert df is not None
    assert df.shape[0] > 0

    return df


@lru_cache(maxsize=None)
def owid_vaccine_to_vaccine_name(owid_vaccine):
    df = vaccine_mappings()

    assert owid_vaccine in df.index

    return df.loc[owid_vaccine]['ours']
