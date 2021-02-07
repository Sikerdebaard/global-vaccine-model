import argparse
from pathlib import Path

from covid19.utils.owid import locations, country_data
from covid19.utils.country import country_alpha3_to_name

import pandas as pd


basepath = Path('data') / 'locations'
basepath.mkdir(exist_ok=True)

def main():
    df_countries = locations()
    for alpha3 in df_countries.index.dropna():
        name = country_alpha3_to_name(alpha3=alpha3)

        if not name:
            countrydir = basepath / str(df_countries.loc[alpha3]['location']).lower().strip()
            countrydir.mkdir()

            df_country_data = country_data(alpha3)

            df_metadata = pd.DataFrame(columns=['name', 'value', 'comment']).set_index('name')
            df_metadata.loc['iso_alpha3'] = {'value': alpha3, 'comment': 'AUTOMATICALLY IMPORTED'}
            df_metadata.loc['date_first_dose_administered'] = {'value': df_country_data.index[0].strftime('%Y-%m-%d'), 'comment': 'AUTOMATICALLY IMPORTED'}

            df_metadata.to_csv(countrydir / 'metadata.csv')


def cli():
    parser = argparse.ArgumentParser(description='Add missing countries to the countries dataset')

    args = parser.parse_args()
    main(**vars(args))
