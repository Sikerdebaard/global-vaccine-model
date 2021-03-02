from covid19.strategy.owid_doses_administered import strategy_doses_per_vaccine
import argparse
from pathlib import Path
import pandas as pd


def main(alpha3, out, countrydata, dose_projections, title, subtitle):
    out = Path(out)

    if not out.exists():
        print('Output directory does not exist')
        return 1

    df_country = None
    if countrydata:
        df_country = pd.read_csv(countrydata, index_col=0)
        df_country.index = pd.to_datetime(df_country.index)

    df_doses_by_vaccine = pd.read_csv(dose_projections, index_col=0)

    df_model = strategy_doses_per_vaccine(alpha3, out, df_country=df_country, df_doses_by_vaccine=df_doses_by_vaccine, title=title, subtitle=subtitle)


def cli():
    parser = argparse.ArgumentParser(description='Project number of vaccinated people for a single country')

    parser.add_argument('--alpha3', required=True, help='The ISO alpha 3 notation for the country')
    parser.add_argument('--countrydata', required=True, help='Required country-level data')
    parser.add_argument('--dose-projections', required=True, help='Dose projections')
    parser.add_argument('--out', required=True, help='The output dir for csv, images etc.')

    parser.add_argument('--title', required=False, default=None, help='Chart title.')
    parser.add_argument('--subtitle', required=False, default=None, help='Chart subtitle.')

    args = parser.parse_args()
    main(**vars(args))
