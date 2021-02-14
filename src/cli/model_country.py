from covid19.model_country import model_country
import argparse
from pathlib import Path
import pandas as pd


def main(alpha3, out, countrydata, title, subtitle):
    out = Path(out)

    if not out.exists():
        print('Output directory does not exist')
        return 1

    df_country = None
    if countrydata:
        df_country = pd.read_csv(countrydata, index_col=0)
        df_country.index = pd.to_datetime(df_country.index)

    df_model = model_country(alpha3, out, df_country=df_country, title=title, subtitle=subtitle)


def cli():
    parser = argparse.ArgumentParser(description='Estimate number of vaccinated people for a single country')

    parser.add_argument('--alpha3', required=True, help='The ISO alpha 3 notation for the country')
    parser.add_argument('--countrydata', required=False, default=None, help='URL to CSV with custom country data. This overrules the OWID data.')
    parser.add_argument('--out', required=True, help='The output dir for csv, images etc.')

    parser.add_argument('--title', required=False, default=None, help='Chart title.')
    parser.add_argument('--subtitle', required=False, default=None, help='Chart subtitle.')

    args = parser.parse_args()
    main(**vars(args))
