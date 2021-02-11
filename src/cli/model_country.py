from covid19.model_country import model_country
import argparse
from pathlib import Path


def main(alpha3, out):
    out = Path(out)

    if not out.exists():
        print('Output directory does not exist')
        return 1

    df_model = model_country(alpha3, out)


def cli():
    parser = argparse.ArgumentParser(description='Estimate number of vaccinated people for a single country')

    parser.add_argument('--alpha3', required=True, help='The ISO alpha 3 notation for the country')
    parser.add_argument('--out', required=True, help='The output dir for csv, images etc.')

    args = parser.parse_args()
    main(**vars(args))
