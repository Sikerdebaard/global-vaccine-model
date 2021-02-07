from covid19.model_country import model_country
import argparse


def main(alpha3):
    df_model = model_country(alpha3)

    from pprint import pprint
    pprint(df_model)


def cli():
    parser = argparse.ArgumentParser(description='Estimate number of vaccinated people for a single country')

    parser.add_argument('--alpha3', required=True, help='The ISO alpha 3 notation for the country')

    args = parser.parse_args()
    main(**vars(args))
