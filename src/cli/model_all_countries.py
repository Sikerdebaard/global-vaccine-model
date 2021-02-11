import argparse
from cli.model_country import main as model_country
from covid19.utils.country import all_countries_alpha3
import pycountry


def main(out):
    countries = all_countries_alpha3()

    for country in countries:
        print(f'Working on {pycountry.countries.get(alpha_3=country).name}')
        model_country(country, out)


def cli():
    parser = argparse.ArgumentParser(description='Estimate number of vaccinated people for all countries')

    parser.add_argument('--out', required=True, help='The output dir for csv, images etc.')

    args = parser.parse_args()
    main(**vars(args))
