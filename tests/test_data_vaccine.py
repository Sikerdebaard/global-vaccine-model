import pytest
import warnings
from fixtures.owid import owid_vaccines, owid_vaccine_mappings

from covid19.utils.vaccine import metadata_for_vaccine
from pathlib import Path


def _check_path(path):
    if not path.exists():
        warnings.warn(f'File/folder {path} does not exist')
        return False
    return True


def test_owid_mappings(owid_vaccines, owid_vaccine_mappings):
    print(owid_vaccine_mappings)
    for vaccine in owid_vaccines:
        assert vaccine in owid_vaccine_mappings.index, f'{vaccine} not in owid vaccine mappings {owid_vaccine_mappings.index}'

    for vaccine in owid_vaccine_mappings.index:
        assert vaccine in owid_vaccines, f'{vaccine} not in owid vaccine list {owid_vaccines}'


def test_owid_file_folder_mappings(owid_vaccine_mappings):
    for folder in owid_vaccine_mappings['ours'].values:
        vaccine_folder = (Path('data/vaccine') / folder)

        _check_path(vaccine_folder)

def test_vaccine_metadata(owid_vaccine_mappings):
    for vaccine in owid_vaccine_mappings['ours'].values:
        vaccine_folder = (Path('data/vaccine') / vaccine)
        metadata_file = vaccine_folder / 'metadata.csv'

        if _check_path(metadata_file):
            df_meta = metadata_for_vaccine(vaccine)

            assert 'doses' in df_meta.index

            if int(df_meta['value'].loc['doses']) > 1:
                assert 'interval_between_doses' in df_meta.index
