import pytest
from covid19.utils.owid import locations, vaccines, vaccine_mappings


@pytest.fixture
def owid_locations():
    return locations()

@pytest.fixture
def owid_vaccines():
    return vaccines()

@pytest.fixture
def owid_vaccine_mappings():
    return vaccine_mappings()

