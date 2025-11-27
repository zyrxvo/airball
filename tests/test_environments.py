import pytest

import airball

################################################
# INITIALIZATION TESTS  ##############
################################################


@pytest.mark.parametrize(
    "env",
    (
        airball.OpenCluster(),
        airball.LocalNeighborhood(),
        airball.GlobularCluster(),
        airball.GalacticBulge(),
        airball.GalacticCore(),
    ),
)
def test_preset_environment_initialization(env):
    assert env is not None
