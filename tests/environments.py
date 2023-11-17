import pytest
import airball

################################################
################################################
##########  INITIALIZATION TESTS  ##############
################################################
################################################

def test_preset_environment_initialization():
    oc = airball.OpenCluster()
    ln = airball.LocalNeighborhood()
    gc = airball.GlobularCluster()
    gb = airball.GalacticBulge()
    gk = airball.GalacticCore()