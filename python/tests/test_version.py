from importlib.metadata import version

import tilefish


def test_version_matches_the_package_metadata():
    assert tilefish.__version__ == version("tilefish")
