"""Shared pytest configuration for the STRATA test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long acceptance test (deselect with -m 'not slow')"
    )
