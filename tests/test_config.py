import pytest

from src.config.settings import ConfigError, load_settings


def test_default_config_loads():
    s = load_settings()
    assert s.target == "Converted"
    assert len(s.input_columns) == 17
    assert set(s.id_categorical_columns) <= set(s.categorical_columns)


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "nope.yaml")


def test_malformed_config_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed")
    with pytest.raises(ConfigError):
        load_settings(bad)


def test_empty_config_raises(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ConfigError):
        load_settings(empty)


def test_missing_key_raises():
    s = load_settings()
    with pytest.raises(ConfigError):
        s.section("does_not_exist")
