import pytest

from src.data.loader import DataValidationError, SessionDataLoader


def test_load_real_dataset(raw_df):
    assert raw_df.shape == (12000, 18)
    assert raw_df["Converted"].isin([0, 1]).all()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SessionDataLoader(path=tmp_path / "missing.csv").load()


def test_missing_column_raises(raw_df, tmp_path):
    p = tmp_path / "d.csv"
    raw_df.drop(columns=["PageValues"]).to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="PageValues"):
        SessionDataLoader(path=p).load()


def test_non_binary_target_raises(raw_df, tmp_path):
    p = tmp_path / "d.csv"
    bad = raw_df.copy()
    bad.loc[0, "Converted"] = 7
    bad.to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="binary"):
        SessionDataLoader(path=p).load()


def test_empty_file_raises(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("")
    with pytest.raises(DataValidationError):
        SessionDataLoader(path=p).load()


def test_split_is_stratified_and_reproducible(raw_df):
    loader = SessionDataLoader()
    X_tr, X_te, y_tr, y_te = loader.split(raw_df)
    assert len(X_te) == 2400
    assert abs(y_tr.mean() - y_te.mean()) < 0.005
    assert "Converted" not in X_tr.columns
    X_tr2, *_ = loader.split(raw_df)
    assert X_tr.index.equals(X_tr2.index)
