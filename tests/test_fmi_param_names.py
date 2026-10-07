"""fetch_fmi_data must not keep responses whose parameter names are empty or unknown.

On 2025-04-29 16:00-19:00 FMI served three parameters with an empty name. The
per-row dict collapsed them into one '' key (later written as 'Unnamed: 15'),
so Horizontal visibility and Cloud amount were lost for that window.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

_GOOD = {"Air temperature": {"value": 1.0}, "Cloud amount": {"value": 3.0}}
_BAD = {"Air temperature": {"value": 1.0}, "": {"value": 7.0}}


def _response(variables):
    data = {datetime(2024, 1, 1, 0): {"Helsinki": variables}}
    return SimpleNamespace(data=data, location_metadata={"Helsinki": {"fmisid": 1}})


def _fetch(responses):
    from src.fetchers.FMI import FMIDataFetcher
    calls = iter(responses)
    with patch("src.fetchers.FMI.download_stored_query", lambda *a, **k: next(calls)), \
         patch("src.fetchers.FMI.time.sleep"):
        return FMIDataFetcher().fetch_fmi_data(
            "18,55,35,75", datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1))[0]


def test_empty_parameter_name_is_retried():
    df = _fetch([_response(_BAD), _response(_GOOD)])
    assert "" not in df.columns
    assert df.loc[0, "Cloud amount"] == 3.0


def test_persistent_bad_names_are_kept_after_last_attempt(capsys):
    df = _fetch([_response(_BAD)] * 3)
    assert len(df) == 1
    assert "unexpected parameter names" in capsys.readouterr().out
