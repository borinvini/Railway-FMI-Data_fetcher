import os
import pandas as pd
import pytest
from unittest.mock import patch

from src.processors.DataLoader import DataLoader


WEATHER_OBS = {
    'closest_ems': 'Helsinki Kaisaniemi',
    'Air temperature': -5.0,
    'Snow depth': 15.0,
}

MATCHED_STOP_1 = {
    'stationName': 'Helsinki asema',
    'stationShortCode': 'HKI',
    'stationUICCode': 1,
    'countryCode': 'FI',
    'type': 'DEPARTURE',
    'trainStopping': True,
    'commercialStop': True,
    'commercialTrack': '9',
    'cancelled': False,
    'scheduledTime': '2024-01-01T04:57:00.000Z',
    'actualTime': '2024-01-01T04:57:21.000Z',
    'differenceInMinutes': 0,
    'differenceInMinutes_offset': 0,
    'differenceInMinutes_eachStation_offset': 0,
    'causes': [],
    'trainReady': None,
    'weather_observations': WEATHER_OBS,
}

MATCHED_STOP_2 = {
    'stationName': 'Tampere asema',
    'stationShortCode': 'TPE',
    'stationUICCode': 160,
    'countryCode': 'FI',
    'type': 'ARRIVAL',
    'trainStopping': True,
    'commercialStop': True,
    'commercialTrack': '2',
    'cancelled': False,
    'scheduledTime': '2024-01-01T06:57:00.000Z',
    'actualTime': '2024-01-01T06:57:21.000Z',
    'differenceInMinutes': 2,
    'differenceInMinutes_offset': 2,
    'differenceInMinutes_eachStation_offset': 2,
    'causes': [],
    'trainReady': None,
    'weather_observations': {},
}

MATCHED_TRAIN_ROW = {
    'trainNumber': 1,
    'departureDate': '2024-01-01',
    'operatorUICCode': 10,
    'operatorShortCode': 'vr',
    'trainType': 'IC',
    'trainCategory': 'Long-distance',
    'commuterLineID': None,
    'runningCurrently': False,
    'cancelled': False,
    'version': 123456,
    'timetableType': 'REGULAR',
    'timetableAcceptanceDate': '2023-11-02T05:57:22.000Z',
    'timeTableRows': str([MATCHED_STOP_1, MATCHED_STOP_2]),
}


@pytest.fixture
def loader(tmp_path):
    with patch.object(DataLoader, '_check_data_folder'):
        dl = DataLoader()
    dl.output_folder = str(tmp_path)
    dl.data_folder = str(tmp_path)
    return dl


def make_matched_csv(tmp_path, filename, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


class TestConvertMatchedToFlat:

    def test_output_row_count_equals_total_stops(self, loader, tmp_path):
        make_matched_csv(tmp_path, 'matched_data_2024_01.csv', [MATCHED_TRAIN_ROW])

        loader.convert_matched_to_flat()

        flat_path = os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv')
        assert os.path.exists(flat_path)
        flat_df = pd.read_csv(flat_path)
        assert len(flat_df) == 2  # MATCHED_STOP_1 + MATCHED_STOP_2

    def test_train_level_columns_repeated_for_each_stop(self, loader, tmp_path):
        make_matched_csv(tmp_path, 'matched_data_2024_01.csv', [MATCHED_TRAIN_ROW])

        loader.convert_matched_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv'))
        assert list(flat_df['trainNumber']) == [1, 1]
        assert list(flat_df['trainType']) == ['IC', 'IC']
        assert list(flat_df['departureDate']) == ['2024-01-01', '2024-01-01']

    def test_stop_cancelled_renamed_from_cancelled(self, loader, tmp_path):
        make_matched_csv(tmp_path, 'matched_data_2024_01.csv', [MATCHED_TRAIN_ROW])

        loader.convert_matched_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv'))
        assert 'stop_cancelled' in flat_df.columns
        assert list(flat_df['stop_cancelled']) == [False, False]

    def test_weather_observations_flattened_to_columns(self, loader, tmp_path):
        make_matched_csv(tmp_path, 'matched_data_2024_01.csv', [MATCHED_TRAIN_ROW])

        loader.convert_matched_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv'))
        # MATCHED_STOP_1 has weather data
        assert 'closest_ems' in flat_df.columns
        assert 'Air temperature' in flat_df.columns
        assert 'Snow depth' in flat_df.columns
        assert flat_df['closest_ems'].iloc[0] == 'Helsinki Kaisaniemi'
        assert flat_df['Air temperature'].iloc[0] == -5.0
        assert flat_df['Snow depth'].iloc[0] == 15.0

    def test_stop_without_weather_observations_produces_nan(self, loader, tmp_path):
        make_matched_csv(tmp_path, 'matched_data_2024_01.csv', [MATCHED_TRAIN_ROW])

        loader.convert_matched_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv'))
        # MATCHED_STOP_2 has empty weather_observations — weather cols should be NaN
        assert pd.isna(flat_df['closest_ems'].iloc[1])
        assert pd.isna(flat_df['Air temperature'].iloc[1])

    def test_weather_observations_not_a_column(self, loader, tmp_path):
        make_matched_csv(tmp_path, 'matched_data_2024_01.csv', [MATCHED_TRAIN_ROW])

        loader.convert_matched_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv'))
        assert 'weather_observations' not in flat_df.columns

    def test_skips_existing_flat_file(self, loader, tmp_path):
        make_matched_csv(tmp_path, 'matched_data_2024_01.csv', [MATCHED_TRAIN_ROW])

        flat_path = os.path.join(str(tmp_path), 'matched_data_flat_2024_01.csv')
        pd.DataFrame([{'sentinel': 'do_not_overwrite'}]).to_csv(flat_path, index=False)
        mtime_before = os.path.getmtime(flat_path)

        loader.convert_matched_to_flat()

        assert os.path.getmtime(flat_path) == mtime_before
