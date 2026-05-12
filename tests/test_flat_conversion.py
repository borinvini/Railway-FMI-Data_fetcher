import ast
import os
import pandas as pd
import pytest
from unittest.mock import patch

from src.processors.DataLoader import DataLoader


STOP_1 = {
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
    'causes': [],
    'trainReady': {'source': 'KUPLA', 'accepted': True, 'timestamp': '2024-01-01T04:51:10.000Z'},
}

STOP_2 = {
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
    'causes': [{'categoryCode': 'X1'}],
    'trainReady': None,
}

TRAIN_ROW = {
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
    'timeTableRows': str([STOP_1, STOP_2]),
}


@pytest.fixture
def loader(tmp_path):
    with patch.object(DataLoader, '_check_data_folder'):
        dl = DataLoader()
    dl.output_folder = str(tmp_path)
    dl.data_folder = str(tmp_path)
    return dl


def make_train_csv(tmp_path, filename, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


class TestConvertTrainsToFlat:

    def test_output_row_count_equals_total_stops(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_path = os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv')
        assert os.path.exists(flat_path)
        flat_df = pd.read_csv(flat_path)
        assert len(flat_df) == 2  # STOP_1 + STOP_2

    def test_train_level_columns_repeated_for_each_stop(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv'))
        assert list(flat_df['trainNumber']) == [1, 1]
        assert list(flat_df['trainType']) == ['IC', 'IC']
        assert list(flat_df['departureDate']) == ['2024-01-01', '2024-01-01']

    def test_stop_cancelled_renamed_from_cancelled(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv'))
        assert 'stop_cancelled' in flat_df.columns
        assert list(flat_df['stop_cancelled']) == [False, False]

    def test_causes_stored_as_string(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv'))
        assert flat_df['causes'].dtype == object  # stored as string
        assert flat_df['causes'].iloc[0] == '[]'
        assert flat_df['causes'].iloc[1] == "[{'categoryCode': 'X1'}]"

    def test_train_ready_stored_as_string_or_none(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        loader.convert_trains_to_flat()

        flat_df = pd.read_csv(os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv'))
        # STOP_1 has trainReady dict, STOP_2 has None
        assert isinstance(flat_df['trainReady'].iloc[0], str)
        assert pd.isna(flat_df['trainReady'].iloc[1])

    def test_skips_existing_flat_file(self, loader, tmp_path):
        train_file = make_train_csv(tmp_path, 'all_trains_data_2024_01.csv', [TRAIN_ROW])
        loader.train_files = [train_file]

        # Pre-create flat file with known content
        flat_path = os.path.join(str(tmp_path), 'all_trains_data_flat_2024_01.csv')
        pd.DataFrame([{'sentinel': 'do_not_overwrite'}]).to_csv(flat_path, index=False)
        mtime_before = os.path.getmtime(flat_path)

        loader.convert_trains_to_flat()

        assert os.path.getmtime(flat_path) == mtime_before  # file not touched
