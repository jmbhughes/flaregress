import numpy as np
from sunpy.sun import solar_cycle_number
import pandas as pd


class DataChunker:
    """
    For a given flare event, request all necessary data, align different cadences, and pick from samples
    """

    def __init__(self, database, streams=('meta', 'f107', 'xrs'), num_historical_goes=5):
        self.goes_k = num_historical_goes
        self.database = database.entries
        self.streams = streams

    def chunk(self, test_stream='xrsa'):
        """
        Divide data stream up into small flare sections
        :return: a pair of chunked inputs and outputs
        :rtype: (np.ndarray, np.ndarray)
        """

        chunks = []
        test = []
        for entry in self.database:
            for chunk_start_index in range(entry['xrs'].shape[0] - self.goes_k):
                chunk = dict()
                if 'f107' in self.streams:
                    if entry['f107'].values.shape[0] > 0:
                        chunk['f107'] = np.mean(entry['f107'].values)
                    else:
                        chunk['f017'] = np.nan

                if 'meta' in self.streams:
                    chunk['time_start'] = entry['meta'].time_start.timestamp()
                    chunk['pos_x'] = entry['meta'].pos_x
                    chunk['pos_y'] = entry['meta'].pos_y
                    chunk['solar_cycle_number'] = solar_cycle_number(entry['meta']['time_start'])
                    chunk['time_diff'] = (entry['xrs'].index[chunk_start_index] -
                                          entry['meta'].time_start).total_seconds()

                if 'xrs' in self.streams:
                    for i, index in enumerate(range(chunk_start_index, chunk_start_index + self.goes_k)):
                        chunk['xrsa_{:05d}'.format(i)] = entry['xrs']['xrsa'][index]
                        chunk['xrsb_{:05d}'.format(i)] = entry['xrs']['xrsb'][index]
                    test.append(entry['xrs'][test_stream][chunk_start_index + self.goes_k:].values)
                chunks.append(chunk)
        return pd.DataFrame(chunks), test
