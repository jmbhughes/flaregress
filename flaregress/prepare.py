from .io import GOESXrayRetriever
from urllib.error import HTTPError
import numpy as np


class DataChunker:
    """
    For a given flare event, request all necessary data, align different cadences, and pick from samples
    """

    def __init__(self, num_historical_goes=5):
        self.goes_k = num_historical_goes

    def chunk(self, streams, events, verbose=True):
        """
        Divide data stream up into small flare sections
        :param streams: list of data streams to pull from
        :type streams: list
        :param events: set of events to chunk on
        :type events: pd.DataFrame
        :param verbose: whether to output a progress estimate while processing
        :type verbose: boolean
        :return: a pair of chunked inputs and outputs
        :rtype: (np.ndarray, np.ndarray)
        """
        if len(streams) > 1:
            raise NotImplementedError("Currently only supports single GOES stream")
        if type(streams[0]) != GOESXrayRetriever:
            raise NotImplementedError("Currently only supports GOES streams")

        gxr = streams[0]
        X, Y = [], []
        count = 0
        for index, event in events.iterrows():
            if verbose:
                print(count)

            try:
                data = gxr.retrieve(event['time_start'], event['time_end'])
            except (Exception, HTTPError):  # for some reason it couldn't fetch so ignore it
                pass
            else:
                times, xrsb = data.index, data['xrsb'].values
                for i in range(self.goes_k, len(times)):
                    X.append(xrsb[i - self.goes_k:i])
                    Y.append(xrsb[i])
            count += 1
        return np.array(X), np.array(Y)