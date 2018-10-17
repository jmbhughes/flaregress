from abc import ABC, abstractmethod
import urllib.request
from datetime import datetime
from collections import namedtuple
import pandas as pd
import os


class DataRetriever(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def retrieve(self, start_time, end_time):
        pass


class GOESXrayRetriever(DataRetriever):
    def __init__(self):
        DataRetriever.__init__(self)

    def retrieve(self, start_time, end_time):
        pass


class ListHandler(ABC):
    """
    A generic handler that fetches lists of flares, allows them to be saved to file, and then reloaded
    """
    def __init__(self, url=None):
        self.url = url
        self.contents = None

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def fetch(self):
        pass

    @abstractmethod
    def save(self, path):
        pass


class RhessiListHandler(ListHandler):
    def __init__(self, url="https://hesperia.gsfc.nasa.gov/hessidata/dbase/hessi_flare_list.txt"):
        ListHandler.__init__(self, url)

    def fetch(self):
        """
        fetch the Rhessi table from the specified url
        :return: processed list
        :rtype: pd.DataFrame
        """
        # fetch the page contents
        page = urllib.request.urlopen(self.url).read()
        lines = page.decode("utf-8").split("\n")
        start_line, end_line = 7, -41  # lines[7:-41] contain the table while everything else is comments

        FlareRecord = namedtuple("FlareRecord",
                                 ["number", "time_start", "time_peak", "time_end", "duration",
                                  "max_rate", "total_counts", "pos_x", "pos_y", "radial", "ar"])

        def process_line(line):
            """
            Process a single entry in the Rhessi table into a clean format
            :param line: a line in the rhessi table
            :type line: str
            :return: a clean record of the Rhessi table line with entries converted to the appropriate types,
                    e.g. dates are in DateTime and numbers are ints
            :rtype: FlareRecord
            """
            month_numbers = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                             "Jul": 7,
                             "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
            entries = line.split()

            # the entry number
            number = int(entries[0])

            # process start date
            start_date = entries[1]
            start_time = entries[2]
            day, month, year = start_date.split("-")
            hour, minute, second = start_time.split(":")
            start = datetime(int(year), month_numbers[month], int(day), int(hour), int(minute), int(second))

            # process peak time
            peak_time = entries[3]
            hour, minute, second = peak_time.split(":")
            peak = datetime(int(year), month_numbers[month], int(day), int(hour), int(minute), int(second))

            # process end time
            end_time = entries[4]
            hour, minute, second = end_time.split(":")
            end = datetime(int(year), month_numbers[month], int(day), int(hour), int(minute), int(second))

            # process extra entry contents
            duration = int(entries[5])
            max_rate = int(entries[6])
            total = int(entries[7])
            energy = entries[8]
            pos_x = int(entries[9])
            pos_y = int(entries[10])
            radial = int(entries[11])
            ar = entries[12]
            return FlareRecord(number, start, peak, end, duration, max_rate, total, pos_x, pos_y, radial, ar)

        records = [process_line(line) for line in lines[start_line:end_line]]
        df = pd.DataFrame(records)
        self.contents = df
        return df

    def load(self, path):
        """
        Loads a saved list from drive
        :param path: location of list
        :type path: str
        :return: loaded list
        :rtype: pd.DataFrame
        """
        if os.path.isfile(path):
            self.contents = pd.read_csv(path, parse_dates = ['time_start', 'time_peak', 'time_end'])
            return self.contents
        else:
            raise FileNotFoundError("{} was not found".format(path))

    def save(self, path):
        """
        Saves a list to file
        :param path: where to save
        :type path: str
        """
        if self.contents is not None:
            pd.to_csv(path, index=False)
        else:
            raise RuntimeError("A list must be fetched or loaded before saving")