from abc import ABC, abstractmethod
import urllib.request
from datetime import datetime
from collections import namedtuple
import os
import sunpy.lightcurve
from sunpy.time import TimeRange
from datetime import timedelta
from sunpy.instr.goes import get_goes_event_list
import math
from multiprocessing import Pool
import pandas as pd
import deepdish as dd


class DatabaseHandler:
    """
    Handles fetching all data and saving it for future usage
    """
    def __init__(self):
        self.entries = []

    def build(self, flare_list, retrievers):
        for index, event in flare_list.iterrows():
            self.entries.append({retriever.name: retriever.retrieve(event.time_start, event.time_end)
                                 for retriever in retrievers})
            self.entries[-1]['meta'] = event

    def save(self, path):
        dd.io.save(path, self.entries)

    def load(self, path):
        self.entries = dd.io.load(path)


class DataRetriever(ABC):
    """
    A generic interface to retrieve data
    """
    def __init__(self, database_path, save_directory):
        """
        Create a data retriever
        :param database_path: where the CSV database indicating where previously fetched files is stored
        :param save_directory: where all the files that are fetched for the first time should be saved
        """
        self.name = None

        # handle the database
        self.database_path = os.path.expandvars(os.path.expanduser(database_path))  # remove unix special characters
        if os.path.isfile(self.database_path):  # check if it exists and if so open it
            self.database = pd.read_csv(self.database_path, parse_dates=['start_time', 'end_time'])
        elif os.path.isdir(os.path.dirname(self.database_path)):
            # if it doesn't exist try to create it in the directory it was expected
            self.database = pd.DataFrame({"start_time": [], "end_time": [], "filename": []})
        else:  # the database directory does not even exist
            msg = "Requested database that does not exist"
            msg += "Could not create since {} is not valid directory".format(os.path.dirname(self.database_path))
            raise NotADirectoryError(msg)

        # make sure the save directory is valid
        if not os.path.isdir(save_directory):
            raise NotADirectoryError("Passed non-existent save directory: {}".format(save_directory))
        self.save_directory = save_directory

    @abstractmethod
    def retrieve(self, start_time, end_time):
        """
        Fetch all data between start and end times of this time
        :param start_time: initial request time
        :type start_time: datetime.datetime
        :param end_time: final request time
        :type end_time: datetime.datetime
        :return: data as panda
        :rtype: pd.DataFrame
        """
        pass

    @abstractmethod
    def get_from_file(self, start_time, end_time):
        """
        Load the file from a local file specified in the database
        :param start_time: initial request time
        :type start_time: datetime.datetime
        :param end_time: final request time
        :type end_time: datetime.datetime
        :return: data if it exists, otherwise None
        :rtype: None or pd.DataFrame
        """
        pass

    def find_file(self, start_time, end_time):
        """
        Search the database to see if a correct example exists
        :param start_time: initial request time
        :type start_time: datetime
        :param end_time: final request time
        :type end_time: datetime
        :return: filename if it exists, otherwise None
        :rtype: str or None
        """
        candidates = self.database[(self.database.start_time <= start_time) & (self.database.end_time >= end_time)]
        if candidates.shape[0] > 0:  # found a candidate
            fn = candidates.iloc[0].filename
            return fn
        return None

    def close(self):
        """
        Close the retriever and update the database
        """
        if self.database is not None:
            self.database.to_csv(self.database_path, index=False)


class LASPf107Retriever(DataRetriever):
    """
    Retrieves the 10.7 cm data from LASP
    http://lasp.colorado.edu/lisird/data/noaa_radio_flux/
    """
    def __init__(self, database_path=None, save_directory=None):
        DataRetriever.__init__(self, database_path, save_directory)
        self.name = 'f107'

    def retrieve(self, start_time, end_time):
        """
        Fetch all data between start and end times of this time
        :param start_time: initial request time
        :type start_time: datetime.datetime
        :param end_time: final request time
        :type end_time: datetime.datetime
        :return: data as panda
        :rtype: pd.DataFrame
        """

        # only one sample per day so make sure the times span a day
        if (end_time - start_time).total_seconds() < (24 * 60 * 60) - 1:
            start_time = datetime(start_time.year, start_time.month, start_time.day)
            end_time = datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59)

        curve = self.get_from_file(start_time, end_time)
        if curve is None:
            start_str = start_time.strftime("%Y-%m-%dT%H:%M")
            end_str = end_time.strftime("%Y-%m-%dT%H:%M")
            url = "http://lasp.colorado.edu/lisird/latis/"
            url += "noaa_radio_flux.csv?time>={}&time<={}".format(start_str, end_str)

            with urllib.request.urlopen(url) as response:
                curve = pd.read_csv(response, index_col=[0], parse_dates=[0])
                curve = curve.rename(index=str, columns={'f107 (solar flux unit (SFU))':'f107'})
                del curve.index.name

                if self.save_directory:  # if saving files
                    save_name = os.path.join(self.save_directory,
                                             "laspf107_{}_{}.csv".format(int(start_time.timestamp()),
                                                                         int(end_time.timestamp())))
                    curve.to_csv(save_name)  # save as a CSV and update the database
                    self.database = self.database.append({"start_time": start_time,
                                                          "end_time": end_time,
                                                          "filename": save_name}, ignore_index=True)
        return curve

    def get_from_file(self, start_time, end_time):
        """
        Load the file from a local file specified in the database
        :param start_time: initial request time
        :type start_time: datetime.datetime
        :param end_time: final request time
        :type end_time: datetime.datetime
        :return: data if it exists, otherwise None
        :rtype: None or pd.DataFrame
        """
        if self.database is not None:  # a database exists
            fn = self.find_file(start_time, end_time)  # a valid file in the time window exists
            if fn and os.path.isfile(fn):  # the file wasn't misplaced
                curve = pd.read_csv(fn, index_col=0, parse_dates=[0])
                return curve
            elif fn:  # the file was misplaced so remove it from the database
                candidates = self.database[(self.database.start_time <= start_time) &
                                           (self.database.end_time >= end_time)]
                self.database.drop(candidates.iloc[0].name)
                return None
            else:  # no file ever existed
                return None
        else:  # no database exists
            return None


class GOESXrayRetriever(DataRetriever):
    """
    Retrieve GOES XRS light curves using Sunpy
    """
    def __init__(self, database_path=None, save_directory=None):
        """
        Create a data retriever
        :param database_path: where the CSV database indicating where previously fetched files is stored
        :param save_directory: where all the files that are fetched for the first time should be saved
        """
        DataRetriever.__init__(self, database_path, save_directory)
        self.name = 'xrs'

    def retrieve(self, start_time, end_time):
        """
        Fetch all data between start and end times of this time
        :param start_time: initial request time
        :type start_time: datetime.datetime
        :param end_time: final request time
        :type end_time: datetime.datetime
        :return: data as panda
        :rtype: pd.DataFrame
        """

        # attempt to fetch it locally
        curve = self.get_from_file(start_time, end_time)
        if curve is None:  # if the local fetch failed, fetch from internet
            times = TimeRange(start_time, end_time)
            try:
                curve = sunpy.lightcurve.GOESLightCurve.create(times).data
            except:
                curve = pd.DataFrame(columns=['xrsa', 'xrsb'])

            if self.save_directory:  # if saving files
                save_name = os.path.join(self.save_directory, "goesxrs_{}_{}.csv".format(int(start_time.timestamp()),
                                                                                         int(end_time.timestamp())))
                curve.to_csv(save_name)  # save as a CSV and update the database
                self.database = self.database.append({"start_time": start_time,
                                                      "end_time": end_time,
                                                      "filename": save_name}, ignore_index=True)
        return curve

    def get_from_file(self, start_time, end_time):
        """
        Load the file from a local file specified in the database
        :param start_time: initial request time
        :type start_time: datetime.datetime
        :param end_time: final request time
        :type end_time: datetime.datetime
        :return: data if it exists, otherwise None
        :rtype: None or pd.DataFrame
        """
        if self.database is not None:  # a database exists
            fn = self.find_file(start_time, end_time)  # a valid file in the time window exists
            if fn and os.path.isfile(fn):  # the file wasn't misplaced
                curve = pd.read_csv(fn, index_col=0, parse_dates=[0])
                return curve
            elif fn:  # the file was misplaced so remove it from the database
                candidates = self.database[(self.database.start_time <= start_time) &
                                           (self.database.end_time >= end_time)]
                self.database.drop(candidates.iloc[0].name)
                return None
            else:  # no file ever existed
                return None
        else:  # no database exists
            return None


class ListHandler(ABC):
    """
    A generic handler that fetches lists of flares, allows them to be saved to file, and then reloaded
    """
    def __init__(self, url=None):
        self.url = url
        self.contents = None

    @abstractmethod
    def load(self, path):
        """
        Loads a list from a local location
        :param path: where file is locally stored
        :type path: str
        :return: loaded list
        :rtype: pd.DataFrame
        """
        pass

    @abstractmethod
    def fetch(self):
        """
        Fetch a list for the first time from the internet
        :return: loaded list
        :rtype: pd.DataFrame
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves a list locally for quicker access
        :param path: where to save
        :type path: str
        """
        pass


class GOESListHandler(ListHandler):

    def __init__(self):
        ListHandler.__init__(self, None)
        pass

    def load(self, path):
        """
        Loads a saved list from drive
        :param path: location of list
        :type path: str
        :return: loaded list
        :rtype: pd.DataFrame
        """
        if os.path.isfile(path):
            self.contents = pd.read_csv(path, parse_dates=['time_start', 'time_peak', 'time_end'])
            return self.contents
        else:
            raise FileNotFoundError("{} was not found".format(path))

    def fetch(self, start_time=None, end_time=None, threaded=True):
        """
        Get the list
        :param start_time: when the list begins
        :type start_time: datetime
        :param end_time: when the list ends
        :type end_time: datetime
        :param threaded: whether to multithread the request
        :type threaded: bool
        :return: a complete list of flares between start_time and end_time
        :rtype: pd.DataFrame
        """
        # set the default start and end times
        if start_time is None:
            start_time = datetime(1975, 1, 1)

        if end_time is None:
            end_time = datetime.now()

        # if the request is for more than a month chunk it up into month sized requests and combine
        if (end_time - start_time) > timedelta(days=30):
            num_steps = math.ceil((end_time-start_time) / timedelta(days=30))
            intervals = [(start_time + i*timedelta(days=30),
                          start_time + (i+1)*timedelta(days=30)) for i in range(0, num_steps)]
            intervals[-1] = (start_time + (num_steps - 1) * timedelta(days=30), end_time)

            # if threading is requested
            if threaded:
                pool = Pool()
                frames = pool.starmap(self._fetch, intervals)
            else:  # no threading
                frames = [self._fetch(*interval) for interval in intervals]
            df = pd.concat(frames).reset_index(drop=True)  # drop the prior index and just use the new one
        else:  # request was for less than a month so just ask for it all
            df = self._fetch(start_time, end_time)
        self.contents = df
        return df

    @staticmethod
    def _fetch(start_time, end_time):
        """
        A helpher method of fetch to actually perform an individual request
        :param start_time: when list begins
        :type start_time: datetime
        :param end_time: when list ends
        :type end_time: datetime
        :return: list of flares between start_time and end_time
        :rtype: pd.DataFrame
        """
        response = get_goes_event_list(TimeRange(start_time, end_time))
        if response == []:  # no events were found
            df = pd.DataFrame(columns=['time_start', 'time_end', 'time_peak', 'ar', 'pos_x', 'pos_y'])
        else:
            df = pd.DataFrame(response)
            df = df.rename(index=str, columns={"start_time": "time_start",
                                               "end_time": "time_end",
                                               "peak_time": "time_peak",
                                               "noaa_active_region": "ar"})
            df = df.drop(['event_date'], axis=1)
            df['pos_x'] = [e[0] for e in df['goes_location']]
            df['pos_y'] = [e[1] for e in df['goes_location']]
            df = df.drop(['goes_location'], axis=1)
        return df

    def save(self, path):
        """
        Saves a list to file
        :param path: where to save
        :type path: str
        """
        if self.contents is not None:
            self.contents.to_csv(path, index=False)
        else:
            raise RuntimeError("A list must be fetched or loaded before saving")


class RHESSIListHandler(ListHandler):
    """
    Fetches and loads the RHESSI flare catalog.
    Able to save it locally and reopen from there in the future.
    """

    def __init__(self, url="https://hesperia.gsfc.nasa.gov/hessidata/dbase/hessi_flare_list.txt"):
        """
        Create a handler
        :param url: the url for the flare catalog if need to fetch it for the first time
        :type url: str
        """
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
            self.contents.to_csv(path, index=False)
        else:
            raise RuntimeError("A list must be fetched or loaded before saving")
