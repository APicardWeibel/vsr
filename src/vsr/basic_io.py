"""
Shared functions for basic read/write to human readable format
"""

import json
import os
from typing import Any, Callable, Optional

import dill as dl
import numpy as np
import pandas as pd


class SaveLoad:
    """Save and Load information.
    This class is constructed around 2 main functions,
    load and save (optional). 'load' should open a file and
    return an object, which can be saved to a file through 'save'.

    Other attributes:
        ext, the extension of the saving files.
    """

    def __init__(
        self,
        load: Callable[[str], Any],
        save: Optional[Callable[[str, Any], str]] = None,
        extension: Optional[str] = None,
    ):
        self.__load = load
        self.__save = save
        self.__ext = extension

    def check_ext(self, path):
        if self.__ext is not None:
            assert path[-len(self.__ext) :] == self.__ext

    @property
    def ext(self):
        """Extension of file"""

    def save(self, path: str, obj):
        """Save object to path"""
        # Check that extension is coherent with path
        self.check_ext(path)

        if self.__save is None:
            # Assumes that obj has a save method
            return obj.save(path)
        return self.__save(path, obj)

    def load(self, path: str, optional: bool = False):
        """Load object from path
        Arsg:
            path: path to load
            optional: behavior if file does not exist
                (if optional, returns None, else raise
                FileNotFound)
        """
        if not os.path.exists(path):
            if optional:
                return None
            raise FileNotFoundError(f"Could not find {path}")
        self.check_ext(path)
        return self.__load(path)


# str IO
def write_str(path: str, x: str):
    """Write str to a txt file"""
    with open(path, "w", encoding="utf-8") as file:
        file.write(x)


def load_str(path: str) -> str:
    """Read str from txt file"""
    with open(path, "r", encoding="utf-8") as file:
        x = file.read()
    return x


rw_str = SaveLoad(save=write_str, load=load_str, extension="txt")


# int IO
def write_int(path: str, x: int):
    """Write int to a txt file"""
    with open(path, "w", encoding="utf-8") as file:
        file.write(str(x))


def load_int(path: str) -> int:
    with open(path, "r", encoding="utf-8") as file:
        x = int(file.read())
    return x


rw_int = SaveLoad(save=write_int, load=load_int, extension="txt")


def write_bool(path: str, x: bool):
    return write_int(path, int(x))


def load_bool(path: str) -> bool:
    return bool(load_int(path))


rw_bool = SaveLoad(save=write_bool, load=load_bool, extension="txt")


def write_float(path: str, x: float) -> str:
    """Write float to a txt file"""
    with open(path, "w", encoding="utf-8") as file:
        file.write(str(x))
    return path


def load_float(path: str) -> float:
    """Read float from txt file"""
    with open(path, "r", encoding="utf-8") as file:
        x = float(file.read())
    return x


rw_flt = SaveLoad(save=write_float, load=load_float, extension="txt")


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.dtype):
            return str(o)
        if isinstance(o, np.ndarray):
            return {"__ndarray__": o.tolist(), "dtype": o.dtype}
        return super().default(o)


def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and "__ndarray__" in dct:
        return np.array(dct["__ndarray__"], dtype=dct["dtype"])
    return dct


class PdEncoder(NpEncoder):
    def default(self, o):
        if isinstance(o, pd.Index):
            return {"__pd_Index__": o.to_list()}
        if isinstance(o, pd.DatetimeIndex):
            return {"__pd_DatetimeIndex__": o.to_list()}
        if isinstance(o, pd.Timestamp):
            return str(o)
        if isinstance(o, pd.DataFrame):
            return {
                "__pd_DataFrame__": o.to_numpy(),
                "columns": o.columns,
                "index": o.index,
            }
        return super().default(o)


def json_pd_obj_hook(dct):
    if isinstance(dct, dict) and "__pd_Index__" in dct:
        return pd.Index(dct["__pd_Index__"])

    if isinstance(dct, dict) and "__pd_DataFrame__" in dct:
        return pd.DataFrame(
            dct["__pd_DataFrame__"], columns=dct["columns"], index=dct["index"]
        )
    if isinstance(dct, dict) and "__pd_DatetimeIndex__" in dct:
        return pd.DatetimeIndex(dct["__pd_DatetimeIndex__"])
    return json_numpy_obj_hook(dct)


def write_arr(path: str, arr: np.ndarray):
    """Write np.ndarray to json file as a dictionnary of shape and data.
    Resulting file can be loaded through load_arr function

    Args:
        path: str, where the array should be written
        arr: np.ndarray, array to be written
    Returns:
        path
    """
    # Force convert to array to allow function to be used
    # for np.ndarray convertibles input
    arr_clean = np.asarray(arr)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(arr_clean, file, cls=NpEncoder)
    return path


def load_arr(path: str) -> np.ndarray:
    """Load array written as json file containing a dictionnary of shape and data.
    Args:
        path: str, where the array is written
    Returns:
        loaded array"""
    with open(path, "r", encoding="utf-8") as file:
        arr_dsc = json.load(file, object_hook=json_numpy_obj_hook)

    return arr_dsc


rw_arr = SaveLoad(save=write_arr, load=load_arr, extension="json")


def write_dl(path: str, obj) -> str:
    with open(path, "wb") as file:
        dl.dump(obj, file)
    return path


def load_dl(path: str):
    with open(path, "rb") as file:
        obj = dl.load(file)
    return obj


rw_dl = SaveLoad(save=write_dl, load=load_dl, extension="dl")


def write_json_like(path: str, obj) -> str:
    with open(path, "w") as file:
        json.dump(obj, file, cls=PdEncoder)
    return path


def load_json_like(path: str):
    with open(path, "r") as file:
        obj = json.load(file, object_hook=json_pd_obj_hook)
    return obj


rw_jsonlike = SaveLoad(save=write_json_like, load=load_json_like, extension="json")
