import os
from pathlib import Path


def get_project_root(current_directory=None, max_depth:int=20):
    """
    Returns the project root directory for the project at the provided directory.

    :returns: path to the root of the project
    :rtype: string
    """

    if current_directory is None:
        current_directory = os.getcwd()
    home = str(Path.home())

    for _ in range(max_depth):
        if current_directory == home:
            raise Exception("Could not find project root directory.")

        content = os.listdir(current_directory)
        if ".gitignore" in content:
            return current_directory
        current_directory = os.path.dirname(current_directory)

    raise Exception("Could not find project root directory.")


def abs_path(folder:str, path:str=None):
    if path is None:
        path = ""

    return os.path.join(get_project_root(), folder, path)


def datap(path:str=None):
    return abs_path("data", path)


def modelp(path:str=None):
    return abs_path("models", path)


def cachep(path:str=None):
    return abs_path("cache", path)


def outputp(path:str=None):
    return abs_path("output", path)