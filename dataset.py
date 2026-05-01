# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="GBXneGBgt2OeLyBC5BSJ")
project = rf.workspace("nicolai-hoirup-nielsen").project("logo-detector-cgxef")
version = project.version(2)
dataset = version.download("yolokeras")
                