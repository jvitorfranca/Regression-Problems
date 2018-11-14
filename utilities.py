import autosklearn.regression as acs
import sklearn as sk
import pandas as pd
import requests, zipfile
import os
from random import randint


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
