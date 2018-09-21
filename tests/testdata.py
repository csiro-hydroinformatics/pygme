import os
import re
import zipfile
import numpy as np

from hydrodiy.io import csv

FHERE = os.path.dirname(os.path.abspath(__file__))

FINPZ = os.path.join(FHERE, 'input_data.zip')
FOUTPZ = os.path.join(FHERE, 'output_data.zip')

def read(filename, source='input', has_dates=False):
    ''' Get data from zip file '''

    # Get zip file name
    fzip = FINPZ if source == 'input' else FOUTPZ

    # Read data
    with zipfile.ZipFile(fzip, 'r') as archive:
        archive.extract(filename, path=FHERE)
        fout = os.path.join(FHERE, filename)

    # Read data
    if has_dates:
        data, _ = csv.read_csv(fout, index_col=0, \
                        parse_dates=True)
    else:
        data, _ = csv.read_csv(fout)

    # Clean folder
    os.remove(fout)

    # Clean column names
    data.columns = [re.sub('"', '', cn) for cn in data.columns]

    return data


