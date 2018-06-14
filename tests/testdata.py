import os
import zipfile

FHERE = os.path.dirname(os.path.abspath(__file__))

def check_files_exist(files, folder):
    ''' Check folder and files exist '''
    if not os.path.exists(folder):
        return False

    for f in files:
        ff = os.path.join(folder, f)
        if not os.path.exists(ff):
            return False

    return True


def check_inputs():
    ''' Check if input files exist '''

    finputs = os.path.join(FHERE, 'input_data')

    files = ['KNNTEST_H111_A0030501.csv', \
                'KNNTEST_H125_473.csv', \
                'KNNTEST_H212_803003.csv', \
                'KNNTEST_H57_108003A.csv']

    for i in range(1, 21):
        files.append('input_data_{0:02d}.csv'.format(i))
        files.append('input_data_monthly_{0:02d}.csv'.format(i))

    return check_files_exist(files, finputs)


def check_outputs():
    ''' Check if output files exist '''
    foutputs = os.path.join(FHERE, 'output_data')

    files = []
    for i in range(1, 21):
        for model in ['GR2M', 'GR4J', 'GR6J', 'HBV']:
            files.append(model+'_params_{0:02d}.csv'.format(i))
            files.append(model+'_timeseries_{0:02d}.csv'.format(i))

    return check_files_exist(files, foutputs)


def unzip(data='input'):
    ''' Unzip input data '''
    if not data in ['input', 'output']:
        raise ValueError('Expected data in [input/output], got '+data)

    fz = os.path.join(FHERE, data+'_data.zip')
    if not os.path.exists(fz):
        raise ValueError(data + ' data file {0} does not exist. '+\
                    'Please run get_obs_data.py script')

    with zipfile.ZipFile(fz, 'r') as archive:
        archive.extractall(path=FHERE)


def check_all():
    ''' process all test data '''
    if not check_inputs():
        unzip('input')

    if not check_outputs():
        unzip('output')


