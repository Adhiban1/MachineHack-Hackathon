from zipfile import ZipFile
import os

def unzip(loc, target):
    if not os.path.exists(target):
        os.mkdir(target)
    with ZipFile(loc, 'r') as zObject:
	    zObject.extractall(path=target)

unzip('Participants_Data_DSSC_2023.zip', 'dataset')