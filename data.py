import numpy
import urllib.request
from pyunpack import Archive
import os
from sgfmill import sgf
import urllib 
from py7zr import unpack_7zarchive

#!pip install pyunpack
#!pip install patool
#!pip install sgfmill
                                                       
download_link = 'https://github.com/featurecat/go-dataset/raw/master/9d/9d.7z'        
file_name = 'go_data.7z'
data_path = 'go_data'
urllib.request.urlretrieve(download_link, file_name)
data_dir = 'go_data/9d'
file_names = os.listdir(data_dir)
print(file_names[:5])

with open(os.path.join(data_dir, file_names[0]), 'rb') as f:
  game = sgf.Sgf_game.from_bytes(f.read())

game.get_main_sequence()