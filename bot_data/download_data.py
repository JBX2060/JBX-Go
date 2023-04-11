import os
import urllib.request
import tarfile
import argparse
import urllib.request

def get_folder_name(folder_name):
    # If folder name is not specified, find the highest numbered kata folder and increment it
    if not folder_name:
        kata_folders = [name for name in os.listdir('.') if name.startswith('kata')]
        if len(kata_folders) > 0:
            latest_folder = sorted(kata_folders)[-1]
            try:
                folder_num = int(latest_folder[4:])
            except ValueError:
                folder_num = 0
            folder_name = f'kata{folder_num + 1}'
        else:
            folder_name = 'kata'
    # If folder name is specified, use it
    else:
        folder_name = folder_name
    
    return folder_name

def append_used_data(date_str_list):
    # If the used_data.txt file exists, append the date_strs to it
    if os.path.exists('used_data.txt'):
        with open('used_data.txt', 'a') as f:
            for date_str in date_str_list:
                f.write(date_str + '\n')
    # If the used_data.txt file doesn't exist, create it and append the date_strs to it
    else:
        with open('used_data.txt', 'w') as f:
            for date_str in date_str_list:
                f.write(date_str + '\n')

import urllib.request

def download_and_extract(date_str_list, file_type='training', folder_name=None):
    # Construct the absolute path to the bot_data directory
    bot_data_dir = os.path.dirname(os.path.abspath(__file__))

    for date_str in date_str_list:
        file_name = date_str + '.tar.bz2'
        file_url = f'https://katagoarchive.org/kata1/{file_type}games/{file_name}'
        print(file_url)
        file_path = os.path.join(bot_data_dir, file_name)

        # Set a custom user agent
        req = urllib.request.Request(
            file_url,
            headers={
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1'
            }
        )

        # Download the file
        with urllib.request.build_opener().open(req) as response:
            with open(file_path, 'wb') as outfile:
                outfile.write(response.read())

        # Extract the file
        with tarfile.open(file_path) as tar:
            tar.extractall(bot_data_dir)

    # Append date_strs to used_data.txt
    append_used_data(date_str_list)




if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Download and extract files from Katago Archive.')
    parser.add_argument('date_str_list', nargs='+', type=str, help='The list of date_strs, separated by space.')
    parser.add_argument('-t', '--file_type', type=str, help='The type of the file (e.g., ratinggames).', default='training')
    parser.add_argument('-f', '--folder_name', type=str, help='The name of the folder to store the downloaded files.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call download_and_extract function with command line arguments
    download_and_extract(args.date_str_list, args.file_type, args.folder_name)
