from dagshub import get_repo_bucket_client
import argparse
import pandas as pd
from list_files import load_files
import config_file.constants as const
from upload_file import info_file_string
from download_file import download
from upload_file import upload
import os
import hashlib


def rename_object(username, bucketname, key_name, new_key):
    path_for_temp_file = '.tmp-rename'
    download(username, bucketname, path_for_temp_file, key_name)
    upload(username, bucketname, path_for_temp_file, new_key)
    os.remove(path_for_temp_file)


def reverse_df_to_key_default(keyname: str, list_of_files: pd.DataFrame) -> str:
    df_key = list_of_files.loc[keyname]
    key_string_format = info_file_string(keyname, df_key["Source"], df_key["Creation Date"], df_key["Template"])
    return key_string_format
 

def make_new_key(files_in_data,keyname,break_flag):   
    df_key_to_break = files_in_data[files_in_data.index == keyname]
    if break_flag == True:
        new_template_key = df_key_to_break["Template"][0] + "?"
    else:
        new_template_key = df_key_to_break["Template"][0][:-1]
    new_key = info_file_string(df_key_to_break.index[0],df_key_to_break["Source"][0],df_key_to_break["Creation Date"][0],new_template_key)
    return new_key


def break_key(username,bucketname,keyname):
    files_in_data = load_files(username,bucketname)
    if keyname not in files_in_data.index:
        print(const.ERROR_DATA_NOT_FOUND)
        return
    else:
        template = files_in_data.loc[keyname, "Template"]
        new_key = info_file_string(keyname=keyname,
                                   source=files_in_data.loc[keyname, "Source"],
                                   creation_date=files_in_data[keyname, "Creation Date"],
                                   template=(template + "?") if template[-1] != "?" else template)
        rename_object(username,bucketname,keyname,new_key)

def unbreak_key(username,bucketname,keyname):
    files_in_data = load_files(username,bucketname)
    if keyname not in files_in_data.index:
        print(const.ERROR_DATA_NOT_FOUND)
        return
    else:
        template = files_in_data.loc[keyname, "Template"]
        new_key = info_file_string(keyname=keyname,
                                   source=files_in_data.loc[keyname, "Source"],
                                   creation_date=files_in_data[keyname, "Creation Date"],
                                   template=template if template[-1] != "?" else template[:-1])
        rename_object(username,bucketname,keyname,new_key)


def main_break():
    parser = argparse.ArgumentParser(description="enter key to break file")

    username = 'yoacal.data.science' 
    bucketname = 'exp-repo'
    parser.add_argument("keyname", type=str, help="key name for the file to break") 
    args = parser.parse_args()
    break_key(username,bucketname,args.keyname)


def main_unbreak():
    parser = argparse.ArgumentParser(description="enter key to unbreak file")

    username = 'yoacal.data.science' 
    bucketname = 'exp-repo'
    parser.add_argument("keyname", type=str, help="key name for the file to unbreak") 
    args = parser.parse_args()
    unbreak_key(username,bucketname,args.keyname)

    
if __name__ == "__main__":
    main_break()
