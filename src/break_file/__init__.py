import argparse
import pandas as pd
from list_files import load_files
import config_file.constants as const
from upload_file import info_file_string
from download_file import download
from upload_file import upload
import os
from get_client import boto_client
from dagshub import get_repo_bucket_client

def rename_object(username, bucketname, key_name, new_key):
    path_for_temp_file = '.tmp-rename'
    files_in_data = load_files(username,bucketname)
    old_key = info_file_string(key_name, files_in_data.loc[key_name, "Source"], files_in_data.loc[key_name, "Creation Date"], files_in_data.loc[key_name, "Template"])
    download(username, bucketname, path_for_temp_file, key_name)
    boto_client.delete_object(Bucket=bucketname, Key=old_key)
   # boto_client = get_repo_bucket_client(username + "/" + bucketname)
    upload(bucketname, path_for_temp_file, new_key)
    os.remove(path_for_temp_file)


def reverse_df_to_key_default(keyname: str, list_of_files: pd.DataFrame) -> str:
    df_key = list_of_files.loc[keyname]
    key_string_format = info_file_string(keyname, df_key["Source"], df_key["Creation Date"], df_key["Template"])
    return key_string_format
 

def break_key(username,bucketname,keyname):
    files_in_data = load_files(username,bucketname)
    if keyname not in files_in_data.index:
        print(f"{keyname} wasn't found in bucket. Please check for typos.")
        return
    else:
        template = files_in_data.loc[keyname, "Template"]
        if template[-1] == "?":
            return
        new_key = info_file_string(name=keyname,
                                   source=files_in_data.loc[keyname, "Source"],
                                   creation_date=files_in_data.loc[keyname, "Creation Date"],
                                   template=template + "?")
        rename_object(username,bucketname,keyname,new_key)

def unbreak_key(username,bucketname,keyname):
    files_in_data = load_files(username,bucketname)
    if keyname not in files_in_data.index:
        print(f"{keyname} wasn't found in bucket. Please check for typos.")
        return
    else:
        template = files_in_data.loc[keyname, "Template"]
        if template[-1] != "?":
            return
        new_key = info_file_string(name=keyname,
                                   source=files_in_data.loc[keyname, "Source"],
                                   creation_date=files_in_data.loc[keyname, "Creation Date"],
                                   template=template[:-1])
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
