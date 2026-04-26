from list_files import load_files
import argparse
from get_client import boto_client
import config_file.constants as const
from dagshub import get_repo_bucket_client
from upload_file import info_file_string

def remove(username, bucketname, key_name):
    files_in_data = load_files(username,bucketname)
    if key_name not in files_in_data.index:
        print(f"Key '{key_name}' not found in the bucket '{bucketname}'.")
        return
    
    if files_in_data.loc[key_name, "Template"][-1] != "?":
        print(const.ERROR_REMOVE_NOT_BREAKABLE)
        return
    old_key = info_file_string(key_name, files_in_data.loc[key_name, "Source"], files_in_data.loc[key_name, "Creation Date"], files_in_data.loc[key_name, "Template"])
    boto_client.delete_object(Bucket=bucketname, Key=old_key)

def main():
    parser = argparse.ArgumentParser(description="Remove a file from a DagsHub bucket.")
    # Hardcoded username and bucketname
    username = 'yoacal.data.science'
    bucketname = 'exp-repo'

    parser.add_argument("key", type=str, help="Key of the file in the bucket to remove (only can remove if it end with '?')")


    args = parser.parse_args()

    remove(username, bucketname, args.key)

if __name__ == "__main__":
    main()

