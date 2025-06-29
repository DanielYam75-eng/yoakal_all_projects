import io
from dagshub import get_repo_bucket_client
import pandas as pd
from list_files import load_files
from upload_file import info_file_string

def read(key_name):
    username = 'yoacal.data.science'
    bucketname = 'exp-repo'

    existing_files = load_files(username, bucketname)
    if key_name not in existing_files.index:
        raise ValueError(f"Key '{key_name}' not found in the bucket '{bucketname}'.")

    key_row = existing_files.loc[key_name]

    boto_client = get_repo_bucket_client(username + "/" + bucketname)

    key = info_file_string(key_name, key_row['Source'], key_row['Creation Date'], key_row['Template'])

    obj = boto_client.get_object(Bucket=bucketname, Key=key)

    data = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return data

