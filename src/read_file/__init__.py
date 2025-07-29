import io
import pandas as pd
from list_files import load_files
from upload_file import info_file_string
from get_client import boto_client

def read(key_name: str, requested_templates: str | list[str] | None = None, **kwargs) -> pd.DataFrame:
    username = 'yoacal.data.science'
    bucketname = 'exp-repo'
    if requested_templates is str:
        requested_templates = [requested_templates]
    existing_files = load_files(username, bucketname)
    if key_name not in existing_files.index:
        raise ValueError(f"Key '{key_name}' not found in the bucket '{bucketname}'.")

    key_row = existing_files.loc[key_name]

    # When to raise an error:
    if requested_templates is not None and key_row['Template'] not in requested_templates:
        raise ValueError(f"Template mismatch: expected one of '{requested_templates}', got '{key_row['Template']}'.")

    key = info_file_string(key_name, key_row['Source'], key_row['Creation Date'], key_row['Template'])
    obj = boto_client.get_object(Bucket=bucketname, Key=key)

    data = pd.read_csv(io.BytesIO(obj['Body'].read()), **kwargs)
    return data
