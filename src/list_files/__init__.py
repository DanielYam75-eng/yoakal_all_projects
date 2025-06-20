from dagshub import get_repo_bucket_client
import argparse
from datetime import datetime
import pandas as pd


def list_objects(username, bucketname):
    boto_client = get_repo_bucket_client(username + "/" + bucketname)
    response = boto_client.list_objects_v2(Bucket=bucketname)
    return response

def getting_contents_response(response):
    contents_response = response.get('Contents')
    return contents_response

def extract_files(contents_response):
    list_of_files = pd.DataFrame(columns=['Key', 'LastModified', 'Size'])
    for i in range(len(contents_response)):
        list_of_files.loc[i] = [contents_response[i].get("Key"),pd.to_datetime(contents_response[i].get("LastModified")).strftime('%d-%m-%Y'),contents_response[i].get("Size")]
    list_of_files=list_of_files.set_index('Key')
    return list_of_files

def main():
    username = 'yoacal.data.science'
    bucketname = 'new-repo'

    response = list_objects(username, bucketname)
    contents_response = getting_contents_response(response)
    list_of_files = extract_files(contents_response)

    print(list_of_files)

if __name__ == "__main__":
    main()
