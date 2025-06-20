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

def get_key_info(contents_response):
        Name = contents_response.split("^")[0].split("=")[1]
        Source = contents_response.split("^")[1].split("=")[1]
        Creation_Date = contents_response.split("^")[2].split("=")[1]
        Template = contents_response.split("^")[3].split("=")[1]
        return Name, Source, Creation_Date, Template


def extract_data_on_files(contents_response):
    list_of_files = pd.DataFrame(columns=['Name', "Source" , "Creation Date", "Template" ,'Last Modified', 'Size'])
    for i in range(len(contents_response)):
        Name, Source, Creation_Date, Template = get_key_info(contents_response[i].get("Key"))
        list_of_files.loc[i] = [Name, Source, Creation_Date, Template, pd.to_datetime(contents_response[i].get("LastModified")).strftime('%d-%m-%Y'),contents_response[i].get("Size")]
    return list_of_files

def main():
    username = 'yoacal.data.science'
    bucketname = 'new-repo'

    response = list_objects(username, bucketname)
    contents_response = getting_contents_response(response)
    list_of_files = extract_data_on_files(contents_response)

    print(list_of_files)

if __name__ == "__main__":
    main()