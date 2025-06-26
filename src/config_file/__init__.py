from dagshub import get_repo_bucket_client
from datetime import datetime
import pandas as pd
from tabulate import tabulate
import argparse


def list_objects(username, bucketname):
    boto_client = get_repo_bucket_client(username + "/" + bucketname)
    response = boto_client.list_objects_v2(Bucket=bucketname)
    return response

def getting_contents_response(response):
    contents_response = response.get('Contents')
    return contents_response

def get_key_info(contents_response):
    parts = contents_response.split("^")
    name = parts[0]
    source, creation_Date, template = (part.split("=")[1] for part in parts[1:])
    return name, source, creation_Date, template

def get__key_name(contents_response):
    parts = contents_response.split("^")
    name = parts[0]
    return name

def extract_data_on_files(contents_response,required_template=None):
    list_of_files = pd.DataFrame(columns=['Name', "Source" , "Creation Date", "Template" ,'Last Modified', 'Size'])
    for i in range(len(contents_response)):
        name, source, creation_Date, template = get_key_info(contents_response[i].get("Key"))
        if required_template is None:
            list_of_files.loc[i] = [name, source, creation_Date, template, pd.to_datetime(contents_response[i].get("LastModified")).strftime('%d-%m-%Y'),contents_response[i].get("Size")]
        elif required_template == template:
            list_of_files.loc[i] = [name, source, creation_Date, template, pd.to_datetime(contents_response[i].get("LastModified")).strftime('%d-%m-%Y'),contents_response[i].get("Size")]
    return list_of_files

def load_files(username, bucketname,required_template=None):
    response = list_objects(username, bucketname)
    contents_response = getting_contents_response(response)
    list_of_files = extract_data_on_files(contents_response,required_template)
    list_of_files = list_of_files.set_index("Name")
    list_of_files=list_of_files.sort_values(by=["Last Modified"])
    return list_of_files

def main():
    parser = argparse.ArgumentParser(description="specify what kind of template you want")

    username = 'yoacal.data.science'
    bucketname = 'exp-repo'

    parser.add_argument("-t", "--template", type=str, help="template of files to list")
    args = parser.parse_args()

    list_of_files = load_files(username, bucketname,args.template)

    print(tabulate(list_of_files, tablefmt='plain'))

if __name__ == "__main__":
    main()


