import pandas as pd
from tabulate import tabulate
import argparse
from get_client import boto_client

def list_objects(username, bucketname):
    response = boto_client.list_objects_v2(Bucket=bucketname)
    return response

def getting_contents_response(response):
    contents_response = response.get('Contents', [])
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
            list_of_files.loc[i] = [name, source, creation_Date, template, pd.to_datetime(contents_response[i].get("LastModified")).strftime('%Y-%m-%d'),contents_response[i].get("Size")]
        elif required_template == template:
            list_of_files.loc[i] = [name, source, creation_Date, template, pd.to_datetime(contents_response[i].get("LastModified")).strftime('%Y-%m-%d'),contents_response[i].get("Size")]
    return list_of_files

def load_files(username, bucketname,required_template=None):
    response = list_objects(username, bucketname)
    contents_response = getting_contents_response(response)
    list_of_files = extract_data_on_files(contents_response,required_template)
    list_of_files = list_of_files.set_index("Name")
    list_of_files=list_of_files.sort_values(by=["Last Modified"], ascending=False)
    return list_of_files

def human_readable_size(size):
    if size >= 1048576:
        return f"{size / 1048576:.2f}M"
    elif size >= 1024:
        return f"{size / 1024:.2f}K"
    else:
        return f"{size}B"

def main():
    parser = argparse.ArgumentParser(description="specify what kind of template you want")

    username = 'yoacal.data.science'
    bucketname = 'exp-repo'

    parser.add_argument("-t", "--template", type=str, help="List only files of template <template>", required=False)
    parser.add_argument("--all", action="store_true", help="List all files, instead of only the 10 most recently modified")

    args = parser.parse_args()

    list_of_files = load_files(username, bucketname,args.template)
    list_of_files.columns = ['Source', 'Creation Date', 'Template', 'Last Modified', 'Size']
    list_of_files['Size'] = list_of_files['Size'].apply(human_readable_size)

    if args.all:
        print(tabulate(list_of_files, headers='keys', tablefmt='plain'))
    else:
        last_10_modified_files = list_of_files.head(10)
        print(tabulate(last_10_modified_files, headers='keys', tablefmt='plain'))


if __name__ == "__main__":
    main()


