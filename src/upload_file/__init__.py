from dagshub import get_repo_bucket_client
import argparse
from datetime import datetime
import re
import pandas as pd
from list_files import load_files
from list_files import get__key_name
import hashlib
import numpy as np

ERROR_INVALID_NAME = "Must be 1-128 characters and contain only letters, digits, '-', '_', or '.' and not include '^' or '='."
ERROR_INVALID_DATE = "Invalid date format. Please enter the date in YYYY-MM-DD format."
ERROR_INVALID_KEY_NAME = "this key name already in the system. choose diffrent type of key name"
ERROR_DATA_ALREADY_IN_THE_SYSTEM = "this data already in the system. "
PROMPT_FOR_TEMPLATE = "(type 'list' to list all allowed templates, prefix your template with the characters 'ad-hoc-' to enter a free form template): "
ERROR_TEMPLATE_DISALLOWED = "The chosen template is disallowed, pleasechoose another one. You may type 'list' to list allowed templates"
ERROR_TEMPLATE_NOT_KNOWN = "This template is not known. To use ad-hoc template prefix its name with 'ad-hoc-'"

allowed_templates = pd.DataFrame(
        [["monim-no-re-1", 'BI', True],
         ["invoices-to-po-1", 'BW', False],
         ["po-1", 'BW', False],
         ["monim-by-posting-date-1", 'BW', True],
         ["po-foreign-currency-1", 'BW', False],
         ["po-foreign-currency-2", 'BW', True],
         ["invoices-to-po-2", 'BW', True],
         ["invoices-to-po-foreign-currency-1", 'BW', True],
         ["po-2", 'BW', True],
         ["ZH-foreign-currency-1", 'BI', True],],
        columns=['template', 'source', 'allowed'])


def upload(username, bucketname, filepath, key):
    boto_client = get_repo_bucket_client(username + "/" + bucketname)
    boto_client.upload_file(filepath, bucketname, key)

def info_file_string(name,source,creation_date,template):
    info_file_source = f"source={source}"
    info_file_creation_date = f"creation_date={creation_date}"
    info_file_template = f"template={template}"
    combined_file_info = f"{name}^{info_file_source}^{info_file_creation_date}^{info_file_template}"
    return combined_file_info

def is_valid_name(name):
    if not (1 <= len(name) <=128):
        return False

    if '^' in name or '=' in name:
        return False

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        return False
    return True

def get_valid_input_template(prompt, validator, error_message):
    while True:
        user_response = input(prompt)
        if user_response == "list":
            print(allowed_templates)
        elif user_response.startswith('ad-hoc-'):
            if validator(user_response):
                return user_response, None
            else:
                print(error_message)
        elif user_response in allowed_templates['template'].values:
            if np.any(allowed_templates.loc[allowed_templates['template'] == user_response, 'allowed']):
                set_source = allowed_templates.loc[allowed_templates['template'] == user_response, 'source'].values[0]
                return user_response , set_source
            else:
                print(ERROR_TEMPLATE_DISALLOWED)
        else:
            print(ERROR_TEMPLATE_NOT_KNOWN)


def get_valid_input_source(prompt, validator, error_message):
    while True:
        user_response = input(prompt)
        if validator(user_response):
            return user_response
        else:
            print(error_message)

def get_valid_date(prompt, error_message):
    default_date = datetime.today().strftime('%Y-%m-%d')
    while True:
        date = input(f"{prompt} [default: {default_date}]: ") or default_date
        try:
            return pd.to_datetime(date, format="%Y-%m-%d")
        except ValueError:
            print(error_message)

def is_valid_key_name(key_name,username,bucketname):
    if key_name in load_files(username,bucketname).index:
        return False
    return True

def check_md5_valid(username,bucketname,filepath):
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    hash_md5_specific_file = hashlib.md5(file_bytes).hexdigest().strip('"')

    boto_client = get_repo_bucket_client(username + "/" + bucketname)
    response = boto_client.list_objects_v2(Bucket=bucketname)

    for Contents in response.get('Contents'):
        etag_specific_data = Contents.get('ETag').strip('"')
        if hash_md5_specific_file == etag_specific_data:
            return get__key_name(Contents.get("Key"))
    return None

def main():
    parser = argparse.ArgumentParser(description="upload a file to a DagsHub bucket.")

    username = 'yoacal.data.science'
    bucketname = 'new-repo'

    parser.add_argument("keyname", type=str, help="key name for the uploaded file") 
    parser.add_argument("-i", "--input", type=str, help="local path of file to input")

    args = parser.parse_args()

    matching_key = check_md5_valid(username, bucketname, args.input)
    if matching_key is not None:
        print(ERROR_DATA_ALREADY_IN_THE_SYSTEM + f" under the key name: {matching_key}")
        return

    if not is_valid_name(args.keyname):
        print(f"Invalid key name. {ERROR_INVALID_NAME}")
        return
    
    if not is_valid_key_name(args.keyname,username,bucketname):
        print(ERROR_INVALID_KEY_NAME)
        return

    template , set_source = get_valid_input_template("Enter template: " + PROMPT_FOR_TEMPLATE, is_valid_name, f"Invalid template.{ERROR_INVALID_NAME}")

    if set_source is not None:
        source = set_source
    else:
        source = get_valid_input_source("Enter source: ", is_valid_name, f"Invalid source.{ERROR_INVALID_NAME}")

    creation_date = get_valid_date("Enter creation date (format: YYYY-MM-DD): ",ERROR_INVALID_DATE)
    

    info_of_file_as_string = info_file_string(args.keyname,source,creation_date,template)
    upload(username, bucketname, args.input, info_of_file_as_string)

if __name__ == "__main__":
    main()