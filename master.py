import re
import os
import argparse
import pandas as pd
import subprocess
from read_file import read


parser = argparse.ArgumentParser()
parser.add_argument("--local_path",  "-l", type = str,  required = True)
parser.add_argument("--remote_path", "-r", type = str,  required = True)
parser.add_argument("--target_year", "-y", type = str,  required = True)
parser.add_argument("--seed",        "-s", type = str,  required = False, default = "1")
parser.add_argument("--train",       "-t", type = str,  required = False, default = "1")
args = parser.parse_args()


LDIR = args.local_path
RDIR = args.remote_path
YEAR = args.target_year
THRSHOLD = 1000
VAL = "volume"
GROUP = "doc"



data : pd.DataFrame = read(RDIR)
data = data.groupby(GROUP)


def program(name: str, group: pd.DataFrame):

    data_path   = os.path.join(LDIR, f"{name}.csv")
    model_path  = os.path.join(LDIR, f"{name}_model.pt")
    output_path = LDIR + name

    group.to_csv(data_path, index = False)

    print(f"Processing {name}...")

    try: output = subprocess.run(["python", "program.py", "-i", data_path, "-o", output_path, "-m", model_path, "-y", YEAR, "-s", args.seed, "-t", args.train])
    except: print(output.stderr)

    print(f"Finished processing {name}.")

    os.remove(data_path)
    print(f"Removed temporary files for {name}.")


def summarize_files(keyword: str):

    pattern = re.compile(rf"^[A-Z]{{2}} {re.escape(keyword)} {YEAR}.csv$")

    files = [file for file in os.listdir(LDIR) if pattern.match(file)]

    print(f"Total {keyword}: {sum(pd.read_csv(os.path.join(LDIR, file)).select_dtypes('number').sum().sum() for file in files):.2e}")




for name, group in data:
    if len(group) >= THRSHOLD:
        program(name, group)

program("rest", data.filter(lambda x: len(x) < THRSHOLD).reset_index(drop = True))
print("All groups processed.")
summarize_files("forecast")
summarize_files("actual")
print(f"Total forecast and actual data for {YEAR} saved successfully.")