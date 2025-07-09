import os
import argparse
import pandas as pd
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--dir",         "-d", type = str,  required = True)
parser.add_argument("--path",        "-p", type = str,  required = True)
parser.add_argument("--target_year", "-y", type = str,  required = True)
parser.add_argument("--seed",        "-s", type = str,  required = False, default = "1")
parser.add_argument("--train",       "-t", type = str,  required = False, default = "1")
args = parser.parse_args()


PATH = args.dir + os.sep + args.path
YEAR = args.target_year
THRSHOLD = 1000
GROUP = "doc"


data =  pd.read_csv(PATH).groupby(GROUP)


def program(name: str, group: pd.DataFrame):

    data_path   = os.path.join(args.dir, f"{name}.csv")
    model_path  = os.path.join(args.dir, f"{name}_model.pt")
    output_path = args.dir + name

    group.to_csv(data_path, index = False)

    print(f"Processing {name}...")

    try: output = subprocess.run(["python", "program.py", "-i", data_path, "-o", output_path, "-m", model_path, "-y", YEAR, "-s", args.seed, "-t", args.train])
    except: print(output.stderr)

    print(f"Finished processing {name}.")

    os.remove(data_path)
    print(f"Removed temporary files for {name}.")


for name, group in data:
    if len(group) >= THRSHOLD:
        program(name, group)


program("rest", data.filter(lambda x: len(x) < THRSHOLD).reset_index(drop = True))
    

print("All groups processed.")

forecast_files = [file for file in os.listdir(args.dir) if "forecast" in file and YEAR in file]
actual_files   = [file for file in os.listdir(args.dir) if "actual"   in file and YEAR in file]


print(f"Total sum: {sum(pd.read_csv(os.path.join(args.dir, file)).select_dtypes('number').sum().sum() for file in forecast_files):.2e}")

print(f"Total forecast and actual data for {YEAR} saved successfully.")