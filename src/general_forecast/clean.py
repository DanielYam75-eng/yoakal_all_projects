import subprocess
def main():
    subprocess.run(["rm", "*forcast*.csv"])
    subprocess.run(["rm", "*actual*.csv"])
    subprocess.run(["rm", "*full*.csv"])
    subprocess.run(["rm", "*grades*.csv"])
    subprocess.run(["rm", "*result*.csv"])

if __name__ == "__main__":
    main()
    print("Done")