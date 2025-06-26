import config_file.template as temp
from tabulate import tabulate

def main():
    template_data = temp.allowed_templates.loc[temp.allowed_templates['allowed'] ==  True, ["template", "source"]]
    print(tabulate(template_data, headers='keys', tablefmt='plain', showindex=False))


if __name__ == "__main__":
    main()