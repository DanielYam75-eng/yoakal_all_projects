import pandas as pd
from read_file import read

def get_ZH_tuples(in_data: pd.DataFrame) -> pd.DataFrame:
    in_data['000'] = in_data['000'].astype(str).str.replace(',', '').astype(float)
    in_data['110'] = in_data['110'].astype(str).str.replace(',', '').astype(float)
    in_data = in_data.set_index(['doc_id', 'doc_item'])
    in_data = in_data.unstack()
    out_data = pd.DataFrame(index=in_data.index, columns=['MOF_class_out', 'MOF_class_in', 'value', 'year', 'month', 'law'])
    out_data['year'] = in_data[('year', 1)]
    out_data['MOF_class_out'] = in_data[('MOF_class', 1)]
    out_data['MOF_class_in'] = in_data[('MOF_class', 2)]
    out_data['value'] = in_data[('110', 1)]
    out_data['month'] = in_data[('month', 1)]
    out_data['law'] = in_data[('law', 1)]
    return out_data


def main(input_path):

    input_data = read(input_path, sep='\t')
    output_data = get_ZH_tuples(input_data)
    output_data = output_data.dropna(subset=['MOF_class_out', 'MOF_class_in', 'value', 'year', 'month', 'law'])
    output_data['MOF_class_out'] = output_data['MOF_class_out'].astype(str)
    output_data['MOF_class_in'] = output_data['MOF_class_in'].astype(str)
    output_data['year'] = output_data['year'].astype(int).astype(str)
    output_data['month'] = output_data['month'].astype(int).astype(str)
    output_data['law'] = output_data['law'].astype(int).astype(str)
    output_data['value'] = output_data['value'].astype(float)
    
    return output_data


if __name__ == "__main__":
    main()
        