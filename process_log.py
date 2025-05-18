import pandas as pd
import re
import argparse

def parse_log_lines(lines):
    ref_line = lines[0]
    sample_line = lines[1]
    fid_line = lines[2]
    
    match = re.search(r'.*/KV(\d+)/(\w+)KVMask/(\w+)TokenMask/.*', sample_line)
    if match:
        KV = int(match.group(1))
        KVMask = match.group(2)
        TokenMask = match.group(3)
        
        fid_match = re.search(r'FID score: ([\d.]+)', fid_line)
        if fid_match:
            Fid = float(fid_match.group(1))
            return KV, KVMask, TokenMask, Fid
    return None

def process_log_file(file_path):
    data = []

    with open(file_path, 'r') as file:
        lines = []
        for i, line in enumerate(file):
            lines.append(line.strip())
            if len(lines) == 3:
                parsed_data = parse_log_lines(lines)
                if parsed_data:
                    data.append(parsed_data)
                lines = []

    df = pd.DataFrame(data, columns=["KV", "KVMask", "TokenMask", "FID"])

    return df

def generate_markdown_tables(df, row_order=None, column_order=None):
    markdown_tables = {}
    
    for KV in df['KV'].unique():
        kv_df = df[df['KV'] == KV]
        
        pivot_table = kv_df.pivot_table(index='KVMask', columns='TokenMask', values='FID', aggfunc='mean')

        if row_order:
            pivot_table = pivot_table.reindex(row_order, axis=0)
        if column_order:
            pivot_table = pivot_table.reindex(column_order, axis=1)
        
        markdown = f"KV={KV}\n\n"
        markdown += "|| " + " | ".join(pivot_table.columns) + " |\n"
        markdown += "| --- " * (len(pivot_table.columns) + 1) + "|\n"
        
        for idx, row in pivot_table.iterrows():
            markdown += f"| {idx} | " + " | ".join(f"{value:.4f}" for value in row) + " |\n"
        
        markdown_tables[KV] = markdown
    
    return markdown_tables

def main():
    parser = argparse.ArgumentParser(description="Process log file and generate markdown tables.")
    parser.add_argument('--log_file', type=str, required=True, help="Path to the log file.")
    parser.add_argument('--table_path', type=str, required=True, help="Path to save the markdown table result.")
    
    args = parser.parse_args()

    log_path = args.log_file
    table_path = args.table_path

    df = process_log_file(log_path)
    row_order = ['Random', 'MaxCosine', 'MinCosine', 'Fixed']
    column_order = ['Height', 'EvenCluster', 'UnevenCluster', 'Random']
    markdown_tables = generate_markdown_tables(df, row_order=row_order, column_order=column_order)

    with open(table_path, 'w') as result_file:
        for KV, table in markdown_tables.items():
            result_file.write(table)
            result_file.write("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
