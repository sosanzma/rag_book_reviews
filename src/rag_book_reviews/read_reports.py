import os

def read_reports():
    reports = {}
    report_types = ['goodreads', 'reddit']
    
    for report_type in report_types:
        filename = f"{report_type}_report.txt"
        filepath = os.path.join('reports', filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if content:
                reports[report_type] = content
            else:
                print(f"Warning: {filename} is empty.")
        else:
            print(f"Warning: {filename} not found.")
    return reports
