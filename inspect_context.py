import zipfile
import xml.etree.ElementTree as ET
import csv
import os
from collections import Counter

def read_docx(file_path):
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = [node.text for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error reading docx: {e}"

def inspect_dataset(file_path):
    try:
        print("\n--- DATASET INSPECTION ---")
        print(f"File: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Columns: {header}")
            
            rows = list(reader)
            total_rows = len(rows)
            print(f"Total Rows: {total_rows}")
            
            # Assuming label is the second column (index 1) based on previous view_file
            labels = [row[1] for row in rows if len(row) > 1]
            label_counts = Counter(labels)
            
            print("\nLabel Distribution:")
            for label, count in label_counts.items():
                print(f"{label}: {count}")
                
            print("\nSample (first 3 rows):")
            for i in range(min(3, len(rows))):
                print(rows[i])
                
    except Exception as e:
        print(f"Error reading dataset: {e}")

def main():
    base_dir = r"H:\The Thesis"
    synopsis_path = os.path.join(base_dir, "Synopsis_-_Zahid_Mahmood_L1F23MSDS0001 final.docx")
    dataset_path = os.path.join(base_dir, "cyberbullying_tweets.csv")

    print("--- SYNOPSIS CONTENT ---")
    synopsis_text = read_docx(synopsis_path)
    print(synopsis_text)

    inspect_dataset(dataset_path)

if __name__ == "__main__":
    main()
