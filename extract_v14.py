import os
import re

def extract_files(bible_path):
    with open(bible_path, 'r', encoding='latin-1') as f:
        content = f.read()

    # Split by file markers
    # Marker example: ================================================================================
    # FILE: AGENTS\instruction.txt
    file_sections = re.split(r'={80}', content)
    
    for section in file_sections:
        if 'FILE:' not in section:
            continue
            
        lines = section.strip().split('\n')
        file_path_line = [l for l in lines if l.startswith('FILE:')][0]
        file_path = file_path_line.replace('FILE:', '').strip()
        
        # Normalize path for OS
        file_path = file_path.replace('\\', os.sep)
        
        # Find file content between -----BEGIN FILE----- and -----END FILE-----
        file_content_match = re.search(r'-----BEGIN FILE-----\n(.*?)\n-----END FILE-----', section, re.DOTALL)
        if file_content_match:
            file_data = file_content_match.group(1)
            
            # Create directories if they don't exist
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f_out:
                f_out.write(file_data)
            print(f"Extracted: {file_path}")

if __name__ == "__main__":
    extract_files('bible_v14.txt')
