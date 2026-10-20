import os

def convert_crlf_to_lf(root_dir):
    """
    Recursively finds all .py files in the given directory and 
    converts their CRLF line endings to LF.
    """
    count = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(subdir, file)
                
                # Read content and replace CRLF (\r\n) with LF (\n)
                with open(file_path, 'rb') as f:
                    content = f.read()
                    
                # Using .replace() safely updates the bytes in-memory
                new_content = content.replace(b'\r\n', b'\n')
                
                # Only write to the file if the content actually changed
                if new_content != content:
                    with open(file_path, 'wb') as f:
                        f.write(new_content)
                    print(f"Converted: {file_path}")
                    count += 1
                    
    print(f"\nDone! Processed and converted {count} Python file(s).")

if __name__ == '__main__':
    # "." represents the current directory
    current_directory = '.'
    convert_crlf_to_lf(current_directory)
