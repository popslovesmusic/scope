import os

def list_project_contents(path):
    """
    Recursively lists the file and directory structure of a given path,
    excluding virtual environments and dependency directories.
    
    Args:
        path (str): The root directory to start the listing from.
    """
    # Directories to exclude (common dependency/cache directories)
    exclude_dirs = {
        '.venv', 'venv', 'env',  # Virtual environments
        'node_modules',           # Node.js dependencies
        '__pycache__',           # Python cache
        '.git',                  # Git repository data
        '.pytest_cache',         # Pytest cache
        '.mypy_cache',          # MyPy cache
        '.tox',                 # Tox testing
        '.coverage',            # Coverage data
        'htmlcov',              # Coverage HTML reports
        '.idea', '.vscode',     # IDE directories
        'site-packages'         # Python packages
    }
    
    for root, dirs, files in os.walk(path):
        # Remove excluded directories from dirs list to prevent os.walk from entering them
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Calculate the depth of the current directory
        level = root.replace(path, '').count(os.sep)
        # Create an indent for visual hierarchy
        indent = ' ' * 4 * level
        
        # Don't print the root directory name on first iteration
        if level == 0:
            print(f"Project structure for: {os.path.basename(path)}/\n")
        else:
            print(f'{indent}?? {os.path.basename(root)}/')
        
        subindent = ' ' * 4 * (level + 1)
        
        # Print files in the current directory, excluding common temporary/cache files
        exclude_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.egg-info'}
        exclude_files = {'.DS_Store', 'Thumbs.db', '.gitkeep'}
        
        for f in files:
            # Skip files with excluded extensions or names
            file_ext = os.path.splitext(f)[1].lower()
            if file_ext not in exclude_extensions and f not in exclude_files:
                print(f'{subindent}?? {f}')

if __name__ == "__main__":
    # Use the current working directory
    current_directory = os.getcwd()
    list_project_contents(current_directory)