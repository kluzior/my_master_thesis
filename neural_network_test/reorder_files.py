import os

def rename_png_files(folder_path):
    # Get a list of all PNG files in the folder
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    # Sort the files alphabetically (or by any custom criteria)
    files.sort()

    # Rename each file in increasing order with leading zeros
    for i, filename in enumerate(files):
        new_name = f"{i + 1:03}.png"  # Format number as three digits with leading zeros
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        
        os.rename(src, dst)
        print(f'Renamed {filename} to {new_name}')

# Specify the path to the folder containing the PNG files
folder_path = 'images/images_26_06_24/objects/refs/obj5'

# Call the function to rename the PNG files
rename_png_files(folder_path)
