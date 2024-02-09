import gzip
import shutil
import os
import glob as gb
from pathlib import Path
from Support_scripts.paths_via_class import DataPaths


if __name__ == '__main__':

    date = '2024-01-04'
    main_dir = date[0:4]
    data_dir = f'{date[0:4]}_{date[5:7]}_{date[8:]}sun'
    path_obj = DataPaths(date, data_dir, main_dir)

    az_dict = {'+24': '_01', '+20': '_02', '+16': '_03', '+12': '_04', '+08': '_05', '+04': '_06', '+00': '_07',
               '-04': '_08', '-08': '_09', '-12': '_10', '-16': '_11', '-20': '_12', '-24': '_13'}

    paths = gb.glob(str(Path(path_obj.primary_dir_path, "*bin.gz")))    # Set Pattern in glob() function
    if paths:
        for s in paths:
            with gzip.open(s, 'rb') as f_in:
                if not os.path.exists(s[: -3]):
                    with open(s[: -3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                else:
                    print(f'File {s[: -3]} exists')

        for s in paths:
            if os.path.exists(s):
                os.remove(s)
                print(f"{s} удален.")
    else:
        print('Paths to files not found')

    paths = gb.glob(str(Path(path_obj.primary_dir_path, "*.bin")))  # Set Pattern in glob() function
    for s in paths:
        s_num = s[-7:-4]
        s_new = str(Path(path_obj.primary_dir_path, date + az_dict[s_num] + s[-7::]))
        os.rename(s, s_new)
    pass
