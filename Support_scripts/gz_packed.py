import gzip
import io
import os, sys, shutil
import numpy as np
import glob as gb
from pathlib import Path
from Support_scripts.paths_via_class import DataPaths


if __name__ == '__main__':

    date = '2023-12-15'
    main_dir = date[0:4]
    data_dir = f'{date[0:4]}_{date[5:7]}_{date[8:]}sun'
    path_obj = DataPaths(date, data_dir, main_dir)

    az_dict = {'+24': '_01', '+20': '_02', '+16': '_03', '+12': '_04', '+08': '_05', '+04': '_06', '+00': '_07',
               '-04': '_08', '-08': '_09', '-12': '_10', '-16': '_11', '-20': '_12', '-24': '_13'}

    paths = gb.glob(str(Path(path_obj.converted_dir_path, "*.npy")))  # Set Pattern in glob() function
    if paths:
        for s in paths:
            filename_in = s
            filename_out = f'{s}.gz'
            if not os.path.exists(filename_out):
                with open(Path(s), 'rb') as inp:
                    current_file = np.load(inp, allow_pickle=True)

                # обратите внимание как открывается выходной файл `gzip.open()`
                with open(filename_in, "rb") as fin, gzip.open(filename_out, "wb") as fout:
                    # Читает файл по частям, экономя оперативную память
                    shutil.copyfileobj(fin, fout)

                print(f"Несжатый размер: {str(s)} - {os.stat(filename_in).st_size}")
                print(f"Сжатый размер: {os.stat(filename_out).st_size}")

            if os.path.exists(s) & os.path.exists(f'{s}.gz'):
                os.remove(s)
                print(f"{s} удален.")
            # with gzip.open(filename_out, "rb") as fin:
            #     data = np.load(fin, allow_pickle=True)
            #     # data = fin.read()
            #     print(f"Несжатый размер: {sys.getsizeof(data)}")
