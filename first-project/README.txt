1. Requirements
python >=3.10.7

matplotlib==3.8.2
numpy==1.26.2
opencv_python==4.8.1.78

2. How to run tasks
In folder `data/` there are: a folder named `imagini_auxiliare` where there is empty table image (it needs to stay there for the script to work)
and a folder named `templates`, where are some template images for template matching.

In file `run_project.py`, at line 74 there is a call for above-defined function `run_double_double_domino_detector`.
Change path_test and path_write_folder parameters before running.
Parameters:
- path_test=PATH_FOLDER_TEST  # path to folder containing images and moves files
- path_write_folder='fisiere_solutie'  # writes the results in this folder (20 * number of games .txt files) => if None => path_write_folder = path_test
- write=True  # write results, default False (they are just returned from this function, not written in files)
- games=None  # if None => games = [1, 2, 3, 4, 5]

Warning: if some files named `1_01.txt` ... `5_20.txt` are there, they will be overwritten!
All tasks are run by command `python run_project.py`.


