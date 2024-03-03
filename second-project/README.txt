1.Requirements:

certifi==2023.11.17
charset-normalizer==3.3.2
contourpy==1.2.0
cycler==0.12.1
filelock==3.13.1
fonttools==4.47.0
fsspec==2023.12.2
idna==3.6
imageio==2.33.1
importlib-resources==6.1.1
install==1.3.5
Jinja2==3.1.2
joblib==1.3.2
kiwisolver==1.4.5
lazy_loader==0.3
MarkupSafe==2.1.3
matplotlib==3.8.2
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.2
opencv-python==4.8.1.78
opencv-python-rolling==5.0.0.20221015
packaging==23.2
Pillow==10.1.0
pyparsing==3.1.1
python-dateutil==2.8.2
requests==2.31.0
scikit-image==0.22.0
scikit-learn==1.3.2
scipy==1.11.4
six==1.16.0
sympy==1.12
threadpoolctl==3.2.0
tifffile==2023.12.9
torch==2.1.2
torchvision==0.16.2
typing_extensions==4.9.0
urllib3==2.1.0
zipp==3.17.0

2.How to run the project:

2.1. 
    Add test images in `/data/test/` directory.
2.2
    In `src` directory there is a file named `run_tasks.py`.
    In order to run the project, you have to navigate into src project and from there to run in the terminal `python3 run_tasks.py`.
    This command runs both tasks.
    Tasks can be run separately but carefully because task 2 depends on task 1, in the sense that task 1 saves some output in `results/task1/` THAT WILL BE USED BY TASK 2.

Results will be saved in `results/task1/` and `results/task2/`.


Task 1 - around     15-20 	minutes to run on 200 test images
Task 2 - less than  2 		minutes to run on 200 test images (in reality task 2 runs on intermediary detections of task 1 - see documentation)

Note: in `task1_2_dataset.zip` there are images used for training.
