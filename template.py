import os
from pathlib import Path

list_of_file = [
    ".git/workflow/.gitkeep",
    "src/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_monitoring.py",
    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/prediction_pipeline.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",
    "Dockerfile",
    "app.py",
]

for filepath in list_of_file:
    filepath = Path(filepath)
    filedir,_ = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize() == 0):
        with open(filepath,'wb') as f:
            pass
    else:
        print('File Already Exists')
