from pathlib import Path
from utils.studies import load_studies, select_best_architecture, group_best_trial
from utils.parser import compare_studies_parser

args = compare_studies_parser()
root_path:str = args.root_path

# TODO: change path to folder with db here
db_path = Path(root_path) / "results" / "studies" 
df = load_studies(path=db_path)
_ = select_best_architecture(data=df)
_, _ = group_best_trial(data=df)
