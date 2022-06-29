import optuna
import pandas as pd

def load_studies(path):
    studyPaths = list(path.glob(f'*.db'))
    studyPaths = sorted(studyPaths)
    studyDfs = []
    for studyPath in studyPaths:
        print(f"Loading Study from path: {studyPath}")
        studies = optuna.study.get_all_study_summaries(storage=f"sqlite:///{str(studyPath)}")
        loaded_study = optuna.load_study(study_name=studies[0].study_name, storage=f"sqlite:///{str(studyPath)}")
        trials = loaded_study.trials_dataframe()
        studyDfs.append(trials)
    df = pd.concat(studyDfs)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={"value": "Objective Value", "user_attrs_architecture": "Architecture", "user_attrs_encoder_name": "Feature Extractor"}, inplace=True)
    return df

def select_best_architecture(data):
    df = data[data["Objective Value"] == data["Objective Value"].min()]
    print(f"best Model: \n{df[['Objective Value', 'Architecture', 'Feature Extractor']]}")
    return df

def group_best_trial(data):
    best_val = data.groupby(["Architecture","Feature Extractor"]).min()
    best_val = best_val[['Objective Value']].reset_index().copy()
    std = data.groupby(["Architecture","Feature Extractor"]).std()
    std = std[['Objective Value']].reset_index().copy()
    std.columns = ["Architecture","Feature Extractor", "std(Objective Value)"]
    print("\n")
    print("Best trial per architecture and feature extractor:")
    print(best_val)
    print("\n")
    print("Standard Deviation of the studies:")
    print(std)
    return best_val, std