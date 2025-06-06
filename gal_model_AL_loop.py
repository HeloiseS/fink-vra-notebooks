from glob import glob
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
from datetime import datetime
import os
from finkvra.utils.features import make_features as fvra_make_features
from finkvra.utils.labels import cli_label_one_object as fvra_cli_label_one_object
import json
from mlflow.tracking import MlflowClient

# Make sure you start the server FROM THE FINK-VRA-NOTEBOOKS DIRECTORY: mlflow server --host 127.0.0.1 --port 6969

# TODO: put paths and constants in config file
# ------------
# Constants
# ------------
label2galclass = {'real': np.nan, 
                  'extragal': 0, 
                  'gal': 1, 
                  'agn': 0,
                  'bogus': np.nan, 
                  'varstar': 1,
                  None: np.nan
                 }


EXPERIMENT = "gal_model_AL"
SAMPLING_STRATEGY = "uncertainty"


# --------------
# 0. Set up 
# --------------


mlflow.set_tracking_uri("http://127.0.0.1:6969")
mlflow.set_experiment(EXPERIMENT)

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT)

# Get the run idea of the last SUCCESSFUL run
experiment_id = experiment.experiment_id

runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=1,
)

if not runs:
    raise RuntimeError("No successful (FINISHED) runs found.")

last_run = runs[0]
prev_run_id = last_run.info.run_id
print("Last successful run ID:", prev_run_id)


# -------------------
# 1. Load data
# -------------------
files = sorted(glob('/home/stevance/Data/FinkZTFStream/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
data = pd.concat(dfs, ignore_index=True)


# -------------------
# 2. Make our features
# -------------------
X, meta = fvra_make_features(data)
# remove samples that have no positive detections
valid_candid = list(X[X.ndets > 0].index.values)
X = X.loc[valid_candid]
meta = meta.loc[valid_candid]

# -------------------
# 3. Load previous training set and exclude
# -------------------
# Artifact: we log and load training IDs as a CSV
previous_ids_path = f"{EXPERIMENT}_training_ids.csv"
previous_ids_df = pd.read_csv(previous_ids_path)
CURRENT_ROUND= previous_ids_df.iloc[-1]['round'] + 1

train_ids = previous_ids_df["candid"].tolist()
X_pool = X.drop(index=train_ids, errors='ignore')
meta_pool = meta.drop(index=train_ids, errors='ignore')

# -------------------
# 4. Load previous model and predict
# -------------------
mlflow_uri = mlflow.get_tracking_uri()
model_subpath = "gal_model"
model_uri = f"runs:/{prev_run_id}/{model_subpath}"
clf = mlflow.sklearn.load_model(model_uri)
y_pred = clf.predict_proba(X_pool)[:, 1]  
y_pred_pool = pd.DataFrame( np.vstack((X_pool.index.values.astype(str), y_pred)).T, columns= ['candid', 'pred']).set_index('candid')


# -------------------
# 6. Active Learning sampling with dynamic labeling
# -------------------
y_pred_pool['uncertainty_score'] = np.abs(y_pred_pool["pred"].astype(float) - 0.5) 

labels = pd.read_csv('~/Data/FinkZTFStream/labeled.csv', index_col=0)

new_labels = []
new_label_candid = []
new_sample_candid = []

N_to_sample = 10
N_i = 0

for _candid in y_pred_pool.index.astype(np.int64):
    try: 
        try:
            classification = label2galclass[labels.loc[_candid].label]
        except TypeError:
            print("Shit! You got duplicate labels in Data/FinkZTFStream/labeld.csv")
            print(_candid)
            exit()
        if not np.isnan(classification):
            new_sample_candid.append(_candid)
            N_i += 1
    except KeyError: 
        # if _candid not in labels then labels.loc[_candid] will throw a KeyError 
        # and we can activate the logic below
        _objectId = meta.loc[_candid].objectId
        
        # this is where we need the labeling 
        label = fvra_cli_label_one_object(_objectId)
        if label is None: 
            continue
        new_labels.append(label)
        new_label_candid.append(_candid)
        classification = label2galclass[label]
        if not np.isnan(classification):
            new_sample_candid.append(_candid)
            N_i += 1
        
    if N_i == N_to_sample:
        break

if N_i < N_to_sample:
    print(f'not enough - N = {N_i}')
    
timestamp = datetime.utcnow().isoformat()
new_labels_df = pd.DataFrame.from_dict({
                                         'objectId': meta.loc[np.array(new_label_candid).astype(np.int64)].objectId,
                                          'label':  new_labels,
                                        'timestamp': timestamp
                                       })

updated_labels = pd.concat((labels, new_labels_df))

updated_labels.to_csv('~/Data/FinkZTFStream/labeled.csv')

new_ids_df = pd.DataFrame({'candid': np.array(new_sample_candid).astype(np.int64),
                           'round': CURRENT_ROUND,
                          })

train_ids_df = pd.concat([previous_ids_df, new_ids_df]).reset_index(drop=True)

# TODO - add {SAMPLING STRATEGY} to name?
train_ids_df.to_csv(f'./{EXPERIMENT}_training_ids.csv', index=False)

# -------------------
# 7. Make the y_train X_train for new round
# -------------------

# TODO: I think I need to save those somewhere so I can save them to artifacts in ML FLow
y_train = updated_labels.loc[train_ids_df.candid].label.map(label2galclass)
X_train = X.loc[train_ids_df.candid]


# -------------------
# 8. Train!
# -------------------

ARTIFACT_PATH = "gal_model"
MODEL_TAG = f"{ARTIFACT_PATH}_round_{CURRENT_ROUND}"

# TODO - add precision, recall, F1 score
with mlflow.start_run(run_name=f"round_{CURRENT_ROUND}_{SAMPLING_STRATEGY}"):

    # Log metadata
    meta_info = {
        "round": int(CURRENT_ROUND),
        "timestamp": datetime.utcnow().isoformat(),
        "n_train": int(X_train.shape[0]),
        "sampling_strategy": str(SAMPLING_STRATEGY),
        "model_tag": str(MODEL_TAG)
    }

    with open("meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)
    mlflow.log_artifact("meta.json")

    # Train model
    # Logging parameters
    mlflow.log_params(params)
    
    clf_new = HistGradientBoostingClassifier(max_iter=100, 
                                             l2_regularization=10,
                                             random_state=42,
                                             learning_rate=0.1)
    clf_new.fit(X_train.values, y_train.values)

    # Evaluate on training set
    acc = accuracy_score(y_train, clf_new.predict(X_train.values))
    mlflow.log_metric("train_accuracy", acc)

    # Log model
    signature = infer_signature(X_train, clf_new.predict(X_train))
    mlflow.sklearn.log_model(
        clf_new,
        artifact_path=ARTIFACT_PATH,
        signature=signature,
        input_example=X_train.iloc[:2]
    )

    # Save training state
    mlflow.log_artifact(f"{EXPERIMENT}_training_ids.csv")
    # TODO: Save the data as well
