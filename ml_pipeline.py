
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

# Pretty-print function
def pprint(obj):
    print(json.dumps(obj, indent=2))

# Load parameters from JSON file
with open(r'C:\Users\DELL\Downloads\DS_Assignment - internship\Screening Test - DS\algoparams_from_ui.json') as file:

    params = json.load(file)

# Extract parameters
target_info = params['design_state_data']['target']
feature_handling = params['design_state_data']['feature_handling']
dataset_name = params['design_state_data']['session_info']['dataset']
feature_reduction_config = params['design_state_data']['feature_reduction']
algorithms = params['design_state_data']['algorithms']

# Display extracted information
print("\nTarget:")
pprint(target_info)

print("\nFeature Handling:")
pprint(feature_handling)

print(f"\nDataset name: {dataset_name}")

# Load dataset
#df = pd.read_csv(dataset_name)
# Correcting to use an absolute path
#dataset_name = params['design_state_data']['session_info']['dataset']
full_path = r'C:\Users\DELL\Downloads\DS_Assignment - internship\Screening Test - DS\\' + dataset_name
df = pd.read_csv(full_path)


# === Data Preprocessing ===
def preprocess_data(df, feature_handling):
    for col, feature in feature_handling.items():
        if not feature['is_selected']:
            df.drop(columns=[col], inplace=True)

        if feature["feature_variable_type"] == "numerical":
            missing_values_action = feature['feature_details']["missing_values"]
            impute_method = feature['feature_details'].get('impute_with', None)

            if missing_values_action == "Impute":
                if impute_method == "Average of values":
                    df[col] = df[col].fillna(df[col].mean())
                elif impute_method == "custom":
                    impute_value = feature['feature_details']['impute_value']
                    df[col] = df[col].fillna(impute_value)
                else:
                    raise ValueError(f"Unknown imputation method: {impute_method}")

        elif feature["feature_variable_type"] == "text":
            unique_labels = {key: idx for idx, key in enumerate(df[col].unique())}
            df[col] = df[col].map(unique_labels)
        else:
            raise ValueError(f"Unknown feature type: {feature['feature_variable_type']}")

    return df

df = preprocess_data(df, feature_handling)

# === Feature Reduction ===
def perform_feature_reduction(df, config, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if config['feature_reduction_method'] == "Tree-based":
        model_cls = RandomForestRegressor if target_info['type'] == "regression" else RandomForestClassifier
        model = model_cls(n_estimators=int(config['num_of_trees']), max_depth=int(config['depth_of_trees']))
        sel = SelectFromModel(model)
        sel.fit(X, y)

        # Keep top features based on importance
        important_features = sel.estimator_.feature_importances_
        top_indices = np.argsort(important_features)[::-1][:int(config['num_of_features_to_keep'])]
        selected_columns = list(df.columns[top_indices]) + [target_col]
        return df[selected_columns]

    elif config['feature_reduction_method'] == "Correlation with target":
        correlations = df.corr()[target_col].abs().drop(target_col)
        top_features = correlations.nlargest(int(config['num_of_features_to_keep'])).index
        return df[top_features.to_list() + [target_col]]

    elif config['feature_reduction_method'] == "Principal Component Analysis":
        pca = PCA(n_components=int(config['num_of_features_to_keep']))
        X_pca = pca.fit_transform(X)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        return pd.DataFrame(X_pca)

    return df

target_col = target_info['target']
df = perform_feature_reduction(df, feature_reduction_config, target_col)

# === Model Training ===
def train_model(algorithm_params, X, y):
    model_name = algorithm_params.get('model_name')
    if model_name == "Random Forest Regressor":
        param_grid = {
            'n_estimators': range(algorithm_params["min_trees"], algorithm_params["max_trees"] + 1),
            'max_depth': range(algorithm_params["min_depth"], algorithm_params["max_depth"] + 1),
            'min_samples_leaf': range(algorithm_params["min_samples_per_leaf_min_value"], algorithm_params["min_samples_per_leaf_max_value"] + 1)
        }
        model = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1)
        model.fit(X, y)
        print(f"Best parameters: {model.best_params_}")
        print(f"Best score: {model.best_score_}")
        return model.best_estimator_

    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(**algorithm_params)
        model.fit(X, y)
        return model

    # Add other algorithms here as needed
    return None

# Train selected model
X = df.drop(columns=[target_col]).values
y = df[target_col].values

for algo_name, algo_params in algorithms.items():
    if not algo_params.pop('is_selected', False):
        continue
    model = train_model(algo_params, X, y)
    print(f"Trained Model: {algo_params['model_name']}")
    pprint(algo_params)
    if model:
        break

print("Process Completed.")

