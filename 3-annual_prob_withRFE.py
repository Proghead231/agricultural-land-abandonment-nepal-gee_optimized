import sys
import os
import logging
from helpers import landsat_composites
from helpers import config
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import joblib
from pathlib import Path
import ee
ee.Authenticate()
ee.Initialize(project="ee-joshisur231")
import geemap

log_filepath = Path(r"code\outputs\RF\logs\random_forest_training.log")
log_filepath.parent.mkdir(parents=True, exist_ok=True)
log_filepath.touch(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, mode='w'), # Logs to the file
        logging.StreamHandler(sys.stdout)            # Logs to the console
    ]
)

config.ROI  = ee.FeatureCollection("projects/ee-joshisur231/assets/pa_effectiveness/nepal_boundary").first().geometry()
base_out_dir = r"code/outputs/RF/results/with_landsat7/global"
class_report_dir = os.path.join(base_out_dir, "classification_reports")
feature_imp_dir = os.path.join(base_out_dir, "feature_importance")
cv_result_dir = os.path.join(base_out_dir, "cv_results")
model_dir = os.path.join(base_out_dir, "saved_models")
predictions_dir = os.path.join(base_out_dir, "test_predictions")
conf_matrix_dir = os.path.join(base_out_dir, "confusion_matrices")
rfecv_results_dir = os.path.join(base_out_dir, "rfecv_results")
rfecv_ranking_dir = os.path.join(base_out_dir, "rfecv_ranking")
best_params_dir = os.path.join(base_out_dir, "best_params")
rfecv_model_dir = os.path.join(base_out_dir, "saved_rfecv_models")

os.makedirs(class_report_dir, exist_ok=True)
os.makedirs(feature_imp_dir, exist_ok=True)
os.makedirs(cv_result_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(conf_matrix_dir, exist_ok=True)
os.makedirs(rfecv_results_dir, exist_ok=True)
os.makedirs(rfecv_ranking_dir, exist_ok=True)
os.makedirs(best_params_dir, exist_ok=True)
os.makedirs(rfecv_model_dir, exist_ok=True)

train_samples_dir = r"code/outputs/RF/training_samples_with_predictors/training_samples_with_l7"
train_samples_files = sorted([os.path.join(train_samples_dir, f) for f in os.listdir(train_samples_dir)])
predictors = [
    #'blue_mean', 'blue_median', 'blue_p25', 'blue_p75', 'blue_stdDev',
    'green_median', 'green_p25', 'green_p75', #'green_stdDev','green_mean',
    'red_median', 'red_p25', 'red_p75', #'red_stdDev','red_mean', 
    'nir_median', 'nir_p25', 'nir_p75', #'nir_stdDev','nir_mean', 
    'swir1_median', 'swir1_p25', 'swir1_p75', #'swir1_stdDev', 'swir1_mean', 
    'swir2_median', 'swir2_p25', 'swir2_p75', #'swir2_stdDev', 'swir2_mean', 
    'ndvi_median', 'ndvi_p25', 'ndvi_p75', 'ndvi_iqr', #'ndvi_stdDev','ndvi_mean',
    'evi_median', 'evi_p25', 'evi_p75', 'evi_iqr', #'evi_stdDev','evi_mean', 
    'msavi_median', 'msavi_p25', 'msavi_p75', 'msavi_iqr', #'msavi_stdDev','msavi_mean',
    'ndmi_median', 'ndmi_p25', 'ndmi_p75', #'ndmi_stdDev', ndmi_mean
    'slope', 'tri_dtm', #'tri_srtm',
    'ndvi_median_asm', 'ndvi_median_contrast', 'ndvi_median_corr', 'ndvi_median_dent',
    'ndvi_median_diss', 'ndvi_median_dvar', 'ndvi_median_ent','ndvi_median_idm', 
    'ndvi_median_imcorr1', 'ndvi_median_imcorr2', 'ndvi_median_inertia', 'ndvi_median_maxcorr', 
    'ndvi_median_prom', 'ndvi_median_savg', 'ndvi_median_sent', 'ndvi_median_shade',
    'ndvi_median_svar', 'ndvi_median_var'   
]

target = 'stbl_crop'
crs = "EPSG:4326"
scale = 30
#Initial Checks
expected_years = set(range(2000, 2025))
actual_years = set(int(f.split("_")[-1].split(".")[0]) for f in os.listdir(train_samples_dir) if f.endswith(".csv"))
missing_years = expected_years - actual_years
if missing_years:
    logging.warning(f"Missing training samples for years: {sorted(list(missing_years))}")
    logging.info(f"Continuing with the available training data for years: {sorted(list(actual_years))}")

for train_file in train_samples_files:
    year = int(train_file.split("/")[-1].split("_")[-1].split(".")[0])
    if year in [2000, 2001, 2002, 2003, 2004, 2005, 2006, 
    2007, 2008, 2009, 2010, 2018, 2019, 2020, 2021,
    2022, 2023, 2024]:
        logging.info(f"Skipping year {year} as it is already exported")
        continue
    samples_df = pd.read_csv(train_file)
    MIN_PX_COUNT = 6 
    before = len(samples_df)
    samples_df = samples_df[samples_df['px_count'] >= MIN_PX_COUNT]
    after = len(samples_df)
    logging.info(f"Removed {before - after} samples with px_count < {MIN_PX_COUNT}")
        
    # checks
    if samples_df[predictors].shape != (2118, len(predictors)):
        logging.warning(f"Some data might be missing in the {train_file} | Shape: {samples_df.shape}")
    
    null_vals = samples_df.isna().sum()
    if null_vals.any():
        logging.warning(f"{null_vals[predictors][null_vals[predictors] > 0].sum()} missing values in {train_file} | Columns: {null_vals[null_vals > 0].index.tolist()}")        
        logging.info(f"Dropping rows with missing values in {train_file}")
        samples_df = samples_df.dropna(subset=predictors)

    X = samples_df[predictors]
    y = samples_df[target]

    logging.info(f"Imbalance check for year {year}")
    logging.info(f"Percentage of stable crop: {(y.sum() / y.shape[0]) * 100:.2f}%")
    logging.info(f"Percentage of stable noncrop: {((y == 0).sum() / y.shape[0]) * 100:.2f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=45)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=45)
    logging.info(f"Shape of X_train: {X_train.shape}")
    logging.info(f"Shape of X_test: {X_test.shape}")
    logging.info(f"Shape of y_train: {y_train.shape}")
    logging.info(f"Shape of y_test: {y_test.shape}")

    param_grid = {
        'n_estimators': [50, 100, 200, 300, 400, 500, 800], 
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_leaf_nodes': [None, 100, 200, 500],
        'max_samples': [0.5, 0.63, 0.8]
    }

    # cannot use the following logic because the RF in GEE doesnt have a class weight param
    # class_counts = samples_df[target].value_counts()
    # class_percentages = samples_df[target].value_counts(normalize=True) * 100
    # majority_share = class_percentages.max()
    # if majority_share >= 65:
    #     logging.info(f"Class imbalance, the majority class takes up {majority_share:.1f}%. Using class_weight='balanced'")
    #     rf_model = RandomForestClassifier(class_weight='balanced', random_state=45)
    # else:
    #     logging.info(f"Class balance, the majority class takes up {majority_share:.1f}%. Not using class_weight='balanced'")
    #     rf_model = RandomForestClassifier(n_estimators = 100, random_state=45)
    rf_model = RandomForestClassifier(n_estimators = 100, random_state=45)
    # RFECV wraps the base model
    rfecv = RFECV(
        estimator=rf_model,
        step=3,
        cv=kfold,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )
    rfecv.fit(X_train, y_train)
    selected_features = X_train.columns[rfecv.support_].tolist()
    logging.info(f"Selected {len(selected_features)} features for year {year}: {selected_features}")

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    logging.info("Starting Hyperparameter Tuning on selected features...")
    
    rf_model = RandomForestClassifier(random_state=45)
    # RandomizedSearchCV wraps the base RF model to tune on selected features
    rf_search = RandomizedSearchCV(
        estimator=rf_model, 
        param_distributions=param_grid, 
        n_iter=100, 
        cv=kfold, 
        verbose=3, 
        random_state=45, 
        n_jobs=-1,
        scoring="f1"
    )
    rf_search.fit(X_train_sel, y_train)
    
    best_estimator = rf_search.best_estimator_
    # Get the best parameters directly
    best_params = rf_search.best_params_

    # Predict using the best estimator on the selected features
    y_pred = best_estimator.predict(X_test_sel)
    y_pred_proba = best_estimator.predict_proba(X_test_sel)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual_NonCrop', 'Actual_Crop'], columns=['Predicted_NonCrop', 'Predicted_Crop'])
    
    # Keep the original index so we can join back to the samples for spatial error analysis
    test_results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted_Class': y_pred,
        'Predicted_Probability': y_pred_proba
    }, index=y_test.index)
    
    # Grab any extra metadata columns (like coordinates or system:index) from the original dataframe
    meta_cols = [c for c in samples_df.columns if c not in predictors + [target]]
    if meta_cols:
        test_results_df = test_results_df.join(samples_df.loc[y_test.index, meta_cols])

    results_df = pd.DataFrame(rf_search.cv_results_)
    class_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    
    feat_imp = pd.DataFrame({
        'feature': selected_features,
        'importance': best_estimator.feature_importances_
    })
    perm_imp = permutation_importance(
    best_estimator, X_test_sel, y_test,
    n_repeats=10, scoring='f1', random_state=45, n_jobs=-1
    )
    perm_imp_df = pd.DataFrame({
        'feature': selected_features,
        'importance_mean': perm_imp.importances_mean,
        'importance_std': perm_imp.importances_std
    }).sort_values('importance_mean', ascending=False)

    
    class_report.to_csv(os.path.join(class_report_dir, f"classification_report_{year}.csv"))
    feat_imp.to_csv(os.path.join(feature_imp_dir, f"feature_importance_{year}.csv"))
    perm_imp_df.to_csv(os.path.join(feature_imp_dir, f"perm_feature_importance_{year}.csv"))
    results_df.to_csv(os.path.join(cv_result_dir, f"cv_results_{year}.csv"))
    # Save test predictions with index=True to preserve the original row indices
    test_results_df.to_csv(os.path.join(predictions_dir, f"test_predictions_{year}.csv"), index=True)
    cm_df.to_csv(os.path.join(conf_matrix_dir, f"confusion_matrix_{year}.csv"))
    joblib.dump(best_estimator, os.path.join(model_dir, f"rf_model_{year}.joblib"))
    joblib.dump(rfecv, os.path.join(rfecv_model_dir, f"rfecv_model_{year}.joblib"))
    
    # Save RFECV performance curve (Accuracy vs. Number of Features)
    # Filter for 1D arrays to avoid ValueError: Per-column arrays must each be 1-dimensional
    rfecv_results_1d = {k: v for k, v in rfecv.cv_results_.items() if hasattr(v, 'ndim') and v.ndim == 1}
    rfecv_curve = pd.DataFrame(rfecv_results_1d)
    rfecv_curve.to_csv(os.path.join(rfecv_results_dir, f"rfecv_curve_{year}.csv"), index=False)
    
    # Save the elimination ranking of ALL original features
    rfecv_ranking = pd.DataFrame({
        'feature': predictors,
        'ranking': rfecv.ranking_,
        'selected': rfecv.support_
    }).sort_values(by='ranking')
    rfecv_ranking.to_csv(os.path.join(rfecv_ranking_dir, f"rfecv_ranking_{year}.csv"), index=False)
    
    # Save a clean version of the final hyperparameters
    pd.Series(best_params).to_csv(os.path.join(best_params_dir, f"best_params_{year}.csv"), header=["value"])
    
    #Checks before using the modelbagFraction
    def check_f1_feat():
        check = True
        class_report = pd.read_csv(os.path.join(class_report_dir, f"classification_report_{year}.csv"), index_col=0)
        feature_imp = pd.read_csv(os.path.join(feature_imp_dir, f"feature_importance_{year}.csv"))
        
        # 1. Check Overall Accuracy (F1)
        overall_f1 = class_report.loc['macro avg', 'f1-score']
        if overall_f1 < 0.85:
            logging.warning(f"Overall F1 score for year {year} is {overall_f1:.2f} (less than 0.85)")
            check = False
            
        # 2. Check Feature Importance Health
        top_features = feature_imp.sort_values(by='importance', ascending=False)
        
        # A. Check for single-feature dominance (e.g., > 40% importance)
        top_1_imp = top_features.iloc[0]['importance']
        top_1_name = top_features.iloc[0]['feature']
        if top_1_imp > 0.40:
            logging.warning(f"Model relies too heavily on a single feature ({top_1_name}: {top_1_imp:.2f}). Possible proxy bias.")
            check = False
            
        # B. Ensure physical phenology metrics are driving the model
        top_5_names = top_features.head(5)['feature'].tolist()
        has_phenology = any(('iqr' in feat or 'p75' in feat or 'p25' in feat) for feat in top_5_names)
        if not has_phenology:
            logging.warning(f"Top 5 features do not include temporal metrics (iqr, p75). The model might be ignoring agricultural cycles.")
            check = False
            
        return check

    check_before = check_f1_feat()
    if check_before == False:
        logging.warning(f"{year} has low F1 score or high single feature importance")

    samples_train = ee.FeatureCollection(f"projects/ee-joshisur231/assets/agriculture_abandonment_nepal_revised/training_samples_with_l7/trainSamp_{year}")\
        .filter(ee.Filter.gte('px_count', MIN_PX_COUNT))\
        .filter(ee.Filter.notNull(selected_features))

    landsat_image_collection = "projects/ee-joshisur231/assets/agriculture_abandonment_nepal_revised/landsat_images_for_prediction"
    image_name = f'predictors_{year}'
    asset_id = f'{landsat_image_collection}/{image_name}'
    l = ee.Image(asset_id)
    mf = best_params["max_features"]
    if mf == "log2":
        vars_per_split = max(1, round(math.log2(len(selected_features))))
    elif mf == "sqrt":
        vars_per_split = max(1, round(math.sqrt(len(selected_features))))
    else:
        vars_per_split = max(1, round(mf * len(selected_features)))

    gee_params = {
            'numberOfTrees': best_params['n_estimators'],
            'minLeafPopulation': best_params['min_samples_leaf'],
            'variablesPerSplit': vars_per_split,
            'bagFraction': best_params['max_samples'],
            'seed': 45
    }
    if best_params['max_leaf_nodes'] is not None:
        gee_params['maxNodes'] = best_params['max_leaf_nodes']

    rf_classifier = ee.Classifier.smileRandomForest(**gee_params)\
        .setOutputMode('MULTIPROBABILITY')\
        .train(features = samples_train, classProperty="stbl_crop", inputProperties=selected_features)

    classified = l.classify(rf_classifier)
    probabilities = classified.arrayFlatten([["noncrop_prob", "crop_prob"]])

    geemap.ee_export_image_to_asset(
        image=probabilities.select("crop_prob")
            .addBands(l.select("px_count"))
            .addBands(l.select("quality_flag"))
            .clip(config.ROI),
        description=f'crop_prob_{year}',
        assetId=f'projects/ee-joshisur231/assets/agriculture_abandonment_nepal_revised/crop_prob_ic/crop_prob_{year}',
        region=config.ROI,
        scale=scale,
        crs='EPSG:32645',
        maxPixels=1e13
    )