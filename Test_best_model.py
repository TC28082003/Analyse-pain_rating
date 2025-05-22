import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def preprocess_and_train_s_v_r():
    try:
        df_raw = pd.read_csv('Pain_python.csv')
    except FileNotFoundError:
        print("Error: Pain_python.csv not found. Make sure it's in the correct directory.")
        return None, None, np.nan, np.nan, 0, 0.0

    # 2. Preprocessing
    df_t0 = df_raw[df_raw['Timepoint'] == 'T0'].copy()
    df_t2 = df_raw[df_raw['Timepoint'] == 'T2'].copy()
    df_t2_target = df_t2[['Number', 'Pain rating']].copy()
    df_t2_target.rename(columns={'Pain rating': 'Pain_rating_T2'}, inplace=True)
    merged_df = pd.merge(df_t0, df_t2_target, on='Number', how='inner')
    merged_df.drop(columns=['Timepoint'], inplace=True, errors='ignore')
    if 'CD57+NK cells' in merged_df.columns:
        merged_df.drop(columns=['CD57+NK cells'], inplace=True)
    merged_df.dropna(subset=['Pain_rating_T2'], inplace=True)
    patient_numbers = merged_df['Number']
    merged_df_features = merged_df.drop(columns=['Number', 'Pain_rating_T2'])
    categorical_cols = merged_df_features.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Q28. JointP' not in categorical_cols and 'Q28. JointP' in merged_df_features.columns:
        categorical_cols.append('Q28. JointP')
    numerical_cols = merged_df_features.select_dtypes(include=np.number).columns.tolist()

    # 3. Data Splitting
    X = merged_df_features
    y = merged_df['Pain_rating_T2']
    X_train_df, X_test_df, y_train, y_test, numbers_train, numbers_test = train_test_split(
        X, y, patient_numbers, test_size=0.2, random_state=42
    )

    # Imputation
    for col in numerical_cols:
        median_val = X_train_df[col].median()
        X_train_df.loc[:, col] = X_train_df[col].fillna(median_val)
        X_test_df.loc[:, col] = X_test_df[col].fillna(median_val)
    if 'Q28. JointP' in X_train_df.columns:
        X_train_df.loc[:, 'Q28. JointP'] = X_train_df['Q28. JointP'].astype(str)
        X_test_df.loc[:, 'Q28. JointP'] = X_test_df['Q28. JointP'].astype(str)
    for col in categorical_cols:
        if col in X_train_df.columns:
            mode_val = X_train_df[col].mode()[0] if not X_train_df[col].mode().empty else 'Unknown'
            X_train_df.loc[:, col] = X_train_df[col].fillna(mode_val)
            X_test_df.loc[:, col] = X_test_df[col].fillna(mode_val)
            X_train_df.loc[:, col] = X_train_df[col].astype(str)
            X_test_df.loc[:, col] = X_test_df[col].astype(str)

    X_train_encoded = pd.get_dummies(X_train_df, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_df, columns=categorical_cols, drop_first=True)
    train_cols = X_train_encoded.columns
    test_cols = X_test_encoded.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test_encoded[c] = 0
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train_encoded[c] = 0
    X_test_encoded = X_test_encoded[train_cols]

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # 5. Train SVR
    svr_tuned = SVR(C=100, epsilon=0.5, gamma='scale', kernel='rbf')
    svr_tuned.fit(X_train_scaled, y_train)

    # 6. Predictions
    y_pred_svr_tuned = svr_tuned.predict(X_test_scaled)

    # 7. Output DataFrame
    results_df = pd.DataFrame({
        'Number': numbers_test,
        'Pain_rating_T2_actual': y_test,
        'Pain_rating_T2_predict': y_pred_svr_tuned
    })
    results_df['Difference'] = results_df['Pain_rating_T2_predict'] - results_df['Pain_rating_T2_actual']
    results_df.sort_values(by='Number', inplace=True)

    mae_svr_tuned = mean_absolute_error(y_test, y_pred_svr_tuned)
    rmse_svr_tuned = np.sqrt(mean_squared_error(y_test, y_pred_svr_tuned))
    abs_diff = np.abs(results_df['Difference'])
    count_less_than_1 = np.sum(abs_diff < 1.0)
    percentage_less_than_1 = (count_less_than_1 / len(y_test)) * 100

    # For local execution:
    print(f"Tuned SVR - MAE: {mae_svr_tuned:.4f}")
    print(f"Tuned SVR - RMSE: {rmse_svr_tuned:.4f}")
    print(f"Tuned SVR - Count of abs(Difference) < 1.0: {count_less_than_1}")
    print(f"Tuned SVR - Percentage of abs(Difference) < 1.0: {percentage_less_than_1:.2f}%")
    results_df.to_csv('final_predictions_report1.csv', index=False)

    return results_df  # Or other relevant outputs for further use

if __name__ == '__main__':
    preprocess_and_train_s_v_r()