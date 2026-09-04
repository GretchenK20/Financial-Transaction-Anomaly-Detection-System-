/*
  Gold: final feature mart for ML model consumption.
  One row per transaction. All numeric features are ready for the
  PyTorch autoencoder and XGBoost without further preprocessing.
*/
with features as (
    select * from {{ ref('int_transaction_features') }}
)

select
    transaction_id,
    time,
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
    v21, v22, v23, v24, v25, v26, v27, v28,
    amount,
    log_amount,
    hour_of_day,
    is_night_transaction,
    amount_bucket,
    amount_zscore,
    class,
    current_timestamp as feature_generated_at
from features
