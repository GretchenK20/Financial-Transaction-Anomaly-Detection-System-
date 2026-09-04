/*
  Silver: transaction-level feature engineering.
  One row per transaction. Derives amount/time features on top of the raw
  V1-V28 PCA components, which pass through unchanged.

  Time is "seconds elapsed since the first transaction in the dataset"
  (spans ~48 hours), so time mod 86400 recovers a within-day clock used
  for hour_of_day / is_night_transaction.
*/
with txns as (
    select * from {{ ref('stg_transactions') }}
),

stats as (
    select
        avg(amount) as amount_mean,
        stddev(amount) as amount_stddev
    from txns
),

featured as (
    select
        t.transaction_id,
        t.time,
        t.v1, t.v2, t.v3, t.v4, t.v5, t.v6, t.v7, t.v8, t.v9, t.v10,
        t.v11, t.v12, t.v13, t.v14, t.v15, t.v16, t.v17, t.v18, t.v19, t.v20,
        t.v21, t.v22, t.v23, t.v24, t.v25, t.v26, t.v27, t.v28,
        t.amount,
        t.class,

        -- Log-scaled amount (compresses the long right tail of transaction amounts)
        ln(t.amount + 1) as log_amount,

        -- Time-of-day features
        cast(floor(mod(t.time, 86400) / 3600) as integer) as hour_of_day,
        case
            when cast(floor(mod(t.time, 86400) / 3600) as integer) between 0 and 5 then 1
            else 0
        end as is_night_transaction,

        -- Categorical amount tier (for reporting/segmentation, not fed to the models)
        case
            when t.amount = 0 then 'zero'
            when t.amount < 10 then 'micro'
            when t.amount < 50 then 'small'
            when t.amount < 200 then 'medium'
            when t.amount < 1000 then 'large'
            else 'very_large'
        end as amount_bucket,

        -- Z-score of amount vs. the full dataset distribution
        (t.amount - s.amount_mean) / nullif(s.amount_stddev, 0) as amount_zscore

    from txns t
    cross join stats s
)

select * from featured
