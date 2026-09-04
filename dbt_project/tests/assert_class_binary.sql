-- Fails if any transaction's Class label is not 0 or 1
select transaction_id, class
from {{ ref('fct_fraud_features') }}
where class not in (0, 1)
