-- Fails if any gold feature row has a null transaction_id, amount, or class
select transaction_id
from {{ ref('fct_fraud_features') }}
where transaction_id is null
   or amount is null
   or class is null
