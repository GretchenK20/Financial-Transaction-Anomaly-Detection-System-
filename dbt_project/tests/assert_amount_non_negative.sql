-- Fails if any transaction has a negative amount.
-- Note: uses >= 0 rather than a strict > 0 check — the real dataset contains
-- 1,825 legitimate $0.00 transactions (e.g. card verification/authorization
-- holds), so a strict > 0 test would fail against real data.
select transaction_id, amount
from {{ ref('fct_fraud_features') }}
where amount < 0
