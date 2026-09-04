-- Fails if any gold feature row has a null patient_id
select patient_id
from {{ ref('fct_patient_risk_features') }}
where patient_id is null
