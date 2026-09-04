-- Fails if any binary flag is not 0 or 1
select patient_id
from {{ ref('fct_patient_risk_features') }}
where has_diabetes not in (0, 1)
   or has_hypertension not in (0, 1)
   or has_cancer not in (0, 1)
   or is_obese not in (0, 1)
