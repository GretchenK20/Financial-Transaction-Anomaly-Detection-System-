-- Fails if any patient has implausible age
select patient_id, age_years
from {{ ref('fct_patient_risk_features') }}
where age_years < 0 or age_years > 130
