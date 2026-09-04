-- Staging view over bronze_patients — no transforms, just aliasing for downstream use
select
    patient_id,
    given_name,
    family_name,
    birth_date,
    age_years,
    gender,
    race,
    ethnicity,
    marital_status,
    city,
    state,
    postal_code,
    deceased,
    ingested_at
from bronze_patients
