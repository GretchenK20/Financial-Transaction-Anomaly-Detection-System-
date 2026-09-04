select
    condition_id,
    patient_id,
    code,
    display,
    system,
    onset_date,
    abatement_date,
    clinical_status,
    category,
    ingested_at
from bronze_conditions
