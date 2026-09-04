select
    observation_id,
    patient_id,
    code,
    display,
    system,
    effective_date,
    status,
    value_numeric,
    value_unit,
    value_text,
    category,
    ingested_at
from bronze_observations
