/*
  Silver: latest vital signs per patient.
  LOINC codes: 8302-2 height, 29463-7 weight, 39156-5 BMI,
               8480-6 systolic BP, 8462-4 diastolic BP,
               55284-4 BP panel, 2093-3 cholesterol
*/
with obs as (
    select * from {{ ref('stg_observations') }}
    where category = 'vital-signs'
      and value_numeric is not null
      and status = 'final'
),

latest_per_code as (
    select
        patient_id,
        code,
        display,
        value_numeric,
        value_unit,
        effective_date,
        row_number() over (
            partition by patient_id, code
            order by effective_date desc
        ) as rn
    from obs
),

pivoted as (
    select
        patient_id,
        max(case when code = '8302-2' then value_numeric end)  as height_cm,
        max(case when code = '29463-7' then value_numeric end) as weight_kg,
        max(case when code = '39156-5' then value_numeric end) as bmi,
        max(case when code = '8480-6' then value_numeric end)  as systolic_bp,
        max(case when code = '8462-4' then value_numeric end)  as diastolic_bp,
        max(case when code = '2093-3' then value_numeric end)  as cholesterol_total,
        max(effective_date) as last_vital_date
    from latest_per_code
    where rn = 1
    group by patient_id
)

select
    patient_id,
    height_cm,
    weight_kg,
    bmi,
    systolic_bp,
    diastolic_bp,
    cholesterol_total,
    -- Derived risk flags
    case when bmi >= 30 then 1 else 0 end               as is_obese,
    case when systolic_bp >= 140 then 1 else 0 end       as elevated_systolic,
    case when cholesterol_total >= 240 then 1 else 0 end as high_cholesterol,
    last_vital_date
from pivoted
