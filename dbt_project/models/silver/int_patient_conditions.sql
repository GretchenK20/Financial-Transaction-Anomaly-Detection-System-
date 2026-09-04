/*
  Silver: patient-level condition flags used for risk feature engineering.
  One row per patient with binary flags for high-risk chronic conditions.
  Condition codes are SNOMED CT from Synthea's standard modules.
*/
with conditions as (
    select * from {{ ref('stg_conditions') }}
    where clinical_status = 'active'
),

patient_condition_flags as (
    select
        patient_id,
        -- Cardiovascular
        max(case when code in ('44054006','73211009') then 1 else 0 end) as has_diabetes,
        max(case when code in ('38341003','59621000') then 1 else 0 end) as has_hypertension,
        max(case when code in ('53741008','22298006') then 1 else 0 end) as has_coronary_artery_disease,
        max(case when code in ('84114007','43878008') then 1 else 0 end) as has_heart_failure,
        -- Respiratory
        max(case when code in ('195967001','13645005') then 1 else 0 end) as has_asthma_or_copd,
        -- Renal
        max(case when code in ('709044004','431855005') then 1 else 0 end) as has_ckd,
        -- Mental health
        max(case when code in ('35489007','370143000') then 1 else 0 end) as has_depression,
        max(case when code in ('197480006','300604001') then 1 else 0 end) as has_anxiety,
        -- Oncology
        max(case when code like '363%' or display ilike '%cancer%' or display ilike '%malignant%'
            then 1 else 0 end) as has_cancer,
        -- Counts
        count(*) as total_active_conditions,
        count(distinct date_trunc('year', onset_date)) as condition_onset_years
    from conditions
    group by patient_id
)

select * from patient_condition_flags
