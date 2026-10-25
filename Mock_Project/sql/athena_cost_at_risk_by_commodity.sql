-- Amazon Athena, database metals_db: 12-month cost at risk by commodity and clause type
SELECT
    index_metal_id AS commodity,
    metal_clause_type,
    COUNT(DISTINCT contract_id) AS active_contracts,
    ROUND(SUM(cost_at_risk_12m_eur), 2) AS total_cost_at_risk_eur
FROM "metals_db"."fact_cost_at_risk"
GROUP BY index_metal_id, metal_clause_type
ORDER BY total_cost_at_risk_eur DESC;
