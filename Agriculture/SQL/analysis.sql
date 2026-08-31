-- 1.AVG yields across all years
SELECT crop,
ROUND(AVG(yield_tonnes_ha), 3) AS avg_yield,
ROUND(MIN(yield_tonnes_ha), 3) AS min_yield,
ROUND(MAX(yield_tonnes_ha), 3) AS max_yield
FROM yields
GROUP BY crop
ORDER BY avg_yield DESC;

-- 2.TOP 10 COUNTIES BY Maize yield
SELECT county,
ROUND(AVG(yield_tonnes_ha),3) AS avg_maize_yield,
COUNT(*) AS seasons_recorded
FROM yields
WHERE crop = 'Maize'
GROUP BY county
ORDER BY avg_maize_yield DESC
LIMIT 10;

-- 3.SEASONS COMPARISON 
SELECT season,crop,
ROUND(AVG(yield_tonnes_ha),3) AS avg_yield,
ROUND(AVG(rainfall_mm),3) AS avg_rainfall
FROM yields
GROUP BY season,crop
ORDER BY crop,avg_yield DESC;

-- 4.FERTILISER  IMPACT
SELECT 
CASE WHEN fertiliser_use_kg_ha < 20 THEN 'Low(0-20 kg/ha)'
     WHEN fertiliser_use_kg_ha < 60 THEN 'Medium(20-60 kg/ha)'
	 WHEN fertiliser_use_kg_ha < 100 THEN 'High (60-100 kg/ha)'
	 ELSE ' Very High (100+)' END AS fert_tier,
ROUND(AVG(yield_tonnes_ha),3) AS avg_yield,
COUNT(*) AS records
FROM yields WHERE crop = 'Maize'
GROUP BY fert_tier ORDER BY avg_yield DESC;

-- 5.Yield trend by decade
SELECT (year/10*10) AS decade,
ROUND(AVG(yield_tonnes_ha),3) AS avg_yield,
ROUND(AVG(rainfall_mm),1) AS avg_rainfall
FROM yields 
WHERE crop = 'Maize'
GROUP BY decade ORDER BY decade;