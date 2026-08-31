-- overall churn rate
SELECT COUNT(*) AS customers, 
SUM(Churn) AS churned,
ROUND(AVG(Churn)*100,2) AS churn_pct
FROM churn_clean;

 -- Churn by contract
SELECT Contract, COUNT(*) AS n,
ROUND(AVG(churn)*100,2) AS Churn_pct
FROM churn GROUP BY Contract ORDER BY churn_pct DESC;

 -- Revenue at Risk
 SELECT
 ROUND(SUM(CASE WHEN Churn=1 THEN MonthlyCharges ELSE 0 END),2) AS monthly_at_risk,
 ROUND(SUM(CASE WHEN Churn=1 THEN TotalCharges ELSE 0 END),2) AS lifetime_at_risk
 FROM churn_clean;

-- avg tenure and charges: churned vs stayed
SELECT Churn,
   ROUND(AVG(tenure),1) AS avg_tenure,
   ROUND(AVG(MonthlyCharges),2) AS avg_monthly
   From churn_clean GROUP BY Churn;
   
   -- high risk new customers
   SELECT COUNT(*) AS high_risk,
   ROUND(AVG(Churn)*100,2) AS churn_pct
   FROM churn_clean
   WHERE NewCustomer=1 AND MonthlyCharges>70;




