SHOW GLOBAL VARIABLES LIKE 'local_infile';

LOAD DATA LOCAL INFILE "C:\Users\niluc\OneDrive\Desktop\Superstore_cleaned.csv"
INTO TABLE superstore_cleaned 
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


ALTER TABLE superstore_cleaned
MODIFY COLUMN `row id` INT,
MODIFY COLUMN `order date` DATE,
MODIFY COLUMN `ship mode` VARCHAR(50),
MODIFY COLUMN `customer id` VARCHAR(50),
MODIFY COLUMN `customer name` VARCHAR(50),
MODIFY COLUMN segment VARCHAR(50),
MODIFY COLUMN country VARCHAR(50),
MODIFY COLUMN city VARCHAR(50),
MODIFY COLUMN state VARCHAR(50),
MODIFY COLUMN `postal code` VARCHAR(20),
MODIFY COLUMN category VARCHAR(50),
MODIFY COLUMN `sub-category` VARCHAR(50),
MODIFY COLUMN sales DECIMAL (10,2),
MODIFY COLUMN quantity INT,
MODIFY COLUMN discount DECIMAL (5,2),
MODIFY COLUMN profit DECIMAL (10,2);

 -- summary
SELECT
ROUND(SUM(sales),2) AS total_sales,
ROUND(SUM(profit),2) AS total_profit,
COUNT(`order id`) AS Total_orders,
ROUND(AVG(Profit_margin),2) AS AVG_margin_pct
FROM superstore_cleaned;

  -- sales by category
SELECT category,
ROUND(SUM(sales),2) AS sales,
ROUND(SUM(profit),2) AS profit,
ROUND(SUM(profit)/SUM(sales)*100,2) AS Margin_pct
FROM superstore_cleaned
GROUP BY category
ORDER BY sales DESC;

-- top 10 selling products
SELECT `Product Name`,
ROUND(sum(sales),2) AS sales,
ROUND(SUM(profit),2) AS profit
FROM superstore_cleaned
GROUP BY `product Name`
ORDER BY sales DESC
limit 10;

-- sales trend by year

SELECT 
`Order Year` AS year,
ROUND(SUM(sales),2) AS total_sales,
count(`order id`) AS Total_orders,
AVG(sales) AS avg_order
FROM superstore_cleaned
GROUP BY `order year`
ORDER BY `order year`;

-- by region
SELECT region,
ROUND(SUM(sales),2) AS Revenue,
ROUND(SUM(profit),2) AS Profit
FROM superstore_cleaned
GROUP BY Region
ORDER BY Revenue DESC;

-- by segment
SELECT segment,
ROUND(SUM(sales),2) AS sales,
ROUND(SUM(profit),2) AS profit,
COUNT(`order id`)  AS Orders,
 ROUND(AVG(sales),2) AS AVG_order_value
FROM superstore_cleaned
GROUP BY segment
ORDER BY sales DESC;



select count(*) FROM superstore_cleaned