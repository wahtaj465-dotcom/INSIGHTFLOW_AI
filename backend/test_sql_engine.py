import pandas as pd

from services.sql_engine import SQLEngine


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/sales_processed.csv"
)

sql_engine = SQLEngine()

sql_engine.register_dataframe(
    df,
    table_name="sales"
)


# ============================================================
# TEST 1 — BASIC SELECT
# ============================================================

query1 = """
SELECT *
FROM sales
LIMIT 5;
"""

print("\nQUERY 1:")
print(repr(query1))

print("\nVALIDATION:")
print(sql_engine.validate_query(query1))

result = sql_engine.execute_query(query1)

print("\nRESULT:")
print(result)


# ============================================================
# TEST 2 — ORDERS BY REGION
# ============================================================

query2 = """
SELECT
    region,
    COUNT(*) AS total_orders
FROM sales
GROUP BY region
ORDER BY total_orders DESC;
"""

print("\n\nQUERY 2:")
print(repr(query2))

print("\nVALIDATION:")
print(sql_engine.validate_query(query2))

result = sql_engine.execute_query(query2)

print("\nRESULT:")
print(result)


# ============================================================
# TEST 3 — PRODUCT ANALYSIS
# ============================================================

query3 = """
SELECT
    product,
    COUNT(*) AS orders,
    SUM(quantity) AS total_quantity,
    AVG(price) AS average_price
FROM sales
GROUP BY product
ORDER BY orders DESC;
"""

print("\n\nQUERY 3:")
print(repr(query3))

print("\nVALIDATION:")
print(sql_engine.validate_query(query3))

result = sql_engine.execute_query(query3)

print("\nRESULT:")
print(result)


# ============================================================
# TEST 4 — DELETE
# ============================================================

query4 = """
DELETE FROM sales;
"""

print("\n\nQUERY 4:")
print(repr(query4))

print("\nVALIDATION:")
print(sql_engine.validate_query(query4))

result = sql_engine.execute_query(query4)

print("\nRESULT:")
print(result)


# ============================================================
# TEST 5 — DROP
# ============================================================

query5 = """
DROP TABLE sales;
"""

print("\n\nQUERY 5:")
print(repr(query5))

print("\nVALIDATION:")
print(sql_engine.validate_query(query5))

result = sql_engine.execute_query(query5)

print("\nRESULT:")
print(result)


# ============================================================
# TEST 6 — MULTIPLE STATEMENTS
# ============================================================

query6 = """
SELECT * FROM sales;
DROP TABLE sales;
"""

print("\n\nQUERY 6:")
print(repr(query6))

print("\nVALIDATION:")
print(sql_engine.validate_query(query6))

result = sql_engine.execute_query(query6)

print("\nRESULT:")
print(result)


# ============================================================
# FINAL CHECK
# ============================================================

query7 = """
SELECT COUNT(*) AS row_count
FROM sales;
"""

print("\n\nFINAL QUERY:")
print(repr(query7))

print("\nVALIDATION:")
print(sql_engine.validate_query(query7))

result = sql_engine.execute_query(query7)

print("\nRESULT:")
print(result)


# ============================================================
# CLOSE DATABASE
# ============================================================

sql_engine.close()