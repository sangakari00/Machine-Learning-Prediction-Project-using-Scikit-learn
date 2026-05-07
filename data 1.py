from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, to_date, month

# Create Spark session
spark = SparkSession.builder \
    .appName("BigDataSalesAnalysis") \
    .getOrCreate()

# Fix date parsing issue
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Load dataset
df = spark.read.csv(
    "C:\\Users\\l\\OneDrive\\Desktop\\programming\\file python\\data_analysis.csv",
    header=True,
    inferSchema=True
)
df.show(5)

# Rename column
df = df.withColumnRenamed("Unit Price", "Unit_Price")

# Check columns
print(df.columns)

# Check rows
print("Total rows:", df.count())
print("Rows without nulls:", df.dropna().count())

# Clean numeric columns
df = df.withColumn("Unit_Price", regexp_replace("Unit_Price", "[$,]", "").cast("double")) \
       .withColumn("Sales", regexp_replace("Sales", "[$,]", "").cast("double")) \
       .withColumn("Cost", regexp_replace("Cost", "[$,]", "").cast("double"))

# Convert date (ONLY ONCE)
df = df.withColumn("OrderDate", to_date("OrderDate", "EEEE, MMMM d, yyyy"))

# Create Profit
df = df.withColumn("Profit", col("Sales") - col("Cost"))

# Check date
df.select("OrderDate").show(5)

# Total analysis
df.selectExpr(
    "sum(Sales) as Total_Sales",
    "sum(Cost) as Total_Cost",
    "sum(Profit) as Total_Profit"
).show()

# Create Month column
df = df.withColumn("Month", month("OrderDate"))

# Monthly analysis
df.groupBy("Month") \
  .sum("Sales") \
  .orderBy("Month") \
  .show()

# quantity
df.groupBy("Quantity") \
  .sum("Sales") \
  .orderBy("Quantity") \
  .show()

# sales territory
df.groupBy("SalesTerritoryKey") \
  .sum("Sales") \
  .orderBy("sum(Sales)", ascending=False) \
  .show()

# Cache dataframe in memory
df.cache()

# Repartition data
df = df.repartition(4)

# Show execution plan
df.explain()