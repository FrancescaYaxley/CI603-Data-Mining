import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import LinearSVC, NaiveBayes, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.fpm import FPGrowth


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
#os.environ["HADOOP_HOME"] = r"C:\hadoop"


spark = SparkSession.builder \
    .appName("WineQualityDataMining") \
    .config("spark.pyspark.python", sys.executable) \
    .config("spark.pyspark.driver.python", sys.executable) \
    .getOrCreate()


#loading datasets
red   = spark.read.csv(r"C:\Users\CESCA\OneDrive - University of Brighton\year3\CI603 Data Mining\wine\winequality-red.csv", header=True, inferSchema=True, sep=";")
white = spark.read.csv(r"C:\Users\CESCA\OneDrive - University of Brighton\year3\CI603 Data Mining\wine\winequality-white.csv", header=True, inferSchema=True, sep=";")

#combine datasets, red being 0 and white being 1
red   = red.withColumn("wine_type",   F.lit(0))   # 0 = red
white = white.withColumn("wine_type", F.lit(1))   # 1 = white
df    = red.union(white)

feature_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol", "wine_type"
]

print(f"Total records: {df.count()}")
print(f"Red wine: {red.count()}")
print(f"White wine: {white.count()}")
#df.describe().show() - removed - too busy to tell

print("Quality Score Distribution")
df.groupBy("quality").count().orderBy("quality").show()

# preprocessing

#to see which values to use for low, medium scores using standard deviation and mean
#simpler to visualise and understand
df.select("fixed acidity").describe().show()
df.select("volatile acidity").describe().show()
df.select("citric acid").describe().show()
df.select("residual sugar").describe().show()
df.select("chlorides").describe().show()
df.select("free sulfur dioxide").describe().show()
df.select("total sulfur dioxide").describe().show()
df.select("density").describe().show()
df.select("pH").describe().show()
df.select("alcohol").describe().show()
df.select("sulphates").describe().show()
df.select("quality").describe().show()
df.select("wine_type").describe().show()

#checking for missing values
print("Missing Values per Column")
df.select([
    F.count(F.when(F.isnull(c), c)).alias(c)
    for c in df.columns
]).show()

#gathering statistics for z score calculation
stats = df.select([
    *[F.mean(F.col(c)).alias(f"{c}_mean") for c in feature_cols], #* for error 
    *[F.stddev(F.col(c)).alias(f"{c}_stddev") for c in feature_cols]
]).collect()[0]

#creating new df to hold z scores for outlier detection
df_z = df
for c in feature_cols:
    mean_val = stats[f"{c}_mean"]
    std_val  = stats[f"{c}_stddev"]
    df_z = df_z.withColumn(f"{c}_zscore", (F.col(c) - mean_val) / std_val)

#find the maximum z score for each feature to identify potential outliers
print("Max z score per feature")
df_z.select([F.max(F.abs(F.col(f"{c}_zscore"))).alias(c) for c in feature_cols]).show()

#anomaly detection - looking for z score over 3
#over three is extreme outlier
#z score on chlorides
chloride_stats = df.select(
    F.mean("chlorides").alias("mean_cl"),
    F.stddev("chlorides").alias("std_cl")
).collect()[0]

#calculation mean divided by standard deviation
df = df.withColumn(
    "chloride_z",
    (F.col("chlorides") - chloride_stats["mean_cl"]) / chloride_stats["std_cl"]
)


outliers = df.filter(F.abs(F.col("chloride_z")) > 3)
print(f"Anomaly Detection: {outliers.count()} outlier wines. Score over 3 in chlorides")
outliers.select("quality", "chlorides", "chloride_z", "wine_type").show(10)

#remove z score column and outliers
df = df.drop("chloride_z")

#aggregate quality into 3 classes for classification
#low = 0 quality 3–4, medium = 1 quality 5–6, high = 2 quality 7–9
q = F.col("quality")
labelExpr = (F.when(q <= 4, 0) #low quality
     .when(q <= 6, 1) #medium quality
     .otherwise(2) #everything else becomes high quality
)
df = df.withColumn("quality_label", labelExpr)


print("Class Distribution (0=Low, 1=Medium, 2=High)")
df.groupBy("quality_label").count().orderBy("quality_label").show() #showing the distribution of the classes after labelling

#assemble feature vector


assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
df = assembler.transform(df)

#standardisation of features - mean of 0 and stand deviation of 1
scaler_model = StandardScaler(inputCol="raw_features", outputCol="features",
    withMean=True, withStd=True
).fit(df)
df = scaler_model.transform(df)

print("Preprocessing complete")

#dimensionality reduction with PCA

#fit with all 12 components to inspect explained variance
pca_full = PCA(k=12, inputCol="features", outputCol="pca_all")
pca_full_model = pca_full.fit(df)
ev = pca_full_model.explainedVariance
cum = [sum(ev[:i+1]) for i in range(len(ev))]

print("PCA")
#looping until 100%
#visual to see at which point variance reaches over 90%
for i, (e, c) in enumerate(zip(ev, cum)):
    print(f"  PC{i+1:2d}: {e*100:10.2f}%   cumulative percentage: {c*100:10.2f}%")

#retain enough components to explain >= 90% of variance
n_comp = next(i + 1 for i, c in enumerate(cum) if c >= 0.90)
print(f"Keeping {n_comp} principal components")

pca_model = PCA(k=n_comp, inputCol="features", outputCol="pca_features").fit(df)
df_pca    = pca_model.transform(df)

#classification

#train/test split - 80/20
(train, test) = df_pca.select("pca_features", "quality_label").randomSplit([0.8, 0.2], seed=42)
print(f"Train size: {train.count()}. Test size: {test.count()}")

#SVM
lsvc = LinearSVC(labelCol="quality_label", featuresCol="pca_features",
                    maxIter=100, regParam=0.1)
ovr = OneVsRest(classifier=lsvc, labelCol="quality_label", featuresCol="pca_features")
svm_mdl = ovr.fit(train)
svm_pred = svm_mdl.transform(test)

svm_acc = MulticlassClassificationEvaluator( labelCol="quality_label", predictionCol="prediction", metricName="accuracy"
).evaluate(svm_pred)
svm_f1  = MulticlassClassificationEvaluator( labelCol="quality_label", predictionCol="prediction", metricName="f1"
).evaluate(svm_pred)

print(f"SVM Results")
print(f"Accuracy : {svm_acc*100:.2f}%")
print(f"F1 Score : {svm_f1:.2f}")
svm_pred.groupBy("quality_label", "prediction").count().orderBy("quality_label", "prediction").show()

#naive bayes
nb_model = NaiveBayes(labelCol="quality_label", featuresCol="pca_features",
                      smoothing=1.0, modelType="gaussian").fit(train)
nb_pred  = nb_model.transform(test)

nb_acc = MulticlassClassificationEvaluator( labelCol="quality_label", predictionCol="prediction", metricName="accuracy"
).evaluate(nb_pred)
nb_f1  = MulticlassClassificationEvaluator( labelCol="quality_label", predictionCol="prediction", metricName="f1"
).evaluate(nb_pred)

print(f"Naive Bayes Results")
print(f"Accuracy : {nb_acc*100:.2f}%")
print(f"F1 Score : {nb_f1:.2f}")

print("Classifier Comparison")
print(f"  {'Classifier':<10} {'Accuracy':>10} {'F1 Score':>10}")
print(f"  {'SVM':<10} {svm_acc*100:>10.2f}% {svm_f1:>10.2f}")
print(f"  {'Naive Bayes':<10} {nb_acc*100:>10.2f}% {nb_f1:>10.2f}")

#association analysis using FPGrowth



#discretise continuous features into categorical items
#only continuous features used
#alcohol, volatile acidity, residual sugar, sulphates are important features 
df_fp = df \
    .withColumn("alcohol",
        F.when(F.col("alcohol") < 9.5,  "alcohol_low")
         .when(F.col("alcohol") < 11.5, "alcohol_medium")
         .otherwise("alcohol_high")) \
    .withColumn("vol_acid",
        F.when(F.col("volatile acidity") < 0.3, "va_low")
         .when(F.col("volatile acidity") < 0.6, "va_medium")
         .otherwise("va_high")) \
    .withColumn("sugar",
        F.when(F.col("residual sugar") < 2.5,  "sugar_low")
         .when(F.col("residual sugar") < 10.0, "sugar_medium")
         .otherwise("sugar_high")) \
    .withColumn("sulphates",
        F.when(F.col("sulphates") < 0.5,  "sulphates_low")
         .when(F.col("sulphates") < 0.75, "sulphates_medium")
         .otherwise("sulphates_high")) \
    .withColumn("total_sulfur_dioxide", 
        F.when(F.col("total sulfur dioxide") < 50,  "tsd_low")
         .when(F.col("total sulfur dioxide") < 160, "tsd_medium")
         .otherwise("tsd_high")) \
    .withColumn("quality", #quality is the consequent/target
        F.when(F.col("quality") <= 4, "quality_low")
         .when(F.col("quality") <= 6, "quality_medium")
         .otherwise("quality_high")) \
    .withColumn("wine_type", #this gives context to rules for red and white wine
        F.when(F.col("wine_type") == 0, "red_wine")
         .otherwise("white_wine"))

#creating an array of items for FPGrowth
item_cols = ["alcohol", "vol_acid", "sugar",
             "sulphates", "total_sulfur_dioxide", "quality", "wine_type"]
df_fp = df_fp.withColumn("items", F.array(*[F.col(c) for c in item_cols]))

#FPGrowth model with minimum support of 10% and confidence of 60%
#must show up in at least 10% of wines
#rules must be correct at least 60% of the time
fp_model = FPGrowth(itemsCol="items", minSupport=0.02, minConfidence=0.4).fit(df_fp) #changed to be able to show high quality rules

print(f"FPGrowth Results")
print(f"Frequent itemsets : {fp_model.freqItemsets.count()}") #number of itemsets that appear in at least 10% of wines
print(f"Association rules : {fp_model.associationRules.count()}") #number of rules that have confidence of at least 60%

#sorted by confidence
print("Top 20 association rules")
fp_model.associationRules.orderBy(F.desc("confidence")).show(20, truncate=False) #show top 20 rules by confidence

print("Rules predicting wine quality") #show top 20 rules that predict quality
fp_model.associationRules.filter( F.array_contains(F.col("consequent"), "quality_high") |
    F.array_contains(F.col("consequent"), "quality_medium") |
    F.array_contains(F.col("consequent"), "quality_low")
).orderBy(F.desc("confidence")).show(20, truncate=False)

print("Rules predicting high quality wine")
#to find patterns that lead to high quality wine
fp_model.associationRules.filter(F.array_contains(F.col("consequent"), "quality_high")).orderBy(F.desc("confidence")).show(20, truncate=False)

#summary of results
print("Summary of results")
print(f"Dataset: {df.count()} wines (red + white)")
print(f"Total wines: {df.count()}")
print(f"Features: {len(feature_cols)} physicochemical attributes")
print(f"PCA components: {n_comp}")
print(f"SVM  accuracy/F1: {svm_acc*100:.2f}% / {svm_f1:.2f}") #getting percentage
print(f"Naive Bayes accuracy/F1: {nb_acc*100:.2f}% / {nb_f1:.2f}") #getting percentage
print(f"FP-Growth itemsets: {fp_model.freqItemsets.count()}")
print(f"FP-Growth rules: {fp_model.associationRules.count()}")

spark.stop()
