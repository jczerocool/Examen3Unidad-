//1 importar una sesion de spark
import org.apache.spark.sql.SparkSession
//2 utilice las lineas de codigo para reportar errores reducidos
import org.apache.log4j._
//3 cree una instancia de la sesion spark
Logger.getLogger("org").setLevel(Level.ERROR)
val spark = SparkSession.buider().getOrCreate()
// 4 importar la libreria de kmeans para el algoritmo de agrupamiento
import  org.apache.spark.ml.clustering.KMeans
//5 cargar el dataset de wholesale customer data
val df = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")
df.printSchema
//6 seleccionar las siguientes columnas para el conjunto de entrenamiento
val feature_data = df.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
//7 importar vectorassembler and vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
//8 crear un nuevo objeto vectorassembler
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
//9 utilice el objeto assembler para transfomrar feature_data
val training_data = assembler.transform(feature_data).select("features")
//10 crear un modelo  kmenas con k=3
val kmeans = new KMeans().setK(3)
//11 evaluar los grupos utilizando wssse
val model = kmeans.fit(training_data)
//12 mostrar los resultados
val WSSSE = model.computeCost(training_data)
println(s"resultado de suma de errores:  ${WSSSE} ")
