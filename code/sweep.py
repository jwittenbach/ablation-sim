#! /usr/bin/env spark-submit
from pyspark import SparkConf, SparkContext

conf=SparkConf().setMaster('local[4]')
sc = SparkContext(conf=conf)

data = sc.parallelize([1,2,3])
print data.collect()