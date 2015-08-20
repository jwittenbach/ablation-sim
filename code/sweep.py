#! /usr/bin/env spark-submit
from pyspark import SparkConf, SparkContext
import os

with open(os.path.expanduser("~/spark-master")) as f:
    master = f.read().replace('\n', '')
conf = SparkConf().setAppName('sweep').setMaster(master)
sc = SparkContext(conf=conf)
