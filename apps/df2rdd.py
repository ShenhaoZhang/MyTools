import pandas as pd
from pyspark.sql import DataFrame
from pyspark.rdd import RDD
from pyspark import Row

def table_to_PandasDF_rdd(table:DataFrame, groupby_key:list):
    """
    将SparkDataFrame转换为分组的RDD

    Parameters
    ----------
    df : SparkDataFrame
        Spark DataFrame
    groupby_key : list
        分组的字段
    numPartitions : int
        分区数量

    Returns
    -------
    RDD
        Key-Value RDD，Key为分组的字段，Value为Pandas DataFrame
    """
    # 防止groupby_key为单个字符串
    groupby_key = groupby_key if isinstance(groupby_key,list) else [groupby_key]
        
    rdd = table.rdd
    out_rdd = (
        rdd
        .keyBy(lambda row: tuple([row[key] for key in groupby_key]))
        .mapValues(lambda row: row.asDict())
        .groupByKey()
        .mapValues(pd.DataFrame)
    )
    return out_rdd

def PandasDF_rdd_to_table(rdd:RDD):
    """
    将Pandas DataFrame的RDD转换为pyspark的DataFrame

    Parameters
    ----------
    parid_rdd : RDD
        value是pandas DataFrame的RDD

    Returns
    -------
    pyspark DataFrame
        数据
    """
    # TODO 输入可以不是pair rdd
    # TODO 输入是pari rdd时，保留key
    rdd = rdd.flatMapValues(lambda df:map(lambda row:Row(**row[1].to_dict()),df.iterrows()))
    rdd = rdd.map(lambda rdd:rdd[1])
    return rdd.toDF()