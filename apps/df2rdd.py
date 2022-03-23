import pandas as pd

def df_to_rdd(df, groupby_key,numPartitions=50):
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
    if not isinstance(groupby_key,list):
        groupby_key = [groupby_key]
        
    rdd = df.rdd
    out_rdd = (
        rdd
        .repartition(numPartitions)
        .keyBy(lambda row: tuple([row[key] for key in groupby_key]))
        .mapValues(lambda row: [row.asDict()])
        .reduceByKey(lambda r1, r2: r1+r2)
        .mapValues(lambda r: pd.DataFrame(r))
    )
    return out_rdd

def rdd_to_df(rdd):
    pass