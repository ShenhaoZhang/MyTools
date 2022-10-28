import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score,mean_absolute_percentage_error,mean_absolute_error
import plotnine as gg
    
def plot_model_metric(y_ture:np.ndarray, y_pred:np.ndarray, ts:np.ndarray = None, name:str = 'y', type = 'pred_point') -> gg.ggplot:
    """
    预测效果的评价图

    Parameters
    ----------
    y_ture : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    ts : np.ndarray, optional
        时间索引，默认为0开始的连续值, by default None
    name : str, optional
        预测指标的名称, by default 'y'
    type : str, optional
        图形的类型, by default 'point'
            pred_point  : 真实值与预测值的散点图
            pred_line   : 真实值与预测值的时间序列图
            resid_line  : 残差的时间序列图

    Returns
    -------

    """
    sample        = len(y_ture)
    residual      = y_ture - y_pred
    residual_mean = np.round(np.mean(residual),4)
    residual_skew = np.round(stats.skew(residual),4)
    residual_kurt = np.round(stats.kurtosis(residual,fisher=True),4)
    residual_iqr  = np.quantile(residual,q=0.75) - np.quantile(residual,q=0.25)
    
    # 异常值的定义是残差大于1.5倍IQR的预测样本点
    outlier_index = np.abs(residual) > (1.5 * residual_iqr)
    outlier_count = np.sum(outlier_index)
    outlier_pct   = np.round(outlier_count / sample * 100,4)

    # 格式化数值文本
    def get_str_metric(metric_fun):
        metric =  str(round(metric_fun(y_true=y_ture,y_pred=y_pred),4))
        return metric    
    r2 = get_str_metric(r2_score)
    mae = get_str_metric(mean_absolute_error)
    mape = get_str_metric(mean_absolute_percentage_error)
    
    # 图形的注释
    caption = f'''
    Sample_N : {sample}   Outlier_N : {outlier_count}   Outlier_Pct : {outlier_pct}%
    Residual   Mean : {residual_mean}   Skew : {residual_skew}   Kurt_Fisher : {residual_kurt}
    R2 : {r2}   MAE : {mae}   MAPE : {mape}
    '''
    if ts is None:
        ts = np.arange(len(y_ture))
    data = pd.DataFrame({'y_true':y_ture,'y_pred':y_pred})
    
    if type == 'pred_point':
        title = f'True_{name} VS Predict_{name}'
        plot = (
            gg.ggplot(data)
            + gg.aes(x='y_true',y='y_pred')
            + gg.geom_point()
            + gg.geom_point(gg.aes(color='"Outlier"'),data=lambda dt:dt.loc[outlier_index])
            + gg.geom_abline(intercept=0,slope=1,size=1,color='red')
            + gg.geom_smooth(method='lm',color='blue')
            + gg.scale_color_manual(values=['red'],name=' ')
            + gg.labs(title=title,
                    caption=caption,
                    x=f'True_{name}',
                    y=f'Pred_{name}')
            + gg.theme(title=gg.element_text(ha='left'))
        )
        
    elif type == 'pred_line':
        title = f'Time Series for True_{name} and Predict_{name}'
        plot = (
            data
            .assign(ts=ts)
            .pipe(gg.ggplot)
            + gg.geom_line(gg.aes(x='ts',y='y_true',color='"True"'))
            + gg.geom_line(gg.aes(x='ts',y='y_pred',color='"Predict"'))
            + gg.geom_point(gg.aes(x='ts',y='y_pred',color='"Outlier"'),data=lambda dt:dt.loc[outlier_index])
            + gg.scale_color_manual(values=['black','green','red'],name=' ')
            + gg.labs(title=title,
                      caption=caption,
                      x=f'Time',
                      y=f'{name}')
            + gg.theme(title=gg.element_text(ha='left'))
        )
    
    elif type == 'resid_line':
        title = f'Time Series for Residual'
        plot = (
            data
            .assign(ts = ts,resid = residual)
            .pipe(gg.ggplot)
            + gg.geom_line(gg.aes(x='ts',y='resid'))
            + gg.geom_point(gg.aes(x='ts',y='resid',color='"Outlier"'),data=lambda dt:dt.loc[outlier_index])
            + gg.geom_hline(yintercept=0,color='red',size=1)
            + gg.geom_hline(gg.aes(linetype='"+- 1.5 * IQR"',yintercept=residual_iqr * 1.5),color='green',size=0.5)
            + gg.geom_hline(gg.aes(linetype='"+- 1.5 * IQR"',yintercept=-residual_iqr * 1.5),color='green',size=0.5)
            + gg.scale_linetype_manual(name=' ',values=['--','--'])
            + gg.scale_color_manual(values=['red'])
            + gg.labs(title=title,
                      caption=caption,
                      color=' ',
                      x=f'Time',
                      y=f'{name}')
            + gg.theme(title=gg.element_text(ha='left'))
        )
    
    return plot
