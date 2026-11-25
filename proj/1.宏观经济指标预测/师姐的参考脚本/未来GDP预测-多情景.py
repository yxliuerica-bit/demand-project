# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:07:40 2025

基于Cai et al., 2022 GDP预测结果

@author: Yuxi Liu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['font.size'] = 12 # 设置字体大小
plt.rcParams['axes.unicode_minus'] = False # 使坐标轴刻度标签正常显示正负号

fi_history = '/lustre/home/litong/intern/raw_data/城镇人口分省年度数据.csv'
fi_p0_hbase = '/lustre/home/litong/intern/output/城镇人口预测/population_predictions_基准.csv'
#fi_p1 = 'C:/Users/admin/Desktop/预测原始数据/GDP_Projection_CWJ.xlsx'
fi_p1 = '/lustre/home/litong/intern/output/城镇人口预测/population_predictions_低增长.csv'
fi_p2 = '/lustre/home/litong/intern/output/城镇人口预测/population_predictions_中增长.csv'
fi_p3 = '/lustre/home/litong/intern/output/城镇人口预测/population_predictions_高增长.csv'

df_history = pd.read_csv(fi_history,index_col=0)
df_p0_hbase = pd.read_csv(fi_p0_hbase,index_col=0)
df_p1 = pd.read_csv(fi_p1,index_col=0)
df_p2 = pd.read_csv(fi_p2,index_col=0)
df_p3 = pd.read_csv(fi_p3,index_col=0)

scenarios = ['历史趋势外推','低速发展','稳定发展','高速发展']; nscenario = 4
years = np.arange(2010,2036,1); nyear = len(years)
provlist_cn = ['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']
provlist_en = ['Beijing','Tianjin','Hebei','Henan','Shandong','Shanxi','Inner Mongolia','Shaanxi']

# 绘图选项：
eva_provincial_result = 0
plot_nation_vs_region = 0
plot_provincial_bar = 1

if eva_provincial_result:
    for ip,prov in enumerate(provlist_cn):
        dfout = pd.DataFrame(np.zeros((nscenario,nyear)),index=scenarios,columns=years)
        dfout.loc[:,np.arange(2010,2025)] = df_history.loc[np.arange(2010,2025),prov].values
        dfout.loc['logistic prediction',np.arange(2025,2036)] = df_p0_hbase.loc[np.arange(2025,2036),prov].values
        
        '''
        # scale Cai et al.结果，使得基准年一致
        gdp_2024 = df_history.loc[2024,prov]
        scale_factor_s1 = gdp_2024/(df_p1.loc[(df_p1.V1==provlist_en[ip]) & (df_p1.V2=='SSP1'),2024].values[0]/10000)
        scale_factor_s2 = gdp_2024/(df_p1.loc[(df_p1.V1==provlist_en[ip]) & (df_p1.V2=='SSP3'),2024].values[0]/10000)
        scale_factor_s3 = gdp_2024/(df_p1.loc[(df_p1.V1==provlist_en[ip]) & (df_p1.V2=='SSP5'),2024].values[0]/10000)
        
        dfout.loc['Low',np.arange(2010,2036)] = df_p1.loc[(df_p1.V1==provlist_en[ip]) & (df_p1.V2=='SSP1'),np.arange(2010,2036).tolist()].values[0]/10000*scale_factor_s1
        dfout.loc['Stable',np.arange(2010,2036)] = df_p1.loc[(df_p1.V1==provlist_en[ip]) & (df_p1.V2=='SSP3'),np.arange(2010,2036).tolist()].values[0]/10000*scale_factor_s2
        dfout.loc['High',np.arange(2010,2036)] = df_p1.loc[(df_p1.V1==provlist_en[ip]) & (df_p1.V2=='SSP5'),np.arange(2010,2036).tolist()].values[0]/10000*scale_factor_s3
        '''
        
        dfout.loc['Low',np.arange(2025,2036)] = df_p1.loc[np.arange(2025,2036),prov].values
        dfout.loc['Stable',np.arange(2025,2036)] = df_p2.loc[np.arange(2025,2036),prov].values
        dfout.loc['High',np.arange(2025,2036)] = df_p3.loc[np.arange(2025,2036),prov].values
       
        # 生成折线图
        dfplot = dfout.T
        dfplot.plot(kind='line',  # 指定折线图类型
                   figsize=(10, 6),  # 设置图像大小
                   title=prov,  # 标题
                   grid=True,  # 显示网格
                   marker='o')  # 数据点标记为圆圈
        plt.show()
    
if plot_nation_vs_region:

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5, 6))
    
    scenarios = ['历史趋势外推','低速发展','稳定发展','高速发展']; nscenario = 4
    years = np.arange(2015,2036,1); nyear = len(years)
    province = df_history.columns
    df_history['全国'] = df_history[province].sum(axis=1)
    df_history['京津冀及周边8省市'] = df_history[provlist_cn].sum(axis=1)
    
    dfout = pd.DataFrame(np.zeros((nscenario,nyear)),index=scenarios,columns=years)
    dfout.loc[:,np.arange(2015,2025)] = df_history.loc[np.arange(2015,2025),'全国'].values/10000
    dfout.loc['历史趋势外推',np.arange(2025,2036)] = df_p0_hbase.loc[np.arange(2025,2036),'全国'].values/10000
    dfout.loc['低速发展',np.arange(2025,2036)] = df_p1.loc[np.arange(2025,2036),'全国'].values/10000
    dfout.loc['稳定发展',np.arange(2025,2036)] = df_p2.loc[np.arange(2025,2036),'全国'].values/10000
    dfout.loc['高速发展',np.arange(2025,2036)] = df_p3.loc[np.arange(2025,2036),'全国'].values/10000
    
    dfregion = pd.DataFrame(np.zeros((nscenario,nyear)),index=scenarios,columns=years)
    dfregion.loc[:,np.arange(2015,2025)] = df_history.loc[np.arange(2015,2025),'京津冀及周边8省市'].values/10000
    dfregion.loc['历史趋势外推',np.arange(2025,2036)] = df_p0_hbase.loc[np.arange(2025,2036),'京津冀及周边8省市'].values/10000
    dfregion.loc['低速发展',np.arange(2025,2036)] = df_p1.loc[np.arange(2025,2036),'京津冀及周边8省市'].values/10000
    dfregion.loc['稳定发展',np.arange(2025,2036)] = df_p2.loc[np.arange(2025,2036),'京津冀及周边8省市'].values/10000
    dfregion.loc['高速发展',np.arange(2025,2036)] = df_p3.loc[np.arange(2025,2036),'京津冀及周边8省市'].values/10000
    
    scelist = ['低速发展','稳定发展','高速发展','历史趋势外推']
    colorlist = ['#1868B2','#018A67','#F3A332','#BF1D2D']
    for is_,sce in enumerate(scelist):
        ax1.plot(np.arange(2015,2036),dfout.loc[sce,:],color=colorlist[is_],marker='o',markersize=4,label=sce)
        ax2.plot(np.arange(2015,2036),dfregion.loc[sce,:],color=colorlist[is_],marker='o',markersize=4,label=sce)

    ax1.grid(axis='y', lw=0.8, ls='--')
    ax2.grid(axis='y', lw=0.8, ls='--')
    ax1.axvline(x=2024,ls='--')
    ax2.axvline(x=2024,ls='--')
    ax1.set_xlim(2015,2035)
    ax2.set_xlim(2015,2035)
    ax1.legend(frameon=False,loc='upper left')
    ax2.legend(frameon=False,loc='upper left')
    ax1.set_ylabel('全国城镇人口(万人)')
    ax2.set_ylabel('京津冀及周边8省市城镇人口(万人)')
    plt.tight_layout()
    plt.savefig('/lustre/home/litong/intern/output/城镇人口预测/image/全国及8省市城镇人口结果对比.png',dpi=500)
    plt.show()
    
if plot_provincial_bar:

    # 1. 读取CSV文件
    file_path = fi_p0_hbase
    df = pd.read_csv(file_path, index_col=0)
    
    # 2. 筛选所需省份和年份
    target_provinces = ['北京市', '天津市', '河北省', '河南省', '山东省', '山西省', '内蒙古自治区', '陕西省']
    colors = ['#DEECF6','#AFC8E2','#E2F2CD','#B6DAA7','#F9D5D5','#EF98A1','#FBE3C0','#FBC99A']
    
    # 提取2024-2035年的数据
    df_filtered = df.loc[2024:2035, target_provinces].copy()/10000 # convert to 亿元
    
    # 3. 准备绘图数据
    years = df_filtered.index
    provinces = df_filtered.columns
    
    # 创建堆叠数据
    bottom = np.zeros(len(years))
    
    # 4. 创建图表
    plt.figure(figsize=(4.5, 2.8))
    
    # 绘制堆叠柱状图
    for i, province in enumerate(provinces):
        values = df_filtered[province].values
        plt.bar(years, values, bottom=bottom, label=province, color=colors[i])
        bottom += values
    
    # 5. 添加标签和标题
    plt.ylabel('城镇人口(万人)', fontsize=9)
    plt.yticks(np.arange(0,71,10),fontsize=9)
    plt.xticks(years, rotation=90,fontsize=9)
    
    # 6. 添加图例和总人口标签
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1.02),fontsize=9)
    plt.tight_layout()
    plt.ylim(0,70)
    
    # 在柱子顶部添加总人口标签
    for year, total in zip(years, bottom):
        plt.text(year, total+1, f'{total:.1f}', 
                 ha='center', va='bottom', fontsize=9,rotation=90)
    
    # 7. 保存和显示
    plt.savefig('/lustre/home/litong/intern/output/城镇人口预测/image/京津冀及周边8省市城镇人口堆叠图.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    # ===== 新增部分：2024年人口饼图 =====

    # 提取2024年数据
    data_2024 = df_filtered.loc[2024]
    
    # 创建饼图
    plt.figure(figsize=(3.5, 3.5))
    
    # 绘制饼图
    patches, texts, autotexts = plt.pie(
        data_2024, 
        labels=data_2024.index, 
        autopct='%1.1f%%', 
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )

    
    # 调整百分比标签位置和样式
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
    
    # 添加图例
    #plt.legend(
    #    patches, 
    #    [f'{province} ({value:.2f}亿人)' for province, value in zip(data_2024.index, data_2024)],
    #    loc='best',
    #    fontsize=9
    #)
    
    # 确保饼图是圆形
    plt.axis('equal')
    
    # 保存饼图
    plt.savefig('/lustre/home/litong/intern/output/城镇人口预测/image/2024年京津冀及周边8省市城镇人口饼图.png', 
                dpi=500, bbox_inches='tight')
    plt.show()