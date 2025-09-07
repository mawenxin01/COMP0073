#!/usr/bin/env python3
"""
CSV数据预处理脚本
用于标准化gender和arrival_transport字段
"""

import pandas as pd
import numpy as np

def standardize_gender(df):
    """标准化gender字段"""
    # 将gender统一转换为大写
    df['gender'] = df['gender'].str.upper()
    
    # 标准化性别值
    gender_mapping = {
        'M': 'Male',
        'F': 'Female',
        'MALE': 'Male',
        'FEMALE': 'Female'
    }
    
    df['gender'] = df['gender'].map(gender_mapping).fillna('Unknown')
    return df

def standardize_arrival_transport(df):
    """标准化arrival_transport字段"""
    # 将arrival_transport统一转换为大写
    df['arrival_transport'] = df['arrival_transport'].str.upper()
    
    # 标准化交通方式值
    transport_mapping = {
        'AMBULANCE': 'AMBULANCE',
        'WALK IN': 'WALK IN',
        'WALK-IN': 'WALK IN',
        'WALKIN': 'WALK IN',
        'HELICOPTER': 'HELICOPTER',
        'OTHER': 'OTHER',
        'UNKNOWN': 'UNKNOWN'
    }
    
    df['arrival_transport'] = df['arrival_transport'].map(transport_mapping).fillna('WALK IN')
    return df

def preprocess_csv(input_file, output_file=None):
    """
    预处理CSV文件
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径（如果为None，则覆盖原文件）
    """
    print(f"读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始gender值: {df['gender'].value_counts()}")
    print(f"原始arrival_transport值: {df['arrival_transport'].value_counts()}")
    
    # 标准化数据
    df = standardize_gender(df)
    df = standardize_arrival_transport(df)
    
    print(f"\n处理后gender值: {df['gender'].value_counts()}")
    print(f"处理后arrival_transport值: {df['arrival_transport'].value_counts()}")
    
    # 保存文件
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False)
    print(f"\n保存到: {output_file}")

if __name__ == "__main__":
    # 处理主要数据文件
    input_file = "data_processing/processed_data/triage_final.csv"
    
    # 创建备份
    backup_file = "data_processing/processed_data/triage_final_backup.csv"
    
    # 处理数据
    preprocess_csv(input_file, output_file=None)  # 直接覆盖原文件
    
    print("数据预处理完成！")

