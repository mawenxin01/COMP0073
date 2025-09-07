#!/usr/bin/env python3
"""
数据预处理步骤2：为数据文件添加时间段编码标签

功能：
1. 为 triage_final.csv 添加时间段编码列
2. 为 triage_with_keywords.csv 添加时间段编码列

时间段编码：
- 凌晨（0–6点）: 0
- 上午（6–12点）: 1  
- 下午（12–18点）: 2
- 晚上（18–24点）: 3

作者: Medical Triage Assistant Team
版本: 1.0
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, Union, List, Optional
import warnings


class TimeFeatureProcessor:
    """统一的时间特征处理器"""
    
    # 时间段编码映射
    TIME_PERIOD_MAPPING = {
        'dawn': {'range': (0, 6), 'code': 0, 'name': '凌晨'},
        'morning': {'range': (6, 12), 'code': 1, 'name': '上午'},
        'afternoon': {'range': (12, 18), 'code': 2, 'name': '下午'},
        'evening': {'range': (18, 24), 'code': 3, 'name': '晚上'}
    }
    
    # 默认值设置
    DEFAULT_VALUES = {
        'time_period': 2        # 默认下午时段
    }
    
    @classmethod
    def extract_time_features(cls, time_input: Union[str, pd.Series, List]) -> Union[Dict, pd.DataFrame]:
        """
        从时间输入提取标准化时间特征
        
        Args:
            time_input: 时间输入，支持：
                - 单个时间字符串 (str)
                - pandas Series 
                - 时间字符串列表 (List)
                
        Returns:
            单个字典或DataFrame，包含以下特征：
            - time_period: 时间段编码 (0-3)
        """
        
        # 处理pandas Series
        if isinstance(time_input, pd.Series):
            features_list = []
            for time_str in time_input:
                features_list.append(cls._extract_single_time_features(time_str))
            return pd.DataFrame(features_list, index=time_input.index)
        
        # 处理列表
        elif isinstance(time_input, list):
            features_list = []
            for time_str in time_input:
                features_list.append(cls._extract_single_time_features(time_str))
            return pd.DataFrame(features_list)
        
        # 处理单个字符串或None值
        else:
            return cls._extract_single_time_features(time_input)
    
    @classmethod
    def _extract_single_time_features(cls, time_str: Union[str, None]) -> Dict:
        """
        提取单个时间字符串的特征
        
        Args:
            time_str: 时间字符串，格式为 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            时间特征字典
        """
        try:
            # 处理空值或无效值
            if pd.isna(time_str) or time_str == '' or time_str is None:
                return cls.DEFAULT_VALUES.copy()
            
            # 解析时间字符串
            dt = datetime.strptime(str(time_str).strip(), '%Y-%m-%d %H:%M:%S')
            
            # 提取基础特征
            hour = dt.hour
            time_period = cls._get_time_period_code(hour)
            
            return {
                'time_period': time_period
            }
            
        except (ValueError, TypeError) as e:
            # 时间解析失败，使用默认值
            warnings.warn(f"时间解析失败: {time_str}, 错误: {e}, 使用默认值")
            return cls.DEFAULT_VALUES.copy()
    
    @classmethod
    def _get_time_period_code(cls, hour: int) -> int:
        """
        根据小时获取时间段编码
        
        Args:
            hour: 小时 (0-23)
            
        Returns:
            时间段编码 (0-3)
        """
        for period_info in cls.TIME_PERIOD_MAPPING.values():
            start, end = period_info['range']
            if start <= hour < end:
                return period_info['code']
        
        # 如果小时超出范围，返回默认值
        return cls.DEFAULT_VALUES['time_period']
    
    @classmethod
    def get_time_period_name(cls, code: int) -> str:
        """
        根据编码获取时间段名称
        
        Args:
            code: 时间段编码 (0-3)
            
        Returns:
            时间段名称
        """
        for period_info in cls.TIME_PERIOD_MAPPING.values():
            if period_info['code'] == code:
                return period_info['name']
        return "未知时段"
    
    @classmethod
    def add_time_features_to_dataframe(cls, 
                                     df: pd.DataFrame, 
                                     time_column: str = 'intime',
                                     prefix: str = '') -> pd.DataFrame:
        """
        为DataFrame添加时间特征列
        
        Args:
            df: 输入DataFrame
            time_column: 时间列名，默认'intime'
            prefix: 特征列名前缀，可选
            
        Returns:
            添加了时间特征的DataFrame
        """
        if time_column not in df.columns:
            warnings.warn(f"列 '{time_column}' 不存在，跳过时间特征提取")
            return df.copy()
        
        print(f"⏰ 从列 '{time_column}' 提取时间特征...")
        
        # 提取时间特征
        time_features = cls.extract_time_features(df[time_column])
        
        # 添加前缀（如果指定）
        if prefix:
            time_features.columns = [f"{prefix}{col}" for col in time_features.columns]
        
        # 合并到原DataFrame
        result_df = pd.concat([df, time_features], axis=1)
        
        feature_names = list(time_features.columns)
        print(f"✅ 时间特征提取完成，新增特征: {feature_names}")
        
        return result_df
    
    @classmethod
    def get_feature_descriptions(cls) -> Dict[str, str]:
        """
        获取时间特征的详细描述
        
        Returns:
            特征描述字典
        """
        # 构建时间段描述
        period_descriptions = []
        for info in cls.TIME_PERIOD_MAPPING.values():
            period_descriptions.append(f"{info['name']}: {info['code']}")
        period_desc = ", ".join(period_descriptions)
        
        return {
            'time_period': f'时间段编码 - {period_desc}'
        }


def add_time_labels_to_file(input_file_path: str, output_file_path: str = None, backup: bool = True):
    """
    为数据文件添加时间段编码标签
    
    Args:
        input_file_path: 输入文件路径
        output_file_path: 输出文件路径，如果为None则覆盖原文件
        backup: 是否创建备份文件
    """
    
    print(f"📂 处理文件: {input_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(input_file_path):
        print(f"❌ 文件不存在: {input_file_path}")
        return False
    
    try:
        # 1. 读取数据
        print("📖 读取数据中...")
        df = pd.read_csv(input_file_path, low_memory=False)
        print(f"✅ 数据读取完成，共 {len(df):,} 行，{len(df.columns)} 列")
        
        # 2. 检查是否已存在时间段编码列
        if 'time_period' in df.columns:
            print("⚠️ 时间段编码列已存在，将覆盖现有数据")
        
        # 3. 检查时间列
        time_column = 'intime'
        if time_column not in df.columns:
            print(f"❌ 时间列 '{time_column}' 不存在于数据中")
            print(f"可用列: {list(df.columns)}")
            return False
        
        # 4. 创建备份（如果需要）
        if backup and output_file_path != input_file_path:
            backup_path = input_file_path.replace('.csv', '_backup.csv')
            if not os.path.exists(backup_path):
                print(f"💾 创建备份文件: {backup_path}")
                df.to_csv(backup_path, index=False)
        
        # 5. 提取时间特征
        print("⏰ 提取时间段编码...")
        original_columns = df.columns.tolist()
        
        # 使用统一的时间处理器
        df_with_time = TimeFeatureProcessor.add_time_features_to_dataframe(df, time_column)
        
        # 6. 分析时间段分布
        print("\n📊 时间段分布统计:")
        try:
            # 确保time_period列是数值类型
            df_with_time['time_period'] = pd.to_numeric(df_with_time['time_period'], errors='coerce')
            time_period_counts = df_with_time['time_period'].value_counts().sort_index()
            total_count = len(df_with_time)
            
            for period_code, count in time_period_counts.items():
                if pd.notna(period_code):  # 检查是否为有效值
                    period_name = TimeFeatureProcessor.get_time_period_name(int(period_code))
                    percentage = count / total_count * 100
                    print(f"  {period_name}({period_code}): {count:,} 例 ({percentage:.1f}%)")
        except Exception as e:
            print(f"⚠️ 时间段统计时出错: {e}")
            print("  继续处理...")
        
        # 7. 检查缺失值处理
        missing_time = df[time_column].isna().sum()
        if missing_time > 0:
            print(f"\n⚠️ 发现 {missing_time:,} 个缺失的时间值，已使用默认值填充（下午时段）")
        
        # 8. 保存结果
        if output_file_path is None:
            output_file_path = input_file_path
        
        print(f"💾 保存处理结果到: {output_file_path}")
        df_with_time.to_csv(output_file_path, index=False)
        
        # 9. 验证保存结果
        print("✅ 验证保存结果...")
        saved_df = pd.read_csv(output_file_path, nrows=5)
        required_time_features = ['time_period']
        
        for feature in required_time_features:
            if feature in saved_df.columns:
                print(f"  ✓ {feature} 列已成功添加")
            else:
                print(f"  ❌ {feature} 列添加失败")
        
        print(f"✅ 文件处理完成！")
        print(f"   原始列数: {len(original_columns)}")
        print(f"   新增列数: {len(df_with_time.columns) - len(original_columns)}")
        print(f"   最终列数: {len(df_with_time.columns)}")
        print(f"   新增列: time_period (时间段编码: 0=凌晨, 1=上午, 2=下午, 3=晚上)")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")
        return False


def main():
    """主函数：处理所有目标数据文件"""
    
    print("🏥 医疗分诊数据时间标签预处理工具")
    print("=" * 60)
    
    # 定义文件路径
    data_dir = os.path.join(os.path.dirname(__file__), "processed_data")
    
    target_files = [
        {
            'name': 'triage_final.csv',
            'path': os.path.join(data_dir, 'triage_final.csv'),
            'description': '最终分诊数据'
        },
        {
            'name': 'triage_with_keywords.csv', 
            'path': os.path.join(data_dir, 'triage_with_keywords.csv'),
            'description': '包含关键词的分诊数据'
        }
    ]
    
    # 处理每个文件
    success_count = 0
    for file_info in target_files:
        print(f"\n📋 处理 {file_info['description']} ({file_info['name']})...")
        print("-" * 50)
        
        if add_time_labels_to_file(file_info['path'], backup=True):
            success_count += 1
            print(f"✅ {file_info['name']} 处理成功")
        else:
            print(f"❌ {file_info['name']} 处理失败")
    
    # 总结
    print("\n" + "=" * 60)
    print(f"🎉 处理完成！成功处理 {success_count}/{len(target_files)} 个文件")
    
    if success_count == len(target_files):
        print("✅ 所有数据文件已成功添加时间段编码标签")
        print("\n💡 现在可以在所有分诊方法中直接使用以下时间特征：")
        print("  📌 time_period: 时间段编码 (0=凌晨, 1=上午, 2=下午, 3=晚上)")
        print(f"\n🔄 建议更新所有方法文件以使用预处理好的时间特征列")
    else:
        print("⚠️ 部分文件处理失败，请检查错误信息并重试")


if __name__ == "__main__":
    main()
