#!/usr/bin/env python3
"""
Stock Hotness & Concept Service - 个股热度与板块查询
基于Tushare Pro API (同花顺热度接口)
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict
import tushare as ts
import asyncio
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ThsHotService:
    """Tushare同花顺热度及概念服务"""
    
    def __init__(self):
        self.name = "Stock Hotness & Concept Service"
        # Use token from environment variable
        self.token = os.getenv('TUSHARE_TOKEN')
        
        if not self.token:
            logging.warning("TUSHARE_TOKEN not found in environment variables")
        
        try:
            if self.token:
                ts.set_token(self.token)
                self.pro = ts.pro_api(self.token)
                logging.info("ThsHotService initialized")
            else:
                self.pro = None
                logging.warning("ThsHotService initialized without token (some features may fail)")
        except Exception as e:
            logging.error(f"Failed to initialize Tushare API: {str(e)}")
            self.pro = None

    async def get_stock_info(self, stock_name: str, trade_date: str = None) -> Dict[str, Any]:
        """
        根据股票名称获取热度值和所属概念板块
        
        Args:
            stock_name: 股票名称 (如 '宁德时代' 或 '宁德时代,贵州茅台')
            trade_date: 交易日期 (YYYYMMDD)，默认当天或最近交易日
            
        Returns:
            Dict: 包含热度和概念数据的字典 (data字段为列表)
        """
        try:
            if not self.pro:
                return {"success": False, "error": "Tushare not initialized"}

            # 如果未提供日期，默认使用今天
            if not trade_date:
                trade_date = datetime.now().strftime("%Y%m%d")

            logging.info(f"Fetching info for stock(s): {stock_name} on date: {trade_date}")
            
            # 获取同花顺热度榜单 (市场='热股')
            # 注意：API一次返回整个榜单，我们在内存中过滤
            df = self.pro.ths_hot(trade_date=trade_date, market='热股', fields='ts_code,ts_name,hot,concept')
            
            if df is None or df.empty:
                # 尝试回退一天（如果是周末或节假日可能无数据，简单重试机制）
                # 这里简单处理：如果当天没数据，提示可能非交易日
                return {
                    "success": False, 
                    "error": f"No data found for date {trade_date}. It might be a non-trading day or data not updated yet.",
                    "trade_date": trade_date
                }
            
            # 处理输入的股票名称，支持逗号分隔
            target_names = [name.strip() for name in stock_name.split(',') if name.strip()]
            
            # 过滤特定股票
            target_stocks = df[df['ts_name'].isin(target_names)]
            
            if target_stocks.empty:
                return {
                    "success": True,
                    "found": False,
                    "message": f"Stocks '{stock_name}' not found in top hot list for {trade_date}.",
                    "data": []
                }
            
            records = []
            for _, row in target_stocks.iterrows():
                # 转字典
                record = row.to_dict()
                
                # 处理NaN
                cleaned_record = {}
                for k, v in record.items():
                    if pd.isna(v):
                        cleaned_record[k] = None
                    else:
                        cleaned_record[k] = v
                
                # 格式化概念字段 (API返回可能是字符串形式的列表 "['A', 'B']")
                # 如果是字符串，尝试解析；如果是列表直接使用
                concepts = cleaned_record.get('concept')
                if isinstance(concepts, str):
                    try:
                        # 简单的字符串清理，或者使用eval (需谨慎)
                        # 假设格式是标准的JSON-like list
                        import ast
                        concepts = ast.literal_eval(concepts)
                        cleaned_record['concept'] = concepts
                    except:
                        pass # 保持原样
                
                records.append(cleaned_record)

            return {
                "success": True,
                "found": True,
                "data": records,
                "metadata": {
                    "trade_date": trade_date,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "count": len(records)
                }
            }
            
        except Exception as e:
            logging.error(f"Error querying stock info: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_ths_index_daily(self, ts_code: str, trade_date: str = None, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        获取同花顺板块指数日线行情
        
        Args:
            ts_code: 指数代码 (如 '865001.TI' 或 '865001.TI,885751.TI')
            trade_date: 交易日期 (YYYYMMDD)
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            Dict: 包含板块指数行情的字典
        """
        try:
            if not self.pro:
                return {"success": False, "error": "Tushare not initialized"}
            
            logging.info(f"Fetching THS daily index: {ts_code}, trade_date={trade_date}, start={start_date}, end={end_date}")
            
            # 构造参数字典，过滤掉None
            params = {'ts_code': ts_code}
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            params['fields'] = 'ts_code,trade_date,open,close,high,low,pct_change,vol,turnover_rate,total_mv,float_mv'
            
            df = self.pro.ths_daily(**params)
            
            if df is None or df.empty:
                return {
                    "success": True,
                    "found": False,
                    "message": "No data found for the given criteria.",
                    "data": []
                }
            
            records = []
            for _, row in df.iterrows():
                record = row.to_dict()
                # 处理NaN
                cleaned_record = {}
                for k, v in record.items():
                    if pd.isna(v):
                        cleaned_record[k] = None
                    else:
                        cleaned_record[k] = v
                records.append(cleaned_record)
                
            return {
                "success": True,
                "found": True,
                "data": records,
                "metadata": {
                    "count": len(records),
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            logging.error(f"Error querying THS daily index: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_hot_money_detail(self, ts_code: str = None, trade_date: str = None, hm_name: str = None, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        获取游资明细 (Hot Money Detail)
        
        Args:
            ts_code: 股票代码 (支持逗号分隔的多个代码)
            trade_date: 交易日期 (YYYYMMDD)
            hm_name: 游资名称
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            Dict: 包含游资明细的字典
        """
        try:
            if not self.pro:
                return {"success": False, "error": "Tushare not initialized"}
            
            logging.info(f"Fetching hot money detail: ts_code={ts_code}, trade_date={trade_date}, hm_name={hm_name}")
            
            # 构造参数字典
            params = {}
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            if hm_name:
                params['hm_name'] = hm_name
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            params['fields'] = 'trade_date,ts_code,ts_name,buy_amount,sell_amount,net_amount,hm_name,hm_orgs,tag'
            
            df = self.pro.hm_detail(**params)
            
            if df is None or df.empty:
                return {
                    "success": True,
                    "found": False,
                    "message": "No hot money data found for the given criteria.",
                    "data": []
                }
            
            records = []
            for _, row in df.iterrows():
                record = row.to_dict()
                # 处理NaN
                cleaned_record = {}
                for k, v in record.items():
                    if pd.isna(v):
                        cleaned_record[k] = None
                    else:
                        cleaned_record[k] = v
                records.append(cleaned_record)
                
            return {
                "success": True,
                "found": True,
                "data": records,
                "metadata": {
                    "count": len(records),
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            logging.error(f"Error querying hot money detail: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_hot_money_list(self, name: str = None) -> Dict[str, Any]:
        """
        获取游资名录 (Hot Money List)
        
        Args:
            name: 游资名称 (可选，模糊匹配)
            
        Returns:
            Dict: 包含游资名录的字典
        """
        try:
            if not self.pro:
                return {"success": False, "error": "Tushare not initialized"}
            
            logging.info(f"Fetching hot money list: name={name}")
            
            # 构造参数字典
            params = {}
            if name:
                params['name'] = name
            
            params['fields'] = 'name,desc,orgs'
            
            df = self.pro.hm_list(**params)
            
            if df is None or df.empty:
                return {
                    "success": True,
                    "found": False,
                    "message": "No hot money list found.",
                    "data": []
                }
            
            records = []
            for _, row in df.iterrows():
                record = row.to_dict()
                # 处理NaN
                cleaned_record = {}
                for k, v in record.items():
                    if pd.isna(v):
                        cleaned_record[k] = None
                    else:
                        cleaned_record[k] = v
                records.append(cleaned_record)
                
            return {
                "success": True,
                "found": True,
                "data": records,
                "metadata": {
                    "count": len(records),
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            logging.error(f"Error querying hot money list: {str(e)}")
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async def test():
        service = ThsHotService()
        print("Testing ThsHotService...")
        
        # 测试用例
        # 注意：需要一个有效的历史交易日来保证有数据，周末可能无数据
        test_date = '20240315' 
        stock_name = '宁德时代,贵州茅台'
        
        print(f"Querying {stock_name} on {test_date}...")
        res = await service.get_stock_info(stock_name, test_date)
        
        import json
        print(json.dumps(res, ensure_ascii=False, indent=2))
        
        # 测试不存在的股票
        print("\nQuerying non-existent stock...")
        res2 = await service.get_stock_info("不存在的股票", test_date)
        print(json.dumps(res2, ensure_ascii=False, indent=2))

        # 测试同花顺板块指数
        print("\nTesting get_ths_index_daily...")
        index_code = '865001.TI' # 示例代码
        print(f"Querying index {index_code}...")
        # 只需要最近几天的
        start_date_test = '20240101'
        end_date_test = '20240110'
        res3 = await service.get_ths_index_daily(ts_code=index_code, start_date=start_date_test, end_date=end_date_test)
        print(json.dumps(res3, ensure_ascii=False, indent=2))
        
        # 测试逗号分隔输入
        print("\nTesting get_ths_index_daily with comma-separated string...")
        index_codes = '865001.TI,885751.TI'
        res4 = await service.get_ths_index_daily(ts_code=index_codes, start_date=start_date_test, end_date=end_date_test)
        print(f"Found {len(res4.get('data', []))} records for multiple codes")
        
        # 测试游资明细
        print("\nTesting get_hot_money_detail...")
        # 注意：hm_detail需要权限或积分，且数据可能有延迟
        hm_date = '20230815'
        print(f"Querying hot money detail on {hm_date}...")
        res5 = await service.get_hot_money_detail(trade_date=hm_date)
        print(f"Found {len(res5.get('data', []))} hot money records")
        if res5.get('data'):
            print(res5['data'][0])
            
        # 测试游资名录
        print("\nTesting get_hot_money_list...")
        print("Querying hot money list...")
        res6 = await service.get_hot_money_list()
        print(f"Found {len(res6.get('data', []))} hot money names")
        if res6.get('data'):
            print(res6['data'][0])

    asyncio.run(test())
