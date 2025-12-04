#!/usr/bin/env python3
"""
Tushare Stock Data Service.
Provides historical price and key indicators for A-share market.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    logging.warning("tushare not installed. Tushare stock data service will not be available.")


class TushareStockService:
    """Tushare Stock Service for A-share market data."""

    def __init__(self, token: Optional[str] = None):
        """Initialize Tushare service with API token."""
        self.name = "Tushare Stock Data Service"
        self.version = "1.0.0"
        self.token = token or os.getenv('TUSHARE_TOKEN')
        
        if not self.token:
            logging.warning("TUSHARE_TOKEN not found in environment variables")

        if not TUSHARE_AVAILABLE:
            logging.error("tushare library is not available. Please install with: pip install tushare")
            self.pro = None
        else:
            try:
                if self.token:
                    # 全局设置token，用于ts.pro_bar()等全局函数
                    ts.set_token(self.token)
                    self.pro = ts.pro_api(self.token)
                    logging.info(f"Tushare Stock Service v{self.version} initialized successfully")
                else:
                    self.pro = None
                    logging.warning("Tushare initialized without token")
            except Exception as e:
                logging.error(f"Failed to initialize Tushare API: {str(e)}")
                self.pro = None

    async def get_price_data(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        freq: str = 'D',
        adj: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get historical stock price data.
        
        Args:
            ts_code: Stock code (e.g., '000001.SZ', '600519.SH').
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
            freq: Frequency 'D' (Day), 'W' (Week), 'M' (Month).
            adj: Adjustment type: None (default), 'qfq' (pre-adj), 'hfq' (post-adj).
        
        Returns:
            Dict: Contains 'data' list with OHLC, volume, amount.
        """
        try:
            if not TUSHARE_AVAILABLE or self.pro is None:
                return {
                    "success": False,
                    "error": "tushare library is not installed or API initialization failed",
                    "data": [],
                    "metadata": {}
                }
            
            # 参数验证
            if not ts_code or not ts_code.strip():
                raise ValueError("ts_code is required and cannot be empty")
            
            # 处理ts_code，支持逗号分隔
            ts_code = ts_code.strip().upper().replace(' ', '')
            
            # 验证freq参数
            valid_freqs = ['D', 'W', 'M']
            if freq not in valid_freqs:
                raise ValueError(f"freq must be one of {valid_freqs}, got: {freq}")
            
            # 验证adj参数
            valid_adjs = [None, 'qfq', 'hfq']
            if adj not in valid_adjs:
                raise ValueError(f"adj must be one of {valid_adjs}, got: {adj}")
            
            # 日期格式转换：YYYY-MM-DD -> YYYYMMDD
            formatted_start = None
            formatted_end = None
            
            if start_date:
                try:
                    dt = datetime.strptime(start_date, "%Y-%m-%d")
                    formatted_start = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid start_date format: {start_date}. Expected: YYYY-MM-DD")
            
            if end_date:
                try:
                    dt = datetime.strptime(end_date, "%Y-%m-%d")
                    formatted_end = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid end_date format: {end_date}. Expected: YYYY-MM-DD")
            
            # 日志记录
            logging.info(
                f"Tushare request - Code: {ts_code}, "
                f"Date: {formatted_start or 'default'} to {formatted_end or 'today'}, "
                f"Freq: {freq}, Adj: {adj or 'None'}"
            )
            
            # 根据参数调用相应的API
            df = None
            
            if adj is not None:
                # 使用pro_bar获取复权数据
                freq_map = {'D': 'D', 'W': 'W', 'M': 'M'}
                
                # 如果是多个股票代码，pro_bar可能不支持一次性请求（取决于内部实现），
                # 为稳妥起见，如果是逗号分隔的多个代码，我们循环获取并合并
                if ',' in ts_code:
                    codes = [c for c in ts_code.split(',') if c]
                    dfs = []
                    for code in codes:
                        try:
                            sub_df = ts.pro_bar(
                                ts_code=code,
                                start_date=formatted_start,
                                end_date=formatted_end,
                                freq=freq_map[freq],
                                adj=adj
                            )
                            if sub_df is not None and not sub_df.empty:
                                dfs.append(sub_df)
                        except Exception as e:
                            logging.warning(f"Error fetching pro_bar for {code}: {e}")
                    
                    if dfs:
                        df = pd.concat(dfs, ignore_index=True)
                else:
                    df = ts.pro_bar(
                        ts_code=ts_code,
                        start_date=formatted_start,
                        end_date=formatted_end,
                        freq=freq_map[freq],
                        adj=adj
                    )
            else:
                # 使用专门的接口获取未复权数据
                if freq == 'D':
                    df = self.pro.daily(
                        ts_code=ts_code,
                        start_date=formatted_start,
                        end_date=formatted_end
                    )
                elif freq == 'W':
                    df = self.pro.weekly(
                        ts_code=ts_code,
                        start_date=formatted_start,
                        end_date=formatted_end
                    )
                elif freq == 'M':
                    df = self.pro.monthly(
                        ts_code=ts_code,
                        start_date=formatted_start,
                        end_date=formatted_end
                    )
            
            if df is None or df.empty:
                logging.warning(f"No data returned for {ts_code}")
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "ts_code": ts_code,
                        "record_count": 0,
                        "start_date": start_date,
                        "end_date": end_date,
                        "freq": freq,
                        "adj": adj,
                        "provider": "tushare",
                        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "message": "No data available for the specified parameters"
                    }
                }
            
            # 数据转换
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    
                    # 处理日期字段
                    if col == 'trade_date' and value:
                        try:
                            record['date'] = datetime.strptime(str(value), "%Y%m%d").strftime("%Y-%m-%d")
                        except:
                            record['date'] = str(value)
                    # 处理NaN值
                    elif value is None or (hasattr(value, '__float__') and str(value) == 'nan'):
                        record[col] = None
                    # 处理数值类型
                    elif isinstance(value, (int, float)):
                        record[col] = float(value)
                    # 其他类型
                    else:
                        record[col] = str(value)
                
                records.append(record)
            
            # 构建metadata
            actual_start = records[0].get('date', start_date) if records else start_date
            actual_end = records[-1].get('date', end_date) if records else end_date
            
            metadata = {
                "ts_code": ts_code,
                "record_count": len(records),
                "start_date": actual_start,
                "end_date": actual_end,
                "freq": freq,
                "adj": adj,
                "provider": "tushare",
                "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "columns": list(df.columns),
            }
            
            logging.info(
                f"Successfully fetched {len(records)} records for {ts_code} "
                f"from {actual_start} to {actual_end}"
            )
            
            return {
                "success": True,
                "data": records,
                "metadata": metadata
            }
            
        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            return {
                "success": False,
                "error": str(ve),
                "data": [],
                "metadata": {
                    "ts_code": ts_code if 'ts_code' in locals() else None,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            logging.error(f"Unexpected error in get_price_data: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "data": [],
                "metadata": {
                    "ts_code": ts_code if 'ts_code' in locals() else None,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }

    async def get_key_indicators(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get key stock indicators including turnover rate and volume ratio.
        
        Args:
            ts_code: Stock code (e.g., '000001.SZ' or comma-separated list).
            trade_date: Specific trade date 'YYYY-MM-DD'.
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
        
        Returns:
            Dict: Contains turnover_rate, volume_ratio, circ_mv, etc.
        """
        try:
            if not TUSHARE_AVAILABLE or self.pro is None:
                return {
                    "success": False,
                    "error": "tushare library is not installed or API initialization failed",
                    "data": [],
                    "metadata": {}
                }
            
            # 至少需要ts_code或trade_date之一
            if not ts_code and not trade_date:
                raise ValueError("At least one of ts_code or trade_date is required")
            
            if ts_code:
                ts_code = ts_code.strip().upper().replace(' ', '')
            
            # 日期格式转换
            formatted_trade_date = None
            formatted_start = None
            formatted_end = None
            
            if trade_date:
                try:
                    dt = datetime.strptime(trade_date, "%Y-%m-%d")
                    formatted_trade_date = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid trade_date format: {trade_date}. Expected: YYYY-MM-DD")
            
            if start_date:
                try:
                    dt = datetime.strptime(start_date, "%Y-%m-%d")
                    formatted_start = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid start_date format: {start_date}. Expected: YYYY-MM-DD")
            
            if end_date:
                try:
                    dt = datetime.strptime(end_date, "%Y-%m-%d")
                    formatted_end = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid end_date format: {end_date}. Expected: YYYY-MM-DD")
            
            # 日志记录
            logging.info(
                f"Tushare daily_basic request - Code: {ts_code or 'all'}, "
                f"Trade date: {formatted_trade_date or 'N/A'}, "
                f"Date range: {formatted_start or 'N/A'} to {formatted_end or 'N/A'}"
            )
            
            # 调用API
            df = self.pro.daily_basic(
                ts_code=ts_code,
                trade_date=formatted_trade_date,
                start_date=formatted_start,
                end_date=formatted_end
            )
            
            if df is None or df.empty:
                logging.warning(f"No daily indicator data returned")
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "ts_code": ts_code,
                        "record_count": 0,
                        "trade_date": trade_date,
                        "start_date": start_date,
                        "end_date": end_date,
                        "provider": "tushare",
                        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "message": "No data available for the specified parameters"
                    }
                }
            
            # 数据转换
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    
                    # 处理日期字段
                    if col == 'trade_date' and value:
                        try:
                            record['date'] = datetime.strptime(str(value), "%Y%m%d").strftime("%Y-%m-%d")
                        except:
                            record['date'] = str(value)
                    # 处理NaN值
                    elif value is None or (hasattr(value, '__float__') and str(value) == 'nan'):
                        record[col] = None
                    # 处理数值类型
                    elif isinstance(value, (int, float)):
                        record[col] = float(value)
                    # 其他类型
                    else:
                        record[col] = str(value)
                
                records.append(record)
            
            # 构建metadata
            metadata = {
                "ts_code": ts_code,
                "record_count": len(records),
                "trade_date": trade_date,
                "start_date": start_date,
                "end_date": end_date,
                "provider": "tushare",
                "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "columns": list(df.columns),
            }
            
            logging.info(f"Successfully fetched {len(records)} daily indicator records")
            
            return {
                "success": True,
                "data": records,
                "metadata": metadata
            }
            
        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            return {
                "success": False,
                "error": str(ve),
                "data": [],
                "metadata": {
                    "ts_code": ts_code if 'ts_code' in locals() else None,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            logging.error(f"Unexpected error in get_daily_indicators: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "data": [],
                "metadata": {
                    "ts_code": ts_code if 'ts_code' in locals() else None,
                    "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }


    async def get_money_flow(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get THS money flow data (buy/sell amount by order size).

        Args:
            ts_code: Stock code (e.g., '000001.SZ' or comma-separated list).
            trade_date: Specific trade date 'YYYY-MM-DD'.
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.

        Returns:
            Dict: Contains buy/sell amounts for small/medium/large/super_large orders.
        """
        try:
            if not TUSHARE_AVAILABLE or self.pro is None:
                return {
                    "success": False,
                    "error": "tushare library is not installed or API initialization failed",
                    "data": [],
                    "metadata": {}
                }
            
            # Require at least ts_code or trade_date
            if not ts_code and not trade_date:
                raise ValueError("At least one of ts_code or trade_date is required")
            
            if ts_code:
                ts_code = ts_code.strip().upper().replace(' ', '')
            
            # Date format conversion
            formatted_trade_date = None
            formatted_start = None
            formatted_end = None
            
            if trade_date:
                try:
                    dt = datetime.strptime(trade_date, "%Y-%m-%d")
                    formatted_trade_date = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid trade_date format: {trade_date}. Expected: YYYY-MM-DD")
            
            if start_date:
                try:
                    dt = datetime.strptime(start_date, "%Y-%m-%d")
                    formatted_start = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid start_date format: {start_date}. Expected: YYYY-MM-DD")
            
            if end_date:
                try:
                    dt = datetime.strptime(end_date, "%Y-%m-%d")
                    formatted_end = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid end_date format: {end_date}. Expected: YYYY-MM-DD")
            
            # Logging
            logging.info(
                f"Tushare moneyflow_ths request - Code: {ts_code or 'all'}, "
                f"Date: {formatted_trade_date or 'range'}"
            )
            
            # Call API
            df = self.pro.moneyflow_ths(
                ts_code=ts_code,
                trade_date=formatted_trade_date,
                start_date=formatted_start,
                end_date=formatted_end
            )
            
            if df is None or df.empty:
                logging.warning(f"No money flow data returned")
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "ts_code": ts_code,
                        "trade_date": trade_date,
                        "provider": "tushare",
                        "message": "No data available"
                    }
                }
            
            # Convert data
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    if col == 'trade_date' and value:
                        try:
                            record['date'] = datetime.strptime(str(value), "%Y%m%d").strftime("%Y-%m-%d")
                        except:
                            record['date'] = str(value)
                    elif value is None or (hasattr(value, '__float__') and str(value) == 'nan'):
                        record[col] = None
                    elif isinstance(value, (int, float)):
                        record[col] = float(value)
                    else:
                        record[col] = str(value)
                records.append(record)
            
            return {
                "success": True,
                "data": records,
                "metadata": {
                    "ts_code": ts_code,
                    "record_count": len(records),
                    "columns": list(df.columns)
                }
            }
            
        except Exception as e:
            logging.error(f"Error in get_money_flow: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "metadata": {}
            }


    async def get_top_list(
        self,
        trade_date: str,
        ts_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get Dragon Tiger List (Top List) data.

        Args:
            trade_date: Trade date 'YYYY-MM-DD' (Required).
            ts_code: Stock code (e.g., '000001.SZ' or comma-separated list).

        Returns:
            Dict: Contains net_amount, reason, buy/sell amounts.
        """
        try:
            if not TUSHARE_AVAILABLE or self.pro is None:
                return {
                    "success": False,
                    "error": "tushare library is not installed or API initialization failed",
                    "data": [],
                    "metadata": {}
                }
            
            if not trade_date:
                raise ValueError("trade_date is required for top_list")
            
            # Date format conversion
            try:
                dt = datetime.strptime(trade_date, "%Y-%m-%d")
                formatted_trade_date = dt.strftime("%Y%m%d")
            except ValueError:
                raise ValueError(f"Invalid trade_date format: {trade_date}. Expected: YYYY-MM-DD")
            
            if ts_code:
                ts_code = ts_code.strip().upper().replace(' ', '')
            
            # Logging
            logging.info(
                f"Tushare top_list request - Date: {formatted_trade_date}, Code: {ts_code or 'all'}"
            )
            
            # Call API
            df = self.pro.top_list(
                trade_date=formatted_trade_date,
                ts_code=ts_code
            )
            
            if df is None or df.empty:
                logging.warning(f"No top list data returned")
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "trade_date": trade_date,
                        "ts_code": ts_code,
                        "provider": "tushare",
                        "message": "No data available"
                    }
                }
            
            # Convert data
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    if col == 'trade_date' and value:
                        try:
                            record['date'] = datetime.strptime(str(value), "%Y%m%d").strftime("%Y-%m-%d")
                        except:
                            record['date'] = str(value)
                    elif value is None or (hasattr(value, '__float__') and str(value) == 'nan'):
                        record[col] = None
                    elif isinstance(value, (int, float)):
                        record[col] = float(value)
                    else:
                        record[col] = str(value)
                records.append(record)
            
            return {
                "success": True,
                "data": records,
                "metadata": {
                    "trade_date": trade_date,
                    "record_count": len(records),
                    "columns": list(df.columns)
                }
            }
            
        except Exception as e:
            logging.error(f"Error in get_top_list: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "metadata": {}
            }


    async def get_top_inst(
        self,
        trade_date: str,
        ts_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get Dragon Tiger List Institution details (Top Inst).

        Args:
            trade_date: Trade date 'YYYY-MM-DD' (Required).
            ts_code: Stock code (e.g., '000001.SZ' or comma-separated list).

        Returns:
            Dict: Contains exalter, side (buy/sell), amounts, reason.
        """
        try:
            if not TUSHARE_AVAILABLE or self.pro is None:
                return {
                    "success": False,
                    "error": "tushare library is not installed or API initialization failed",
                    "data": [],
                    "metadata": {}
                }
            
            if not trade_date:
                raise ValueError("trade_date is required for top_inst")
            
            # Date format conversion
            try:
                dt = datetime.strptime(trade_date, "%Y-%m-%d")
                formatted_trade_date = dt.strftime("%Y%m%d")
            except ValueError:
                raise ValueError(f"Invalid trade_date format: {trade_date}. Expected: YYYY-MM-DD")
            
            if ts_code:
                ts_code = ts_code.strip().upper().replace(' ', '')
            
            # Logging
            logging.info(
                f"Tushare top_inst request - Date: {formatted_trade_date}, Code: {ts_code or 'all'}"
            )
            
            # Call API
            df = self.pro.top_inst(
                trade_date=formatted_trade_date,
                ts_code=ts_code
            )
            
            if df is None or df.empty:
                logging.warning(f"No top inst data returned")
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "trade_date": trade_date,
                        "ts_code": ts_code,
                        "provider": "tushare",
                        "message": "No data available"
                    }
                }
            
            # Convert data
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    if col == 'trade_date' and value:
                        try:
                            record['date'] = datetime.strptime(str(value), "%Y%m%d").strftime("%Y-%m-%d")
                        except:
                            record['date'] = str(value)
                    elif value is None or (hasattr(value, '__float__') and str(value) == 'nan'):
                        record[col] = None
                    elif isinstance(value, (int, float)):
                        record[col] = float(value)
                    else:
                        record[col] = str(value)
                records.append(record)
            
            return {
                "success": True,
                "data": records,
                "metadata": {
                    "trade_date": trade_date,
                    "record_count": len(records),
                    "columns": list(df.columns)
                }
            }
            
        except Exception as e:
            logging.error(f"Error in get_top_inst: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "metadata": {}
            }


    async def get_limit_list(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        limit_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get Limit List (Up/Down limit, broken limit).

        Args:
            trade_date: Specific trade date 'YYYY-MM-DD'.
            ts_code: Stock code (e.g., '000001.SZ' or comma-separated list).
            limit_type: 'U' (Up limit), 'D' (Down limit), 'Z' (Broken limit).
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.

        Returns:
            Dict: Contains limit_times, up_stat, fd_amount, open_times.
        """
        try:
            if not TUSHARE_AVAILABLE or self.pro is None:
                return {
                    "success": False,
                    "error": "tushare library is not installed or API initialization failed",
                    "data": [],
                    "metadata": {}
                }
            
            # Require at least one date param or ts_code
            if not any([trade_date, start_date, end_date, ts_code]):
                 raise ValueError("At least one of trade_date, start_date, end_date or ts_code is required")

            if ts_code:
                ts_code = ts_code.strip().upper().replace(' ', '')
            
            # Date format conversion
            formatted_trade_date = None
            formatted_start = None
            formatted_end = None
            
            if trade_date:
                try:
                    dt = datetime.strptime(trade_date, "%Y-%m-%d")
                    formatted_trade_date = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid trade_date format: {trade_date}. Expected: YYYY-MM-DD")
            
            if start_date:
                try:
                    dt = datetime.strptime(start_date, "%Y-%m-%d")
                    formatted_start = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid start_date format: {start_date}. Expected: YYYY-MM-DD")
            
            if end_date:
                try:
                    dt = datetime.strptime(end_date, "%Y-%m-%d")
                    formatted_end = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid end_date format: {end_date}. Expected: YYYY-MM-DD")
            
            # Logging
            logging.info(
                f"Tushare limit_list_d request - Date: {formatted_trade_date or 'range'}, Type: {limit_type}"
            )
            
            # Call API
            # limit_list_d supports various combinations
            df = self.pro.limit_list_d(
                trade_date=formatted_trade_date,
                ts_code=ts_code,
                limit_type=limit_type,
                start_date=formatted_start,
                end_date=formatted_end
            )
            
            if df is None or df.empty:
                logging.warning(f"No limit list data returned")
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "trade_date": trade_date,
                        "limit_type": limit_type,
                        "provider": "tushare",
                        "message": "No data available"
                    }
                }
            
            # Convert data
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    if col == 'trade_date' and value:
                        try:
                            record['date'] = datetime.strptime(str(value), "%Y%m%d").strftime("%Y-%m-%d")
                        except:
                            record['date'] = str(value)
                    elif value is None or (hasattr(value, '__float__') and str(value) == 'nan'):
                        record[col] = None
                    elif isinstance(value, (int, float)):
                        record[col] = float(value)
                    else:
                        record[col] = str(value)
                records.append(record)
            
            return {
                "success": True,
                "data": records,
                "metadata": {
                    "trade_date": trade_date,
                    "record_count": len(records),
                    "columns": list(df.columns)
                }
            }
            
        except Exception as e:
            logging.error(f"Error in get_limit_list: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "metadata": {}
            }

    async def get_limit_list_ths(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        limit_type: Optional[str] = '涨停池',
        market: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get Limit List (THS) - 同花顺涨跌停榜单.
        
        Args:
            trade_date: Specific trade date 'YYYY-MM-DD'.
            ts_code: Stock code (e.g., '000001.SZ').
            limit_type: Type ('涨停池', '连扳池', '冲刺涨停', '炸板池', '跌停池'). Default: '涨停池'.
            market: Market ('HS', 'GEM', 'STAR').
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
            
        Returns:
            Dict: Contains ths limit list data.
        """
        try:
            if not TUSHARE_AVAILABLE or self.pro is None:
                return {
                    "success": False,
                    "error": "tushare library is not installed or API initialization failed",
                    "data": [],
                    "metadata": {}
                }
            
            # Date format conversion
            formatted_trade_date = None
            formatted_start = None
            formatted_end = None
            
            if trade_date:
                try:
                    dt = datetime.strptime(trade_date, "%Y-%m-%d")
                    formatted_trade_date = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid trade_date format: {trade_date}. Expected: YYYY-MM-DD")
            
            if start_date:
                try:
                    dt = datetime.strptime(start_date, "%Y-%m-%d")
                    formatted_start = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid start_date format: {start_date}. Expected: YYYY-MM-DD")
            
            if end_date:
                try:
                    dt = datetime.strptime(end_date, "%Y-%m-%d")
                    formatted_end = dt.strftime("%Y%m%d")
                except ValueError:
                    raise ValueError(f"Invalid end_date format: {end_date}. Expected: YYYY-MM-DD")
            
            if ts_code:
                ts_code = ts_code.strip().upper().replace(' ', '')

            # Logging
            logging.info(
                f"Tushare limit_list_ths request - Date: {formatted_trade_date or 'range'}, Type: {limit_type}"
            )
            
            # Call API
            df = self.pro.limit_list_ths(
                trade_date=formatted_trade_date,
                ts_code=ts_code,
                limit_type=limit_type,
                market=market,
                start_date=formatted_start,
                end_date=formatted_end
            )
            
            if df is None or df.empty:
                logging.warning(f"No THS limit list data returned")
                return {
                    "success": True,
                    "data": [],
                    "metadata": {
                        "trade_date": trade_date,
                        "limit_type": limit_type,
                        "provider": "tushare",
                        "message": "No data available"
                    }
                }
            
            # Convert data
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    value = row[col]
                    if col in ['trade_date', 'first_lu_time', 'last_lu_time', 'first_ld_time', 'last_ld_time'] and value:
                        try:
                            # Some are YYYYMMDD, some might be timestamps, try basic YYYYMMDD first
                             if len(str(value)) == 8 and str(value).isdigit():
                                record[col] = datetime.strptime(str(value), "%Y%m%d").strftime("%Y-%m-%d")
                             else:
                                record[col] = str(value)
                        except:
                            record[col] = str(value)
                    elif value is None or (hasattr(value, '__float__') and str(value) == 'nan'):
                        record[col] = None
                    elif isinstance(value, (int, float)):
                        record[col] = float(value)
                    else:
                        record[col] = str(value)
                records.append(record)
            
            return {
                "success": True,
                "data": records,
                "metadata": {
                    "trade_date": trade_date,
                    "record_count": len(records),
                    "columns": list(df.columns)
                }
            }
            
        except Exception as e:
            logging.error(f"Error in get_limit_list_ths: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "metadata": {}
            }


if __name__ == "__main__":
    import asyncio
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_tushare_service():
        """测试Tushare服务"""
        # 初始化服务（自动从环境变量获取token）
        service = TushareStockService()
        
        target_date = '2025-11-27'
        print("\n" + "="*80)
        print(f"测试 Tushare Stock Data Service - 目标日期: {target_date}")
        print("="*80)
        
        # 测试1：获取日线数据
        print(f"\n[测试1] 获取平安银行和贵州茅台日线数据 ({target_date})...")
        result1 = await service.get_price_data(
            ts_code='000001.SZ,600519.SH',
            start_date=target_date,
            end_date=target_date,
            freq='D'
        )
        print(f"成功: {result1['success']}")
        print(f"记录数: {result1['metadata'].get('record_count', 0)}")
        if result1['success'] and result1['data']:
            print(f"样本数据(前2): {result1['data'][:2]}")
        
        # 测试2：获取周线数据
        print("\n[测试2] 获取平安银行周线数据...")
        result2 = await service.get_price_data(
            ts_code='000001.SZ',
            start_date='2025-01-01',
            end_date=target_date,
            freq='W'
        )
        print(f"成功: {result2['success']}")
        print(f"记录数: {result2['metadata'].get('record_count', 0)}")
        
        # 测试3：获取前复权数据
        print(f"\n[测试3] 获取平安银行和贵州茅台前复权日线数据 ({target_date})...")
        result3 = await service.get_price_data(
            ts_code='000001.SZ,600519.SH',
            start_date=target_date,
            end_date=target_date,
            freq='D',
            adj='qfq'
        )
        print(f"成功: {result3['success']}")
        print(f"记录数: {result3['metadata'].get('record_count', 0)}")
        
        # 测试4：获取打板关键指标（换手率、量比等）
        print(f"\n[测试4] 获取平安银行关键打板指标 ({target_date})...")
        result4 = await service.get_key_indicators(
            ts_code='000001.SZ',
            start_date=target_date,
            end_date=target_date
        )
        print(f"成功: {result4['success']}")
        print(f"记录数: {result4['metadata'].get('record_count', 0)}")
        if result4['success'] and result4['data']:
            sample = result4['data'][0]
            print(f"样本日期: {sample['date']}")
            print(f"收盘价: {sample['close']}")
            print(f"换手率: {sample['turnover_rate']}%")
            print(f"量比: {sample['volume_ratio']}")
            print(f"流通市值: {sample['circ_mv']}万元")
        
        # 测试5：获取资金流向数据
        print(f"\n[测试5] 获取平安银行资金流向数据 ({target_date})...")
        result5 = await service.get_money_flow(
            ts_code='000001.SZ',
            start_date=target_date,
            end_date=target_date
        )
        print(f"成功: {result5['success']}")
        print(f"记录数: {result5['metadata'].get('record_count', 0)}")
        if result5['success'] and result5['data']:
            sample = result5['data'][0]
            print(f"样本日期: {sample['date']}")
            print(f"大单买入: {sample.get('buy_lg_amount', 'N/A')}")
            print(f"大单买入率: {sample.get('buy_lg_amount_rate', 'N/A')}")
        
        # 测试6：获取龙虎榜数据
        print(f"\n[测试6] 获取龙虎榜数据 ({target_date})...")
        result6 = await service.get_top_list(
            trade_date=target_date
        )
        print(f"成功: {result6['success']}")
        print(f"记录数: {result6['metadata'].get('record_count', 0)}")
        if result6['success'] and result6['data']:
            print(f"前3条龙虎榜数据:")
            for i, item in enumerate(result6['data'][:3]):
                print(f"{i+1}. 代码:{item['ts_code']} 名称:{item.get('name','N/A')} 净买入:{item.get('net_amount','N/A')}万 上榜理由:{item.get('reason','N/A')}")
        
        # 测试7：获取龙虎榜机构明细
        print(f"\n[测试7] 获取龙虎榜机构明细 ({target_date})...")
        result7 = await service.get_top_inst(
            trade_date=target_date
        )
        print(f"成功: {result7['success']}")
        print(f"记录数: {result7['metadata'].get('record_count', 0)}")
        if result7['success'] and result7['data']:
            print(f"前3条机构明细:")
            for i, item in enumerate(result7['data'][:3]):
                side_str = "卖出" if str(item.get('side')) == '1' else "买入"
                print(f"{i+1}. 代码:{item['ts_code']} 机构:{item.get('exalter','N/A')} 类型:{side_str} 净买入:{item.get('net_buy','N/A')}元")
        
        # 测试8：获取涨跌停数据
        print(f"\n[测试8] 获取涨停板数据 ({target_date})...")
        result8 = await service.get_limit_list(
            trade_date=target_date,
            limit_type='U'
        )
        print(f"成功: {result8['success']}")
        print(f"记录数: {result8['metadata'].get('record_count', 0)}")
        if result8['success'] and result8['data']:
            print(f"前3个涨停股:")
            for i, item in enumerate(result8['data'][:3]):
                print(f"{i+1}. 代码:{item['ts_code']} 名称:{item.get('name','N/A')} 连板:{item.get('limit_times','N/A')} 封单:{item.get('fd_amount','N/A')}元")
        
        # 测试9：获取同花顺涨跌停数据
        print(f"\n[测试9] 获取同花顺涨跌停数据 ({target_date})...")
        result9 = await service.get_limit_list_ths(
            trade_date=target_date,
            limit_type='涨停池'
        )
        print(f"成功: {result9['success']}")
        print(f"记录数: {result9['metadata'].get('record_count', 0)}")
        if result9['success'] and result9['data']:
             print(f"前3个同花顺涨停股:")
             for i, item in enumerate(result9['data'][:3]):
                print(f"{i+1}. 代码:{item['ts_code']} 描述:{item.get('lu_desc','N/A')} 状态:{item.get('status','N/A')} 标签:{item.get('tag','N/A')}")

        print("\n" + "="*80)
        print("测试完成")
        print("="*80 + "\n")
    
    # 运行测试
    asyncio.run(test_tushare_service())
