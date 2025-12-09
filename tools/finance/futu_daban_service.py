#!/usr/bin/env python3
"""
Futu Daban Indicator Service - 富途打板指标服务
基于 Futu OpenD API，提供实时分时、盘口及逐笔分析
"""
import logging
import time
import argparse
from typing import Any, Dict, List
from datetime import datetime
import os
import pandas as pd
from dotenv import load_dotenv

# Explicit imports from futu to avoid star import issues
try:
    from futu import (
        OpenQuoteContext, KLType, AuType, SubType, RET_OK
    )
    FUTU_AVAILABLE = True
except ImportError:
    FUTU_AVAILABLE = False
    logging.warning("futu-api not installed. Service will not be available.")

# Load environment variables
load_dotenv()

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    logging.warning("tushare not installed. Name mapping may fail.")

class DBMonitor:
    """
    打板监控单元 (单股)
    负责计算: MA/VWAP, 趋势结构, 盘口压力, 资金流向
    """
    def __init__(self, code: str, name: str, ctx):
        self.code = code
        self.name = name
        self.ctx = ctx
        self.indicators = {}
        # self.score_val = 0 # Removed
        
    def update_kline(self, n_bars=1000, output_n=10, target_date=None):
        """
        更新1分钟K线，计算 MA, VWAP, 趋势, 影线
        Args:
            target_date: 'YYYY-MM-DD', if provided, filter only this date. 
                         Defaults to current date if None.
        """
        if not self.ctx:
            self.indicators['K线状态'] = "未连接"
            return

        # Fetch 1-min K-line (Use None for unadjusted price to match board price)
        ret, df = self.ctx.get_cur_kline(self.code, n_bars, KLType.K_1M, AuType.NONE)
        if ret != RET_OK:
            logging.warning(f"[{self.code}] K-line fetch failed: {df}")
            self.indicators['K线状态'] = f"获取失败: {df}"
            return

        if df.empty:
            self.indicators['K线状态'] = "无数据"
            return

        # Ensure sorted by time
        df = df.sort_values('time_key').reset_index(drop=True)
        
        # Filter future bars
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = df[df['time_key'] <= now_str].reset_index(drop=True)
        
        # Filter by target date (Default to Today if not specified)
        if not target_date:
            target_date = datetime.now().strftime("%Y-%m-%d")
            
        df = df[df['time_key'].str.startswith(target_date)].reset_index(drop=True)

        if df.empty:
            self.indicators['K线状态'] = f"无{target_date}数据"
            # Clear indicators to avoid stale data
            self.indicators['time_key'] = []
            return

        # 1. 计算 MA3, MA5
        df['ma3'] = df['close'].rolling(3).mean()
        df['ma5'] = df['close'].rolling(5).mean()
        
        # 2. 计算 VWAP (Intraday Cumulative - Resets Daily)
        # Identify "Today" based on the last bar's date
        last_time = df['time_key'].iloc[-1]
        last_date = last_time.split(' ')[0] # 'YYYY-MM-DD'
        
        # Create a mask for the current day
        day_mask = df['time_key'].str.startswith(last_date)
        
        # Calculate cumulative sums only for the current day
        # We use .loc to assign back to the original dataframe safely
        day_df = df.loc[day_mask].copy()
        day_df['cum_amount'] = day_df['turnover'].cumsum()
        day_df['cum_vol'] = day_df['volume'].cumsum()
        day_df['vwap'] = day_df['cum_amount'] / day_df['cum_vol']
        
        # Assign VWAP back to original df
        df.loc[day_mask, 'vwap'] = day_df['vwap']
        
        # For non-today rows, VWAP might be NaN or incorrect (based on previous logic), 
        # but we mainly care about the recent output which is likely "today".
        # Ideally, we should do this group-by date if we wanted full history correctness,
        # but for performance on just the "last output_n", this is sufficient if output_n is small.
        # If output_n spans across midnight, the previous day's VWAP won't be recalculated here 
        # (it will stay NaN or previous value), but that's an edge case.


        # Slice the last N rows for output
        subset = df.tail(output_n).copy().reset_index(drop=True)
        
        # Store Time Series (Fundamental K-line Data)
        self.indicators['time_key'] = subset['time_key'].tolist()
        self.indicators['现价'] = [round(float(x), 3) for x in subset['close']]
        self.indicators['分时成交量'] = [int(x) for x in subset['volume']]
        self.indicators['分时成交额'] = [round(float(x), 2) for x in subset['turnover']]
        
        # Calculate Pct Change Series (Requires prev_close from update_tick or day_kline)
        prev_close = self.indicators.get('昨收', 0.0)
        if prev_close > 0:
            self.indicators['涨跌幅强度'] = [f"{round((x - prev_close) / prev_close * 100, 2)}%" for x in subset['close']]
        else:
            # Fallback if prev_close not available
            self.indicators['涨跌幅强度'] = ["0.00%" for _ in subset['close']]

        # Indicators
        self.indicators['分时_MA3'] = [round(float(x), 3) if pd.notna(x) else None for x in subset['ma3']]
        self.indicators['分时_MA5'] = [round(float(x), 3) if pd.notna(x) else None for x in subset['ma5']]
        self.indicators['VWAP'] = [round(float(x), 3) if pd.notna(x) else None for x in subset['vwap']]
        
        # Pre-calculate rolling ma5 for volume on full df to ensure subset has valid history
        df['vol_ma5'] = df['volume'].rolling(5).mean()
            
        # 3. Trend Structure (Slope)
        df['ma5_slope'] = (df['ma5'] - df['ma5'].shift(2)) / 2
        
        def get_trend(slope):
            if pd.isna(slope):
                return "数据不足"
            if slope > 0.01:
                return "强上升"
            elif slope > 0:
                return "弱上升"
            elif slope > -0.01:
                return "震荡"
            else:
                return "下降"
            
        df['trend_str'] = df['ma5_slope'].apply(get_trend)
        
        # 4. Shadow
        def get_shadow(row):
            body = abs(row['close'] - row['open'])
            upper = row['high'] - max(row['open'], row['close'])
            lower = min(row['open'], row['close']) - row['low']
            if upper > body * 2:
                return "长上影(抛压)"
            if lower > body * 2:
                return "长下影(承接)"
            if body > (row['open'] * 0.01):
                return "大实体"
            return "正常"
            
        df['shadow_str'] = df.apply(get_shadow, axis=1)
        
        # 5. Vol Trend
        def get_vol_trend(row):
            if pd.isna(row['vol_ma5']) or row['vol_ma5'] == 0:
                return "平量"
            
            # Check for very low volume (start of bar?)
            ratio = row['volume'] / row['vol_ma5']
            res = "平量"
            
            if ratio > 2.0:
                res = "爆量"
                if row['close'] < row['open']:
                    res += "(下杀)"
                else:
                    res += "(拉升)"
            elif ratio < 0.5:
                res = "缩量"
            
            # Refine "Shrink" if it's extremely small (likely incomplete)
            if ratio < 0.1:
                res = "缩量(初始)"

            if row['close'] > row['open'] and row['volume'] < row['vol_ma5']:
                res += "(无量上冲)"
            return res
            
        df['vol_trend_str'] = df.apply(get_vol_trend, axis=1)
        
        # Extract subset lists
        subset_idx = df.index[-output_n:]
        self.indicators['分时趋势结构'] = df.loc[subset_idx, 'trend_str'].tolist()
        self.indicators['上下影线结构'] = df.loc[subset_idx, 'shadow_str'].tolist()
        self.indicators['分时量能趋势'] = df.loc[subset_idx, 'vol_trend_str'].tolist()

        # 6. 成交金额趋势 (Series)
        self.indicators['成交金额趋势'] = ["流入" if r['close'] > r['open'] else "流出" for _, r in df.loc[subset_idx].iterrows()]

        # 7. Rebound Count (Cumulative? or Snapshot?) 
        prev_close = self.indicators.get('昨收')
        rebound_count = 0
        if prev_close and prev_close > 0:
            cross_up = (df['close'].shift(1) < prev_close) & (df['close'] > prev_close)
            rebound_count = cross_up.sum()
        self.indicators['反抽次数'] = int(rebound_count)

        # 8. OPEN 强度 (Scalar, defined at open)
        if not df.empty:
            first_bar = df.iloc[0]
            op_pct = (first_bar['close'] - first_bar['open']) / first_bar['open'] * 100
            op_strength = "平"
            if op_pct > 1.0:
                op_strength = "极强"
            elif op_pct > 0.5:
                op_strength = "强"
            elif op_pct < -1.0:
                op_strength = "极弱"
            elif op_pct < -0.5:
                op_strength = "弱"
            self.indicators['OPEN强度'] = f"{op_strength}({round(op_pct, 2)}%)"

        # 3.1 高低点结构 (Check last 3 highs of the *latest* moment for scalar alert? 
        high_structure = "正常"
        if len(df) >= 3:
            h1 = df['high'].iloc[-1]
            h2 = df['high'].iloc[-2]
            h3 = df['high'].iloc[-3]
            if h1 < h2 < h3:
                high_structure = "高点下降(风险)"
        self.indicators['高点结构'] = high_structure
        
        # 3.2 早盘低点
        self.indicators['早盘低点'] = float(df['low'].min())

    def update_tick(self):
        """
        更新实时报价 (Snapshot)，作为 Tick 的替代
        """
        if not self.ctx:
            self.indicators['行情状态'] = "未连接"
            return

        ret, df = self.ctx.get_stock_quote([self.code])
        if ret != RET_OK:
            logging.warning(f"[{self.code}] Quote fetch failed: {df}")
            self.indicators['行情状态'] = f"获取失败: {df}"
            return
            
        if df.empty:
            self.indicators['行情状态'] = "无数据"
            return
            
        quote = df.iloc[0]
        
        # 1. 涨跌幅强度
        curr_price = quote['last_price']
        prev_close = quote['prev_close_price']
        
        # Avoid division by zero
        if prev_close and prev_close > 0:
            chg_pct = (curr_price - prev_close) / prev_close * 100
        else:
            chg_pct = 0.0
        
        self.indicators['涨跌幅强度'] = f"{round(chg_pct, 2)}%"
        self.indicators['现价'] = float(curr_price)
        self.indicators['换手率'] = float(quote['turnover_rate'])
        self.indicators['昨收'] = float(prev_close)
        
        # 2. 主动买卖盘 (Proxy using current turnover/volume direction)
        self.indicators['成交量'] = int(quote['volume'])
        self.indicators['成交额'] = round(float(quote['turnover']) / 10000, 2) # 万元

    def update_day_kline(self):
        """
        更新日K线，获取昨日高低点
        """
        if not self.ctx:
            self.indicators['日K状态'] = "未连接"
            return
            
        ret, df = self.ctx.get_cur_kline(self.code, 2, KLType.K_DAY, AuType.NONE)
        if ret == RET_OK and not df.empty and len(df) >= 2:
            # Index -1 is Today, -2 is Yesterday (usually)
            prev = df.iloc[-2]
            self.indicators['昨日最高'] = float(prev['high'])
            self.indicators['昨日最低'] = float(prev['low'])
        else:
             self.indicators['昨日最高'] = "N/A"
             self.indicators['昨日最低'] = "N/A"

    def update_order_book(self):
        """
        更新买卖队列，判断托单/压单
        """
        if not self.ctx:
            self.indicators['盘口状态'] = "未连接"
            return

        # Fetch top 10 levels
        ret, data = self.ctx.get_order_book(self.code, num=10)
        if ret != RET_OK:
            logging.warning(f"[{self.code}] Order book fetch failed: {data}")
            self.indicators['盘口状态'] = f"获取失败: {data}"
            return
            
        bids = data.get('Bid', []) # [(price, volume, count), ...]
        asks = data.get('Ask', [])
        
        if not bids or not asks:
            self.indicators['托单压单'] = "无盘口数据"
            return
            
        total_bid_vol = sum([item[1] for item in bids])
        total_ask_vol = sum([item[1] for item in asks])
        
        ratio = total_bid_vol / total_ask_vol if total_ask_vol > 0 else 10.0
        
        desc = "均衡"
        if ratio > 2:
            desc = "托单极强"
        elif ratio > 1.2:
            desc = "托单强"
        elif ratio < 0.5:
            desc = "压单极强"
        elif ratio < 0.8:
            desc = "压单强"
        
        self.indicators['托单压单'] = f"{desc} (委比:{round(ratio, 2)})"
        self.indicators['买一量'] = int(bids[0][1]) if bids else 0
        self.indicators['卖一量'] = int(asks[0][1]) if asks else 0

class FutuDabanService:
    """
    富途打板指标服务
    封装 FutuOpenD 的常用接口
    """
    def __init__(
        self, 
        host: str = '127.0.0.1', 
        port: int = 11111,
    ):
        """
        初始化富途客户端
        Args:
            host: FutuOpenD 服务地址
            port: FutuOpenD 服务端口
        """
        self.host = host
        self.port = port
        self.unlock_pwd = os.getenv('FUTU_UNLOCK_PWD')
        
        self.ctx = None
        self.token = os.getenv('TUSHARE_TOKEN')
        self.ts_pro = None
        
        if TUSHARE_AVAILABLE and self.token:
            ts.set_token(self.token)
            self.ts_pro = ts.pro_api()

    def _init_ctx(self):
        try:
            # Connect
            self.ctx = OpenQuoteContext(host=self.host, port=self.port)
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Futu OpenD: {e}")
            return False

    def _close_ctx(self):
        if self.ctx:
            self.ctx.close()
            self.ctx = None

    def _sanitize(self, obj):
        """
        Recursively convert numpy types to Python native types for JSON serialization
        """
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'item'): # Numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'): # Numpy arrays
            return obj.tolist()
        return obj

    def _get_code_by_name(self, names: List[str]) -> Dict[str, str]:
        """
        Map stock names to Futu codes (e.g., '利欧股份' -> 'SZ.002131')
        """
        if not self.ts_pro:
            return {}
            
        name_map = {}
        try:
            df = self.ts_pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
            if df is not None and not df.empty:
                for name in names:
                    row = df[df['name'] == name]
                    if not row.empty:
                        ts_code = row.iloc[0]['ts_code']
                        # Convert 002131.SZ -> SZ.002131
                        parts = ts_code.split('.')
                        if len(parts) == 2:
                            futu_code = f"{parts[1]}.{parts[0]}"
                            name_map[name] = futu_code
        except Exception as e:
            logging.error(f"Error mapping names: {e}")
            
        return name_map

    def get_daban_indicators_realtime(self, stock_names: str, output_limit: int = 10, target_date: str = None) -> Dict[str, Any]:
        """
        获取打板核心因子 (基于 Futu 实时数据)
        
        Args:
            stock_names: 股票名称
            output_limit: 返回最近 N 条
            target_date: 指定日期 'YYYY-MM-DD' (默认当天)
            
        Returns:
            Dict: 包含个股打板因子的详细数据字典
            
        | 指标名称 | 含义与作用 |
        |----------|------------|
        | 涨跌幅强度 | 实时涨跌幅，判断资金预期 |
        | 现价/昨收 | 基础价格信息 |
        | 换手率 | 实时换手率 |
        | 成交量/额 | 实时成交数据 |
        | 分时 MA3/MA5 | 1分钟K线均线，判断分时强弱、回踩力度 |
        | VWAP | 分时成交均价，判断主力成本线 (跌破VWAP为弱) |
        | 分时趋势结构 | MA5斜率判断：强上升/弱上升/震荡/下降 |
        | 上下影线结构 | 判断K线形态：长上影(抛压)/长下影(承接)/大实体 |
        | 分时量能趋势 | 爆量(拉升/下杀)/缩量/平量/无量上冲 |
        | 成交金额趋势 | 简单资金流向：流入(收>开)/流出(收<开) |
        | 反抽次数 | 跌破昨日收盘价后反弹突破的次数 |
        | OPEN强度 | 开盘第1分钟涨幅：极强/强/平/弱/极弱 |
        | 高点结构 | 最近3根K线高点比较，判断高点是否下降(风险) |
        | 早盘低点 | 当日最低价，作为防守位参考 |
        | 托单压单 | 买卖盘口委比分析：托单强/压单强 |
        | 买一/卖一量 | 盘口第一档挂单量 |
        """
        names_list = [n.strip() for n in stock_names.split(',') if n.strip()]
        if not names_list:
            return {"success": False, "error": "No stock names provided"}

        # 1. Map names
        name_code_map = self._get_code_by_name(names_list)
        futu_codes = list(name_code_map.values())
        
        if not futu_codes:
            return {"success": False, "error": "Could not map any names to codes (Check Tushare token or names)"}
            
        if not self._init_ctx():
             return {"success": False, "error": "Could not connect to Futu OpenD"}
             
        results = []
        try:
            # 2. Subscribe
            # K_1M, K_DAY, RT_DATA (for Quote/Tick), ORDER_BOOK
            sub_types = [SubType.K_1M, SubType.K_DAY, SubType.QUOTE, SubType.ORDER_BOOK]
            for code in futu_codes:
                for st in sub_types:
                    self.ctx.subscribe([code], [st], subscribe_push=False)
            
            # Allow subscription to register
            time.sleep(1) 
            
            # 3. Process each stock
            for name in names_list:
                code = name_code_map.get(name)
                if not code:
                    continue
                    
                monitor = DBMonitor(code, name, self.ctx)
                
                # Update Data
                # Order matters: Tick (Snapshot) gets prev_close needed for Rebound Count in Kline
                monitor.update_tick() 
                monitor.update_day_kline()
                monitor.update_kline(n_bars=1000, output_n=output_limit, target_date=target_date) 
                monitor.update_order_book()
                
                results.append({
                    "name": name,
                    "code": code,
                    "indicators": monitor.indicators
                })
                
        except Exception as e:
            logging.error(f"Error in processing: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Unsubscribe all
            if self.ctx:
                try:
                    self.ctx.unsubscribe_all()
                except Exception:
                    pass
            self._close_ctx()
            
        # Determine data date string for filename
        if target_date:
            date_str = target_date.replace('-', '')
        else:
            date_str = datetime.now().strftime("%Y%m%d")

        return self._sanitize({
            "success": True,
            "data": results,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "data_date": date_str
            }
        })

def save_to_csv(data: Dict[str, Any], output_dir: str):
    """
    将打板数据保存为CSV文件
    """
    if not data.get('success') or not data.get('data'):
        logging.warning("无数据可保存")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"创建目录: {output_dir}")

    # Use data_date from metadata if available, else current time
    data_date = data.get('metadata', {}).get('data_date')
    if not data_date:
        data_date = datetime.now().strftime("%Y%m%d")

    for item in data['data']:
        name = item.get('name', 'Unknown')
        indicators = item.get('indicators', {})
        
        if not indicators:
            continue

        # 分离时间序列数据和标量数据
        time_series_data = {}
        scalar_data = {}
        
        # 假设 time_key 是主键，用来确定行数
        time_keys = indicators.get('time_key', [])
        if not time_keys:
            logging.warning(f"[{name}] 无分时数据，跳过CSV保存")
            continue
            
        n_rows = len(time_keys)
        
        # Define snapshot fields (only show in last row)
        snapshot_fields = {
            '托单压单', '买一量', '卖一量', 
            '反抽次数', '高点结构', '早盘低点'
        }
        
        # Define excluded fields (debug/status info, not in CSV)
        exclude_fields = {
            'score', '日K状态', 'K线状态', '盘口状态', '行情状态'
        }
        
        for k, v in indicators.items():
            if k in exclude_fields:
                continue
                
            if isinstance(v, list) and len(v) == n_rows:
                time_series_data[k] = v
            else:
                scalar_data[k] = v
        
        try:
            df = pd.DataFrame(time_series_data)
            
            # 将标量数据作为常数列添加
            for k, v in scalar_data.items():
                if k in snapshot_fields:
                    # Only show in last row, leave history empty
                    df[k] = [None] * (n_rows - 1) + [v]
                else:
                    # Normal broadcast for true scalars like 昨收, OPEN强度
                    df[k] = v
            
            # 重新排序列，确保 time_key 在前
            cols = ['time_key'] + [c for c in df.columns if c != 'time_key']
            df = df[cols]
            
            filename = f"{name}_{data_date}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logging.info(f"已保存: {filepath}")
            
        except Exception as e:
            logging.error(f"[{name}] 保存CSV失败: {e}")

if __name__ == "__main__":
    # 配置日志输出到控制台
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='富途打板指标获取工具')
    parser.add_argument('stocks', nargs='?', default='利欧股份', help='股票名称，多个用逗号分隔')
    parser.add_argument('--out', default='real_time_daban_indicator', help='输出目录')
    parser.add_argument('--limit', type=int, default=10, help='输出最近N条分时数据')
    parser.add_argument('--date', default=None, help='指定日期 YYYY-MM-DD (默认当天)')
    
    args = parser.parse_args()
    
    service = FutuDabanService()
    logging.info(f"开始获取数据: {args.stocks} 日期: {args.date or '今天'}...")
    
    try:
        res = service.get_daban_indicators_realtime(args.stocks, args.limit, args.date)
        
        if res.get('success'):
            save_to_csv(res, args.out)
            # 简要打印结果
            for item in res['data']:
                print(f"股票: {item['name']}")
        else:
            logging.error(f"获取失败: {res.get('error')}")
            
    except Exception as e:
        logging.error(f"运行出错 (请确保已启动 Futu OpenD): {e}")
