#!/usr/bin/env python3
"""
Ths Daban Indicator Service - 同花顺打板指标服务
基于Tushare Pro API，整合打板策略核心因子
"""
import logging
import asyncio
import os
from datetime import datetime, timedelta
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
    logging.warning("tushare not installed. Service will not be available.")

class ThsDabanService:
    """
    同花顺打板指标服务
    核心功能：提供全方位的打板决策因子，包括封板时间、封单力度、换手率、资金流向、板块地位及市场情绪等。
    """
    
    # 概念过滤黑名单 (过滤掉泛金融属性、指数成分等非题材类概念)
    CONCEPT_IGNORE_LIST = [
        '融资融券', '转融券标的', '融资标的', '融券标的', '融资标的股', '融券标的股',
        '深股通', '沪股通', '港股通', 'HS300_', 'SZ50_', 'ZZ500_',
        '标普道琼斯', '富时罗素', 'MSCI', '证金持股', '汇金持股', '社保重仓',
        '基金重仓', 'QFII重仓', '成份股', '指数', '板块', '含可转债', '新股与次新股'
    ]
    
    def __init__(self, token: Optional[str] = None):
        self.name = "Ths Daban Indicator Service"
        self.token = token or os.getenv('TUSHARE_TOKEN')
        self.pro = None
        self.yuzi_map = {} # name -> {desc, orgs}
        self.yuzi_list = [] # [name1, name2, ...]
        
        if not self.token:
             logging.warning("TUSHARE_TOKEN not found in environment variables")

        if TUSHARE_AVAILABLE:
            try:
                if self.token:
                    ts.set_token(self.token)
                    self.pro = ts.pro_api(self.token)
                    logging.info("ThsDabanService initialized")
                    self._init_yuzi_data()
                else:
                    logging.warning("ThsDabanService initialized without token")
            except Exception as e:
                logging.error(f"Failed to initialize Tushare API: {str(e)}")
    
    def _init_yuzi_data(self):
        try:
            logging.info("Fetching Yuzi list (hm_list)...")
            df = self.pro.hm_list()
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    name = row.get('name') or row.get('hm_name')
                    if not name:
                        continue
                    
                    self.yuzi_map[name] = {
                        "desc": row.get('desc', '暂无描述'),
                        "orgs": row.get('orgs', '')
                    }
                    self.yuzi_list.append(name)
                logging.info(f"Loaded {len(self.yuzi_list)} Yuzi profiles.")
            else:
                logging.warning("hm_list returned empty.")
        except Exception as e:
            logging.warning(f"Failed to fetch hm_list (Check permissions): {e}")
    
    async def _get_code_by_name(self, names: List[str]) -> Dict[str, str]:
        if not self.pro:
            return {}
        name_map = {}
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(None, lambda: self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name'))
            if df is not None and not df.empty:
                for name in names:
                    row = df[df['name'] == name]
                    if not row.empty:
                        name_map[name] = row.iloc[0]['ts_code']
            return name_map
        except Exception as e:
            logging.error(f"Error mapping names to codes: {e}")
            return {}

    async def _fetch_yuzi_detail(self, trade_date: str, ts_code: str, hm_name: str, semaphore: asyncio.Semaphore) -> Optional[dict]:
        async with semaphore:
            try:
                loop = asyncio.get_running_loop()
                df = await loop.run_in_executor(None, lambda: self.pro.hm_detail(trade_date=trade_date, ts_code=ts_code, hm_name=hm_name))
                if df is not None and not df.empty:
                    return df.iloc[0].to_dict()
            except Exception:
                pass 
            return None

    async def _fetch_limit_minute_amount(self, ts_code: str, trade_date: str, first_time: str) -> str:
        if not first_time or len(str(first_time)) < 4:
            return "无封板时间"
        try:
            ft_str = str(first_time).zfill(6)
            hh, mm = ft_str[0:2], ft_str[2:4]
            trade_dt_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
            start_dt = f"{trade_dt_str} {hh}:{mm}:00"
            end_dt = f"{trade_dt_str} {hh}:{mm}:59"
            
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(None, lambda: self.pro.stk_mins(ts_code=ts_code, freq='1min', start_date=start_dt, end_date=end_dt))
            
            if df is not None and not df.empty:
                return f"{round(float(df.iloc[0]['amount'])/10000, 2)}万"
            else:
                return "无分钟数据"
        except Exception as e:
            logging.warning(f"Minute data fetch failed: {e}")
            return "权限/数据错误"

    async def _fetch_stock_concepts(self, ts_code: str) -> str:
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(None, lambda: self.pro.concept_detail(ts_code=ts_code))
            if df is not None and not df.empty:
                concepts = df['concept_name'].unique().tolist()
                filtered_concepts = []
                for c in concepts:
                    is_ignored = False
                    for ignore in self.CONCEPT_IGNORE_LIST:
                        if ignore in c:
                            is_ignored = True
                            break
                    if not is_ignored:
                        filtered_concepts.append(c)
                return ",".join(filtered_concepts[:10]) 
            return ""
        except Exception:
            return ""

    async def _fetch_ths_concepts(self, ts_code: str) -> List[Dict[str, str]]:
        """获取个股所属的同花顺概念板块"""
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(None, lambda: self.pro.ths_member(con_code=ts_code))
            if df is not None and not df.empty:
                concepts = []
                for _, row in df.iterrows():
                    if row.get('ts_code'):
                        concepts.append({'code': row['ts_code'], 'name': '加载中'}) 
                return concepts
            return []
        except Exception as e:
            logging.warning(f"ths_member failed: {e}")
            return []

    async def _fetch_sector_members(self, sector_code: str) -> List[str]:
        """获取板块成分股列表"""
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(None, lambda: self.pro.ths_member(ts_code=sector_code))
            if df is not None and not df.empty:
                return df['con_code'].tolist()
            return []
        except Exception:
            return []

    async def _fetch_stock_hot_rank(self, trade_date: str, stock_names: List[str]) -> Dict[str, Any]:
        """获取个股在同花顺热度榜的排名 (整合自 ThsHotService)"""
        if not self.pro:
            return {}
        try:
            loop = asyncio.get_running_loop()
            # 获取全市场热度榜
            df = await loop.run_in_executor(
                None, 
                lambda: self.pro.ths_hot(trade_date=trade_date, market='热股', fields='ts_code,ts_name,hot,rank')
            )
            
            result = {}
            if df is not None and not df.empty:
                for name in stock_names:
                    # 模糊匹配或精确匹配
                    row = df[df['ts_name'] == name]
                    if not row.empty:
                        item = row.iloc[0]
                        result[name] = {
                            'hot': item.get('hot'),
                            'rank': item.get('rank')
                        }
                    else:
                        result[name] = {'hot': None, 'rank': '未上榜'}
            return result
        except Exception as e:
            logging.warning(f"Failed to fetch hot rank: {e}")
            return {}
            
    async def _fetch_auction_data(self, ts_code: str, trade_date: str) -> Dict[str, Any]:
        """获取个股当日集合竞价数据 (stk_auction)"""
        if not self.pro:
            return {}
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self.pro.stk_auction(ts_code=ts_code, trade_date=trade_date, fields='vol,price,amount,turnover_rate,volume_ratio')
            )
            if df is not None and not df.empty:
                # 理论上只有一行数据，但如果分笔会有多行，stk_auction 返回的是当日汇总还是明细？
                # 根据示例，每天一条记录
                row = df.iloc[0]
                return {
                    "open_price": float(row['price']) if pd.notna(row['price']) else 0,
                    "auction_amount": float(row['amount']) if pd.notna(row['amount']) else 0,
                    "auction_turnover": float(row['turnover_rate']) if pd.notna(row['turnover_rate']) else 0,
                    "auction_vol_ratio": float(row['volume_ratio']) if pd.notna(row['volume_ratio']) else 0,
                    "found": True
                }
            return {"found": False}
        except Exception as e:
            logging.warning(f"Fetch auction data failed: {e}")
            return {"found": False}

    async def _fetch_cyq_chips(self, ts_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """获取官方筹码分布数据 (cyq_chips)"""
        if not self.pro:
            return None
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self.pro.cyq_chips(ts_code=ts_code, trade_date=trade_date)
            )
            if df is not None and not df.empty:
                return df
            return None
        except Exception as e:
            logging.warning(f"Fetch cyq_chips failed: {e}")
            return None

    def _process_cyq_data(self, df_cyq: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """处理官方筹码数据，计算核心指标"""
        if df_cyq is None or df_cyq.empty:
            return {}
            
        # df columns: price, percent
        # Normalize percent just in case (sum should be 100 or 1)
        total_pct = df_cyq['percent'].sum()
        if total_pct == 0:
            return {}
        
        sorted_chips = df_cyq.sort_values('price')
        
        # 1. 获利盘: price < current_price
        profit_df = sorted_chips[sorted_chips['price'] < current_price]
        profit_rate = (profit_df['percent'].sum() / total_pct) * 100
        
        # 2. 平均成本
        avg_cost = (sorted_chips['price'] * sorted_chips['percent']).sum() / total_pct
        
        # 3. 成本集中度
        # Calculate CDF to find cost_15 and cost_85
        sorted_chips['cumsum_pct'] = sorted_chips['percent'].cumsum() / total_pct
        
        cost_15_series = sorted_chips[sorted_chips['cumsum_pct'] >= 0.15]
        cost_85_series = sorted_chips[sorted_chips['cumsum_pct'] >= 0.85]
        
        cost_15 = cost_15_series['price'].iloc[0] if not cost_15_series.empty else 0
        cost_85 = cost_85_series['price'].iloc[0] if not cost_85_series.empty else 0
        
        concentration = 100
        if (cost_85 + cost_15) > 0:
            concentration = (cost_85 - cost_15) / (cost_85 + cost_15) * 100
            
        return {
            "获利盘": round(profit_rate, 2),
            "平均成本": round(avg_cost, 2),
            "筹码集中度": round(concentration, 2),
            "source": "真实筹码(CYQ)"
        }

    def _estimate_chip_distribution_algo(self, df_history: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        [Fallback] 算法估算筹码分布
        基于历史换手率和均价进行筹码衰减计算
        """
        if df_history is None or df_history.empty:
            return {"获利盘": 0, "平均成本": 0, "90%成本区间": "无数据", "筹码集中度": 0, "source": "无数据"}

        # 确保按日期升序
        df = df_history.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
        # 简化版筹码算法
        chip_buckets = {}
        
        # 核心迭代 (假设最近 120 天)
        if len(df) > 120:
            df = df.iloc[-120:]
            
        for _, row in df.iterrows():
            try:
                turnover = float(row['turnover_rate']) / 100.0
                if pd.isna(turnover):
                    continue
                
                vol = float(row.get('vol', 0))
                amount = float(row.get('amount', 0))
                avg_price = (amount * 10 / vol) if vol > 0 else row['close']
                
                decay = 1.0 - turnover
                keys = list(chip_buckets.keys())
                for p in keys:
                    chip_buckets[p] *= decay
                
                price_key = round(avg_price, 2)
                chip_buckets[price_key] = chip_buckets.get(price_key, 0.0) + turnover
            except Exception:
                continue
        
        total_mass = sum(chip_buckets.values())
        if total_mass == 0:
            return {"获利盘": 0, "平均成本": 0, "90%成本区间": "无数据", "筹码集中度": 0, "source": "估算失败"}
        
        sorted_chips = sorted(chip_buckets.items(), key=lambda x: x[0])
        
        profit_mass = sum(mass for p, mass in sorted_chips if p < current_price)
        profit_rate = (profit_mass / total_mass) * 100
        
        weighted_sum = sum(p * mass for p, mass in sorted_chips)
        avg_cost = weighted_sum / total_mass
        
        cum_mass = 0
        cost_5 = 0
        cost_95 = 0
        cost_15 = 0
        cost_85 = 0
        
        for p, mass in sorted_chips:
            cum_mass += mass
            ratio = cum_mass / total_mass
            if cost_5 == 0 and ratio >= 0.05:
                cost_5 = p
            if cost_15 == 0 and ratio >= 0.15:
                cost_15 = p
            if cost_85 == 0 and ratio >= 0.85:
                cost_85 = p
            if cost_95 == 0 and ratio >= 0.95:
                cost_95 = p
            
        concentration = 100
        if (cost_85 + cost_15) > 0:
            concentration = (cost_85 - cost_15) / (cost_85 + cost_15) * 100
            
        return {
            "获利盘": round(profit_rate, 2),
            "平均成本": round(avg_cost, 2),
            "90%成本区间": f"{round(cost_5, 2)} - {round(cost_95, 2)}",
            "筹码集中度": round(concentration, 2),
            "source": "估算筹码(Algo)"
        }

    def _calculate_market_sentiment(self, df_limit: pd.DataFrame) -> Dict[str, Any]:
        """
        计算市场连板情绪指标
        基于当日涨停列表计算：
        1. 最高连板高度
        2. 涨停总家数
        3. 连板家数 (连板数>=2)
        """
        sentiment = {
            "最高连板高度": 0,
            "涨停总家数": 0,
            "连板家数": 0,
            "情绪描述": "无法计算"
        }
        
        if df_limit is None or df_limit.empty:
            return sentiment
            
        try:
            # 过滤掉非涨停状态 (假设 limit_type='U' 且 open_times=0 或者是最后状态是封板)
            # Tushare limit_list_d 通常返回当日有过涨停的，我们需要过滤收盘封住的
            # 这里简单处理，统计所有记录
            
            sentiment["涨停总家数"] = len(df_limit)
            
            if 'limit_times' in df_limit.columns:
                df_limit['limit_times'] = df_limit['limit_times'].fillna(1).astype(int)
                sentiment["最高连板高度"] = df_limit['limit_times'].max()
                sentiment["连板家数"] = len(df_limit[df_limit['limit_times'] >= 2])
            
            # 简单情绪评级
            limit_count = sentiment["涨停总家数"]
            height = sentiment["最高连板高度"]
            
            if limit_count > 100 and height >= 5:
                desc = "情绪火爆"
            elif limit_count > 50 and height >= 3:
                desc = "情绪尚可"
            else:
                desc = "情绪低迷"
                
            sentiment["情绪描述"] = f"{desc} (涨停{limit_count}家, 最高{height}板, 连板{sentiment['连板家数']}家)"
            
        except Exception as e:
            logging.warning(f"Sentiment calculation error: {e}")
            
        return sentiment

    async def _analyze_yesterday_premium(self, target_date: str) -> Dict[str, Any]:
        """
        分析昨日涨停股的今日表现 (赚钱效应/接力情绪)
        利用 limit_list_ths 获取昨日涨停，daily 获取今日涨幅
        """
        if not self.pro:
            return {}
        
        try:
            loop = asyncio.get_running_loop()
            
            # 1. 计算上一个交易日
            df_cal = await loop.run_in_executor(None, lambda: self.pro.trade_cal(exchange='', end_date=target_date, is_open='1', limit=5))
            if df_cal is None or df_cal.empty or len(df_cal) < 2:
                return {"描述": "无交易日历数据"}
            
            df_cal = df_cal.sort_values('cal_date', ascending=False)
            prev_date = df_cal.iloc[1]['cal_date'] # Index 0 是 target_date
            
            # 2. 获取昨日涨停池
            df_prev_limit = await loop.run_in_executor(
                None, 
                lambda: self.pro.limit_list_ths(trade_date=prev_date, limit_type='涨停池', fields='ts_code,name')
            )
            
            if df_prev_limit is None or df_prev_limit.empty:
                return {"描述": f"昨日({prev_date})无涨停数据"}
            
            prev_codes = df_prev_limit['ts_code'].tolist()
            prev_limit_count = len(prev_codes)
            
            # 3. 获取这些股票今日行情
            today_performance = []
            chunk_size = 500
            for i in range(0, len(prev_codes), chunk_size):
                chunk = prev_codes[i:i+chunk_size]
                chunk_str = ",".join(chunk)
                df_today = await loop.run_in_executor(
                    None,
                    lambda: self.pro.daily(trade_date=target_date, ts_code=chunk_str, fields='ts_code,pct_chg,close')
                )
                if df_today is not None and not df_today.empty:
                    today_performance.append(df_today)
            
            if not today_performance:
                return {"描述": "昨日涨停股今日无行情数据"}
                
            df_today_all = pd.concat(today_performance)
            
            # 4. 计算指标
            avg_pct = df_today_all['pct_chg'].mean()
            up_count = len(df_today_all[df_today_all['pct_chg'] > 0])
            # 晋级率 (简单用 > 9.5% 估算)
            limit_count = len(df_today_all[df_today_all['pct_chg'] > 9.5]) 
            
            promotion_rate = (limit_count / prev_limit_count) * 100
            win_rate = (up_count / len(df_today_all)) * 100
            
            desc = f"昨日涨停{prev_limit_count}家, 今日平均涨幅{round(avg_pct, 2)}%, 晋级率{round(promotion_rate, 2)}%({limit_count}家), 赚钱效应{round(win_rate, 2)}%"
            
            return {
                "平均涨幅": round(avg_pct, 2),
                "晋级率": round(promotion_rate, 2),
                "赚钱效应": round(win_rate, 2),
                "描述": desc
            }
            
        except Exception as e:
            logging.warning(f"Yesterday premium analysis failed: {e}")
            return {"描述": "计算失败"}

    async def get_daban_indicators(self, stock_names: str, date: str) -> Dict[str, Any]:
        """
        获取打板核心因子 (Core Strategy Function)
        
        Args:
            stock_names: 股票名称，支持多个，逗号分隔 (e.g., '利欧股份,中信证券').
            date: 查询日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'.
            
        Returns:
            Dict: 包含个股打板因子的详细数据字典.
                  - success (bool): 是否成功
                  - data (list): 个股指标列表
                  - metadata (dict): 元数据

        | 因子名称 | 含义与作用 |
        |----------|------------|
        | 昨日涨停表现 | 市场接力情绪指标：包括昨日涨停股今日平均涨幅、晋级率、赚钱效应 |
        | 竞价金额/换手 | 集合竞价阶段的成交额及换手率（判断开盘抢筹热度） |
        | 竞价量比 | 集合竞价成交量与过去5日均量的比值（>5为爆量，主力异动明显） |
        | 获利盘比例 | 当前价格下处于盈利状态的筹码比例（>90%为新高板，上方无压力） |
        | 筹码集中度 | 90%筹码分布的密集程度（数值越小越集中，<10%为高度控盘） |
        | 首板封板时间 | 第一板涨停时间点（越早越好，反映题材强度和资金力度） |
        | 二板封板时间 | 第二板涨停时间点（综合两天，观察有无加速封板迹象） |
        | 封板次数/炸板次数 | 二板当日封板被打开的次数（0次最好，次数多则封板质量差） |
        | 二板封单比例 | 收盘时涨停封单手数或金额与当日成交的比值（越高表示封板强度越大）|
        | 二板最高封单金额/流通市值 | 越大说明封单金额相对市值越大，有利于连板|
        | 首板换手率 | 第一板当日换手率（%），用于判断首板是缩量板还是放量板 |
        | 二板换手率 | 第二板当日换手率（%），二板缩量/放量情况关系重大 |
        | 成交量变化 | 二板成交额相对于首板的倍数（>1表示放量，<1缩量） |
        | 首板极限成交额 | 首板涨停瞬间单分钟最大成交额（衡量打板时扫单力度） |
        | 龙虎榜机构净买额 | 二板当日龙虎榜机构席位净买入额（正值大额净买有利于晋级） |
        | 龙虎榜知名游资买入 | 是否有知名游资大手笔上榜及净买入（有则加分，显示游资关注） |
        | 主力资金净流入 | 二板当日大单资金净流入金额或占比（判断是否有主力持续加仓） |
        | 板块内连板数排名 | 个股在所属题材板块中的连板高度和强度排名（龙头因子） |
        | 板块热度指标 | 所属概念题材的市场热度（如板块涨停股数量） |
        | 市场连板情绪 | 大盘情绪指标（昨日最高板高度、整体氛围等） |
        | 流通市值 | 流通市值大小（亿元，过大则减分，小盘有利于连板） |
        | 收盘封单金额/流通市值 | 越大说明封单金额相对市值越大，有利于连板 |
        | 个股所属板块连板梯队地位 | 衡量个股在板块内地位 |
        | 个股热度排名 | 个股在同花顺热度榜的实时排名 |
        | 个股所属板块 | 分析板块本身的热度，资金等情况，好的板块个股也好 |
        """
        if not self.pro:
            return {"success": False, "error": "Tushare not initialized"}

        names_list = [n.strip() for n in stock_names.split(',') if n.strip()]
        if not names_list:
            return {"success": False, "error": "No stock names provided"}

        target_date = date.replace('-', '')
        name_code_map = await self._get_code_by_name(names_list)
        ts_codes = list(name_code_map.values())
        if not ts_codes:
            return {"success": False, "error": f"Could not find codes for {stock_names}"}
        
        ts_codes_str = ",".join(ts_codes)
        logging.info(f"Analyzing {names_list} ({ts_codes_str}) for {target_date}")

        loop = asyncio.get_running_loop()
        try:
             target_dt = datetime.strptime(target_date, "%Y%m%d")
             start_date = (target_dt - timedelta(days=20)).strftime("%Y%m%d")
             # 为筹码计算准备更长的历史数据 (120天)
             history_start_date = (target_dt - timedelta(days=180)).strftime("%Y%m%d")
        except ValueError:
             return {"success": False, "error": "Invalid date format"}

        # 2. Batch Data Fetch
        try:
            # 价格数据 (用于筹码计算，需要较长历史)
            df_history_long = await loop.run_in_executor(None, lambda: self.pro.daily(ts_code=ts_codes_str, start_date=history_start_date, end_date=target_date))
            
            # 价格数据 (近期分析)
            df_daily_range = df_history_long[df_history_long['trade_date'] >= start_date].copy() if df_history_long is not None else None
            if df_daily_range is not None:
                df_daily_range = df_daily_range.sort_values('trade_date', ascending=True)

            # 基础指标
            df_basic_range = await loop.run_in_executor(None, lambda: self.pro.daily_basic(ts_code=ts_codes_str, start_date=start_date, end_date=target_date))
            if df_basic_range is not None:
                df_basic_range = df_basic_range.sort_values('trade_date', ascending=True)

            # 个股涨停数据
            df_limit_range = await loop.run_in_executor(None, lambda: self.pro.limit_list_d(ts_code=ts_codes_str, start_date=start_date, end_date=target_date))
            if df_limit_range is not None:
                df_limit_range = df_limit_range.sort_values('trade_date', ascending=True)
            
            # 资金流向
            df_money_t = await loop.run_in_executor(None, lambda: self.pro.moneyflow(ts_code=ts_codes_str, trade_date=target_date))
            
            # 股票基本信息
            df_stock_info = await loop.run_in_executor(None, lambda: self.pro.stock_basic(ts_code=ts_codes_str, fields='ts_code,industry,area,market,name'))

            # 当日全市场涨停列表 (用于板块排名和市场情绪)
            df_limit_all_today = await loop.run_in_executor(None, lambda: self.pro.limit_list_d(trade_date=target_date))

            # 个股热度排名
            hot_rank_map = await self._fetch_stock_hot_rank(target_date, names_list)

            # 市场连板情绪
            market_sentiment = self._calculate_market_sentiment(df_limit_all_today)

            # 昨日涨停表现 (赚钱效应)
            yesterday_premium = await self._analyze_yesterday_premium(target_date)

            # 热门板块 (Top 5 Concept Limit Up)
            hot_sectors_map = {} 
            top5_sectors_display = [] 
            try:
                df_cpt = await loop.run_in_executor(None, lambda: self.pro.limit_cpt_list(trade_date=target_date))
                if df_cpt is not None and not df_cpt.empty:
                    if 'rank' in df_cpt.columns:
                        df_cpt = df_cpt.sort_values('rank', ascending=True)
                    else:
                        df_cpt = df_cpt.sort_values('up_nums', ascending=False)
                        
                    top5 = df_cpt.head(5)
                    for idx, row in top5.iterrows():
                        s_code = row['ts_code']
                        s_name = row['name']
                        s_rank = row.get('rank') if pd.notna(row.get('rank')) else (idx + 1)
                        
                        hot_sectors_map[s_code] = {
                            'name': s_name,
                            'rank': s_rank,
                            'desc': f"{s_name}({row['up_nums']}家, {row['up_stat']})"
                        }
                        top5_sectors_display.append(f"No.{s_rank} {hot_sectors_map[s_code]['desc']}")
            except Exception as e:
                logging.warning(f"Fetch limit_cpt_list failed: {e}")

        except Exception as e:
            logging.error(f"Error fetching data batch: {e}")
            return {"success": False, "error": str(e)}

        # 3. Analysis Loop
        results = []
        for name in names_list:
            ts_code = name_code_map.get(name)
            if not ts_code:
                continue
            
            def get_t_and_prev(df, code, t_date):
                row_t, row_prev = None, None
                if df is not None and not df.empty:
                    df_code = df[df['ts_code'] == code]
                    if not df_code.empty:
                        df_code_sorted = df_code.sort_values('trade_date', ascending=True).reset_index(drop=True)
                        t_rows = df_code_sorted[df_code_sorted['trade_date'] == t_date]
                        if not t_rows.empty:
                            row_t = t_rows.iloc[0]
                            if t_rows.index[0] > 0:
                                row_prev = df_code_sorted.iloc[t_rows.index[0] - 1]
                return row_t, row_prev

            def get_single_row(df, code):
                if df is not None and not df.empty:
                    rows = df[df['ts_code'] == code]
                    if not rows.empty:
                        return rows.iloc[0]
                return None

            row_daily_t, row_daily_prev = get_t_and_prev(df_daily_range, ts_code, target_date)
            row_basic_t, row_basic_prev = get_t_and_prev(df_basic_range, ts_code, target_date)
            row_limit_t, row_limit_prev = get_t_and_prev(df_limit_range, ts_code, target_date)
            row_money_t = get_single_row(df_money_t, ts_code)
            row_info = get_single_row(df_stock_info, ts_code)
            
            # 集合竞价数据 fetch
            auction_data = await self._fetch_auction_data(ts_code, target_date)
            
            # 筹码分布计算 (优先尝试官方接口 cyq_chips)
            current_close = float(row_daily_t['close']) if row_daily_t is not None else 0
            chip_info = {}
            
            # Try official API first
            df_cyq_official = await self._fetch_cyq_chips(ts_code, target_date)
            if df_cyq_official is not None:
                chip_info = self._process_cyq_data(df_cyq_official, current_close)
            
            # Fallback to Algo if failed
            if not chip_info:
                df_chip_history = df_history_long[df_history_long['ts_code'] == ts_code].copy() if df_history_long is not None else None
                chip_info = self._estimate_chip_distribution_algo(df_chip_history, current_close)

            indicators = {}
            
            # --- 竞价与筹码 (新增) ---
            if auction_data.get('found'):
                auc_amt = round(auction_data['auction_amount'] / 10000, 2) # 万元
                auc_to = round(auction_data['auction_turnover'], 2)
                auc_vr = round(auction_data['auction_vol_ratio'], 2)
                indicators['竞价金额/换手'] = f"{auc_amt}万 / {auc_to}%"
                indicators['竞价量比'] = f"{auc_vr}"
            else:
                indicators['竞价金额/换手'] = "无数据/未开通权限"
                indicators['竞价量比'] = "-"

            indicators['获利盘比例'] = f"{chip_info.get('获利盘', 0)}% ({chip_info.get('source', 'N/A')})"
            indicators['筹码集中度'] = f"{chip_info.get('筹码集中度', 0)}%"
            indicators['平均成本'] = f"{chip_info.get('平均成本', 0)}"
            
            # --- 基础打板因子 ---
            indicators['首板封板时间'] = row_limit_prev['first_time'] if row_limit_prev is not None and 'first_time' in row_limit_prev else "非涨停/无数据"
            indicators['二板封板时间'] = row_limit_t['first_time'] if row_limit_t is not None and 'first_time' in row_limit_t else "非涨停/无数据"
            indicators['封板次数/炸板次数'] = int(row_limit_t['open_times']) if row_limit_t is not None and pd.notna(row_limit_t['open_times']) else 0
            
            fd_amount = float(row_limit_t['fd_amount']) if row_limit_t is not None and pd.notna(row_limit_t.get('fd_amount')) else 0
            total_amount = float(row_daily_t['amount']) * 1000 if row_daily_t is not None else 0 
            indicators['二板封单比例'] = round((fd_amount / total_amount) * 100, 2) if total_amount > 0 else 0.0

            circ_mv = float(row_basic_t['circ_mv']) * 10000 if row_basic_t is not None else 0
            indicators['流通市值'] = round(circ_mv / 10000 / 10000, 2) if circ_mv > 0 else None
            indicators['二板最高封单金额/流通市值'] = round((fd_amount / circ_mv) * 100, 4) if circ_mv > 0 else 0.0
            indicators['收盘封单金额/流通市值'] = indicators['二板最高封单金额/流通市值']
            
            indicators['首板换手率'] = float(row_basic_prev['turnover_rate']) if row_basic_prev is not None else None
            indicators['二板换手率'] = float(row_basic_t['turnover_rate']) if row_basic_t is not None else None
            
            vol_t = float(row_daily_t['vol']) if row_daily_t is not None else 0
            vol_prev = float(row_daily_prev['vol']) if row_daily_prev is not None else 0
            indicators['成交量变化'] = round(vol_t / vol_prev, 2) if vol_prev > 0 else None
            
            limit_minute_amt = "无首板数据"
            if row_limit_prev is not None and 'first_time' in row_limit_prev:
                limit_minute_amt = await self._fetch_limit_minute_amount(ts_code, row_limit_prev['trade_date'], row_limit_prev['first_time'])
            indicators['首板极限成交额'] = limit_minute_amt
            
            # --- 资金与游资 ---
            inst_net_buy = 0.0
            yuzi_names_found = []
            semaphore = asyncio.Semaphore(5) 
            tasks = []
            
            if self.yuzi_list:
                for yuzi_name in self.yuzi_list:
                    tasks.append(self._fetch_yuzi_detail(target_date, ts_code, yuzi_name, semaphore))
                yuzi_results = await asyncio.gather(*tasks)
                
                for r in [x for x in yuzi_results if x]:
                    hm_name = r.get('hm_name', '')
                    buy_amt = float(r.get('buy_amount', 0) or 0)
                    sell_amt = float(r.get('sell_amount', 0) or 0)
                    org_name = r.get('hm_orgs') or r.get('org_name') or "未知席位"
                    
                    if '机构' in hm_name or '机构' in org_name:
                        inst_net_buy += (buy_amt - sell_amt)
                    
                    if buy_amt > 0:
                        style_desc = self.yuzi_map.get(hm_name, {}).get('desc', '暂无描述')
                        yuzi_names_found.append(f"【{hm_name}】{org_name}(买{round(buy_amt/10000, 2)}万) [风格:{style_desc}]")
            else:
                yuzi_names_found.append("未加载游资名录")

            indicators['龙虎榜机构净买额'] = f"{round(inst_net_buy / 10000, 2)}万"
            indicators['龙虎榜知名游资买入'] = "\n".join(yuzi_names_found) if yuzi_names_found else "无/未识别到知名游资"

            indicators['主力资金净流入'] = round(float(row_money_t['buy_lg_amount'] + row_money_t['buy_elg_amount'] - row_money_t['sell_lg_amount'] - row_money_t['sell_elg_amount']), 2) if row_money_t is not None else None
            
            # --- 热度与情绪 ---
            indicators['市场连板情绪'] = market_sentiment["情绪描述"]
            indicators['昨日涨停表现'] = yesterday_premium.get("描述", "无数据")
            
            # 个股热度
            hot_info = hot_rank_map.get(name, {})
            indicators['个股热度排名'] = f"Top {hot_info.get('rank', 'N/A')} (热度值: {hot_info.get('hot', 'N/A')})"
            
            # --- 板块与排名 ---
            industry_name = row_info['industry'] if row_info is not None else "未知"
            concept_str = await self._fetch_stock_concepts(ts_code) 
            ths_concepts = await self._fetch_ths_concepts(ts_code)
            
            matched_hot_list = []
            highest_rank_sector = None
            
            for c in ths_concepts:
                c_code = c['code']
                if c_code in hot_sectors_map:
                    hs_info = hot_sectors_map[c_code]
                    c_name = hs_info['name']
                    rank = hs_info['rank']
                    matched_hot_list.append(f"{c_name}(Top {rank})")
                    
                    if highest_rank_sector is None or rank < highest_rank_sector['rank']:
                        highest_rank_sector = {'code': c_code, 'name': c_name, 'rank': rank}
            
            hot_match_str = ", ".join(matched_hot_list) if matched_hot_list else "无"
            indicators['个股所属板块'] = f"行业: {industry_name} | 概念: {concept_str} | 命中热点: {hot_match_str}"
            indicators['板块热度指标'] = "\n".join(top5_sectors_display) if top5_sectors_display else "无数据"
            
            # Rank within Sector
            rank_msg = "无涨停数据"
            status_msg = "非板块龙头"
            target_sector_code = None
            target_sector_name = ""
            
            if highest_rank_sector:
                target_sector_code = highest_rank_sector['code']
                target_sector_name = highest_rank_sector['name']
            
            if target_sector_code:
                member_codes = await self._fetch_sector_members(target_sector_code)
                if member_codes and df_limit_all_today is not None:
                    df_sector_limit = df_limit_all_today[df_limit_all_today['ts_code'].isin(member_codes)].copy()
                    if not df_sector_limit.empty:
                        if 'limit_times' in df_sector_limit.columns:
                            df_sector_limit = df_sector_limit.sort_values(['limit_times', 'first_time'], ascending=[False, True])
                            df_sector_limit['rank'] = range(1, len(df_sector_limit) + 1)
                            
                            my_row = df_sector_limit[df_sector_limit['ts_code'] == ts_code]
                            if not my_row.empty:
                                my_rank = int(my_row.iloc[0]['rank'])
                                my_limit = my_row.iloc[0]['limit_times']
                                total = len(df_sector_limit)
                                rank_msg = f"第{my_rank}名 (板块:{target_sector_name}, 连板{my_limit}, 涨停{total}家)"
                                
                                if my_rank == 1:
                                    status_msg = "板块龙头 (连板高度最高)"
                                elif my_rank <= 3:
                                    status_msg = "板块前排 (龙二/龙三)"
                                else:
                                    status_msg = "板块跟风"
                            else:
                                rank_msg = f"未在{target_sector_name}板块涨停名单中"
            else:
                rank_msg = f"无热门命中 (行业:{industry_name})"

            indicators['板块内连板数排名'] = rank_msg
            indicators['个股所属板块连板梯队地位'] = status_msg
            
            results.append({
                "code": ts_code,
                "name": name,
                "date": target_date,
                "indicators": indicators
            })

        return {
            "success": True,
            "data": results,
            "metadata": { "query_date": target_date }
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    async def test():
        service = ThsDabanService()
        dates = ['20251205'] # Use a recent date
        names = "利欧股份"
        
        for date in dates:
            print(f"\nTesting Daban Analysis for {names} on {date}...")
            result = await service.get_daban_indicators(names, date)
            import json
            print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(test())
