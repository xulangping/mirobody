#!/usr/bin/env python3
"""
Ths Daban Indicator Service - 同花顺打板指标服务
基于Tushare Pro API，专门针对打板（首板、二板等）策略提供深度因子分析
"""
import logging
import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import functools

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
    提供：封板时间、封单比、换手率变化、资金流向、板块地位等核心打板因子
    """
    
    # 概念过滤黑名单 (过滤掉泛金融属性、指数成分等非题材类概念)
    CONCEPT_IGNORE_LIST = [
        '融资融券', '转融券标的', '融资标的', '融券标的', '融资标的股', '融券标的股',
        '深股通', '沪股通', '港股通', 'HS300_', 'SZ50_', 'ZZ500_',
        '标普道琼斯', '富时罗素', 'MSCI', '证金持股', '汇金持股', '社保重仓',
        '基金重仓', 'QFII重仓', '成份股', '指数', '板块', '含可转债'
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
                    if not name: continue
                    
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
        if not self.pro: return {}
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
        if not first_time or len(str(first_time)) < 4: return "无封板时间"
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

    async def get_daban_indicators(self, stock_names: str, date: str) -> Dict[str, Any]:
        if not self.pro: return {"success": False, "error": "Tushare not initialized"}

        names_list = [n.strip() for n in stock_names.split(',') if n.strip()]
        if not names_list: return {"success": False, "error": "No stock names provided"}

        target_date = date.replace('-', '')
        name_code_map = await self._get_code_by_name(names_list)
        ts_codes = list(name_code_map.values())
        if not ts_codes: return {"success": False, "error": f"Could not find codes for {stock_names}"}
        
        ts_codes_str = ",".join(ts_codes)
        logging.info(f"Analyzing {names_list} ({ts_codes_str}) for {target_date}")

        loop = asyncio.get_running_loop()
        try:
             target_dt = datetime.strptime(target_date, "%Y%m%d")
             start_date = (target_dt - timedelta(days=20)).strftime("%Y%m%d")
        except ValueError:
             return {"success": False, "error": "Invalid date format"}

        # 2. Batch Data Fetch
        try:
            df_daily_range = await loop.run_in_executor(None, lambda: self.pro.daily(ts_code=ts_codes_str, start_date=start_date, end_date=target_date))
            if df_daily_range is not None: df_daily_range = df_daily_range.sort_values('trade_date', ascending=True)

            df_basic_range = await loop.run_in_executor(None, lambda: self.pro.daily_basic(ts_code=ts_codes_str, start_date=start_date, end_date=target_date))
            if df_basic_range is not None: df_basic_range = df_basic_range.sort_values('trade_date', ascending=True)

            df_limit_range = await loop.run_in_executor(None, lambda: self.pro.limit_list_d(ts_code=ts_codes_str, start_date=start_date, end_date=target_date))
            if df_limit_range is not None: df_limit_range = df_limit_range.sort_values('trade_date', ascending=True)
            
            df_money_t = await loop.run_in_executor(None, lambda: self.pro.moneyflow(ts_code=ts_codes_str, trade_date=target_date))
            df_stock_info = await loop.run_in_executor(None, lambda: self.pro.stock_basic(ts_code=ts_codes_str, fields='ts_code,industry,area,market,name'))

            df_limit_all_today = await loop.run_in_executor(None, lambda: self.pro.limit_list_d(trade_date=target_date))

            hot_sectors_map = {} 
            top5_sectors_display = [] 
            try:
                df_cpt = await loop.run_in_executor(None, lambda: self.pro.limit_cpt_list(trade_date=target_date))
                if df_cpt is not None and not df_cpt.empty:
                    # 确保按rank排序，如果rank为空则按up_nums降序
                    if 'rank' in df_cpt.columns:
                        df_cpt = df_cpt.sort_values('rank', ascending=True)
                    else:
                        df_cpt = df_cpt.sort_values('up_nums', ascending=False)
                        
                    # Take Top 5
                    top5 = df_cpt.head(5)
                    for idx, row in top5.iterrows():
                        s_code = row['ts_code']
                        s_name = row['name']
                        # 如果有rank用rank，否则用索引+1
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
            if not ts_code: continue
            
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
                    if not rows.empty: return rows.iloc[0]
                return None

            row_daily_t, row_daily_prev = get_t_and_prev(df_daily_range, ts_code, target_date)
            row_basic_t, row_basic_prev = get_t_and_prev(df_basic_range, ts_code, target_date)
            row_limit_t, row_limit_prev = get_t_and_prev(df_limit_range, ts_code, target_date)
            row_money_t = get_single_row(df_money_t, ts_code)
            row_info = get_single_row(df_stock_info, ts_code)
            
            indicators = {}
            indicators['首板封板时间'] = row_limit_prev['first_time'] if row_limit_prev is not None and 'first_time' in row_limit_prev else "非涨停/无数据"
            indicators['二板封板时间'] = row_limit_t['first_time'] if row_limit_t is not None and 'first_time' in row_limit_t else "非涨停/无数据"
            indicators['封板次数/炸板次数'] = int(row_limit_t['open_times']) if row_limit_t is not None and pd.notna(row_limit_t['open_times']) else 0
            
            fd_amount = float(row_limit_t['fd_amount']) if row_limit_t is not None and pd.notna(row_limit_t.get('fd_amount')) else 0
            total_amount = float(row_daily_t['amount']) * 1000 if row_daily_t is not None else 0 
            indicators['二板封单比例'] = round((fd_amount / total_amount) * 100, 2) if total_amount > 0 else 0.0

            circ_mv = float(row_basic_t['circ_mv']) * 10000 if row_basic_t is not None else 0
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
            indicators['流通市值'] = round(circ_mv / 10000 / 10000, 2) if circ_mv > 0 else None
            
            # --- Sector & Rank ---
            industry_name = row_info['industry'] if row_info is not None else "未知"
            concept_str = await self._fetch_stock_concepts(ts_code) 
            ths_concepts = await self._fetch_ths_concepts(ts_code)
            
            matched_hot_list = []
            highest_rank_sector = None
            
            for c in ths_concepts:
                c_code = c['code']
                if c_code in hot_sectors_map:
                    hot_info = hot_sectors_map[c_code]
                    c_name = hot_info['name']
                    rank = hot_info['rank']
                    matched_hot_list.append(f"{c_name}(Top {rank})")
                    
                    if highest_rank_sector is None or rank < highest_rank_sector['rank']:
                        highest_rank_sector = {'code': c_code, 'name': c_name, 'rank': rank}
            
            hot_match_str = ", ".join(matched_hot_list) if matched_hot_list else "无"
            indicators['个股所属板块'] = f"行业: {industry_name} | 概念: {concept_str} | 命中热点: {hot_match_str}"
            indicators['当日热门板块(Top5)'] = "\n".join(top5_sectors_display) if top5_sectors_display else "无数据"
            
            # Rank within Sector
            rank_msg = "无涨停数据"
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
                            else:
                                rank_msg = f"未在{target_sector_name}板块涨停名单中"
            else:
                rank_msg = f"无热门命中 (行业:{industry_name})"

            indicators['板块内连板数排名'] = rank_msg
            
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
        dates = ['20251128']
        names = "通宇通讯"
        
        for date in dates:
            print(f"\nTesting Daban Analysis for {names} on {date}...")
            result = await service.get_daban_indicators(names, date)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(test())
