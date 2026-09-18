from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class BotConfig:
    initial_capital: float = 10_000.0
    risk_pct: float = 0.01
    fee_pct: float = 0.001
    slippage_pct: float = 0.0005
    stop_atr: float = 1.5
    target_atr: float = 3.0

def load_csv(source) -> pd.DataFrame:
    df = pd.read_csv(source, skiprows=1)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for c in ['Open','High','Low','Close','Volume BTC','Volume USDT','tradecount']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Date','Open','High','Low','Close']).sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
    if df.empty: raise ValueError('No valid OHLC rows found.')
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean(); al = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ag / al.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))
    df.loc[(al == 0) & (ag > 0), 'RSI14'] = 100
    df.loc[(ag == 0) & (al > 0), 'RSI14'] = 0
    pc = df['Close'].shift(1)
    tr = pd.concat([(df['High']-df['Low']), (df['High']-pc).abs(), (df['Low']-pc).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df['cross_above_50'] = (df['Close'] > df['SMA50']) & (df['Close'].shift(1) <= df['SMA50'].shift(1))
    df['entry_signal'] = df['cross_above_50'] & (df['SMA50'] > df['SMA200']) & df['RSI14'].between(50,70)
    df['exit_signal'] = df['Close'] < df['SMA50']
    return df

def position_size(equity: float, entry: float, stop: float, risk_pct: float) -> float:
    return min((equity*risk_pct)/max(entry-stop,1e-12), equity/entry)

def backtest(data: pd.DataFrame, cfg: BotConfig, start_idx: int = 0):
    d=data.reset_index(drop=True).copy(); equity=float(cfg.initial_capital); position: Optional[dict]=None; trades=[]; curve=[]
    for i in range(start_idx,len(d)):
        row=d.iloc[i]; mtm=equity if position is None else equity+position['qty']*(row['Close']-position['entry'])
        if position is None and i>start_idx:
            prev=d.iloc[i-1]
            if bool(prev['entry_signal']) and pd.notna(prev['ATR14']):
                entry=float(row['Open'])*(1+cfg.slippage_pct); stop=entry-cfg.stop_atr*float(prev['ATR14']); target=entry+cfg.target_atr*float(prev['ATR14'])
                qty=position_size(equity,entry,stop,cfg.risk_pct)
                if qty>0:
                    entry_fee=qty*entry*cfg.fee_pct; equity-=entry_fee
                    position={'signal_date':prev['Date'],'entry_date':row['Date'],'entry':entry,'qty':qty,'stop':stop,'target':target,'entry_fee':entry_fee,'entry_idx':i}
        if position is not None:
            exit_price=None; reason=None
            if row['Open']<=position['stop']:
                exit_price=float(row['Open'])*(1-cfg.slippage_pct); reason='Stop Loss (Gap)'
            elif row['Open']>=position['target']:
                exit_price=float(row['Open'])*(1-cfg.slippage_pct); reason='Take Profit (Gap)'
            else:
                hs=row['Low']<=position['stop']; ht=row['High']>=position['target']
                if hs and ht: exit_price=position['stop']*(1-cfg.slippage_pct); reason='Stop Loss (Both Hit)'
                elif hs: exit_price=position['stop']*(1-cfg.slippage_pct); reason='Stop Loss'
                elif ht: exit_price=position['target']*(1-cfg.slippage_pct); reason='Take Profit'
                elif bool(row['exit_signal']) and i>position['entry_idx']: exit_price=float(row['Close'])*(1-cfg.slippage_pct); reason='Trend Exit'
            if exit_price is not None:
                gross=(exit_price-position['entry'])*position['qty']; exit_fee=exit_price*position['qty']*cfg.fee_pct; net=gross-position['entry_fee']-exit_fee; equity+=gross-exit_fee
                trades.append({'signal_date':position['signal_date'],'entry_date':position['entry_date'],'exit_date':row['Date'],'entry_price':position['entry'],'exit_price':exit_price,'qty_btc':position['qty'],'stop_loss':position['stop'],'take_profit':position['target'],'gross_pnl':gross,'entry_fee':position['entry_fee'],'exit_fee':exit_fee,'net_pnl':net,'exit_reason':reason}); position=None
        curve.append({'Date':row['Date'],'Equity':mtm if position else equity})
    if position is not None:
        row=d.iloc[-1]; exit_price=float(row['Close'])*(1-cfg.slippage_pct); gross=(exit_price-position['entry'])*position['qty']; exit_fee=exit_price*position['qty']*cfg.fee_pct; equity+=gross-exit_fee
        trades.append({'signal_date':position['signal_date'],'entry_date':position['entry_date'],'exit_date':row['Date'],'entry_price':position['entry'],'exit_price':exit_price,'qty_btc':position['qty'],'stop_loss':position['stop'],'take_profit':position['target'],'gross_pnl':gross,'entry_fee':position['entry_fee'],'exit_fee':exit_fee,'net_pnl':gross-position['entry_fee']-exit_fee,'exit_reason':'End of Test'})
    eq=pd.DataFrame(curve); trades_df=pd.DataFrame(trades)
    if not eq.empty: eq['Peak']=eq['Equity'].cummax(); eq['Drawdown']=eq['Equity']/eq['Peak']-1
    if not trades_df.empty:
        gp=trades_df.loc[trades_df['net_pnl']>0,'net_pnl'].sum(); gl=-trades_df.loc[trades_df['net_pnl']<0,'net_pnl'].sum(); pf=gp/gl if gl>0 else np.inf; wr=float((trades_df['net_pnl']>0).mean())
    else: pf=np.nan; wr=0.0
    ret=eq['Equity'].pct_change().dropna(); sharpe=np.sqrt(365)*ret.mean()/ret.std() if len(ret) and ret.std()>0 else np.nan; mdd=-eq['Drawdown'].min() if not eq.empty else np.nan
    metrics={'Initial Capital':cfg.initial_capital,'Final Capital':float(equity),'Net Profit':float(equity-cfg.initial_capital),'Total Return':float(equity/cfg.initial_capital-1),'Total Trades':len(trades_df),'Winning Trades':int((trades_df['net_pnl']>0).sum()) if not trades_df.empty else 0,'Losing Trades':int((trades_df['net_pnl']<0).sum()) if not trades_df.empty else 0,'Win Rate':wr,'Profit Factor':pf,'Maximum Drawdown':mdd,'Sharpe Ratio':sharpe}
    return metrics,trades_df,eq

def buy_hold(data: pd.DataFrame, initial_capital=10_000.0, fee_pct=.001, slippage_pct=.0005):
    d=data.copy(); entry=float(d.iloc[0]['Open'])*(1+slippage_pct); invest=initial_capital*(1-fee_pct); qty=invest/entry; exit_price=float(d.iloc[-1]['Close'])*(1-slippage_pct); exit_fee=qty*exit_price*fee_pct; final=invest+qty*(exit_price-entry)-exit_fee
    wealth=invest+qty*(d['Close']-entry); dd=-(wealth/wealth.cummax()-1).min(); rr=wealth.pct_change().dropna(); sharpe=np.sqrt(365)*rr.mean()/rr.std() if rr.std()>0 else np.nan
    return {'Initial Capital':initial_capital,'Final Capital':float(final),'Net Profit':float(final-initial_capital),'Total Return':float(final/initial_capital-1),'Maximum Drawdown':float(dd),'Sharpe Ratio':float(sharpe)},wealth
