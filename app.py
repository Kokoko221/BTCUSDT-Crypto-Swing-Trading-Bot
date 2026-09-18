from __future__ import annotations

from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.engine import BotConfig, add_indicators, backtest, buy_hold, load_csv, position_size
from src.decision_rationale import decision_explanation, risk_plan

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / "data" / "Binance_BTCUSDT_d.csv"
BINANCE_URL = "https://api.binance.com/api/v3/klines"

st.set_page_config(page_title="BTC/USDT Swing Trading Bot", page_icon="₿", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1rem;padding-bottom:2rem}
.hero{padding:18px 22px;border:1px solid #263244;border-radius:18px;background:linear-gradient(135deg,#0f172a,#111827 50%,#0b1220);margin-bottom:16px}
.hero h1{margin:0;font-size:2.15rem}.hero p{margin:7px 0 0;color:#94a3b8}
.card{padding:15px 16px;border:1px solid #263244;border-radius:16px;background:#111827;height:100%}
.small{color:#94a3b8;font-size:.82rem}.value{font-size:1.55rem;font-weight:750;margin-top:4px}
.badge{display:inline-block;padding:6px 11px;border-radius:999px;font-weight:750;font-size:.82rem}
.badge-buy{background:#0b3d2e;color:#7ef2bd;border:1px solid #1c6b50}
.badge-wait{background:#3a2c08;color:#f8d775;border:1px solid #67501a}
.badge-exit{background:#4b1820;color:#ff9fa9;border:1px solid #7b2836}
.badge-live{background:#123b22;color:#8ce99a;border:1px solid #285d38}
.badge-paper{background:#172554;color:#93c5fd;border:1px solid #24418a}
.reason{border:1px solid #334155;border-radius:16px;padding:18px;background:#0b1220}
.reason h4{margin:0 0 8px}.muted{color:#94a3b8}
.profit{color:#7ef2bd!important}.loss{color:#ff9fa9!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class='hero'><h1>₿ BTC/USDT Crypto Swing Trading Bot</h1><p>Systematic swing strategy • live market monitoring • paper execution • backtesting • risk controls • decision rationale</p></div>""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "paper_cash": 10000.0,
        "paper_position": None,
        "paper_pending": None,
        "paper_trades": [],
        "paper_last": None,
        "paper_start": 10000.0,
        "paper_peak": 10000.0,
        "paper_snapshot": None,
        "paper_evaluated": False,
        "replay": {
            "active": False,
            "scenario_id": None,
            "current_idx": None,
            "position": None,
            "cash": 10000.0,
            "start_cash": 10000.0,
            "history": [],
            "closed": False,
        },
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()

with st.sidebar:
    st.header("Bot Parameters")
    capital = st.number_input("Starting capital (USDT)", 1000.0, 1_000_000.0, 10_000.0, 500.0)
    risk = st.slider("Risk per trade (%)", 0.25, 2.0, 1.0, 0.25) / 100
    fee = st.number_input("Fee per side (%)", 0.0, 1.0, 0.10, 0.01) / 100
    slip = st.number_input("Slippage per side (%)", 0.0, 1.0, 0.05, 0.01) / 100
    stop_atr = st.number_input("Stop ATR", 0.5, 5.0, 1.5, 0.25)
    target_atr = st.number_input("Target ATR", 0.5, 8.0, 3.0, 0.25)
    oos_days = st.number_input("OOS window (days)", 90, 730, 365, 30)
    st.divider()
    if st.button("Reset Paper Account", use_container_width=True):
        st.session_state.paper_cash = float(capital)
        st.session_state.paper_position = None
        st.session_state.paper_pending = None
        st.session_state.paper_trades = []
        st.session_state.paper_last = None
        st.session_state.paper_start = float(capital)
        st.session_state.paper_peak = float(capital)
        st.session_state.paper_snapshot = None
        st.session_state.paper_evaluated = False
        st.session_state.replay = {
            "active": False, "scenario_id": None, "current_idx": None,
            "position": None, "cash": float(capital), "start_cash": float(capital),
            "history": [], "closed": False,
        }
        st.success("Paper account reset.")
    st.caption("Paper mode never sends real orders or requires a trading API key.")

cfg = BotConfig(capital, risk, fee, slip, stop_atr, target_atr)


def fetch_live():
    response = requests.get(
        BINANCE_URL,
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": 400},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json()
    cols = ["open_time","Open","High","Low","Close","Volume","close_time","quote_volume","trades","tb_base","tb_quote","ignore"]
    d = pd.DataFrame(rows, columns=cols)
    d["Date"] = pd.to_datetime(d["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    d = d[d["Date"] < today].copy()
    for c in ["Open","High","Low","Close","Volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["Date","Open","High","Low","Close","Volume"]].dropna().reset_index(drop=True)


def pct(x):
    return f"{x * 100:.2f}%"


def perf_details(tr):
    if tr is None or tr.empty:
        return {"avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0, "profit_factor": np.nan, "tp": 0, "sl": 0, "trend": 0}
    wins = tr.loc[tr["net_pnl"] > 0, "net_pnl"]
    losses = tr.loc[tr["net_pnl"] < 0, "net_pnl"]
    gp = wins.sum()
    gl = -losses.sum()
    reasons = tr["exit_reason"].astype(str)
    return {
        "avg_win": float(wins.mean()) if len(wins) else 0,
        "avg_loss": float(losses.mean()) if len(losses) else 0,
        "best": float(tr["net_pnl"].max()),
        "worst": float(tr["net_pnl"].min()),
        "profit_factor": float(gp / gl) if gl > 0 else np.inf,
        "tp": int(reasons.str.contains("Take Profit").sum()),
        "sl": int(reasons.str.contains("Stop Loss").sum()),
        "trend": int(reasons.str.contains("Trend Exit").sum()),
    }


def monthly_returns(eq):
    q = eq.set_index("Date")["Equity"].resample("ME").last().pct_change().dropna()
    return pd.DataFrame({"Month": q.index.strftime("%Y-%m"), "Return": q.values})


def render_decision_card(explanation: dict):
    classes = {"BUY": "badge-buy", "WAIT": "badge-wait", "EXIT": "badge-exit"}
    cls = classes[explanation["signal"]]
    st.markdown(
        f"<div class='reason'><span class='badge {cls}'>{explanation['signal']}</span> "
        f"<span style='margin-left:10px'>{explanation['headline']}</span></div>",
        unsafe_allow_html=True,
    )


def render_paper_metrics(position, row):
    start = float(st.session_state.paper_start)
    cash = float(st.session_state.paper_cash)
    if position is None:
        position_value = 0.0
        unrealized = 0.0
        equity = cash
    else:
        position_value = float(position["qty"] * row["Close"])
        unrealized = float((row["Close"] - position["entry"]) * position["qty"] - position["entry_fee"])
        equity = cash + position_value
    realized = float(sum(t["net_pnl"] for t in st.session_state.paper_trades))
    total_pnl = equity - start
    return {
        "cash": cash,
        "position_value": position_value,
        "unrealized": unrealized,
        "realized": realized,
        "total_pnl": total_pnl,
        "equity": equity,
        "return_pct": total_pnl / start if start else 0.0,
    }


def load_historical_replay_data():
    data = load_csv(LOCAL)
    x = add_indicators(data)
    # We reuse the same backtest engine to identify real historical trades.
    elig = x.SMA200.notna() & x.RSI14.notna() & x.ATR14.notna()
    start = int(x.index[elig][0])
    _, trades, _ = backtest(x, cfg, start)
    return x, trades


def replay_reset(start_cash: float):
    st.session_state.replay = {
        "active": False,
        "scenario_id": None,
        "current_idx": None,
        "position": None,
        "cash": float(start_cash),
        "start_cash": float(start_cash),
        "history": [],
        "closed": False,
    }


def replay_open_position(x, idx, cash):
    row = x.iloc[idx]
    prev = x.iloc[idx - 1]
    entry = float(row.Open) * (1 + slip)
    stop = entry - stop_atr * float(prev.ATR14)
    target = entry + target_atr * float(prev.ATR14)
    qty = position_size(cash, entry, stop, risk)
    if qty <= 0:
        return None, cash
    entry_fee = qty * entry * fee
    cash_after = cash - entry_fee
    return {
        "signal_date": prev.Date,
        "entry_date": row.Date,
        "entry": entry,
        "qty": qty,
        "stop": stop,
        "target": target,
        "entry_fee": entry_fee,
    }, cash_after


def replay_manage_position(x, idx, position, cash, allow_trend_exit=True):
    row = x.iloc[idx]
    exit_price = None
    reason = None
    if row.Open <= position["stop"]:
        exit_price = float(row.Open) * (1 - slip); reason = "Stop Loss (Gap)"
    elif row.Open >= position["target"]:
        exit_price = float(row.Open) * (1 - slip); reason = "Take Profit (Gap)"
    else:
        hit_stop = row.Low <= position["stop"]
        hit_target = row.High >= position["target"]
        if hit_stop and hit_target:
            exit_price = position["stop"] * (1 - slip); reason = "Stop Loss (Both Hit)"
        elif hit_stop:
            exit_price = position["stop"] * (1 - slip); reason = "Stop Loss"
        elif hit_target:
            exit_price = position["target"] * (1 - slip); reason = "Take Profit"
        elif allow_trend_exit and bool(row.exit_signal):
            exit_price = float(row.Close) * (1 - slip); reason = "Trend Exit"
    if exit_price is None:
        return position, cash, None
    gross = (exit_price - position["entry"]) * position["qty"]
    exit_fee = exit_price * position["qty"] * fee
    net = gross - position["entry_fee"] - exit_fee
    cash += gross - exit_fee
    trade = {
        "signal_date": position["signal_date"],
        "entry_date": position["entry_date"],
        "exit_date": row.Date,
        "entry_price": position["entry"],
        "exit_price": exit_price,
        "qty_btc": position["qty"],
        "stop_loss": position["stop"],
        "take_profit": position["target"],
        "net_pnl": net,
        "exit_reason": reason,
    }
    return None, cash, trade


def manual_close_position(x, idx, position, cash):
    row = x.iloc[idx]
    exit_price = float(row.Close) * (1 - slip)
    gross = (exit_price - position["entry"]) * position["qty"]
    exit_fee = exit_price * position["qty"] * fee
    net = gross - position["entry_fee"] - exit_fee
    cash += gross - exit_fee
    trade = {
        "signal_date": position["signal_date"],
        "entry_date": position["entry_date"],
        "exit_date": row.Date,
        "entry_price": position["entry"],
        "exit_price": exit_price,
        "qty_btc": position["qty"],
        "stop_loss": position["stop"],
        "take_profit": position["target"],
        "net_pnl": net,
        "exit_reason": "Manual Close",
    }
    return None, cash, trade


t1, t2, t3, t4, t5 = st.tabs(["Dashboard", "Backtest", "Paper Trading", "Decision Rationale", "Strategy"])

# ---------------- Dashboard ----------------
with t1:
    try:
        data = load_csv(LOCAL)
        x = add_indicators(data)
        ex = decision_explanation(x, in_position=st.session_state.paper_position is not None)
        row = x.iloc[-1]
        render_decision_card(ex)

        cards = st.columns(5)
        labels = ["BTC Close", "SMA50", "SMA200", "RSI14", "ATR14"]
        vals = [f"${row['Close']:,.2f}", f"${row['SMA50']:,.2f}", f"${row['SMA200']:,.2f}", f"{row['RSI14']:.2f}", f"${row['ATR14']:,.2f}"]
        subs = [f"As of {row['Date'].date()}", "Medium-term trend", "Long-term trend", "Momentum", "Volatility"]
        for c, label, val, sub in zip(cards, labels, vals, subs):
            with c:
                st.markdown(f"<div class='card'><div class='small'>{label}</div><div class='value'>{val}</div><div class='small'>{sub}</div></div>", unsafe_allow_html=True)

        st.subheader("Market Read")
        a, b, c = st.columns([1.1, 1.1, 2.3])
        with a:
            st.markdown(f"<div class='card'><div class='small'>Market regime</div><div class='value'>{ex['regime']}</div><div class='small'>Setup alignment {ex['alignment_score']}/3</div></div>", unsafe_allow_html=True)
        with b:
            st.markdown(f"<div class='card'><div class='small'>Current decision</div><div class='value'>{ex['signal']}</div><div class='small'>Systematic action</div></div>", unsafe_allow_html=True)
        with c:
            st.markdown(f"<div class='reason'><h4>Why the bot made this decision</h4><div>{ex['headline']}</div><div class='muted' style='margin-top:8px'>{ex['next_action']}</div></div>", unsafe_allow_html=True)

        st.subheader("Price & Trend")
        v = x.tail(500)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=v.Date, open=v.Open, high=v.High, low=v.Low, close=v.Close, name="BTC/USDT"))
        fig.add_trace(go.Scatter(x=v.Date, y=v.SMA50, name="SMA50"))
        fig.add_trace(go.Scatter(x=v.Date, y=v.SMA200, name="SMA200"))
        s = v[v.entry_signal]
        fig.add_trace(go.Scatter(x=s.Date, y=s.Close, mode="markers", name="BUY", marker_symbol="triangle-up", marker_size=10))
        fig.update_layout(height=560, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk Plan")
        if ex["signal"] == "BUY":
            rp = risk_plan(capital, float(row.Close), float(row.ATR14), risk, stop_atr, target_atr)
            z = st.columns(5)
            z[0].metric("Risk Budget", f"${rp['risk_amount']:,.2f}")
            z[1].metric("Entry Reference", f"${row.Close:,.2f}")
            z[2].metric("Stop", f"${rp['stop']:,.2f}")
            z[3].metric("Target", f"${rp['target']:,.2f}")
            z[4].metric("Risk / Reward", f"1 : {rp['risk_reward']:.2f}")
            st.caption("The paper engine enters on the next daily open after a valid signal.")
        else:
            st.info("No new long entry is active. The bot is waiting for the full entry checklist to align.")

        elig = x.SMA200.notna() & x.RSI14.notna() & x.ATR14.notna()
        start = int(x.index[elig][0])
        fm, tr, eq = backtest(x, cfg, start)
        det = perf_details(tr)
        st.subheader("Performance Snapshot")
        k = st.columns(5)
        k[0].metric("Bot P&L", f"${fm['Net Profit']:,.2f}")
        k[1].metric("Return", pct(fm["Total Return"]))
        k[2].metric("Max DD", pct(fm["Maximum Drawdown"]))
        k[3].metric("Win Rate", pct(fm["Win Rate"]))
        k[4].metric("Profit Factor", f"{det['profit_factor']:.2f}" if np.isfinite(det["profit_factor"]) else "∞")
        q1, q2 = st.columns(2)
        eqfig = go.Figure(go.Scatter(x=eq.Date, y=eq.Equity, name="Equity"))
        eqfig.update_layout(title="Equity Curve", height=330)
        q1.plotly_chart(eqfig, use_container_width=True)
        ddfig = go.Figure(go.Scatter(x=eq.Date, y=eq.Drawdown * 100, name="Drawdown"))
        ddfig.update_layout(title="Drawdown (%)", height=330)
        q2.plotly_chart(ddfig, use_container_width=True)
    except Exception as e:
        st.error(f"Dashboard error: {e}")

# ---------------- Backtest ----------------
with t2:
    st.subheader("Backtest Lab")
    up = st.file_uploader("Upload BTCUSDT daily CSV (optional)", type="csv")
    try:
        data = load_csv(up if up is not None else LOCAL)
        x = add_indicators(data)
        elig = x.SMA200.notna() & x.RSI14.notna() & x.ATR14.notna()
        start = int(x.index[elig][0])
        oos_ix = x.index[x.Date >= x.Date.max() - pd.Timedelta(days=int(oos_days))]
        oos_start = max(start, int(oos_ix[0]))
        fm, tr, eq = backtest(x, cfg, start)
        om, otr, oeq = backtest(x, cfg, oos_start)
        bh, _ = buy_hold(x.iloc[start:], capital, fee, slip)
        obh, _ = buy_hold(x.iloc[oos_start:], capital, fee, slip)
        det = perf_details(tr)
        st.caption(f"Dataset: {data.Date.min().date()} → {data.Date.max().date()} | {len(data):,} rows")
        k = st.columns(5)
        k[0].metric("Final Capital", f"${fm['Final Capital']:,.2f}")
        k[1].metric("Net P&L", f"${fm['Net Profit']:,.2f}")
        k[2].metric("Return", pct(fm["Total Return"]))
        k[3].metric("Max DD", pct(fm["Maximum Drawdown"]))
        k[4].metric("Trades", str(fm["Total Trades"]))

        st.subheader("Bot vs Buy & Hold — Full History")
        comp = pd.DataFrame({
            "Metric": ["Final Capital","Net Profit","Return","Max Drawdown","Sharpe","Trades"],
            "Bot": [f"${fm['Final Capital']:,.2f}",f"${fm['Net Profit']:,.2f}",pct(fm['Total Return']),pct(fm['Maximum Drawdown']),f"{fm['Sharpe Ratio']:.2f}",str(fm['Total Trades'])],
            "Buy & Hold": [f"${bh['Final Capital']:,.2f}",f"${bh['Net Profit']:,.2f}",pct(bh['Total Return']),pct(bh['Maximum Drawdown']),f"{bh['Sharpe Ratio']:.2f}","1"],
        })
        st.dataframe(comp, hide_index=True, use_container_width=True)

        st.subheader(f"Out-of-Sample — Last {int(oos_days)} Days")
        oc = pd.DataFrame({
            "Metric": ["Final Capital","Net Profit","Return","Max Drawdown","Sharpe","Trades"],
            "Bot": [f"${om['Final Capital']:,.2f}",f"${om['Net Profit']:,.2f}",pct(om['Total Return']),pct(om['Maximum Drawdown']),f"{om['Sharpe Ratio']:.2f}",str(om['Total Trades'])],
            "Buy & Hold": [f"${obh['Final Capital']:,.2f}",f"${obh['Net Profit']:,.2f}",pct(obh['Total Return']),pct(obh['Maximum Drawdown']),f"{obh['Sharpe Ratio']:.2f}","1"],
        })
        st.dataframe(oc, hide_index=True, use_container_width=True)

        st.subheader("Trade Analytics")
        a,b,c,d,e = st.columns(5)
        a.metric("Avg Winner", f"${det['avg_win']:,.2f}")
        b.metric("Avg Loser", f"${det['avg_loss']:,.2f}")
        c.metric("Best Trade", f"${det['best']:,.2f}")
        d.metric("Worst Trade", f"${det['worst']:,.2f}")
        e.metric("Profit Factor", f"{det['profit_factor']:.2f}" if np.isfinite(det['profit_factor']) else "∞")
        rr = pd.DataFrame({"Exit Reason":["Take Profit","Stop Loss","Trend Exit"],"Trades":[det['tp'],det['sl'],det['trend']]})
        st.plotly_chart(px.bar(rr, x="Exit Reason", y="Trades", title="Trade Exit Reasons"), use_container_width=True)
        mr = monthly_returns(eq)
        if not mr.empty:
            st.plotly_chart(px.bar(mr, x="Month", y="Return", title="Monthly Return Profile"), use_container_width=True)
        st.subheader("Trade Log")
        st.dataframe(tr, hide_index=True, use_container_width=True)
        st.download_button("Download full trade log CSV", tr.to_csv(index=False), "full_trade_log.csv", "text/csv")
    except Exception as e:
        st.error(f"Backtest error: {e}")

# ---------------- Paper Trading ----------------
with t3:
    st.subheader("Paper Trading Terminal")
    st.info("Paper mode only. No real orders are sent.")

    st.markdown("### Live Paper Market")
    st.caption("The market is evaluated only when you press the button. This prevents repeated processing of the same daily candle.")

    if not st.session_state.paper_evaluated:
        st.markdown(
            "<div class='card'><div class='small'>Paper market feed</div><div class='value'>Ready</div>"
            "<div class='small'>Press <b>Evaluate Latest Closed Candle</b> to fetch and process the newest completed daily candle.</div></div>",
            unsafe_allow_html=True,
        )

    evaluate = st.button("Evaluate Latest Closed Candle", type="primary")
    if evaluate:
        try:
            with st.status("Evaluating the latest closed candle…", expanded=True) as status:
                st.write("Connecting to the BTC/USDT market feed…")
                time.sleep(0.20)
                live = fetch_live()
                st.write("Calculating trend, momentum and volatility…")
                time.sleep(0.20)
                x = add_indicators(live)
                row = x.iloc[-1]
                ex = decision_explanation(x, in_position=st.session_state.paper_position is not None)
                latest = row.Date
                st.write("Checking the entry and exit rules…")
                time.sleep(0.20)

                if st.session_state.paper_last is not None and latest == st.session_state.paper_last:
                    st.write("This daily candle has already been evaluated; the paper account was not processed a second time.")
                else:
                    pending = st.session_state.paper_pending
                    pos = st.session_state.paper_position

                    if pending is not None and pos is None and latest > pending["signal_date"]:
                        atr = float(x.iloc[-2].ATR14)
                        entry = float(row.Open) * (1 + slip)
                        stop = entry - stop_atr * atr
                        target = entry + target_atr * atr
                        qty = position_size(st.session_state.paper_cash, entry, stop, risk)
                        if qty > 0:
                            entry_fee = qty * entry * fee
                            st.session_state.paper_cash -= entry_fee
                            st.session_state.paper_position = {
                                "signal_date": pending["signal_date"],
                                "entry_date": latest,
                                "entry": entry,
                                "qty": qty,
                                "stop": stop,
                                "target": target,
                                "entry_fee": entry_fee,
                            }
                        st.session_state.paper_pending = None
                        pos = st.session_state.paper_position

                    if pos is not None:
                        new_pos, new_cash, trade = replay_manage_position(x, len(x) - 1, pos, st.session_state.paper_cash, allow_trend_exit=(pos["entry_date"] < latest))
                        if trade is not None:
                            st.session_state.paper_cash = new_cash
                            st.session_state.paper_trades.append(trade)
                            st.session_state.paper_position = None
                        else:
                            st.session_state.paper_position = new_pos

                    if st.session_state.paper_position is None and ex["signal"] == "BUY":
                        st.session_state.paper_pending = {"signal_date": latest}

                    st.session_state.paper_last = latest

                st.session_state.paper_snapshot = {"data": x, "latest": latest, "explanation": ex}
                st.session_state.paper_evaluated = True
                status.update(label="Evaluation complete", state="complete")
        except requests.RequestException as e:
            st.error(f"Could not reach the public market data feed: {e}")
        except Exception as e:
            st.error(f"Paper trading error: {e}")

    snap = st.session_state.paper_snapshot
    if snap is not None:
        x = snap["data"]
        row = x.iloc[-1]
        ex = snap["explanation"]
        latest = snap["latest"]
        metrics = render_paper_metrics(st.session_state.paper_position, row)
        render_decision_card(ex)

        k = st.columns(7)
        k[0].metric("BTC Close", f"${row.Close:,.2f}")
        k[1].metric("Paper Equity", f"${metrics['equity']:,.2f}")
        k[2].metric("Total P&L", f"${metrics['total_pnl']:,.2f}", delta=pct(metrics["return_pct"]))
        k[3].metric("Realized P&L", f"${metrics['realized']:,.2f}")
        k[4].metric("Unrealized P&L", f"${metrics['unrealized']:,.2f}")
        k[5].metric("Cash", f"${metrics['cash']:,.2f}")
        k[6].metric("Position", f"{st.session_state.paper_position['qty']:.6f} BTC" if st.session_state.paper_position else "None")

        if metrics["total_pnl"] == 0 and not st.session_state.paper_trades and st.session_state.paper_position is None:
            st.caption("P&L is $0.00 because no paper position is open and no trade has been closed. BTC can move without changing account P&L until the system actually enters a position.")

        if st.session_state.paper_position:
            pos = st.session_state.paper_position
            a,b,c,d,e,f = st.columns(6)
            a.metric("Entry", f"${pos['entry']:,.2f}")
            b.metric("Stop", f"${pos['stop']:,.2f}")
            c.metric("Target", f"${pos['target']:,.2f}")
            d.metric("Position Value", f"${metrics['position_value']:,.2f}")
            e.metric("Open P&L", f"${metrics['unrealized']:,.2f}")
            f.button("Close Position", key="live_manual_close", help="Close the open paper position at the latest close.")
            if st.session_state.get("live_manual_close"):
                pos2, cash2, trade = manual_close_position(x, len(x)-1, pos, st.session_state.paper_cash)
                st.session_state.paper_cash = cash2
                st.session_state.paper_trades.append(trade)
                st.session_state.paper_position = None
                st.rerun()
        elif st.session_state.paper_pending:
            st.info(f"Pending BUY from {st.session_state.paper_pending['signal_date'].date()}. It will be simulated at the next daily open when a new completed candle is evaluated.")
        else:
            st.caption("No open or pending trade.")

        st.subheader("Why the Bot Chose This")
        st.markdown(
            f"<div class='reason'><h4>Market view</h4><div>{ex['headline']}</div>"
            f"<div class='muted' style='margin-top:8px'><b>Next action:</b> {ex['next_action']}</div>"
            f"<div class='muted' style='margin-top:8px'>{ex['invalidation']}</div></div>",
            unsafe_allow_html=True,
        )
        st.subheader("Decision Factors")
        st.dataframe(pd.DataFrame(ex["factors"]), hide_index=True, use_container_width=True)

        st.subheader("Live Market View")
        v = x.tail(220)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=v.Date, open=v.Open, high=v.High, low=v.Low, close=v.Close, name="BTC/USDT"))
        fig.add_trace(go.Scatter(x=v.Date, y=v.SMA50, name="SMA50"))
        fig.add_trace(go.Scatter(x=v.Date, y=v.SMA200, name="SMA200"))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.paper_trades:
            pdf = pd.DataFrame(st.session_state.paper_trades)
            st.subheader("Closed Paper Trades")
            st.dataframe(pdf, hide_index=True, use_container_width=True)
            st.download_button("Download paper trade history", pdf.to_csv(index=False), "paper_trade_history.csv", "text/csv")
        else:
            st.info("No paper trade has been closed yet. Completed trades will appear here automatically.")

        st.caption(f"Latest closed candle evaluated: {latest.date()} • Last check: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # ---------------- Historical Market Replay ----------------
    st.divider()
    st.subheader("Historical Market Replay")
    st.caption("Use a real historical setup to watch the same strategy open, manage and close a paper position candle by candle. This is market replay, not a forced signal.")

    try:
        replay_x, replay_trades = load_historical_replay_data()
        if replay_trades.empty:
            st.warning("No completed historical trades are available for replay.")
        else:
            options = []
            for i, t in replay_trades.reset_index(drop=True).iterrows():
                pnl = float(t["net_pnl"])
                sign = "+" if pnl >= 0 else "-"
                label = f"#{i+1}  {pd.Timestamp(t['signal_date']).date()} → {pd.Timestamp(t['exit_date']).date()}  |  {sign}${abs(pnl):,.2f}  |  {t['exit_reason']}"
                options.append((label, i))

            labels = [o[0] for o in options]
            selected_label = st.selectbox("Choose a historical trade", labels, key="replay_trade_select")
            selected_idx = dict(options)[selected_label]
            selected_trade = replay_trades.reset_index(drop=True).iloc[selected_idx]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Signal", pd.Timestamp(selected_trade.signal_date).strftime("%d-%b-%Y"))
            r2.metric("Entry", f"${selected_trade.entry_price:,.2f}")
            r3.metric("Exit", f"${selected_trade.exit_price:,.2f}")
            r4.metric("Historical P&L", f"${selected_trade.net_pnl:,.2f}")

            if st.button("Load Historical Trade", key="load_replay", type="secondary"):
                sig_date = pd.Timestamp(selected_trade.signal_date)
                idx = int(replay_x.index[replay_x.Date == sig_date][0])
                st.session_state.replay = {
                    "active": True,
                    "scenario_id": int(selected_idx),
                    "current_idx": idx,
                    "position": None,
                    "cash": float(capital),
                    "start_cash": float(capital),
                    "history": [],
                    "closed": False,
                }
                st.rerun()

            replay = st.session_state.replay
            if replay["active"] and replay["scenario_id"] == int(selected_idx):
                c1,c2,c3,c4 = st.columns(4)
                if c1.button("▶ Run to Exit", key="run_replay"):
                    exit_date = pd.Timestamp(selected_trade.exit_date)
                    current = replay["current_idx"]
                    with st.status("Running the historical market replay…", expanded=True) as replay_status:
                        progress = st.progress(0.0)
                        total_steps = max(1, int((exit_date - replay_x.iloc[current]["Date"]).days))
                        steps = 0
                        while current < len(replay_x)-1 and replay_x.iloc[current]["Date"] < exit_date:
                            current += 1
                            steps += 1
                            st.write(f"Processing {replay_x.iloc[current]['Date'].date()}…")
                            time.sleep(0.08)
                            if replay["position"] is None and replay_x.iloc[current]["Date"] == pd.Timestamp(selected_trade.entry_date):
                                pos, cash = replay_open_position(replay_x, current, replay["cash"])
                                replay["position"] = pos
                                replay["cash"] = cash
                            elif replay["position"] is not None:
                                pos, cash, trade = replay_manage_position(replay_x, current, replay["position"], replay["cash"])
                                if trade is not None:
                                    replay["history"].append(trade)
                                    replay["position"] = None
                                    replay["cash"] = cash
                            replay["current_idx"] = current
                            progress.progress(min(1.0, steps / total_steps))
                            if replay["history"]:
                                break
                        replay_status.update(label="Replay complete", state="complete")
                    st.session_state.replay = replay
                    st.rerun()

                if c2.button("→ Next Candle", key="next_replay"):
                    current = replay["current_idx"]
                    if current < len(replay_x)-1:
                        current += 1
                        if replay["position"] is None and replay_x.iloc[current]["Date"] == pd.Timestamp(selected_trade.entry_date):
                            pos, cash = replay_open_position(replay_x, current, replay["cash"])
                            replay["position"] = pos
                            replay["cash"] = cash
                            replay["history"].append({"event": "ENTER", "date": replay_x.iloc[current]["Date"], "price": pos["entry"] if pos else None})
                        elif replay["position"] is not None:
                            pos, cash, trade = replay_manage_position(replay_x, current, replay["position"], replay["cash"])
                            if trade is not None:
                                replay["history"].append(trade)
                                replay["position"] = None
                            replay["cash"] = cash
                        replay["current_idx"] = current
                    st.session_state.replay = replay
                    st.rerun()

                if c3.button("Close Simulation", key="close_replay", disabled=(replay["position"] is None)):
                    pos, cash, trade = manual_close_position(replay_x, replay["current_idx"], replay["position"], replay["cash"])
                    replay["position"] = pos
                    replay["cash"] = cash
                    replay["history"].append(trade)
                    replay["closed"] = True
                    st.session_state.replay = replay
                    st.rerun()

                if c4.button("↻ Reset Replay", key="reset_replay"):
                    replay_reset(float(capital))
                    st.rerun()

                cur_row = replay_x.iloc[replay["current_idx"]]
                if replay["position"] is None and not replay["history"]:
                    st.info(f"Replay loaded at signal date {cur_row.Date.date()}. Press Next Candle to simulate the next market session.")
                if replay["position"] is not None:
                    p = replay["position"]
                    open_pnl = (float(cur_row.Close) - p["entry"]) * p["qty"] - p["entry_fee"]
                    s1,s2,s3,s4,s5 = st.columns(5)
                    s1.metric("Replay Date", str(cur_row.Date.date()))
                    s2.metric("Position", f"{p['qty']:.6f} BTC")
                    s3.metric("Open P&L", f"${open_pnl:,.2f}")
                    s4.metric("Stop", f"${p['stop']:,.2f}")
                    s5.metric("Target", f"${p['target']:,.2f}")
                else:
                    s1,s2,s3,s4,s5 = st.columns(5)
                    s1.metric("Replay Date", str(cur_row.Date.date()))
                    s2.metric("Cash", f"${replay['cash']:,.2f}")
                    s3.metric("Position", "None")
                    s4.metric("Closed Trades", str(sum(1 for h in replay["history"] if "exit_date" in h)))
                    s5.metric("Replay P&L", f"${replay['cash']-replay['start_cash']:,.2f}")

                rv = replay_x.loc[max(0, replay["current_idx"]-180):replay["current_idx"]].copy()
                rf = go.Figure()
                rf.add_trace(go.Candlestick(x=rv.Date, open=rv.Open, high=rv.High, low=rv.Low, close=rv.Close, name="BTC/USDT"))
                rf.add_trace(go.Scatter(x=rv.Date, y=rv.SMA50, name="SMA50"))
                rf.add_trace(go.Scatter(x=rv.Date, y=rv.SMA200, name="SMA200"))
                rf.add_vline(x=cur_row.Date, line_width=2, line_dash="dash")
                rf.update_layout(height=470, xaxis_rangeslider_visible=False, title="Historical Market Replay")
                st.plotly_chart(rf, use_container_width=True)

                if replay["history"]:
                    st.subheader("Replay Events")
                    st.dataframe(pd.DataFrame(replay["history"]), hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Replay error: {e}")

# ---------------- Decision Rationale ----------------
with t4:
    st.subheader("Decision Rationale")
    st.caption("Every BUY, WAIT or EXIT state is explained from the same indicators and rules that drive the trading engine.")
    try:
        data = load_csv(LOCAL)
        x = add_indicators(data)
        ex = decision_explanation(x, in_position=st.session_state.paper_position is not None)
        row = x.iloc[-1]
        a,b,c = st.columns(3)
        a.metric("Decision", ex["signal"])
        b.metric("Setup Alignment", f"{ex['alignment_score']}/3")
        c.metric("Market Regime", ex["regime"])
        st.markdown(f"<div class='reason'><h4>Market view</h4><div>{ex['headline']}</div><div class='muted' style='margin-top:8px'><b>Next action:</b> {ex['next_action']}</div><div class='muted' style='margin-top:8px'>{ex['invalidation']}</div></div>", unsafe_allow_html=True)
        st.subheader("Decision Factors")
        st.dataframe(pd.DataFrame(ex["factors"]), hide_index=True, use_container_width=True)
        snap = pd.DataFrame({"Indicator":["Close","SMA50","SMA200","RSI14","ATR14"],"Value":[row.Close,row.SMA50,row.SMA200,row.RSI14,row.ATR14]})
        st.subheader("Indicator Snapshot")
        st.dataframe(snap, hide_index=True, use_container_width=True)
        if ex["signal"] == "BUY":
            rp = risk_plan(capital, float(row.Close), float(row.ATR14), risk, stop_atr, target_atr)
            st.success(f"The complete entry setup is active. Risk budget ${rp['risk_amount']:,.2f}; reference stop ${rp['stop']:,.2f}; target ${rp['target']:,.2f}; planned risk/reward 1:{rp['risk_reward']:.2f}. Execution is scheduled for the next daily open in paper mode.")
        elif ex["signal"] == "EXIT":
            st.error("The short-term trend has failed the exit rule. An open long position should be closed according to the execution rules.")
        else:
            st.warning("The bot is waiting. It will not commit capital until the missing entry condition(s) align.")
    except Exception as e:
        st.error(f"Decision rationale error: {e}")

# ---------------- Strategy ----------------
with t5:
    st.subheader("Strategy & System Design")
    st.markdown("""
### Entry
1. Daily close crosses above SMA50.
2. SMA50 is above SMA200.
3. RSI14 is between 50 and 70.

### Execution
The signal is created after the daily candle closes. Paper/backtest entry is taken at the next daily open.

### Risk management
- 1% of current equity at risk per trade.
- Stop Loss = Entry − 1.5 × ATR14.
- Take Profit = Entry + 3 × ATR14.
- Position size is capped by available spot capital.

### Exits
- Stop Loss.
- Take Profit.
- Close below SMA50.
- End of test.

### Paper trading
Uses public BTC/USDT market data. No real orders are placed.
""")
    st.subheader("System Architecture")
    st.code("Market Data → Indicators → Decision → Risk Management → Backtest / Paper Execution → P&L → Dashboard", language="text")
