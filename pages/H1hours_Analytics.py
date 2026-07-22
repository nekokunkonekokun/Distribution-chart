import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import timedelta, date
import warnings
warnings.filterwarnings('ignore')

# 画面全体のレイアウト設定
st.set_page_config(
    page_title="1H Market Profile & Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- データの取得部分（1時間足用・キャッシュ化＋最新まで自動取得） ---
@st.cache_data(show_spinner="1時間足データを取得中...")
def load_h1_data(ticker, days_ago):
    t_obj = yf.Ticker(ticker)
    total_shares = None
    
    # .info の取得を try-except で囲み、エラーでも落ちないようにする
    try:
        info = t_obj.info
        total_shares = info.get('sharesOutstanding', None)
    except Exception:
        total_shares = None

    # .info で取れなかった場合、少し軽い fast_info も試してみる
    if not total_shares:
        try:
            total_shares = t_obj.fast_info.get('shares', None)
        except Exception:
            total_shares = None
        
    # 終了日は指定せず、periodを使って最新の1時間足まで自動取得（1時間足は最大730日の制限あり）
    df = yf.download(ticker, period=f"{days_ago}d", interval="1h", progress=False)
    
    if df.empty or len(df) < 20:
        return None, None, "Error: 十分な1時間足データが取得できませんでした。日数（days_ago）を大きくしてください。"
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
    
    return df, total_shares, "Success"

# --- サイドバー（設定パネル） ---
st.sidebar.header("📊 1H シミュレーション設定")

ticker = st.sidebar.text_input("銘柄コード (東証は末尾に .T)", value="285A.T")

# 発行済株式数の予備入力欄
fallback_shares = st.sidebar.number_input(
    "発行済株式数 (取得失敗時の予備)", 
    value=10_000_000, 
    step=1_000_000,
    help="Yahoo Financeからの自動取得がブロックされた際、この数値を使って計算します。"
)

# 期間指定（エンドを指定せず、直近何日間かを選択）
days_ago = st.sidebar.slider("データ取得日数 (1時間足)", min_value=10, max_value=100, value=30, step=5)

margin_ratio = st.sidebar.number_input("信用倍率 (買い残 ÷ 売り残)", value=24.0, step=0.1, min_value=0.01)
recent_bars = st.sidebar.slider("比較期間（直近のローソク足の本数）", min_value=1, max_value=50, value=12)

st.sidebar.markdown("---")
st.sidebar.header("📊 バックテスト設定")
profit_threshold = st.sidebar.slider("エントリー閾値: 直近損益比率 (%)", min_value=30.0, max_value=90.0, value=60.0, step=5.0)
holding_bars = st.sidebar.slider("保有期間 (本 / 時間)", min_value=1, max_value=48, value=12, step=1)

# データの読み込みとチェック
df, total_shares, status = load_h1_data(ticker, days_ago)

if "Error" in status:
    st.error(status)
else:
    if not total_shares or total_shares <= 0:
        st.warning(f"⚠️ 発行済株式数の自動取得に失敗しました。サイドバーで指定された予備の数値 ({fallback_shares:,} 株) を使用して計算しています。")
        total_shares = fallback_shares

    # --- 蓄積シミュレーションロジック（1時間足ベース） ---
    recent_border_idx = len(df) - int(recent_bars)

    min_p = float(df['Low'].min())
    max_p = float(df['High'].max())
    bins = np.linspace(min_p, max_p, 101)
    labels = bins[:-1] + (bins[1] - bins[0]) / 2
    bin_width = bins[1] - bins[0]

    total_distribution = np.zeros(len(labels))
    recent_distribution = np.zeros(len(labels))

    for i, (idx, row) in enumerate(df.iterrows()):
        high   = float(row['High'])
        low    = float(row['Low'])
        close  = float(row['Close'])
        volume = float(row['Volume'])

        if volume == 0:
            continue

        turnover_rate = min(volume / total_shares, 1.0)
        total_distribution *= (1 - turnover_rate)
        recent_distribution *= (1 - turnover_rate)

        todays_bins = (labels >= low) & (labels <= high)

        if high == low:
            closest_idx = np.argmin(np.abs(labels - close))
            allocated_volume = np.zeros(len(labels))
            allocated_volume[closest_idx] = volume
            todays_bins = np.zeros(len(labels), dtype=bool)
            todays_bins[closest_idx] = True
        else:
            price_diffs = np.abs(labels[todays_bins] - close)
            max_diff = max(high - close, close - low)
            max_diff = max_diff if max_diff > 0 else 1.0

            weights = 1.0 - (price_diffs / max_diff)
            weights = np.maximum(weights, 0.1)
            allocated_volume = (weights / np.sum(weights)) * volume

        if high == low:
            total_distribution += allocated_volume
            if i >= recent_border_idx:
                recent_distribution += allocated_volume
        else:
            total_distribution[todays_bins] += allocated_volume
            if i >= recent_border_idx:
                recent_distribution[todays_bins] += allocated_volume

    # 指標計算
    current_price = float(df['Close'].iloc[-1])
    long_profit_mask = labels < current_price
    short_profit_mask = labels > current_price

    LONG_RATIO = margin_ratio / (margin_ratio + 1.0)
    SHORT_RATIO = 1.0 / (margin_ratio + 1.0)

    recent_total = np.sum(recent_distribution)
    recent_profit_ratio = (
        (np.sum(recent_distribution[long_profit_mask]) * LONG_RATIO +
         np.sum(recent_distribution[short_profit_mask]) * SHORT_RATIO) / recent_total * 100
    ) if recent_total > 0 else 0.0

    total_sum = np.sum(total_distribution)
    total_profit_ratio = (
        (np.sum(total_distribution[long_profit_mask]) * LONG_RATIO +
         np.sum(total_distribution[short_profit_mask]) * SHORT_RATIO) / total_sum * 100
    ) if total_sum > 0 else 0.0

    peaks, _ = find_peaks(total_distribution, height=np.mean(total_distribution),
                          distance=4, prominence=np.max(total_distribution)*0.05)
    peak_prices = labels[peaks]

    upper_walls = peak_prices[peak_prices > current_price]
    nearest_upper_wall = upper_walls[0] if len(upper_walls) > 0 else None
    dist_to_wall_pct = ((nearest_upper_wall - current_price) / current_price * 100) if nearest_upper_wall else None

    # --- メイン画面の描画 ---
    st.title(f"📈 1H Market Profile & Backtest Dashboard")
    st.subheader(f"銘柄コード: {ticker} (1時間足 / 最新データまで自動取得)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("現在値 (最新)", f"{current_price:,.1f} 円")
    m2.metric(f"直近 {recent_bars}本 損益比率", f"{recent_profit_ratio:.1f} %")
    m3.metric("全体 損益比率", f"{total_profit_ratio:.1f} %")
    m4.metric("最も近い上値の壁", f"{nearest_upper_wall:,.1f} 円" if nearest_upper_wall else "なし", 
              delta=f"あと {dist_to_wall_pct:.1f} %" if dist_to_wall_pct else None, delta_color="inverse")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10})

    # 左グラフ：全体分布
    with col1:
        st.markdown("### 🏛️ 全体コスト分布としこり玉 (1H)")
        fig1, ax1 = plt.subplots(figsize=(7, 5.5))
        ax1.barh(labels, total_distribution, height=bin_width*0.8, color='gray', alpha=0.5, label='Total Cost')
        ax1.axhline(y=current_price, color='blue', linewidth=2, label=f'Current: {current_price:,.1f}')
        for p in peak_prices:
            ax1.axhline(y=p, color='purple', linestyle=':', alpha=0.7)
            ax1.text(np.max(total_distribution)*0.02, p, f'Wall: {p:,.1f}', color='purple', fontsize=10, fontweight='bold')
        ax1.set_xlabel("Accumulated Volume")
        ax1.set_ylabel("Price (Yen)")
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)

    # 右グラフ：直近の損益プロフィール
    with col2:
        st.markdown(f"### ⏳ 直近 {recent_bars}本の損益プロフィール (1H)")
        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
        colors = ['#ff9999' if p < current_price else '#d3d3d3' for p in labels]
        ax2.barh(labels, recent_distribution, height=bin_width*0.8, color=colors, alpha=0.8, label='Recent Active')
        ax2.axhline(y=current_price, color='blue', linewidth=2)
        ax2.set_xlabel("Recent Active Volume")
        ax2.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    # --- バックテストセクション ---
    st.markdown("---")
    st.subheader("📊 1時間足シグナル バックテスト結果")
    st.markdown(f"**ルール**: 直近損益比率が **{profit_threshold}%** を超えたタイミングで買いエントリーし、**{holding_bars}本（時間）後**に決済した場合のシミュレーションです。")

    bt_results = []
    start_idx = max(50, int(recent_bars))

    for i in range(start_idx, len(df) - holding_bars):
        sub_df = df.iloc[:i+1]
        sub_recent_border = len(sub_df) - int(recent_bars)
        
        recent_slice = sub_df.iloc[sub_recent_border:]
        sub_cur = float(sub_df['Close'].iloc[-1])
        
        prof_count = sum(recent_slice['Close'] < sub_cur)
        sim_ratio = (prof_count / len(recent_slice)) * 100 if len(recent_slice) > 0 else 0
        
        if sim_ratio >= profit_threshold:
            entry_price = float(df['Close'].iloc[i])
            exit_price = float(df['Close'].iloc[i + holding_bars])
            ret_pct = ((exit_price - entry_price) / entry_price) * 100
            bt_results.append({
                "Entry Time": df.index[i],
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Return (%)": ret_pct
            })

    if bt_results:
        bt_df = pd.DataFrame(bt_results)
        total_trades = len(bt_df)
        win_trades = len(bt_df[bt_df["Return (%)"] > 0])
        win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
        total_return = bt_df["Return (%)"].sum()
        avg_return = bt_df["Return (%)"].mean()

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("総トレード回数", f"{total_trades} 回")
        bcol2.metric("勝率", f"{win_rate:.1f} %")
        bcol3.metric("平均リターン / 回", f"{avg_return:.2f} %")
        bcol4.metric("累積リターン", f"{total_return:.2f} %")

        st.markdown("#### トレード履歴詳細")
        st.dataframe(bt_df, use_container_width=True)

        bt_df["Cumulative Return"] = bt_df["Return (%)"].cumsum()
        fig_bt, ax_bt = plt.subplots(figsize=(12, 4))
        ax_bt.plot(bt_df["Entry Time"], bt_df["Cumulative Return"], marker='o', color='green', linestyle='-')
        ax_bt.set_title("Backtest Cumulative Return Curve (%)", fontsize=12, fontweight='bold')
        ax_bt.set_xlabel("Entry Time", fontsize=10)
        ax_bt.set_ylabel("Cumulative Return (%)", fontsize=10)
        ax_bt.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig_bt)
    else:
        st.warning("指定した条件（閾値）に一致するトレードシグナルが、この期間内では検出されませんでした。サイドバーの閾値を調整してみてください。")
