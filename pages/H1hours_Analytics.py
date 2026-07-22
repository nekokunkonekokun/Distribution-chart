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
    page_title="1H Market Profile & Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- データの取得部分（1時間足・期間指定対応） ---
@st.cache_data(show_spinner="指定期間の1時間足データを取得中...")
def load_h1_data_by_date(ticker, start_date, end_date):
    t_obj = yf.Ticker(ticker)
    total_shares = None
    
    try:
        info = t_obj.info
        total_shares = info.get('sharesOutstanding', None)
    except Exception:
        total_shares = None

    if not total_shares:
        try:
            total_shares = t_obj.fast_info.get('shares', None)
        except Exception:
            total_shares = None
        
    df = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
    
    if df.empty or len(df) < 10:
        return None, None, "Error: 指定された期間の十分な1時間足データが取得できませんでした。（※1時間足は過去約730日以内のみ取得可能です）"
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
    
    return df, total_shares, "Success"

# --- サイドバー（設定パネル） ---
st.sidebar.header("📊 1H 期間指定・シミュレーション設定")

ticker = st.sidebar.text_input("銘柄コード (東証は末尾に .T)", value="285A.T")

fallback_shares = st.sidebar.number_input(
    "発行済株式数 (取得失敗時の予備)", 
    value=10_000_000, 
    step=1_000_000,
    help="Yahoo Financeからの自動取得がブロックされた際、この数値を使って計算します。"
)

# デフォルトでは常に最新（今日）まで取得するよう、終了日を今日に設定
today = date.today()

# モード切替のチェックボックスを用意（「常に最新まで取得」か「手動で終了日を固定する」か）
auto_latest = st.sidebar.checkbox("終了日を常に最新（今日）にする", value=True, help="チェックを入れると、終了日が自動的に今日（リアルタイム）に設定されます。")

start_date = st.sidebar.date_input("データ取得開始日", value=today - timedelta(days=30))

if auto_latest:
    end_date = today
    st.sidebar.text(f"データ取得終了日: {today} (自動)")
else:
    end_date = st.sidebar.date_input("データ取得終了日", value=today)

margin_ratio = st.sidebar.number_input("信用倍率 (買い残 ÷ 売り残)", value=24.0, step=0.1, min_value=0.01)
recent_bars = st.sidebar.slider("比較期間（直近のローソク足の本数）", min_value=1, max_value=50, value=12)

# データの読み込みとチェック
if start_date >= end_date:
    st.error("開始点は終了点より前の日付を選択してください。")
else:
    # 終了日の翌日までを指定しないと当日分が切れることがあるため、end_dateに1日足して取得するなどの調整も可能ですが、ここでは標準のend_dateを使用
    df, total_shares, status = load_h1_data_by_date(
        ticker, 
        start_date.strftime("%Y-%m-%d"), 
        (end_date + timedelta(days=1)).strftime("%Y-%m-%d") # 終了日当日も含めるための調整
    )
    
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
        st.title(f"📈 1H Market Profile Dashboard")
        st.subheader(f"銘柄コード: {ticker} (期間: {start_date} 〜 {end_date})")

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
