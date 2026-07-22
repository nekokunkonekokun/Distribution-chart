import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# ページの設定
st.set_page_config(
    page_title="1時間足マーケットプロフィール分析",
    page_icon="📈",
    layout="wide"
)

st.title("📈 1時間足 マーケットプロフィール＆しこり玉分析")
st.markdown("指定した銘柄の**1時間足（1h）**データを用いて、需給プロフィールとしこり玉（ピーク）、直近足（12本）の損益状況を可視化します。")

# --- サイドバーの設定 ---
st.sidebar.header("分析パラメータ設定")

ticker_input = st.sidebar.text_input("銘柄コード (例: 285A.T)", value="285A.T")
start_date = st.sidebar.date_input("開始日", value=pd.to_datetime("2026-05-01"))
end_date = st.sidebar.date_input("終了日", value=pd.to_datetime("2026-07-22"))
margin_ratio = st.sidebar.number_input("信用倍率 (買い残÷売り残)", value=24.0, min_0.1=0.1, step=1.0)
recent_bars = st.sidebar.number_input("直近集計本数 (本)", value=12, min_value=1, step=1)

run_button = st.sidebar.button("分析を実行", type="primary")

# --- 分析メイン処理 ---
if run_button:
    with st.spinner(f"{ticker_input} の1時間足データを取得・分析中..."):
        try:
            # 1. データと発行済株式数の取得（1時間足設定）
            t_obj = yf.Ticker(ticker_input)
            info = t_obj.info
            total_shares = info.get('sharesOutstanding', None)

            if not total_shares:
                st.error(f"エラー: {ticker_input} の発行済株式数（sharesOutstanding）が取得できませんでした。")
            else:
                df = yf.download(ticker_input, start=str(start_date), end=str(end_date), interval="1h", progress=False)

                if df.empty or len(df) < 10:
                    st.error("エラー: 十分なデータが取得できませんでした。期間を広げるか、銘柄コードを確認してください。")
                else:
                    # マルチインデックス対策
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])

                    # 直近指定本数（recent_bars）の境界インデックスを設定
                    recent_border_idx = len(df) - int(recent_bars)

                    min_p = float(df['Low'].min())
                    max_p = float(df['High'].max())
                    bins = np.linspace(min_p, max_p, 101)
                    labels = bins[:-1] + (bins[1] - bins[0]) / 2
                    bin_width = bins[1] - bins[0]

                    total_distribution = np.zeros(len(labels))
                    recent_distribution = np.zeros(len(labels))

                    # 2. 蓄積シミュレーションループ（1時間足ベース）
                    for i, (idx, row) in enumerate(df.iterrows()):
                        high   = float(row['High'])
                        low    = float(row['Low'])
                        close  = float(row['Close'])
                        volume = float(row['Volume'])

                        if volume == 0:
                            continue

                        # 回転率ベースの減衰
                        turnover_rate = min(volume / total_shares, 1.0)
                        total_distribution *= (1 - turnover_rate)
                        recent_distribution *= (1 - turnover_rate)

                        # 出来高の分配
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

                    # 3. 信用倍率を反映した損益指標計算
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

                    # ピーク（しこり）検出
                    peaks, _ = find_peaks(total_distribution, height=np.mean(total_distribution),
                                          distance=4, prominence=np.max(total_distribution)*0.05)
                    peak_prices = labels[peaks]

                    upper_walls = peak_prices[peak_prices > current_price]
                    nearest_upper_wall = upper_walls[0] if len(upper_walls) > 0 else None
                    dist_to_wall_pct = ((nearest_upper_wall - current_price) / current_price * 100) if nearest_upper_wall else None

                    # --- 4. メトリクス表示 ---
                    st.success("分析が完了しました！")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("現在値", f"{current_price:,.1f} 円")
                    col2.metric(f"直近 ({recent_bars}本) 損益比率", f"{recent_profit_ratio:.1f} %")
                    col3.metric("全体 損益比率", f"{total_profit_ratio:.1f} %")
                    if nearest_upper_wall:
                        col4.metric("最寄りの上値壁", f"{nearest_upper_wall:,.1f} 円", f"+{dist_to_wall_pct:.1f}%")
                    else:
                        col4.metric("最寄りの上値壁", "なし")

                    # --- 5. グラフ描画 ---
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

                    # 左グラフ：全体分布
                    ax1.barh(labels, total_distribution, height=bin_width*0.8, color='gray', alpha=0.5, label='Total Cost Distribution')
                    ax1.axhline(y=current_price, color='blue', linewidth=2, label=f'Current: {current_price:,.1f}')
                    for p in peak_prices:
                        ax1.axhline(y=p, color='purple', linestyle=':', alpha=0.7)
                        ax1.text(np.max(total_distribution)*0.02, p, f'Wall: {p:,.1f}', color='purple', fontsize=9, fontweight='bold')
                    ax1.set_title(f"Total Cumulative Distribution (1H Interval)", fontsize=12)
                    ax1.set_xlabel("Accumulated Volume (Decayed)", fontsize=10)
                    ax1.set_ylabel("Price (Yen)", fontsize=10)
                    ax1.legend()
                    ax1.grid(True, linestyle='--', alpha=0.3)

                    # 右グラフ：直近指定本数の損益
                    colors = ['#ff9999' if p < current_price else '#d3d3d3' for p in labels]
                    ax2.barh(labels, recent_distribution, height=bin_width*0.8, color=colors, alpha=0.8, label=f'Recent {recent_bars} Bars Distribution')
                    ax2.axhline(y=current_price, color='blue', linewidth=2)
                    ax2.set_title(f"Recent {recent_bars} Bars Trader Profit/Loss Profile", fontsize=12)
                    ax2.set_xlabel("Recent Active Volume", fontsize=10)

                    info_text = (
                        f"[Profit Ratio (Margin: {margin_ratio:.1f})]\n"
                        f"★Recent {recent_bars} Bars: {recent_profit_ratio:.1f}%\n"
                        f"★Total Hold: {total_profit_ratio:.1f}%"
                    )
                    ax2.text(ax2.get_xlim()[1]*0.05, ax2.get_ylim()[0] + (ax2.get_ylim()[1]-ax2.get_ylim()[0])*0.05,
                             info_text, fontsize=11, fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
                    ax2.legend()
                    ax2.grid(True, linestyle='--', alpha=0.3)

                    plt.suptitle(f"1H Market Profile Dashboard: {ticker_input}", fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"予期せぬエラーが発生しました: {str(e)}")
else:
    st.info("左側のサイドバーでパラメータを設定し、[分析を実行] ボタンを押してください。")
                  
