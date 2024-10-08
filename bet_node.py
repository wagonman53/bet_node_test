import numpy as np
import pandas as pd
import streamlit as st
import graph_function


# CSVファイルの読み込み関数
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

#ファイルパスの指定
mda_path = 'https://drive.google.com/uc?export=download&id=1ojUmB6fs4JYyRjD3f2qAEaPqKzW4wdX7'
gto_path = 'https://drive.google.com/uc?export=download&id=1FA4hfKhQB0XgVJ-lNq3NRpiHlN2LMPek'

#ファイルの読み込み
df_mda = load_data(mda_path)
df_gto = load_data(gto_path)

#列の追加
num_categories = ["2-9","10-Q","K","A"]
df_mda["Flop_rank"] = pd.cut(df_mda["Flop_high"],
                      bins=[1,9,12,13,15],
                      labels=num_categories,
                      ordered=True)
df_gto["Flop_rank"] = pd.cut(df_gto["Flop_high"],
                      bins=[1,9,12,13,15],
                      labels=num_categories,
                      ordered=True)
player_categories = ["0-1k","1k-10k","10k-100k","over 100k"]
df_mda["player_rank"] = pd.cut(df_mda["OOP Hands Played"],
                        bins=[0,1000,10000,100000,np.inf],
                        labels=player_categories,
                        ordered=True)
size_categories = ["min-33%","33%-66%","66%-100%","potover"]
df_mda["size_rank"] = pd.cut(df_mda["Flop size 1"],
                             bins=[0,33,66,100,np.inf],
                             labels=size_categories,
                             ordered=True)
df_mda["target"] = (df_mda["Flop action 1"] != 'check').astype(int)
df_gto["target"] = (df_gto["Flop action 1"] != 'check').astype(int)

#欠損値(=ベットしてない)をcheckで置き換え
df_mda["size_rank"] = df_mda["size_rank"].cat.add_categories('check')
df_mda["size_rank"] = df_mda["size_rank"].fillna("check")
df_gto["Flop size 1"] = df_gto["Flop size 1"].fillna(0).round().astype(int)
df_gto["Flop size 1"] = df_gto["Flop size 1"].replace(0, 'check')

#サイドバー設定
feature_col_mda = ["player_rank","Flop_rank","Flop_class"]
feature_col_gto = ["Flop_rank","Flop_class"]

with st.sidebar.form(key="sidebar_form"):
    with st.expander("ヒートマップと表分析の設定"):
        pivot_x = st.selectbox("ヒートマップのx軸",feature_col_mda,index=0)
        pivot_y = st.selectbox("ヒートマップのy軸",feature_col_mda,index=1)
        dfg_col_mda = st.multiselect("表データの特徴量",feature_col_mda,default=feature_col_mda)
        dfg_col_gto = list(set(dfg_col_mda)&set(feature_col_gto))
    with st.expander("アクションと役グラフの設定"):
        player_rank = st.selectbox("プレイヤーランクの選択",["All","0-1k","1k-10k","10k-100k","over 100k"])
        flop_high  = st.selectbox("ハイカードの選択",["All","2-9","10-Q","K","A"])
        flop_class  = st.selectbox("ボード種類の選択",["All","Paired","Monotone","Rainbow","Twotone"])
        mda_size = st.selectbox("MDAのアクションを選択",["All Bets","check"]+size_categories)
        gto_size = st.selectbox("GTOのアクションを選択",["All Bets"]+list(df_gto["Flop size 1"].unique()))
    submit_button = st.form_submit_button(label='更新')

st.header("bet node テスト")
st.text("BBvsBTN3betにおけるBBの最初のアクションを分析対象としています")

#ヒートマップの表示
st.subheader("ピボットヒートマップ")
st.text("ピボットのセルの数値はbet頻度(全サイズ合計)を表しています")
fig1 = graph_function.plot_heatmap(df_mda,df_gto,pivot_x,pivot_y,"target")
st.plotly_chart(fig1, use_container_width=True)

#表データの表示
st.subheader("表分析")
st.text("Targetの数値はピボットと同じくbet頻度です")
dfg = graph_function.plot_table(df_mda,df_gto,dfg_col_mda,dfg_col_gto,"target")
st.dataframe(dfg)

#データの絞り込み
df_mda_filtered = df_mda.copy()
df_gto_filtered = df_gto.copy()
if player_rank != "All":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["player_rank"]==player_rank]
if flop_high != "All":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["Flop_rank"]==flop_high]
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop_rank"]==flop_high]
if flop_class != "All":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["Flop_class"]==flop_class]
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop_class"]==flop_class]

#アクション分布の表示
st.subheader("レンジ全体のアクション頻度")
fig2 = graph_function.plot_action(df_mda_filtered,df_gto_filtered,"size_rank","Flop size 1",size_categories)
st.plotly_chart(fig2, use_container_width=True)

#アクションの絞り込み
if mda_size == "check":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["size_rank"]=="check"]
else:
    df_mda_filtered = df_mda_filtered[df_mda_filtered["size_rank"]!="check"]
if gto_size == "check":
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop size 1"]=="check"]
else:
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop size 1"]!="check"]

#サイズの絞り込み
if mda_size != "All Bets":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["size_rank"]==mda_size]
if gto_size != "All Bets":
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop size 1"]==gto_size]

#SD係数の取得
df_coef,dic_coef = graph_function.get_flop_sd_bias(df_gto_filtered,"OOP")

#バイアスあり役分布の表示
st.subheader("アクション毎の役構成(SDバイアスあり)")
df_gto_filtered_bias = df_gto_filtered[df_gto_filtered["OOP_sd"]==1]
fig3 = graph_function.plot_range(df_mda_filtered,df_gto_filtered_bias,"OOP_Flop_hand_rank")
st.plotly_chart(fig3, use_container_width=True)

#バイアス除去レンジとSD係数の表示
st.subheader("アクション毎の役構成(SDバイアス除去)")
fig4 = graph_function.plot_range_nobias(df_mda_filtered,df_gto_filtered,"OOP_Flop_hand_rank",dic_coef)
st.plotly_chart(fig4, use_container_width=True)
st.subheader("SDバイアス除去に使用した係数")
st.dataframe(df_coef)
