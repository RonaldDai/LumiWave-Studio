import streamlit as st
import numpy as np
import scipy.signal as signal
from scipy.special import erfc
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 页面全局配置与终极 CSS 对齐约束
# ==========================================
st.set_page_config(page_title="LumiWave Studio", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    ::-webkit-scrollbar { display: none; }
    html, body { scrollbar-width: none; -ms-overflow-style: none; overflow-y: hidden; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 0rem !important; max-width: 98% !important; }
    .top-title { font-size: 24px; font-weight: bold; color: #1f77b4; margin-top: -15px; margin-bottom: 8px; border-bottom: 2px solid #eee; padding-bottom: 5px; }
    h3 { margin-top: 0rem !important; margin-bottom: 0.5rem !important; font-size: 1.05rem !important; color: #444; border-bottom: 1px solid #eee; padding-bottom: 0.2rem;}
    div.stMarkdown { margin-bottom: 0px !important; }
    div[data-testid="stVerticalBlock"] > div { padding-bottom: 0rem !important; padding-top: 0rem !important; }
    label { font-size: 0.82rem !important; padding-bottom: 2px !important; margin-bottom: 0px !important; color: #666; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; display: block !important; }
    div[data-testid="stNumberInput"], div[data-testid="stSelectbox"], div[data-testid="stSlider"] { height: 68px !important; display: flex !important; flex-direction: column !important; justify-content: flex-end !important; }
    div[data-testid="stCheckbox"] { height: 68px !important; display: flex !important; flex-direction: column !important; justify-content: flex-end !important; padding-bottom: 8px !important; }
    .result-card { background-color: #f8f9fa; border: 1px solid #ced4da; border-radius: 6px; padding: 5px 10px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); display: flex; flex-direction: column; justify-content: center; min-height: 52px; }
    .result-label { font-size: 0.75rem !important; color: #555 !important; font-weight: 700 !important; margin-bottom: 1px !important; text-transform: uppercase; letter-spacing: 0.5px;}
    .result-value { font-size: 1.15rem !important; font-weight: 800 !important; color: #1f77b4 !important; line-height: 1.1 !important; white-space: nowrap;}
    .result-value.alert { color: #d62728 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="top-title">💠 LumiWave Studio: 光通信物理层数字孪生系统 (Node Probe Edition)</div>', unsafe_allow_html=True)

def sub_header(text):
    st.markdown(f"<div style='font-size:0.8rem; color:#1f77b4; font-weight:bold; margin-top:8px; margin-bottom:2px; border-bottom:1px solid #ddd;'>{text}</div>", unsafe_allow_html=True)

# ==========================================
# 2. 全局物理常数与仿真配置
# ==========================================
KB = 1.38e-23            
TEMP_K = 298             
Q_E = 1.6e-19            

SIM_FS = 40e9            
TARGET_NUM_BITS = 1000000000000  
CHUNK_BITS = 1000000             
OVERLAP_BITS = 1000              
PREAMBLE_BITS = 200000           

FIBER_MBW_MAP = {"OM1": 200, 
                 "OM2": 500, 
                 "OM3": 2000, 
                 "OM4": 4700}
PRBS_TAPS_MAP = {7: (7, 6), 
                 9: (9, 5), 
                 11: (11, 9), 
                 15: (15, 14)} 

# ==========================================
# 3. 基础信号组件
# ==========================================
@st.cache_data
def generate_prbs(order, num_bits):
    tap1, tap2 = PRBS_TAPS_MAP[order]
    register = (1 << order) - 1
    bits = np.zeros(num_bits, dtype=int)
    for i in range(num_bits):
        bits[i] = register & 1
        feedback = ((register >> (tap1 - 1)) ^ (register >> (tap2 - 1))) & 1
        register = (register >> 1) | (feedback << (order - 1))
    return bits

def apply_scrambler(data_bits, prbs_order):
    prbs_seq = generate_prbs(prbs_order, len(data_bits))
    return np.bitwise_xor(data_bits, prbs_seq)

def build_card(label, value, alert=False):
    cls = " alert" if alert else ""
    return f'<div class="result-card"><span class="result-label">{label}</span><span class="result-value{cls}">{value}</span></div>'

# ==========================================
# 4. 物理层单块处理引擎 (含探针提取功能)
# ==========================================
def process_physical_chunk(chunk_bits_with_overlap, global_start_bit, spb, br, 
                           p_tx_mw, p_low_mw, k_nonlin, b_led, a_led, rj_rms_ps, dj_pp_ps, 
                           rin_db, loss_factor, use_disp, kernel, resp_aw, 
                           b_rx, a_rx, rf_eff, c_tot, en_nv, i_mean_est, gn25l96_cfg, 
                           return_nodes=False): # 新增 return_nodes 探针开关
    
    nodes = {}
    
    # 1. 初始基带与抖动
    tx_base_ideal = np.repeat(chunk_bits_with_overlap, spb).astype(float)
    if rj_rms_ps > 0 or dj_pp_ps > 0:
        t_ideal = np.arange(len(tx_base_ideal)) / SIM_FS
        total_jitter = np.zeros(len(tx_base_ideal))
        if rj_rms_ps > 0:
            np.random.seed((global_start_bit // CHUNK_BITS) + 42)
            raw_jitter = np.random.normal(0, 1, len(tx_base_ideal))
            b_j, a_j = signal.butter(1, 0.05, 'low')
            smoothed_jitter = signal.filtfilt(b_j, a_j, raw_jitter)
            total_jitter += smoothed_jitter * (rj_rms_ps * 1e-12 / np.std(smoothed_jitter))
        if dj_pp_ps > 0:
            t_abs = np.arange(global_start_bit * spb, (global_start_bit + len(chunk_bits_with_overlap)) * spb) / SIM_FS
            total_jitter += (dj_pp_ps/2.0)*1e-12 * np.sin(2*np.pi*(br/111.0)*t_abs)
        tx_base = np.interp(t_ideal, t_ideal + total_jitter, tx_base_ideal)
    else: 
        tx_base = tx_base_ideal
        
    if return_nodes: nodes['Node1_Baseband'] = tx_base.copy()

    # 2. FR4 + TX EQ + RLC 寄生
    if gn25l96_cfg['en']:
        if gn25l96_cfg['b_fr4'] is not None:
            tx_base = signal.lfilter(gn25l96_cfg['b_fr4'], 
                                     gn25l96_cfg['a_fr4'], tx_base)
        if gn25l96_cfg['b_eq'] is not None:
            tx_base = signal.lfilter(gn25l96_cfg['b_eq'], 
                                     gn25l96_cfg['a_eq'], tx_base)
            tx_base = np.clip(tx_base, 0, 1) 
        if gn25l96_cfg['b_interconnect'] is not None:
            tx_base = signal.lfilter(gn25l96_cfg['b_interconnect'], 
                                     gn25l96_cfg['a_interconnect'], tx_base)
            
    if return_nodes: nodes['Node2_TxEQ'] = tx_base.copy()

    # 3. 光源发光与信道色散
    tx_p_continuous = p_low_mw + tx_base * (p_tx_mw - p_low_mw)
    tx_p_continuous = tx_p_continuous - k_nonlin * (tx_p_continuous**2) 
    tx_p_continuous = np.maximum(tx_p_continuous, 0) 
    
    sig_mw = signal.lfilter(b_led, a_led, tx_p_continuous)
    np.random.seed((global_start_bit // CHUNK_BITS) + 100)
    sig_mw += np.random.normal(0, 1, len(sig_mw)) * np.sqrt(10**(rin_db/10)*(SIM_FS/2)) * sig_mw
    
    sig_mw *= loss_factor
    if use_disp: sig_mw = np.convolve(sig_mw, kernel, mode='same')
        
    if return_nodes: nodes['Node3_Optical'] = sig_mw.copy()

    # 4. TIA 转换与噪声
    i_photo_A = (sig_mw * 1e-3) * resp_aw
    chunk_len = len(sig_mw)
    freqs = np.fft.rfftfreq(chunk_len, 1/SIM_FS)
    w, h_rx = signal.freqz(b_rx, a_rx, worN=2*np.pi*freqs/SIM_FS)
    
    s_white = (4 * KB * TEMP_K / rf_eff) + (2 * Q_E * (i_mean_est + 10e-9))
    psd = (s_white + (en_nv*1e-9)**2 * (2*np.pi*freqs*c_tot)**2) * (rf_eff**2)
    psd *= np.abs(h_rx)**2 
    np.random.seed((global_start_bit // CHUNK_BITS) + 200)
    noise_V = np.fft.irfft(np.fft.rfft(np.random.normal(0, 1, chunk_len)) * np.sqrt(psd * SIM_FS / 2), n=chunk_len)
    
    v_out_mV_chunk = (signal.lfilter(b_rx, a_rx, i_photo_A * rf_eff) - np.mean(signal.lfilter(b_rx, a_rx, 
                                                                                              i_photo_A * rf_eff)) + noise_V) * 1000

    if return_nodes: nodes['Node4_TIA'] = v_out_mV_chunk.copy()

    # 5. RX CTLE 与 LA
    if gn25l96_cfg['en']:
        if gn25l96_cfg['b_rx_eq'] is not None:
            v_out_mV_chunk = signal.lfilter(gn25l96_cfg['b_rx_eq'], 
                                            gn25l96_cfg['a_rx_eq'], v_out_mV_chunk)
            
        if return_nodes: nodes['Node5_RxEQ'] = v_out_mV_chunk.copy()

        g_lin = 10**(gn25l96_cfg['la_gain'] / 20)
        v_swing = gn25l96_cfg['la_swing']
        
        np.random.seed((global_start_bit // CHUNK_BITS) + 300)
        la_noise_rms = gn25l96_cfg['la_noise'] * 1e-6 * np.sqrt(SIM_FS/2) * 1000
        v_out_mV_chunk += np.random.normal(0, 1, len(v_out_mV_chunk)) * la_noise_rms
        
        v_out_mV_chunk = v_swing * np.tanh((v_out_mV_chunk * g_lin) / v_swing) 
        
        if gn25l96_cfg['b_la'] is not None:
            v_out_mV_chunk = signal.lfilter(gn25l96_cfg['b_la'], 
                                            gn25l96_cfg['a_la'], v_out_mV_chunk)
            
        if return_nodes: nodes['Node6_LA'] = v_out_mV_chunk.copy()
        
    elif return_nodes: # 如果没开 LA，也填充空数据对齐字典
        nodes['Node5_RxEQ'] = v_out_mV_chunk.copy()
        nodes['Node6_LA'] = v_out_mV_chunk.copy()

    # 裁切重叠区
    chunk_out = v_out_mV_chunk[OVERLAP_BITS * spb:]
    if return_nodes:
        for k in nodes: nodes[k] = nodes[k][OVERLAP_BITS * spb:]
        return chunk_out, nodes
    return chunk_out

# ==========================================
# 5. UI 控制面板 (完全贴合信号物理流向重构)
# ==========================================
col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], gap="small")

with col1:
    st.markdown("### 💻 1. 主板与基带")
    sub_header("全局硬件模型")
    en_gn25l96 = st.checkbox("✅ 启用物理全链路", value=True)
    sub_header("数据流生成 (Host)")
    br_gbps = st.slider("数据速率(G)", 0.5, 5.0, 2.0)
    data_source = st.selectbox("基带数据", ["纯随机", "1010序列"])
    use_scrambler = st.checkbox("启用扰码", value=True)
    prbs_order = st.selectbox("PRBS 阶数", [7, 9, 11, 15], index=3)
    sub_header("源端时钟抖动")
    rj_rms_ps = st.number_input("RJ(ps)", 0.0, 50.0, 10.0)
    dj_pp_ps = st.number_input("DJ(ps)", 0.0, 100.0, 30.0)
    sub_header("电域衰减")
    fr4_len_in = st.slider("Host走线FR4损耗(inch)", 0.0, 20.0, 8.0)

with col2:
    st.markdown("### ⚡ 2. 驱动与发光")
    sub_header("TX Input EQ & Driver")
    tx_eq_db = st.slider("TX CTLE 补偿 (dB)", 0.0, 12.0, 4.0)
    tx_bw_ghz = st.slider("TX 级联带宽(GHz)", 1.0, 5.0, 2.7)
    sub_header("封装互连 (Parasitics)")
    ic_bw_ghz = st.slider("驱动-光源寄生带宽(GHz)", 1.0, 10.0, 3.5)
    ic_zeta = st.slider("互连阻尼(ζ小易振铃)", 0.2, 1.0, 0.45)
    sub_header("发光芯片 (O/E)")
    p_tx_mw = st.number_input("峰值光功率(mW)", 0.1, 10.0, 1.0)
    er_db = st.slider("消光比 ER (dB)", 3.0, 15.0, 4.0)
    fc_led_ghz = st.slider("光源物理带宽(GHz)", 0.1, 3.5, 1.3)
    k_nonlin = st.slider("功率压缩系数", 0.0, 0.5, 0.1)
    rin_db = st.slider("光源 RIN(dB/Hz)", -150, -110, -130)

with col3:
    st.markdown("### 🛤️ 3. 信道与探测")
    sub_header("光纤传输 (Channel)")
    dist_m = st.slider("传输距离(m)", 1, 300, 30)
    fiber_type = st.selectbox("光纤类型", ["OM1", "OM2", "OM3", "OM4"], index=3)
    spec_w = st.slider("光源谱宽(nm)", 1, 40, 30)
    disp_coef = st.number_input("色散系数", value=150.0)
    loss_db = st.number_input("光纤插损(dB)", value=60.0)
    sub_header("光电转换 (PD)")
    resp_aw = st.number_input("PD 响应度(A/W)", value=0.2)
    cj_pf = st.number_input("PD 结电容(pF)", 0.01, 1.0, 0.22)

with col4:
    st.markdown("### 🔌 4. 接收前端(TIA)")
    sub_header("TIA 核心参数")
    rf_ohm = st.number_input("跨阻 Rf(Ω)", value=80000, step=5000)
    rin_ohm = st.number_input("输入阻抗 Zin(Ω)", value=435)
    sub_header("寄生与二阶响应")
    c_tia_pf = st.number_input("TIA 寄生电容(pF)", 0.01, 1.0, 0.15)
    tia_zeta = st.slider("TIA 阻尼 ζ", 0.4, 1.0, 0.707)
    sub_header("前端热噪声基底")
    en_nv = st.number_input("等效白噪声(nV)", value=2.0)

with col5:
    st.markdown("### 🎛️ 5. 均衡与限幅")
    sub_header("RX CTLE (高频补偿)")
    rx_eq_db = st.slider("RX EQ 补偿量(dB)", 0.0, 15.0, 3.0)
    rx_eq_bw_ghz = st.slider("RX EQ 峰值频率(G)", 0.5, 5.0, 2.0)
    sub_header("LA (限幅放大器)")
    la_gain_db = st.slider("LA 核心增益(dB)", 10.0, 50.0, 35.0)
    la_swing_mv = st.number_input("LA 硬摆幅(mV)", 100, 800, 250)
    sub_header("LA 滤波与噪声")
    la_bw_ghz = st.slider("LA 级联带宽(GHz)", 1.0, 5.0, 2.5)
    la_noise_nv = st.number_input("LA 附加噪声(nV)", 0.0, 10.0, 1.5)

with col6:
    st.markdown("### 📊 6. 判决监控")
    sub_header("数字统计终端")
    live_stats_placeholder = st.empty()
    with live_stats_placeholder.container():
        st.markdown(build_card("已处理总比特", "0"), unsafe_allow_html=True)
        st.markdown(build_card("实时统计 BER", "等待启动..."), unsafe_allow_html=True)
        st.markdown(build_card("理论极限 BER", "-"), unsafe_allow_html=True)
        st.markdown(build_card("判决级带宽(G)", "-"), unsafe_allow_html=True)
        st.markdown(build_card("模拟 Q因子", "-"), unsafe_allow_html=True)

# ==========================================
# 6. 参数预编译与物理滤波器综合构建区
# ==========================================
br = br_gbps * 1e9
spb = int(SIM_FS / br)
p_low_mw = p_tx_mw / (10**(er_db/10.0))
p_max, p_min = p_tx_mw - k_nonlin * (p_tx_mw**2), p_low_mw - k_nonlin * (p_low_mw**2)
l_km = dist_m / 1000.0
loss_factor = 10**(-loss_db * l_km / 10.0)

i_peak_est = (p_max - p_min) * loss_factor * 1e-3 * resp_aw
i_mean_est = (p_max + p_min) / 2 * loss_factor * 1e-3 * resp_aw
rf_eff = 0.3 / i_peak_est if i_peak_est * rf_ohm > 0.3 else rf_ohm

b_led, a_led = signal.butter(1, min((fc_led_ghz*1e9)/(SIM_FS/2), 0.99), 'low')
c_tot = (cj_pf + c_tia_pf) * 1e-12
fc_rx = 1 / (2 * np.pi * ((rf_eff * rin_ohm)/(rf_eff + rin_ohm)) * c_tot)
fc_rx_safe = min(fc_rx, SIM_FS/2 * 0.95) 
wc_prewarped = 2 * SIM_FS * np.tan((2 * np.pi * fc_rx_safe) / (2 * SIM_FS)) 
b_coeff = 4 * tia_zeta**2 - 2
wn = wc_prewarped / np.sqrt((-b_coeff + np.sqrt(b_coeff**2 + 4)) / 2)
b_rx, a_rx = signal.bilinear([wn**2], [1, 2*tia_zeta*wn, wn**2], fs=SIM_FS)

dt_total = np.sqrt((disp_coef*1e-12*spec_w*l_km)**2 + (0.441/(FIBER_MBW_MAP[fiber_type]*1e6/l_km))**2)
sigma_s = (dt_total * SIM_FS) / (2 * np.sqrt(2 * np.log(2)))
use_disp = sigma_s > 0.5
if use_disp:
    k_size = int(6 * sigma_s) | 1
    x = np.arange(k_size) - (k_size - 1) / 2
    kernel = np.exp(-x**2 / (2 * sigma_s**2))
    kernel = kernel / kernel.sum()
else: kernel = None

gn25l96_cfg = {'en': en_gn25l96}
if en_gn25l96:
    fr4_bw = 15e9 / max(fr4_len_in, 0.1)
    gn25l96_cfg['b_fr4'], gn25l96_cfg['a_fr4'] = signal.butter(1, 
                                                               min(fr4_bw/(SIM_FS/2), 0.99), 'low')
    
    tx_eq_lin = 10**(tx_eq_db / 20)
    if tx_eq_lin > 1.01:
        fp_tx = tx_bw_ghz * 1e9
        fz_tx = fp_tx / tx_eq_lin
        gn25l96_cfg['b_eq'], gn25l96_cfg['a_eq'] = signal.bilinear([1/(2*np.pi*fz_tx), 1], 
                                                                   [1/(2*np.pi*fp_tx), 1], 
                                                                   fs=SIM_FS)
    else:
        gn25l96_cfg['b_eq'], gn25l96_cfg['a_eq'] = None, None
        
    wc_ic = 2 * np.pi * (ic_bw_ghz * 1e9)
    wc_ic_pre = 2 * SIM_FS * np.tan(wc_ic / (2 * SIM_FS))
    b_coeff_ic = 4 * ic_zeta**2 - 2
    wn_ic = wc_ic_pre / np.sqrt((-b_coeff_ic + np.sqrt(b_coeff_ic**2 + 4)) / 2)
    gn25l96_cfg['b_interconnect'], gn25l96_cfg['a_interconnect'] = signal.bilinear([wn_ic**2], 
                                                                                   [1, 2*ic_zeta*wn_ic, wn_ic**2], 
                                                                                   fs=SIM_FS)

    rx_eq_lin = 10**(rx_eq_db / 20)
    if rx_eq_lin > 1.01:
        fp_rx = rx_eq_bw_ghz * 1e9
        fz_rx = fp_rx / rx_eq_lin
        gn25l96_cfg['b_rx_eq'], gn25l96_cfg['a_rx_eq'] = signal.bilinear([1/(2*np.pi*fz_rx), 1], 
                                                                         [1/(2*np.pi*fp_rx), 1], 
                                                                         fs=SIM_FS)
    else:
        gn25l96_cfg['b_rx_eq'], gn25l96_cfg['a_rx_eq'] = None, None

    gn25l96_cfg['b_la'], gn25l96_cfg['a_la'] = signal.butter(2, min(la_bw_ghz*1e9/(SIM_FS/2), 0.99), 'low')
    gn25l96_cfg['la_gain'] = la_gain_db
    gn25l96_cfg['la_swing'] = la_swing_mv
    gn25l96_cfg['la_noise'] = la_noise_nv

# ==========================================
# 7. 启动按钮与图表流式引擎
# ==========================================
st.markdown("---")
eye_charts_placeholder = st.empty()
bathtub_placeholder = st.empty()

if st.button("🚀 启动 10^12 比特全链路物理孪生引擎"):
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    # --- 阶段 1：获取前导码锁定参数与各节点探针 ---
    status_text.text("系统状态: 正在生成初始 Preamble 进行节点探针提取与时钟锁相...")
    
    np.random.seed(42)
    preamble_bits_raw = np.random.randint(0, 2, PREAMBLE_BITS + OVERLAP_BITS)
    if use_scrambler: preamble_bits_raw = apply_scrambler(preamble_bits_raw, prbs_order)
    
    # 【新增】开启 return_nodes 收集所有站点眼图
    preamble_v_out, preamble_nodes = process_physical_chunk(
        preamble_bits_raw, 0, spb, br, p_tx_mw, p_low_mw, k_nonlin, b_led, a_led, 
        rj_rms_ps, dj_pp_ps, rin_db, loss_factor, use_disp, kernel, resp_aw, 
        b_rx, a_rx, rf_eff, c_tot, en_nv, i_mean_est, gn25l96_cfg, return_nodes=True)
    
    # 为每个节点生成 3-UI 宽度的独立眼图数据 (包含群延时自适应中心锁定)
    CDR_SKIP = 5000
    node_eyes = {}
    for name, wave in preamble_nodes.items():
        steady = wave[CDR_SKIP * spb : -spb]
        v_mean = np.mean(steady)
        best_p = np.argmin([np.mean(np.abs(steady[p::spb] - v_mean)) for p in range(spb)])
        best_p = (best_p + spb // 2) % spb
        
        eye_data = []
        for i in range(1000, 1300):
            seg = wave[i*spb - (spb//2) + best_p : i*spb - (spb//2) + best_p + 3*spb]
            if len(seg) == 3*spb: eye_data.append(seg)
        node_eyes[name] = {'data': eye_data, 'bp': best_p}

    # 主信号 CDR 锁定
    steady_v_out = preamble_v_out[CDR_SKIP * spb : -spb]
    v_mean_main = np.mean(steady_v_out)
    best_phase = np.argmin([np.mean(np.abs(steady_v_out[p::spb] - v_mean_main)) for p in range(spb)])
    best_phase = (best_phase + spb // 2) % spb
    
    sampled_preamble = preamble_v_out[CDR_SKIP * spb + best_phase :: spb]
    tx_bits_valid = preamble_bits_raw[OVERLAP_BITS + CDR_SKIP : OVERLAP_BITS + CDR_SKIP + len(sampled_preamble)]
    
    rx_hard_coarse = (sampled_preamble > np.mean(sampled_preamble)).astype(int)
    best_delay, max_matches = 0, 0
    for delay in range(80):
        N = min(len(rx_hard_coarse) - delay, len(tx_bits_valid))
        matches = np.sum(rx_hard_coarse[delay : delay + N] == tx_bits_valid[:N])
        if matches > max_matches: max_matches, best_delay = matches, delay

    N_align = min(len(sampled_preamble) - best_delay, len(tx_bits_valid))
    aligned_rx = sampled_preamble[best_delay : best_delay + N_align]
    aligned_tx = tx_bits_valid[:N_align]

    ones, zeros = aligned_rx[aligned_tx == 1], aligned_rx[aligned_tx == 0]
    mu1, sig1, mu0, sig0 = np.mean(ones), np.std(ones), np.mean(zeros), np.std(zeros)
    q_val = (mu1 - mu0) / (sig1 + sig0 + 1e-12)
    ber_theo = 0.5 * erfc(q_val / np.sqrt(2))
    
    th_min, th_max = max(np.min(aligned_rx), mu0 - 4 * sig0), min(np.max(aligned_rx), mu1 + 4 * sig1)
    thresholds = np.linspace(th_min, th_max, 120) 

    # --- 渲染静态全链路眼图矩阵 (仅渲染一次，不消耗流式计算性能) ---
    with eye_charts_placeholder.container():
        st.markdown("### 👁️ 硬件在环：全链路节点眼图剖析 (3-UI Window)")
        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)
        cols = [r1c1, r1c2, r1c3, r2c1, r2c2, r2c3]
        
        n_keys = ['Node1_Baseband', 
                  'Node2_TxEQ', 
                  'Node3_Optical', 
                  'Node4_TIA', 
                  'Node5_RxEQ', 
                  'Node6_LA']
        n_titles = ["1. 初始基带 (含Jitter)", 
                    "2. TX EQ输出 (补偿FR4+寄生)", 
                    "3. 光纤传输 (含RIN+色散)", 
                    "4. TIA输出 (含高频热噪声)", 
                    "5. RX CTLE (强行掰开眼图)", 
                    "6. LA输出 (非线性硬切)"]
        n_units = ["Volts (V)", 
                   "Volts (V)", 
                   "Power (mW)", 
                   "Amplitude (mV)", 
                   "Amplitude (mV)", 
                   "Amplitude (mV)"]
        
        for idx, col in enumerate(cols):
            fig, ax = plt.subplots(figsize=(5, 3))
            k = n_keys[idx]
            t_axis = np.arange(3*spb) / SIM_FS * 1e12 # 转换为皮秒 (ps)
            
            for seg in node_eyes[k]['data']:
                ax.plot(t_axis, seg, color='#1f77b4', alpha=0.1)
                
            ax.set_title(n_titles[idx], fontsize=10, fontweight='bold')
            ax.set_ylabel(n_units[idx], fontsize=8)
            ax.set_xlabel("Time (ps)", fontsize=8)
            
            # 画 3 个周期的采样中心红线
            for ui in range(3):
                ax.axvline(x=(ui + 0.5)*(1/br)*1e12, 
                           color='red', 
                           linestyle='--', 
                           alpha=0.5)
                
            ax.grid(alpha=0.3)
            plt.tight_layout()
            col.pyplot(fig)
            plt.close(fig)

    # --- 阶段 2：无限流式吞吐引擎 ---
    status_text.text("系统状态: 时钟已锁定！全链路闭环，1,000,000 比特巨型块无尽吞吐中...")
    
    global_err_counts = np.zeros(len(thresholds), dtype=np.int64)
    total_bits_evaluated = 0
    final_bw_display = la_bw_ghz if en_gn25l96 else (fc_rx/1e9)
    
    def render_dashboard():
        with live_stats_placeholder.container():
            opt_errors = np.min(global_err_counts)
            ber_act = opt_errors / total_bits_evaluated if total_bits_evaluated > 0 else 0
            ber_str = f"0 (< 1/{total_bits_evaluated:.1e})" if opt_errors == 0 else f"{ber_act:.2e}"

            st.markdown(build_card("已处理总比特", f"{total_bits_evaluated:,}"), unsafe_allow_html=True)
            st.markdown(build_card("实时统计 BER", ber_str, ber_act > 1e-3), unsafe_allow_html=True)
            st.markdown(build_card("理论极限 BER", f"{ber_theo:.1e}"), unsafe_allow_html=True)
            st.markdown(build_card("判决级带宽(G)", f"{final_bw_display:.2f}"), unsafe_allow_html=True)
            st.markdown(build_card("模拟 Q因子", f"{q_val:.1f}", q_val < 6), unsafe_allow_html=True)
        
        with bathtub_placeholder.container():
            fig2, ax2 = plt.subplots(figsize=(12, 4.0))
            ber_theo_curve = [0.5*erfc((mu1-th)/(np.sqrt(2)*sig1)) + 0.5*erfc((th-mu0)/(np.sqrt(2)*sig0)) for th in thresholds]
            ax2.plot(thresholds, np.clip(ber_theo_curve, 1e-20, 1), color='#1f77b4', linestyle='--', label='Theoretical')
            plot_errors = [e/total_bits_evaluated if e > 0 else 1/total_bits_evaluated for e in global_err_counts]
            ax2.plot(thresholds, plot_errors, color='purple', lw=2, label='Live Counted BER')
            ax2.axhline(y=1/total_bits_evaluated if total_bits_evaluated>0 else 1, color='red', linestyle=':', label='Confidence Limit')
            ax2.set_yscale('log'); ax2.set_ylim(bottom=max(1e-13, 1/(total_bits_evaluated*10)), top=1) 
            ax2.grid(True, which="both", ls="--", alpha=0.4)
            ax2.set_title("Live Streaming Bathtub Curve (Decision End)", fontweight='bold')
            ax2.legend(loc='upper right', fontsize=9)
            plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    # 主循环 (关闭了 return_nodes，保证 100万比特吞吐性能极限)
    for i in range(PREAMBLE_BITS, TARGET_NUM_BITS, CHUNK_BITS):
        np.random.seed((i // CHUNK_BITS) + 777)
        raw_bits = np.random.randint(0, 2, CHUNK_BITS + OVERLAP_BITS)
        if use_scrambler: raw_bits = apply_scrambler(raw_bits, prbs_order)
        
        chunk_v_out = process_physical_chunk(
            raw_bits, i, spb, br, p_tx_mw, p_low_mw, k_nonlin, b_led, a_led, 
            rj_rms_ps, dj_pp_ps, rin_db, loss_factor, use_disp, kernel, resp_aw, 
            b_rx, a_rx, rf_eff, c_tot, en_nv, i_mean_est, gn25l96_cfg, return_nodes=False)
        
        sampled_chunk = chunk_v_out[best_phase :: spb]
        tx_bits_chunk = raw_bits[OVERLAP_BITS : OVERLAP_BITS + len(sampled_chunk)]
        
        align_len = min(len(sampled_chunk) - best_delay, len(tx_bits_chunk))
        if align_len <= 0: continue
        
        aligned_rx_chunk = sampled_chunk[best_delay : best_delay + align_len]
        aligned_tx_chunk = tx_bits_chunk[:align_len]
        
        for t_idx, th in enumerate(thresholds):
            errors = np.sum((aligned_rx_chunk > th).astype(int) != aligned_tx_chunk)
            global_err_counts[t_idx] += errors
            
        total_bits_evaluated += align_len
        
        progress_val = min(total_bits_evaluated / TARGET_NUM_BITS, 1.0)
        progress_bar.progress(progress_val)
        render_dashboard()
        time.sleep(0.01)