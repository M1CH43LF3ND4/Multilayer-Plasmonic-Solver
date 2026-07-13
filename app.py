import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(page_title="Simulasi TMM Fresnel", layout="wide")
st.title("Simulasi Fresnel Multilayer (TMM)")

# ==========================================
# 1. UI: SIDEBAR UNTUK INPUT PARAMETER
# ==========================================
st.sidebar.header("⚙️ Parameter Input")
pol = st.sidebar.radio("Mode Polarisasi", ('P-Pol (TM)', 'S-Pol (TE)'))

n_input = st.sidebar.text_input("n_array (pisahkan koma)", "1.0, 1.45, 1.50")
k_input = st.sidebar.text_input("k_array (pisahkan koma)", "0.0, 0.00, 0.20")
d_input = st.sidebar.text_input("d_array dalam nm (pisahkan koma)", "0, 300, 0")

lam = st.sidebar.slider("Panjang Gelombang (nm)", 300.0, 1000.0, 500.0)
th_choice = st.sidebar.slider("Sudut Datang (°)", 0.0, 89.9, 0.0)
E0 = st.sidebar.number_input("Amplitudo E0 (V/m)", value=100.0)

# Parsing data input dari string ke numpy array
try:
    n_vals = np.array([float(x.strip()) for x in n_input.split(',')])
    k_vals = np.array([float(x.strip()) for x in k_input.split(',')])
    d_vals = np.array([float(x.strip()) for x in d_input.split(',')])
    n_c = n_vals + 1j * k_vals
except ValueError:
    st.sidebar.error("Pastikan format angka benar dan dipisahkan dengan koma.")
    st.stop()

# ==========================================
# 2. FUNGSI FISIKA (TMM N-LAYER GENERAL)
# ==========================================
# Letakkan fungsi get_max_snapshot() milikmu di sini...
def get_max_snapshot(comp_arr):
    if np.max(np.abs(comp_arr)) < 1e-12: 
        return np.zeros_like(comp_arr, dtype=float)
    flat_arr = comp_arr.ravel()
    peak_idx = np.argmax(np.abs(flat_arr))
    opt_phase = np.angle(flat_arr[peak_idx])
    return np.real(comp_arr * np.exp(-1j * opt_phase))

# Letakkan fungsi calc_tmm_global() milikmu secara UTUH di sini...
def calc_tmm_global(pol, n_c, d_nm, th_arr_rad, lam):
    num_layers = len(n_c)
    kx = n_c[0] * np.sin(th_arr_rad).astype(complex)
    
    cos_th = np.zeros((num_layers, len(th_arr_rad)), dtype=complex)
    kz = np.zeros((num_layers, len(th_arr_rad)), dtype=complex)
    for j in range(num_layers):
        cos_th[j] = np.sqrt(1 - (kx / n_c[j])**2)
        kz[j] = (2 * np.pi / lam) * n_c[j] * cos_th[j]
        
    M11, M12 = np.ones_like(th_arr_rad, dtype=complex), np.zeros_like(th_arr_rad, dtype=complex)
    M21, M22 = np.zeros_like(th_arr_rad, dtype=complex), np.ones_like(th_arr_rad, dtype=complex)
    
    for j in range(num_layers - 1):
        if pol == 'S-Pol (TE)':
            rj = (n_c[j]*cos_th[j] - n_c[j+1]*cos_th[j+1]) / (n_c[j]*cos_th[j] + n_c[j+1]*cos_th[j+1])
            tj = (2 * n_c[j]*cos_th[j]) / (n_c[j]*cos_th[j] + n_c[j+1]*cos_th[j+1])
        else:
            rj = (n_c[j+1]*cos_th[j] - n_c[j]*cos_th[j+1]) / (n_c[j+1]*cos_th[j] + n_c[j]*cos_th[j+1])
            tj = (2 * n_c[j]*cos_th[j]) / (n_c[j+1]*cos_th[j] + n_c[j]*cos_th[j+1])
            
        with np.errstate(divide='ignore', invalid='ignore'):
            nM11 = (M11 + M12 * rj) / tj
            nM12 = (M11 * rj + M12) / tj
            nM21 = (M21 + M22 * rj) / tj
            nM22 = (M21 * rj + M22) / tj
        
        if j < num_layers - 2:
            delta = kz[j+1] * d_nm[j+1]
            P11, P22 = np.exp(-1j * delta), np.exp(1j * delta)
            M11, M12 = nM11 * P11, nM12 * P22
            M21, M22 = nM21 * P11, nM22 * P22
        else:
            M11, M12, M21, M22 = nM11, nM12, nM21, nM22
            
    with np.errstate(divide='ignore', invalid='ignore'):
        r_tot = M21 / M11
        t_tot = 1 / M11
        R_pow = np.abs(r_tot)**2
        factor = np.real(n_c[-1] * cos_th[-1]) / np.real(n_c[0] * cos_th[0])
        T_pow = np.abs(t_tot)**2 * factor
        
    R_pow = np.nan_to_num(R_pow, nan=1.0)
    T_pow = np.nan_to_num(T_pow, nan=0.0)
    A_pow = np.clip(1.0 - R_pow - T_pow, 0, 1)
    
    # BARIS INI YANG KEMUNGKINAN HILANG SEBELUMNYA
    return R_pow, T_pow, A_pow, r_tot, t_tot
# ==========================================
# 3. PERHITUNGAN TMM 1D (FRAKSI DAYA & AMPLITUDO)
# ==========================================
theta_array_deg = np.linspace(0, 89.9, 500)
theta_array_rad = np.radians(theta_array_deg)

# Kalkulasi 1D
R_arr, T_arr, A_arr, r_arr, t_arr = calc_tmm_global(pol, n_c, d_vals, theta_array_rad, lam)

# Menghitung Sudut Brewster & Kritis untuk Info Text
n0 = np.real(n_c[0])
n1 = np.real(n_c[1])
n_sub = np.real(n_c[-1])
th_B_1 = np.degrees(np.arctan(n1 / n0))
th_B_sub = np.degrees(np.arctan(n_sub / n0))
str_angles = f"★ Sudut Brewster (Datang -> Layer 1): {th_B_1:.2f}° | Sudut Brewster (Substrat): {th_B_sub:.2f}°"
if n0 > n_sub:
    th_C_sub = np.degrees(np.arcsin(n_sub / n0))
    str_angles += f" | Sudut Kritis: {th_C_sub:.2f}°"

st.info(str_angles)

# Layout Kolom Web untuk Grafik 1D
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(theta_array_deg, R_arr, color='blue', lw=2, label='Reflektansi ($R$)')
    ax1.plot(theta_array_deg, T_arr, color='red', lw=2, label='Transmitansi ($T$)')
    ax1.plot(theta_array_deg, A_arr, color='green', lw=2, label='Absorptansi ($A$)')
    ax1.axvline(x=th_choice, color='gray', linestyle=':')
    ax1.set_title('Fraksi Daya Multilayer (R, T, A)', fontweight='bold')
    ax1.set_xlabel('Sudut Datang (derajat)'); ax1.set_ylabel('Fraksi Daya')
    ax1.set_ylim(-0.05, 1.05); ax1.set_xlim(0, 90)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(theta_array_deg, np.real(r_arr * E0), color='blue', lw=2, label='Pantul ($E_r$)')
    ax2.plot(theta_array_deg, np.real(t_arr * E0), color='red', linestyle='--', lw=2, label='Tembus ($E_t$)')
    ax2.axvline(x=th_choice, color='gray', linestyle=':')
    ax2.axhline(0, color='black', lw=1)
    ax2.set_title('Amplitudo Listrik', fontweight='bold')
    ax2.set_xlabel('Sudut Datang (derajat)'); ax2.set_ylabel('Amplitudo (V/m)')
    ax2.set_ylim(-E0*1.1, E0*1.1); ax2.set_xlim(0, 90)
    ax2.legend(); ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# ==========================================
# 4. TMM BACKPROPAGATION 2D (PROFIL MEDAN)
# ==========================================
num_layers = len(n_c)
total_thickness = np.sum(d_vals[1:-1]) / 1000.0
z_max = max(2.5, total_thickness + 1.0)
x_um = np.linspace(-1.5, 1.5, 120)
z_um = np.linspace(-1.5, z_max, 180) 
X, Z = np.meshgrid(x_um, z_um)

d_um = d_vals / 1000.0
z_bounds = [0.0]
for i in range(1, num_layers - 1):
    z_bounds.append(z_bounds[-1] + d_um[i])

th_single = np.radians(th_choice)
kx = n_c[0] * np.sin(th_single)
cos_th_s = np.zeros(num_layers, dtype=complex)
kz_s = np.zeros(num_layers, dtype=complex)

for j in range(num_layers):
    cos_th_s[j] = np.sqrt(1 - (kx / n_c[j])**2)
    kz_s[j] = (2 * np.pi / lam) * n_c[j] * cos_th_s[j]
    
r_j, t_j = np.zeros(num_layers-1, dtype=complex), np.zeros(num_layers-1, dtype=complex)
for j in range(num_layers - 1):
    if pol == 'S-Pol (TE)':
        r_j[j] = (n_c[j]*cos_th_s[j] - n_c[j+1]*cos_th_s[j+1]) / (n_c[j]*cos_th_s[j] + n_c[j+1]*cos_th_s[j+1])
        t_j[j] = (2 * n_c[j]*cos_th_s[j]) / (n_c[j]*cos_th_s[j] + n_c[j+1]*cos_th_s[j+1])
    else: 
        r_j[j] = (n_c[j+1]*cos_th_s[j] - n_c[j]*cos_th_s[j+1]) / (n_c[j+1]*cos_th_s[j] + n_c[j]*cos_th_s[j+1])
        t_j[j] = (2 * n_c[j]*cos_th_s[j]) / (n_c[j+1]*cos_th_s[j] + n_c[j]*cos_th_s[j+1])
        
_, _, _, r_single, t_single = calc_tmm_global(pol, n_c, d_vals, np.array([th_single]), lam)
v, w = np.zeros(num_layers, dtype=complex), np.zeros(num_layers, dtype=complex)
v[-1], w[-1] = E0 * t_single[0], 0

for j in range(num_layers - 2, -1, -1):
    with np.errstate(divide='ignore', invalid='ignore'):
        v_end = (v[j+1] + r_j[j] * w[j+1]) / t_j[j]
        w_end = (r_j[j] * v[j+1] + w[j+1]) / t_j[j]
    if j > 0:
        delta = kz_s[j] * d_vals[j]
        v[j] = v_end * np.exp(-1j * delta)
        w[j] = w_end * np.exp(1j * delta)
    else:
        v[j], w[j] = v_end, w_end

Ex = np.zeros_like(X, dtype=complex); Ey = np.zeros_like(X, dtype=complex)
Ez = np.zeros_like(X, dtype=complex); Hy = np.zeros_like(X, dtype=complex)
Z0 = 376.7303
kx_um = (2 * np.pi / (lam / 1000.0)) * n_c[0] * np.sin(th_single)
kz_um = kz_s * 1000.0 

U_prop = np.zeros_like(X, dtype=float); V_prop = np.zeros_like(X, dtype=float)

for j in range(num_layers):
    if j == 0:
        mask = Z < z_bounds[0]
        z_loc = Z[mask] - z_bounds[0]
    elif j == num_layers - 1:
        mask = Z >= z_bounds[-1]
        z_loc = Z[mask] - z_bounds[-1]
    else:
        mask = (Z >= z_bounds[j-1]) & (Z < z_bounds[j])
        z_loc = Z[mask] - z_bounds[j-1]
        
    phase_v = np.exp(1j * kz_um[j] * z_loc)
    phase_w = np.exp(-1j * kz_um[j] * z_loc)
    fasa_x = np.exp(1j * kx_um * X[mask])
    
    if pol == 'P-Pol (TM)':
        Ex[mask] = (v[j] * cos_th_s[j] * phase_v - w[j] * cos_th_s[j] * phase_w) * fasa_x
        st_j = n_c[0] * np.sin(th_single) / n_c[j]
        Ez[mask] = (-v[j] * st_j * phase_v - w[j] * st_j * phase_w) * fasa_x
        Hy[mask] = (n_c[j] / Z0) * (v[j] * phase_v + w[j] * phase_w) * fasa_x
    else:
        Ey[mask] = (v[j] * phase_v + w[j] * phase_w) * fasa_x
        Hy[mask] = (-n_c[j] * cos_th_s[j] / Z0) * (v[j] * phase_v - w[j] * phase_w) * fasa_x
        
    if np.imag(cos_th_s[j]) == 0:
        uj = np.real(n_c[0] * np.sin(th_single) / n_c[j])
        vj = -np.real(cos_th_s[j])
        norm = np.sqrt(uj**2 + vj**2)
    else:
        uj, vj, norm = 1.0, 0.0, 1.0
    U_prop[mask] = uj / norm; V_prop[mask] = vj / norm

Ex_real, Ey_real = get_max_snapshot(Ex), get_max_snapshot(Ey)
Ez_real, Hy_real = get_max_snapshot(Ez), get_max_snapshot(Hy)

vmax_val = max(np.max(np.abs(Ex_real)), np.max(np.abs(Ey_real)), np.max(np.abs(Ez_real)))
if vmax_val == 0: vmax_val = E0 if E0 > 0 else 1e-10 
vmax_h = 0.6 * (E0 / 100.0) if E0 > 0 else 1e-10 

# ==========================================
# 5. PLOTTING MEDAN 2D KE STREAMLIT
# ==========================================
st.markdown("### Profil Medan Elektromagnetik 2D")
# Membagi halaman menjadi 4 kolom untuk grafik 2D
col3, col4, col5, col6 = st.columns(4)
step = 15 

def setup_2d_plot(ax, title, z_bounds):
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Posisi X ($\mu$m)')
    ax.set_ylabel('Kedalaman Z ($\mu$m)')
    ax.set_ylim(z_max, -1.5)
    ax.set_xlim(-1.5, 1.5)
    for b_z in z_bounds:
        ax.axhline(b_z, color='white', linestyle='--', lw=1.5, alpha=0.9)

with col3:
    fig_ex, ax_ex = plt.subplots(figsize=(4, 6))
    mesh_ex = ax_ex.pcolormesh(X, Z, Ex_real, cmap='jet', shading='gouraud', vmin=-vmax_val, vmax=vmax_val)
    if pol == 'P-Pol (TM)':
        ax_ex.quiver(X[::step, ::step], Z[::step, ::step], U_prop[::step, ::step], V_prop[::step, ::step], color='red', pivot='mid', scale=20, alpha=0.9)
    setup_2d_plot(ax_ex, 'Profil Medan $E_x$', z_bounds)
    st.pyplot(fig_ex)

with col4:
    fig_ey, ax_ey = plt.subplots(figsize=(4, 6))
    mesh_ey = ax_ey.pcolormesh(X, Z, Ey_real, cmap='jet', shading='gouraud', vmin=-vmax_val, vmax=vmax_val)
    setup_2d_plot(ax_ey, 'Profil Medan $E_y$', z_bounds)
    st.pyplot(fig_ey)

with col5:
    fig_ez, ax_ez = plt.subplots(figsize=(4, 6))
    mesh_ez = ax_ez.pcolormesh(X, Z, Ez_real, cmap='jet', shading='gouraud', vmin=-vmax_val, vmax=vmax_val)
    if pol == 'P-Pol (TM)':
        ax_ez.quiver(X[::step, ::step], Z[::step, ::step], U_prop[::step, ::step], V_prop[::step, ::step], color='red', pivot='mid', scale=20, alpha=0.9)
    setup_2d_plot(ax_ez, 'Profil Medan $E_z$', z_bounds)
    fig_ez.colorbar(mesh_ez, ax=ax_ez, fraction=0.046, pad=0.04).set_label('Amplitudo E (V/m)')
    st.pyplot(fig_ez)

with col6:
    fig_h, ax_h = plt.subplots(figsize=(4, 6))
    mesh_h = ax_h.pcolormesh(X, Z, Hy_real, cmap='jet', shading='gouraud', vmin=-vmax_h, vmax=vmax_h)
    title_h = 'Profil Medan $H_y$' if pol == 'P-Pol (TM)' else 'Profil Medan $H_x$'
    setup_2d_plot(ax_h, title_h, z_bounds)
    fig_h.colorbar(mesh_h, ax=ax_h, fraction=0.046, pad=0.04).set_label('Amplitudo H (A/m)')
    st.pyplot(fig_h)

# ==========================================
# 6. TOMBOL DOWNLOAD DATA 
# ==========================================
# Streamlit memiliki widget khusus untuk mengunduh file
st.markdown("---")
if st.button("Kalkulasi & Siapkan Data Ekstrak (TM & TE)"):
    R_tm, T_tm, A_tm, Er_tm, Et_tm = calc_tmm_global('P-Pol (TM)', n_c, d_vals, theta_array_rad, lam)
    R_te, T_te, A_te, Er_te, Et_te = calc_tmm_global('S-Pol (TE)', n_c, d_vals, theta_array_rad, lam)
    
    data_matrix = np.column_stack((
        theta_array_deg, 
        R_tm, T_tm, A_tm, np.real(Er_tm * E0), np.real(Et_tm * E0),
        R_te, T_te, A_te, np.real(Er_te * E0), np.real(Et_te * E0)
    ))
    
    header_text = 'Sudut(deg)\tR_TM\tT_TM\tA_TM\tEr_TM\tEt_TM\tR_TE\tT_TE\tA_TE\tEr_TE\tEt_TE'
    # Simpan ke memori sebagai string untuk didownload
    import io
    csv_buffer = io.BytesIO()
    np.savetxt(csv_buffer, data_matrix, fmt='%.6f', delimiter='\t', header=header_text, comments='')
    
    st.download_button(
        label="📥 Download File Data_Fresnel_TMM_Lengkap.txt",
        data=csv_buffer.getvalue(),
        file_name="Data_Fresnel_TMM_Lengkap.txt",
        mime="text/plain"
    )