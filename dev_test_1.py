#%%
import warnings
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import statsmodels.api as sm
import xarray as xr
import seaborn as sns
import pytensor.tensor as pt
from pymc.variational.callbacks import CheckParametersConvergence
from pymc.variational.callbacks import Callback

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#%%
# change the working directory to the parent directory of the current file
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir(r"/Users/qian/Desktop/hiwi/luna/luna_modelling")

path = r"/Users/qian/Desktop/hiwi/luna/luna_modelling"
#%%
tx = pd.read_csv(f"{path}/data/data_scales/TX/tx_long.csv")

# T0 data (One off tests)
aist  = pd.read_csv(f"{path}/data/data_scales/T0/aist.csv")         # AIST/General Interest Scale
bfi   = pd.read_csv(f"{path}/data/data_scales/T0/bfi.csv")          # BFI/Big-5 Personality
bg    = pd.read_csv(f"{path}/data/data_scales/T0/background.csv")   # Background data
T0_data= pd.read_csv(f"{path}/data/data_scales/T0/iq.csv")           # IQ Test
kont  = pd.read_csv(f"{path}/data/data_scales/T0/kont.csv")         # Internal and External Controll
math  = pd.read_csv(f"{path}/data/data_scales/T0/math_test.csv")    # Maths ability
mot   = pd.read_csv(f"{path}/data/data_scales/T0/motivation.csv")   # Motivation questionnaire
panas = pd.read_csv(f"{path}/data/data_scales/T0/panas.csv")        # Positive and Negative Affect Scale
tx    = pd.read_csv(f"{path}/data/data_scales/TX/tx_long.csv")

# data grouping
tx_event0 = tx[tx['event'] == 0]
# the proportion of the number of rows for each tx variable is NaN by the number of tx rows
tx_event0.isnull().sum() / tx_event0.shape[0]
# proportion larger than 0.5
tx_event0.columns[tx_event0.isnull().sum() / tx_event0.shape[0] > 0.5]

# change studentID: student1 -> 1
tx_event0['studentID'] = tx_event0['studentID'].str.replace('student', '').astype(int)

# use linear interpolation to fill in the missing values tx_event0 NAN
tx_event0 = tx_event0.interpolate(method='linear')

observed_vars = [
    'Iv1_state', 'Av1_state', 'Uv1_state', 'Co1_state', 'Co2_state',
    'Angst_abbruch_state', 'Angst_scheitern_state', 'Leist_verstehen_state',
    'Leist_bearbeiten_state', 'Leist_stress_state', 'Leist_ueberfordert_state',
    'Wiss_kommilitonen_state', 'Wiss_mathe_state', 'PANP01_state', 'PANP05_state',
    'PANP08_state', 'PANN01_state', 'PANN05_state', 'PANN09_state'
]
# select columns of tx_event0: meas, studentID, observed_vars, event
tx_event0 = tx_event0[['meas', 'studentID', 'event'] + observed_vars]
tx_event0.head()
data= pd.read_csv(f"{path}/data/final_merged_data.csv")
T0_data = data.bfill()
print(T0_data.isnull().sum())
# change studentID: student1 -> 1
tx['studentID'] = tx['studentID'].str.replace('student', '').astype(int)

# Identify students who had at least one event=1 at any meas period
dropout_students = tx[tx['event'] == 1]['studentID'].unique()

# Create a DataFrame with unique student IDs and dropout status
dropout_status = pd.DataFrame({'studentID': tx['studentID'].unique()})
dropout_status['dropout'] = dropout_status['studentID'].isin(dropout_students).astype(int)

dropout_status.head()

# add dropout status in T0_data as new two columns
T0_data['studentID'] = dropout_status['studentID']
T0_data['dropout'] = dropout_status['dropout']
T0_data.head()
# group by time
groups_event0_by_time = tx_event0.groupby('meas')

datasets_event0 = {} 

# for each time, group in groups_event0_by_time
for time, group in groups_event0_by_time:
    datasets_event0[time] = group[observed_vars]

    
#%%
import pymc as pm
import numpy as np

unique_students = T0_data["studentID"].unique().astype("int64")
n_students = len(unique_students)
N = 5

coords = {
    "obs": list(range(len(tx_event0))), 
    "students": unique_students,         
    "indicators": [
        'Iv1_state', 'Av1_state', 'Uv1_state', 'Co1_state', 'Co2_state',
        'Angst_abbruch_state', 'Angst_scheitern_state', 'Leist_verstehen_state',
        'Leist_bearbeiten_state', 'Leist_stress_state', 'Leist_ueberfordert_state',
        'Wiss_kommilitonen_state', 'Wiss_mathe_state', 'PANP01_state', 'PANP05_state',
        'PANP08_state', 'PANN01_state', 'PANN05_state', 'PANN09_state'
    ],
    "indicators_1": ['Iv1_state', 'Av1_state', 'Uv1_state', 'Leist_verstehen_state', 'Leist_bearbeiten_state'],
    "indicators_2": ['Co1_state', 'Co2_state'],
    "indicators_3": ['Angst_abbruch_state', 'Angst_scheitern_state'],
    "indicators_4": ['Leist_stress_state', 'Leist_ueberfordert_state'],
    "indicators_5": ['PANP01_state', 'PANP05_state', 'PANP08_state'],
    "indicators_6": ['PANN01_state', 'PANN05_state', 'PANN09_state'],
    "indicators_7": ['Wiss_kommilitonen_state', 'Wiss_mathe_state'],
    "latent_1": [f"Factor{i+1}" for i in range(7)],
    "between_indicators": ['aist_a', 'bfi_of', 'iq'],
    "between_indicators_8": ['aist_a', 'bfi_of'],
    "between_indicators_9": ['iq'],
    "latent_2": ["iq"],
}

iq_sd = np.std(T0_data["iq"].values.astype("float64"),ddof=1)
with pm.Model(coords=coords) as model:
    # Latent variables η₂i
    eta_2 = pm.Normal("eta_2", mu=0, sigma=iq_sd, dims=("students", "latent_2"))

    # Factor loading matrix Λ₂
    lambda_2 = pm.TruncatedNormal(
        "lambda_2", mu=0, sigma=1, lower=0, dims=("between_indicators_9", "latent_2")
    )

    # Residual standard deviations
    sigma_epsilon_2j = pm.Gamma("sigma_epsilon_2j", alpha=9, beta=4, dims="between_indicators_9")
    #sigma_epsilon_2j = 0
    # Model-predicted mean
    mu_y2i = pm.Deterministic(
        "mu_y2i",
        pm.math.dot(lambda_2, eta_2.T).T,
        dims=("students", "between_indicators_9")
    )

    # Likelihood
    _ = pm.Normal(
        "Y2_likelihood",
        mu=mu_y2i,
        sigma=sigma_epsilon_2j,
        observed=T0_data[coords["between_indicators_9"]].values.astype("float64")  
    )

    # Sampling
    idata = pm.sample(
        draws=500,
        tune=500,
        chains=1,
        target_accept=0.9,
        idata_kwargs={"log_likelihood": True}
    )

    # Posterior predictive
    idata.extend(pm.sample_posterior_predictive(idata))

#pm.model_to_graphviz(model)
model
eta_2_samples = idata.posterior["eta_2"]  # shape: (chains, draws, students, latent_2)
eta_2_samples.var().astype('float64')  
#%%print(eta_2_samples.isel(chain=1, draw=1, students=1, latent_2=0).values.item())
with pm.Model(coords=coords) as model:
    alpha_21j_s1 = pm.Normal('alpha_21j_s1', mu=0, sigma=1, shape=7)
    beta_2j_s1 = pm.TruncatedNormal('beta_2j_s1', mu=0, sigma=1, lower=0, shape=7)  # 7×1
    beta_1j_s1 = pm.Normal('beta_1j_s1', mu=0, sigma=1, shape=7)
    omega_2j_s1 = pm.Normal('omega_2j_s1', mu=0, sigma=1, shape=7)  # 7×1
    # Deltas for dropout students s=2
    delta_alpha_21j_s2 = pm.TruncatedNormal('delta_alpha_21j_s2', mu=0, sigma=1, lower=0, shape=7)
    delta_beta_2j_s2 = pm.Normal('delta_beta_2j_s2', mu=0, sigma=1, shape=7)
    delta_beta_1j_s2 = pm.Normal('delta_beta_1j_s2', mu=0, sigma=1, shape=7)
    delta_omega_2j_s2 = pm.Normal('delta_omega_2j_s2', mu=0, sigma=1, shape=7)
    # Define dropout version (state s=2) parameters
    alpha_21j_s2 = pm.Deterministic('alpha_21j_s2', alpha_21j_s1 + delta_alpha_21j_s2)
    beta_2j_s2 = pm.Deterministic('beta_2j_s2', beta_2j_s1 + delta_beta_2j_s2)
    beta_1j_s2 = pm.Deterministic('beta_1j_s2', beta_1j_s1 + delta_beta_1j_s2)
    omega_2j_s2 = pm.Deterministic('omega_2j_s2', omega_2j_s1 + delta_omega_2j_s2)
     # === Between-level Structural Model ===
    # Latent disturbance term ζ₂i ~ N(0, σ²)
    sigma_zeta_2j = pm.Gamma('sigma_zeta_2j', alpha=9, beta=4, shape=7)
    zeta_2i = pm.Normal('zeta_2i', mu=0, sigma=sigma_zeta_2j, shape=7)

    # α_1is = α_21s + β_2s × η_2i + ζ_2i
    eta_2_matrix = pm.Normal('eta_2_matrix', mu=0, sigma=eta_2_samples.var().astype('float64').item(), shape=(n_students, 1))  # N × 7
    alpha_1is_s1 = pm.Deterministic(
        'alpha_1is_s1',
        alpha_21j_s1[None, :] + pt.dot(eta_2_matrix, beta_2j_s1[None,:]) + zeta_2i
    )

    alpha_1is_s2 = pm.Deterministic(
        'alpha_1is_s2',
        alpha_21j_s2[None, :] + pt.dot(eta_2_matrix, beta_2j_s2[None,:]) + zeta_2i
    )

    # AR(1) matrix for each latent η₁it: B_1is = B_1s + Ω_2s × η_2i
    Omega_2s = pm.Normal('Omega_2s', mu=0, sigma=1, shape=(7, 7))  # Each latent regressed on eta_2i
    B1s= pm.Deterministic(
        'B1s', pt.diag(beta_1j_s1))
    
    # For state 1 and state 2
    # AR(1) autoregressive coefficient vector for s = 1
    B_1is_s1 = pm.Deterministic(
        'B_1is_s1',
        B1s + Omega_2s.T*eta_2_matrix[:,None]  # shape = (N, 7)
    )

    # AR(1) autoregressive coefficient vector for s = 2
    B_1is_s2 = pm.Deterministic(
        'B_1is_s2',
        B1s + Omega_2s.T*eta_2_matrix[:,None]  # shape = (N, 7)
    )
    trace = pm.sample(
        draws         = 500,       # 保存 2 k 样本
        tune          = 500,       # 1 k 预热
        chains        = 1,
        target_accept = 0.9,         # 减少 divergence
        random_seed   = 42,
        return_inferencedata=True, 
        progressbar= True,  # 关键：拿到 xarray 格式
        expectation_verbosity="high"
    )


print(az.summary(trace, var_names=["alpha_1is_s1", "B_1is_s2"]).head())

alpha_1is_s1_mean = trace.posterior["alpha_1is_s1"].mean(dim=("chain", "draw")).values
alpha_1is_s2_mean = trace.posterior["alpha_1is_s2"].mean(dim=("chain", "draw")).values
B_1is_s2_mean     = trace.posterior["B_1is_s2"].mean(dim=("chain", "draw")).values
np.save("alpha_1is_s1_mean.npy", alpha_1is_s1_mean)
np.save("B_1is_s2_mean.npy",     B_1is_s2_mean)
# %%
within_groups = {
    'cog_load':   ['Iv1_state','Av1_state','Uv1_state','Leist_verstehen_state','Leist_bearbeiten_state'],
    'control':    ['Co1_state','Co2_state'],
    'fear':       ['Angst_abbruch_state','Angst_scheitern_state'],
    'stress':     ['Leist_stress_state','Leist_ueberfordert_state'],
    'pos_aff':    ['PANP01_state','PANP05_state','PANP08_state'],
    'neg_aff':    ['PANN01_state','PANN05_state','PANN09_state'],
    'knowledge':  ['Wiss_kommilitonen_state','Wiss_mathe_state']}


with pm.Model(coords=coords) as within_model:
        n_f = len(within_groups)
        # Markov-switching params
        gamma1 = pm.Normal("gamma1", mu=0, sigma=1)
        gamma2 = pm.Normal("gamma2", mu=0, sigma=1)
        gamma3 = pm.Normal("gamma3", mu=0, sigma=1, shape=n_f, dims="latent")
        gamma4 = pm.Normal("gamma4", mu=0, sigma=1, shape=n_f, dims="latent")
        P12    = pm.Beta("P12", alpha=1, beta=9)
        
        S_prev  = pm.ConstantData("S0", np.zeros(len(coords['students']), dtype=int))
        eta_prev = pm.Normal("eta_prev", mu=0, sigma=1, shape=(len(coords['students']), n_f), dims=("students","latent"))
        sigma_zeta_1j = pm.Gamma("sigma_zeta_1j", alpha=9, beta=4, shape=n_f, dims="latent")
        eta2 =trace.posterior['eta_2_matrix'].astype('float64').mean(dim=("chain", "draw")).values
        for t in range(1, max(datasets_event0.keys())+1):
            # transition
            v_it = gamma1 + gamma2*eta2 + pt.dot(eta_prev, gamma3) + pt.dot(eta_prev*eta2[:,None], gamma4)
            p11 = pt.sigmoid(v_it) 
            transition_prob = pt.switch(
                pt.eq(S_prev, 0),
                p11,       # P(保持无倾向)
                1 - P12    # P(从有倾向转回)
            )
            S_t = pm.Bernoulli(f"S_{t}", p=transition_prob, shape=(n_students,))
            # S_t = pm.Bernoulli(f"S_{t}", p=pt.where(pt.eq(S_prev, 0), p11, 1-P12), shape=(117,))
            # state-dependent intercept & AR
            alpha_t = alpha_1is_s1_mean
            B_t     = B_1is_s2_mean
            # update latent
            eta_t   = pm.Normal(f"eta_{t}", mu=alpha_t + B_t*eta_prev, sigma=sigma_zeta_1j, dims=("students","latent"))
            # measurement
            lam_vars = [pm.TruncatedNormal(f"lambda_{fac}_{t}", mu=1, sigma=0.5, lower=0, shape=len(items))
                        for fac, items in within_groups.items()]
            lam_stack = pt.concatenate(lam_vars)
            mu_y      = pt.dot(eta_t, lam_stack)
            eps       = pm.Gamma(f"eps_{t}", alpha=9, beta=4, shape=(117,7,1))
            pm.Normal(f"Y_{t}", mu=mu_y, sigma=eps,
                        observed=datasets_event0[t], dims=("students","indicators"))
            S_prev    = S_t
            eta_prev  = eta_t

# %%
available_times = sorted(datasets_event0.keys())
print(f"Model will use these time points: {available_times}")

with pm.Model(coords=coords) as within_model:
    eps = pm.Gamma("eps", alpha=9, beta=4, dims="indicators")
    n_f = len(within_groups)
    # Markov-switching params
    gamma1 = pm.Normal("gamma1", mu=0, sigma=1)
    gamma2 = pm.Normal("gamma2", mu=0, sigma=1)
    gamma3 = pm.Normal("gamma3", mu=0, sigma=1, shape=n_f, dims="latent")
    gamma4 = pm.Normal("gamma4", mu=0, sigma=1, shape=n_f, dims="latent")
    P12    = pm.Beta("P12", alpha=1, beta=9)
        
    S_prev  = pm.Data("S0", np.zeros(len(coords['students']), dtype=int))
    eta_prev = pm.Normal("eta_prev", mu=0, sigma=1, shape=(len(coords['students']), n_f), dims=("students","latent"))
    sigma_zeta_1j = pm.Gamma("sigma_zeta_1j", alpha=9, beta=4, shape=n_f, dims="latent")
    eta2 =trace.posterior['eta_2_matrix'].astype('float64').mean(dim=("chain", "draw")).values    
    for t in available_times[0:1]:
        # ===== 修正1：转换概率计算 =====
        # 方法1：使用flatten()
        eta2_flat = eta2.flatten()  # 形状 (117,)
        
        # 方法2：使用reshape（正确写法）
        gamma2_reshaped = gamma2.reshape((-1,))  # 注意双重括号
        
        # ===== 修正2：转换概率计算 =====
        v_it = (
            gamma1
            + gamma2_reshaped * eta2_flat  # (117,)*(117,)
            + pt.sum(eta_prev * gamma3, axis=1)  # (117,7)*(7,) -> (117,)
            + pt.sum(
                eta_prev[:, [1,2,3]] * eta2_flat[:,None] * gamma4,  # (117,3)*(117,1)*(3,)
                axis=1
            )
        )
        p11 = pt.sigmoid(v_it)  # 形状 (117,)
        
        # ===== 修正2：状态转换 =====
        transition_prob = pt.switch(
            pt.eq(S_prev, 0),
            p11,          # (117,)
            pt.full_like(p11, 1-P12)  # 显式广播
        )
        S_t = pm.Bernoulli(
            f"S_{t}", 
            p=transition_prob,
            dims="students"  # 明确维度
        )
        
        # ===== 动态因子更新 =====
        # 使用状态依赖参数
        alpha_t = pt.switch(
            S_t[:,None], 
            alpha_1is_s2_mean, 
            alpha_1is_s1_mean
        )  # (117,7)
        
        eta_t = pm.Normal(
            f"eta_{t}",
            mu=alpha_t + (B_1is_s2_mean * eta_prev[...,None]).sum(axis=1),
            sigma=sigma_zeta_1j,
            dims=("students", "latent")
        )
        
        # ===== 测量模型 =====
        lam_vars = [
            pm.TruncatedNormal(
                f"lambda_{fac}_{t}", 
                mu=1, sigma=0.5, lower=0, 
                shape=len(items), 
                dims=["indicators"]
            ) for fac, items in within_groups.items()
        ]
        lam_stack = pt.concatenate(lam_vars, axis=0)  # (n_indicators,)
        
        mu_y = pt.dot(eta_t, lam_stack[None,:])  # (117,7) x (7,n_indicators)
        pm.Normal(
            f"Y_{t}",  # 使用实际时间命名
            mu=mu_y,
            sigma=eps,
            observed=datasets_event0[t],  # 直接使用原始时间键
            dims=("students", "indicators")
        )
        
        # 更新状态
        S_prev, eta_prev = S_t, eta_t
# %%
with within_model:
    # 采样参数建议
    trace = pm.sample(
        draws=500,          # 每个链的采样次数
        tune=500,           # 预热迭代次数
        chains=1,            # 并行链数
        target_accept=0.9,  # 更高的接受率适合复杂模型
        random_seed=42,      # 可重复性
        return_inferencedata=True, 
        progressbar= True,  # 关键：拿到 xarray 格式
        expectation_verbosity="high"
    )

# %%
