#!pip install pysd netCDF4
import pysd, os
import seaborn as sns
from scipy.optimize import minimize, minimize_scalar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filelock import FileLock

def calyear(kappa, theta):
    e = 1+4*theta
    return 12/(kappa*(e+0.001))

def plotlines(tar, model, pb, ax, title, given = [0, 2, 8, 10]):
    
    def runmodel(x, var='alpha'):
        params = pb.copy()
        params[var] = x
        model.reload()
        r = model.run(
                initial_condition='original',
                params=params,
                return_columns=[tar])
        return r
    
    #x = range(37)
    #x = range(t+1)
    
    lb = [round(0.1*i,2) for i in given]
    ys = [runmodel(i)[tar] for i in lb]
    gn = r'$\alpha$'

    x = range(len(ys[0]))


    df = pd.DataFrame({
        "month": np.tile(x, len(lb)),
        tar: np.concatenate(ys),
        gn: np.repeat(lb, len(x))
    })
    
    df[gn] = pd.Categorical([v for v in df[gn]], ordered=True)

    sns.lineplot(data=df, x="month", y=tar, hue=gn, ax=ax, palette="viridis", legend=False)
    ax.set_ylabel(None)
    ax.set_title(title)

    ticks = np.linspace(0, len(ys[0]), 7)
    label = np.linspace(0, 30, 7)
    ax.set_xticks(ticks, label)

    
    for g, color in zip(lb, sns.color_palette("viridis", len(lb))):
        
        sub = df[df[gn] == g]
        ax.text(
            sub["month"].max() + 0.2,              # slightly to the right of last x
            sub.loc[sub["month"].idxmax(), tar],   # last y value
            f"{gn}={g}", 
            va="center", ha="left", fontsize=10
    )

    return ax


def gen_grid(start_params, model, first = 'alpha', second = 'beta', k=50):
    result = dict()
    DM_vals = np.linspace(0.0, 1.0, k) 
    e_vals = np.linspace(0.0, 1.0, k)
    DM_grid, E_grid = np.meshgrid(DM_vals, e_vals)
    TD = np.zeros_like(DM_grid)    
    PBV = np.zeros_like(DM_grid) 
    
    for i in range(DM_grid.shape[0]):
        for j in range(DM_grid.shape[1]):
            dm = float(DM_grid[i,j])
            e = float(E_grid[i,j])

            params = start_params.copy()
            params[first] = dm
            params[second] = e
            
            model.reload()
            r = model.run(
                initial_condition='original',
                params=params,
                return_columns=['perceived_business_value', 'traffic_data']      
            )
            PBV[i,j] = r['perceived_business_value'].iloc[-1]
            TD[i,j] = r['traffic_data'].iloc[-1]
    
    br_idol_cols = np.argmax(PBV, axis=1)  #MB
    br_platform_rows= np.argmax(TD, axis=0) #MA
    NE_idx = [(i, j) for i, j in enumerate(br_idol_cols) if br_platform_rows[j] == i]
    NE_points = [(i, j, float(TD[i,j]), float(PBV[i,j])) for (i, j) in NE_idx]
    print('NE points: ', NE_points)
    for ine in NE_points:
        print('   ', round(DM_vals[ine[1]],2), round(e_vals[ine[0]],2))
              
    result['NE_points'] = NE_points
    result['PBV'] = PBV; result['TD'] = TD
    result['E_grid'] = E_grid; result['DM_grid'] = DM_grid
    result['maxTD'] = [float(TD[i,j]) for (i, j) in NE_idx]
    return result

def drawPBV_DM(res, title, xlabel,
               ylabel = r'DM Allocation $\alpha$ (Idol Agency)'):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # --- First plot (PBV) ---
    cs1 = axes[0].contourf(res['E_grid'], res['DM_grid'], res['PBV'], levels=50, cmap="viridis")
    fig.colorbar(cs1, ax=axes[0])
    axes[0].set_title("Perceived Business Value")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    # --- Second plot (DM) ---
    cs2 = axes[1].contourf(res['E_grid'], res['DM_grid'], res['TD'], levels=50, cmap="viridis")
    fig.colorbar(cs2, ax=axes[1])
    axes[1].set_title("Traffic Data")
    axes[1].set_xlabel(xlabel)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plotlines_step(tar, ax, model, 
                   stepvarvals, pb, 
                   ld=0.1,  
                   given = [0, 2, 8, 10],
                   title = None): 
    
    stepvar='beta'
    def runmodel(a, ld, stepvar, stepvarvals, pb):
        model.reload()
        params = pb.copy()

        params['alpha'] = a
        params['lambda'] = ld
        params['beta'] = stepvarvals
        
        r = model.run(
                initial_condition='original',
                params=params,
                return_columns=[tar])
        return r
    
    
    lb = [round(0.1*i,2) for i in given]
    ys = [runmodel(i, ld=ld, stepvar=stepvar, 
                   stepvarvals=stepvarvals, pb=pb)[tar] for i in lb]
    x = range(len(ys[0]))
    gn = r'$\alpha$'
    df = pd.DataFrame({
        "month": np.tile(x, len(lb)),
        tar: np.concatenate(ys),
        gn: np.repeat(lb, len(x))
    })
    
    df[gn] = pd.Categorical([v for v in df[gn]], ordered=True)

    sns.lineplot(data=df, x="month", y=tar, hue=gn, ax=ax, palette="viridis", legend=False)
    ax.set_ylabel(None)
    ax.set_title(title)
    ticks = np.linspace(0, len(ys[0]), 7)
    label = np.linspace(0, 30, 7)
    ax.set_xticks(ticks, label)
    
    for g, color in zip(lb, sns.color_palette("viridis", len(lb))):
        
        sub = df[df[gn] == g]
        ax.text(
            sub["month"].max() + 0.2,              
            sub.loc[sub["month"].idxmax(), tar],   
            f"{gn}={g}", 
            va="center", ha="left", fontsize=10
    )

    return ax

class findNE:
    def __init__(self, pb, MP, tar, var, c=-1):
        self.start_params = pb
        with FileLock(MP + ".pysd_read.lock"):
            self.model = pysd.read_vensim(MP)
             
        self.tar = tar
        self.var = var
        self.c = c

    def payoffs(self, alpha, beta):
        params = self.start_params.copy()
        params['alpha'] = alpha
        params['beta'] = beta

        if self.c==-1:
            params[self.tar] = self.var
        else: 
            params['Reference Arts'] = params['Reference Arts']*self.c
            params['Reference Discussion'] = params['Reference Discussion']*self.c

        self.model.reload()
        r = self.model.run(
            initial_condition='original',
            params=params,
            return_columns=['perceived_business_value', 'traffic_data']   
        )
        PBV = r['perceived_business_value'].iloc[-1]
        TD = r['traffic_data'].iloc[-1]
        return PBV, TD

    def br1(self, beta):
        res = minimize_scalar(lambda a: -self.payoffs(a, beta)[0], bounds=(0, 1), method="bounded", options={"xatol": 0.01})
        return res.x

    def br2(self, alpha):
        res = minimize_scalar(lambda b: -self.payoffs(alpha, b)[1], bounds=(0, 1), method="bounded", options={"xatol": 0.01})
        return res.x

    def fp_objective(self, x):
        alpha, beta = x
        a_br = self.br1(beta)
        b_br = self.br2(alpha)
        return (alpha - a_br)**2 + (beta - b_br)**2

    def run(self):
        x0 = (0.8, 0.8)
        res = minimize(self.fp_objective, x0=np.array(x0), bounds=[(0,1), (0,1)], method="L-BFGS-B")
        alpha_star, beta_star = res.x
        return alpha_star, beta_star

def findcondition(pb, mp, tar, var, c=-1):
    N = findNE(pb, mp, tar, var, c)
    return N.run()
 
# def run_one_condition(s, pb, cond, k, MP):

#     with FileLock(MP + ".pysd_read.lock"):
#         mod = pysd.read_vensim(MP)
#     update_params = pb.copy()
#     update_params[cond] = float(s)

#     return gen_grid(update_params, mod, first="alpha", second="beta", k=k)


# import os
# def best_alpha_for_theta_kappa(pb, theta, kappa):
#     MP = os.path.abspath("FanEconomy251218.mdl")
#     LP  = MP + ".pysd_read.lock"
#     with FileLock(LP):
#         mod = pysd.read_vensim(MP)
#     alphas = np.arange(0.05, 1.05, 0.05)
#     vals = []
#     for a in alphas:
#         params = pb.copy()
#         params['alpha'] = a
#         mod.reload()
#         r = mod.run(
#                 initial_condition='original',
#                 params=params,
#                 return_columns=['perceived_business_value'])
#         vals.append(r['perceived_business_value'].iloc[-1])
#     vals = np.array(vals)
#     a_star = float(alphas[int(np.argmax(vals))])
#     return float(theta), float(kappa), a_star


def plotlines_draft(tar, model, pb, ax, title, legend=False, given = [0, 2, 8, 10]):
    
    def runmodel(x, var='alpha'):
        params = pb.copy()
        params[var] = x
        model.reload()
        r = model.run(
                initial_condition='original',
                params=params,
                return_columns=[tar])
        return r
    
    #x = range(37)
    #x = range(t+1)
    
    lb = [round(0.1*i,2) for i in given]
    ys = [runmodel(i)[tar] for i in lb]
    gn = r'$\alpha$'

    x = range(len(ys[0]))


    df = pd.DataFrame({
        "month": np.tile(x, len(lb)),
        tar: np.concatenate(ys),
        gn: np.repeat(lb, len(x))
    })
    
    df[gn] = pd.Categorical([v for v in df[gn]], ordered=True)

    alpha_order = [0.0, 0.2, 0.8, 1.0]

    dashes = {
        0.0: "",
        0.2: (4, 2),
        0.8: (6, 2, 1, 2),
        1.0: (1, 2),
    }

    colors = {
        0.0: '#1f1f1f',   # dark gray
        0.2: '#4d4d4d',
        0.8: '#7f7f7f',
        1.0: '#b2b2b2',   # light gray
    }
    markers = {0.0: "o", 0.2: "^", 0.8: "s", 1.0: "X"}

    sns.lineplot(data=df, x="month", y=tar, hue=gn, style=gn,
                 hue_order=alpha_order, style_order=alpha_order,
                 dashes=dashes, 
                 #markers=markers,
                 color=colors,
                 markevery=6, linewidth=2,
                 palette="viridis",
                 ax=ax, legend=legend)
    
    ax.set_ylabel(None)
    ax.set_title(title)

    ticks = np.linspace(0, len(ys[0]), 7)
    label = np.linspace(0, 30, 7)
    ax.set_xticks(ticks, label)


    return ax


def plotlines_step_draft(tar, ax, model, 
                   stepvarvals, pb, legend,
                   ld=0.1,  
                   given = [0, 2, 8, 10],
                   title = None): 
    
    stepvar='beta'
    def runmodel(a, ld, stepvar, stepvarvals, pb):
        model.reload()
        params = pb.copy()

        params['alpha'] = a
        params['lambda'] = ld
        params['beta'] = stepvarvals
        
        r = model.run(
                initial_condition='original',
                params=params,
                return_columns=[tar])
        return r
    
    
    lb = [round(0.1*i,2) for i in given]
    ys = [runmodel(i, ld=ld, stepvar=stepvar, 
                   stepvarvals=stepvarvals, pb=pb)[tar] for i in lb]
    x = range(len(ys[0]))
    gn = r'$\alpha$'
    df = pd.DataFrame({
        "month": np.tile(x, len(lb)),
        tar: np.concatenate(ys),
        gn: np.repeat(lb, len(x))
    })
    
    df[gn] = pd.Categorical([v for v in df[gn]], ordered=True)

    alpha_order = [0.0, 0.2, 0.8, 1.0]

    dashes = {
        0.0: "",
        0.2: (4, 2),
        0.8: (6, 2, 1, 2),
        1.0: (1, 2),
    }

    colors = {
        0.0: '#1f1f1f',   # dark gray
        0.2: '#4d4d4d',
        0.8: '#7f7f7f',
        1.0: '#b2b2b2',   # light gray
    }
    markers = {0.0: "o", 0.2: "^", 0.8: "s", 1.0: "X"}

    sns.lineplot(data=df, x="month", y=tar, hue=gn, style=gn,
                 hue_order=alpha_order, style_order=alpha_order,
                 dashes=dashes, 
                 #markers=markers,
                 color=colors,
                 markevery=6, linewidth=2,
                 palette="viridis",
                 ax=ax, legend=legend)
    ax.set_ylabel(None)
    ax.set_title(title)
    ticks = np.linspace(0, len(ys[0]), 7)
    label = np.linspace(0, 30, 7)
    ax.set_xticks(ticks, label)
    
   
    return ax