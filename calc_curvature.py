import logging 
import argparse
import numpy as np
import matplotlib.pyplot as plt

from modules.visualizer import load_jsonl

"""Calculate curvature-related norms from gap_log and fit power-law relationships."""

def main(args):

    gap_log = load_jsonl(args.jsonl_path)  

    steps    = np.array([g['step'] for g in gap_log])
    sigmas   = np.array([g.get('sigma_hat', g.get('sigma_cur')) for g in gap_log])

    eps_norm = np.array([g.get('eps_theta_norm', np.nan)     for g in gap_log])
    D_norm   = np.array([g.get('Dtheta_norm', np.nan)        for g in gap_log])
    xD_norm  = np.array([g.get('x_minus_D_norm', np.nan)     for g in gap_log])
    Ds_norm  = np.array([g.get('Dsigma_norm', np.nan)        for g in gap_log])
    JD_norm  = np.array([g.get('JD_eps_norm', np.nan)        for g in gap_log])
    ddc_norm = np.array([g.get('ddotx_closed_norm', np.nan)  for g in gap_log])    
    ddf_norm = np.array([g.get('ddotx_fd_norm', np.nan)      for g in gap_log])    
    brk_norm = np.array([g.get('bracket_norm', np.nan)       for g in gap_log])    

    def fit_powerlaw(x, y, mask_extra=None, min_pts=3):
        mask = (~np.isnan(x)) & (~np.isnan(y)) & (x>0) & (y>0)
        if mask_extra is not None: mask &= mask_extra
        if mask.sum() < min_pts: return None
        lx = np.log(x[mask]); ly = np.log(y[mask])
        p, b = np.polyfit(lx, ly, 1)  # y ≈ e^b * x^p
        return {'p': p, 'C': np.exp(b), 'mask': mask}

    # (선택) 극단·라운딩 구간 제외 마스크
    s_min = np.nanmin(sigmas); s_max = np.nanmax(sigmas)
    core = (sigmas > s_min*1.05) & (sigmas < s_max*0.98)

    fits = {
        'eps':         fit_powerlaw(sigmas, eps_norm, mask_extra=core),
        'D':           fit_powerlaw(sigmas, D_norm,   mask_extra=core),
        'x-D':         fit_powerlaw(sigmas, xD_norm,  mask_extra=core),
        'D_sigma':     fit_powerlaw(sigmas, Ds_norm,  mask_extra=core),
        'JDeps':       fit_powerlaw(sigmas, JD_norm,  mask_extra=core),
        'ddot_closed': fit_powerlaw(sigmas, ddc_norm, mask_extra=core),
        'ddot_fd':     fit_powerlaw(sigmas, ddf_norm, mask_extra=core),
        'bracket':     fit_powerlaw(sigmas, brk_norm, mask_extra=core),
    }

    for k,v in fits.items():
        if v is None:
            logging.info(f"{k:12s}: not enough valid points")
        else:
            logging.info(f"{k:12s}: p = {v['p']:.3f}, C = {v['C']:.3e}")

    rho = 7
    if fits.get('bracket') and fits.get('ddot_closed'):
        p_pred = fits['bracket']['p'] + (rho-2)/rho
        logging.info(f"[check] predicted p(ddot_closed) ≈ p(bracket)+{(rho-2)/rho:.3f} = {p_pred:.3f}, "
            f"measured = {fits['ddot_closed']['p']:.3f}")


    plt.figure(figsize=(12,10))

    plt.subplot(231)
    plt.loglog(sigmas, eps_norm, 'o', label='eps')
    if fits['eps']:
        xx = np.linspace(np.nanmin(sigmas[core]), np.nanmax(sigmas[core]), 100)
        plt.loglog(xx, fits['eps']['C']*xx**(fits['eps']['p']), '-', label=f'fit p={fits["eps"]["p"]:.2f}')
    plt.xlabel('sigma'); plt.ylabel('||eps||'); plt.legend()

    plt.subplot(232)
    plt.loglog(sigmas, Ds_norm, 'o', label='D_sigma')
    if fits['D_sigma']:
        xx = np.linspace(np.nanmin(sigmas[core]), np.nanmax(sigmas[core]), 100)
        plt.loglog(xx, fits['D_sigma']['C']*xx**(fits['D_sigma']['p']), '-', label=f'fit p={fits["D_sigma"]["p"]:.2f}')
    plt.xlabel('sigma'); plt.ylabel('||D_sigma||'); plt.legend()

    plt.subplot(233)
    plt.loglog(sigmas, JD_norm, 'o', label='JD_eps')
    if fits['JDeps']:
        xx = np.linspace(np.nanmin(sigmas[core]), np.nanmax(sigmas[core]), 100)
        plt.loglog(xx, fits['JDeps']['C']*xx**(fits['JDeps']['p']), '-', label=f'fit p={fits["JDeps"]["p"]:.2f}')
    plt.xlabel('sigma'); plt.ylabel('||JD_eps||'); plt.legend()

    plt.subplot(234)
    plt.loglog(sigmas, ddc_norm, 'o', label='ddot_closed')
    if fits['ddot_closed']:
        xx = np.linspace(np.nanmin(sigmas[core]), np.nanmax(sigmas[core]), 100)
        plt.loglog(xx, fits['ddot_closed']['C']*xx**(fits['ddot_closed']['p']), '-', label=f'fit p={fits["ddot_closed"]["p"]:.2f}')
    plt.xlabel('sigma'); plt.ylabel('||ddot_closed||'); plt.legend()

    plt.subplot(235)
    plt.loglog(sigmas, ddf_norm, 'o', label='ddot_fd')
    if fits['ddot_fd']:
        xx = np.linspace(np.nanmin(sigmas[core]), np.nanmax(sigmas[core]), 100)
        plt.loglog(xx, fits['ddot_fd']['C']*xx**(fits['ddot_fd']['p']), '-', label=f'fit p={fits["ddot_fd"]["p"]:.2f}')
    plt.xlabel('sigma'); plt.ylabel('||ddot_fd||'); plt.legend()

    plt.subplot(236)
    plt.loglog(sigmas, brk_norm, 'o', label='bracket')
    if fits['bracket']:
        xx = np.linspace(np.nanmin(sigmas[core]), np.nanmax(sigmas[core]), 100)
        plt.loglog(xx, fits['bracket']['C']*xx**(fits['bracket']['p']), '-', label=f'fit p={fits["bracket"]["p"]:.2f}')
    plt.xlabel('sigma'); plt.ylabel('|| (ρ-1)/ρ·eps − JD − Dσ ||'); plt.legend()

    plt.tight_layout(); # plt.show()
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    logging.info("Successfully saved plot to:", args.output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to the gap_log jsonl file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the curvature plot.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)