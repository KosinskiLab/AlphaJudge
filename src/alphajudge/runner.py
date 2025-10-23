from __future__ import annotations
from pathlib import Path
import csv, logging, math
from .parsers import pick_parser
from .core import Complex

def process(directory: str, contact_thresh: float, pae_filter: float, models_to_analyse: str) -> None:
    d = Path(directory)
    parser = pick_parser(d)
    run = parser.parse_run(d)
    models = [run.order[0]] if models_to_analyse == "best" else run.order
    job = d.resolve().name

    rows = []
    for m in models:
        try:
            structure, confidence = run.load_model(m)
            comp = Complex(structure, confidence, contact_thresh, pae_filter)
            global_score = comp.mpDockQ if comp.num_chains > 2 else (
                comp.interfaces[0].pDockQ if comp.interfaces else float('nan')
            )
            for iface in comp.interfaces:
                if iface.num_intf_residues == 0: continue
                if iface.average_interface_pae > pae_filter: continue
                pd2, _ = iface.pDockQ2()
                label = f"{iface.chain1[0].get_parent().id}_{iface.chain2[0].get_parent().id}"
                rows.append({
                    "jobs": job,
                    "model_used": m,
                    "interface": label,
                    "iptm_ptm": float(confidence.iptm_ptm) if confidence.iptm_ptm is not None else float('nan'),
                    "iptm": float(confidence.iptm) if confidence.iptm is not None else float('nan'),
                    "ptm": float(confidence.ptm) if confidence.ptm is not None else float('nan'),
                    "confidence_score": float(confidence.confidence_score) if confidence.confidence_score is not None else float('nan'),
                    "pDockQ/mpDockQ": global_score,
                    "average_interface_pae": iface.average_interface_pae,
                    "interface_average_plddt": iface.average_interface_plddt,
                    "interface_num_intf_residues": iface.num_intf_residues,
                    "interface_polar": iface.polar,
                    "interface_hydrophobic": iface.hydrophobic,
                    "interface_charged": iface.charged,
                    "interface_contact_pairs": iface.contact_pairs,
                    "interface_score": iface.score_complex,
                    "interface_pDockQ2": pd2,
                    "interface_ipSAE": iface.ipsae(),
                    "interface_LIS": iface.lis(),
                    "interface_hb": iface.hb,
                    "interface_sb": iface.sb,
                    "interface_sc": iface.sc,
                    "interface_area": iface.int_area,
                    "interface_solv_en": iface.int_solv_en,
                })
            logging.info("processed model: %s via %s", m, parser.name)
        except Exception as e:
            logging.error("error processing model %s: %s", m, e)

    out = d / "interfaces.csv"
    with out.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        else:
            f.write("")
    logging.info("wrote %s", str(out))
