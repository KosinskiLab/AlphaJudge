import os
import subprocess
import csv
import glob

HBPLUS_PATH = "/Users/admin/Alpha/hbplus/hbplus"
CLEAN_PATH = "/Users/admin/Alpha/hbplus/clean"
BASE_DIR = "test_data/af2/positive_dimers/predictions"
OUTPUT_CSV = "test_data/af2/positive_dimers/benchmarks/HBPLUS.csv"

def run_hbplus_for_complex(pdb_path):
    abs_pdb_path = os.path.abspath(pdb_path)
    complex_name = pdb_path.split("/")[-2]
    work_dir = os.path.dirname(abs_pdb_path)
    pdb_filename = os.path.basename(abs_pdb_path)
    
    # 1. Clean the PDB
    # ./clean requires passing @ before input filenames!
    clean_input = f"@{pdb_filename}\n"
    res = subprocess.run([CLEAN_PATH], input=clean_input.encode(), cwd=work_dir, capture_output=True)
    
    cleaned_pdb = os.path.join(work_dir, pdb_filename.replace(".pdb", ".new"))
    if not os.path.exists(cleaned_pdb):
        print(f"Failed to find cleaned file at {cleaned_pdb} (CWD was {work_dir})")
        print(f"Clean Output: {res.stdout.decode()}")
        print(f"Clean Error: {res.stderr.decode()}")

    # 2. Run HBPLUS
    res = subprocess.run([HBPLUS_PATH, os.path.basename(cleaned_pdb), pdb_filename], cwd=work_dir, capture_output=True)
    if res.returncode != 0:
        print(f"HBPLUS failed for {abs_pdb_path}")
        print(f"Stdout: {res.stdout.decode()[:500]}")
        print(f"Stderr: {res.stderr.decode()[:500]}")
    
    hb2_file = cleaned_pdb.replace(".new", ".hb2")
    if not os.path.exists(hb2_file):
        print(f"Failed to generate .hb2 for {pdb_path}")
        return None
        
    # 3. Parse .hb2
    hb_count = 0
    #sb_count = 0
    
    #sb_donors = {"ARG": ["NH1", "NH2", "NE"], "LYS": ["NZ"], "HIS": ["ND1", "NE2"]}
    #sb_acceptors = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}

    with open(hb2_file, "r") as f:
        lines = f.readlines()
        for line in lines[8:]: # Skip header
            if len(line) < 27: continue
            
            d_chain = line[0]
            #d_res_type = line[6:9].strip()
            #d_atom_type = line[9:13].strip()
            
            a_chain = line[14]
            #a_res_type = line[20:23].strip()
            #a_atom_type = line[23:27].strip()
            
            #categories = line[33:35]
            
            if d_chain != a_chain:
                hb_count += 1

                #if categories == "SS":
                #    is_sb_donor = d_res_type in sb_donors and d_atom_type in sb_donors[d_res_type]
                #    is_sb_acceptor = a_res_type in sb_acceptors and a_atom_type in sb_acceptors[a_res_type]
                #    is_sb_donor_rev = a_res_type in sb_donors and a_atom_type in sb_donors[a_res_type]
                #    is_sb_acceptor_rev = d_res_type in sb_acceptors and d_atom_type in sb_acceptors[d_res_type]
                    
                #    if (is_sb_donor and is_sb_acceptor) or (is_sb_donor_rev and is_sb_acceptor_rev):
                #        sb_count += 1
                        
    # Cleanup tmp files
    if os.path.exists(cleaned_pdb): os.remove(cleaned_pdb)
    if os.path.exists(hb2_file): os.remove(hb2_file)
    debug_dat = os.path.join(work_dir, "hbdebug.dat")
    if os.path.exists(debug_dat): os.remove(debug_dat)

    print(f"Complex: {complex_name}, HB: {hb_count}")
    return {
        "complex": complex_name,
        "hb": hb_count,
        #"sb": sb_count
    }

def main():
    ccp4_csv = "test_data/af2/positive_dimers/benchmarks/CCP4.csv"
    output_merged = "test_data/af2/positive_dimers/benchmarks/HBPLUS_CCP4.csv" # For a unified benchmark CSV file.
    
    if not os.path.exists(ccp4_csv):
        print(f"CCP4 benchmark not found at {ccp4_csv}")
        return

    with open(ccp4_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    print(f"Loaded {len(rows)} complexes from {ccp4_csv}")
    
    final_rows = []
    for i, row in enumerate(rows):
        complex_name = row["complex"]
        pdb_path = os.path.join(BASE_DIR, complex_name, "ranked_0.pdb")
        
        if not os.path.exists(pdb_path):
            print(f"[{i+1}/{len(rows)}] Skipping {complex_name}: PDB not found")
            final_rows.append(row)
            continue
            
        print(f"[{i+1}/{len(rows)}] Processing {complex_name} with HBPLUS...")
        res = run_hbplus_for_complex(pdb_path)
        
        if res:
            row["hb"] = res["hb"]
            #row["sb"] = res["sb"] # Keeping PISA's SB score for now
        
        final_rows.append(row)

    try:
        with open(output_merged, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)
            
        print(f"Successfully generated unified benchmark: {output_merged}")
        
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    main()
