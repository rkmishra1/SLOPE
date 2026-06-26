import os
import sys
import argparse
import time

# Ensure we can import from local 'slope' and other directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulations.simulation_1 import run_simulation_1
from simulations.simulation_2 import run_simulation_2
from real_data.real_data_analysis import run_real_data_analysis

def main():
    parser = argparse.ArgumentParser(description="SLOPE implementation, simulations, and real data analysis master script.")
    parser.add_argument('--mode', type=str, choices=['fast', 'full'], default='fast',
                        help="Run mode: 'fast' (fewer replicates, ~10s execution) or 'full' (more replicates, ~2 mins execution).")
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        SLOPE: Sorted L-One Penalized Estimation              ║")
    print("║     Bogdan et al. (2015) - Python Reference Implementation   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Ensure folders exist
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    
    if args.mode == 'fast':
        num_reps_sim1 = 5
        num_reps_sim2 = 5
        print("Running in FAST mode. (Replicates: Sim1 = 5, Sim2 = 5)\n")
    else:
        num_reps_sim1 = 100
        num_reps_sim2 = 100
        print("Running in FULL mode. (Replicates: Sim1 = 100, Sim2 = 100)\n")
        
    start_time = time.time()
    
    # =============================================================================
    # STEP 1: SIMULATION 1
    # =============================================================================
    print("═══ STEP 1: Running Simulation 1 (Section 1.3.3: Metrics comparison) ═══")
    t0 = time.time()
    run_simulation_1(n=500, p=500, num_reps=num_reps_sim1, seed=42)
    print(f"  ✓ Simulation 1 finished in {time.time() - t0:.2f} seconds\n")
    
    # =============================================================================
    # STEP 2: SIMULATION 2
    # =============================================================================
    print("═══ STEP 2: Running Simulation 2 (Section 3.1: Equicorrelated noise) ═══")
    t0 = time.time()
    run_simulation_2(p=500, num_reps=num_reps_sim2, seed=42)
    print(f"  ✓ Simulation 2 finished in {time.time() - t0:.2f} seconds\n")
    
    # =============================================================================
    # STEP 3: REAL DATA ANALYSIS
    # =============================================================================
    print("═══ STEP 3: Running Real Data Analysis (Diabetes interactions) ═══")
    t0 = time.time()
    coeff_estimates, summary_df = run_real_data_analysis(seed=42)
    print(f"  ✓ Real Data Analysis finished in {time.time() - t0:.2f} seconds\n")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    total_time = time.time() - start_time
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    Analysis Complete!                        ║")
    print(f"║  Total execution time: {total_time:.2f} seconds                  ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Figures generated: 3                                        ║")
    print(f"║    - figures/sim1_metrics.png                                ║")
    print(f"║    - figures/sim2_correlated.png                             ║")
    print(f"║    - figures/real_data_coefs.png                             ║")
    print(f"║  Tables generated: 1                                         ║")
    print(f"║    - tables/real_data_summary.csv                            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

if __name__ == '__main__':
    main()
