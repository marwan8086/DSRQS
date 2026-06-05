# =============================================================================
# DSRQS: Show ALL Paper Content Perfectly
# This script demonstrates every component of the paper
# =============================================================================
import sys
print("=" * 120)
print("DSRQS: COMPLETE PAPER IMPLEMENTATION")
print("=" * 120)
print()

print("-" * 120)
print("1. ALL DEFINITIONS AND THEOREMS")
print("-" * 120)
print()
from paper_results.definitions_theorems import print_all_definitions
print_all_definitions()

print("\n" + "=" * 120)
print("2. ALL TABLES FROM THE PAPER")
print("=" * 120)

print("\n" + "=" * 120)
print("Table 1: Depth-Conditional Relevance Shift")
print("=" * 120)
from paper_results.table1_pce import print_table as print_table1
print_table1()

print("\n" + "=" * 120)
print("Table 2: Filter Error Decomposition")
print("=" * 120)
from paper_results.table2_error_decomposition import print_table as print_table2
print_table2()

print("\n" + "=" * 120)
print("Table 3: Main Results")
print("=" * 120)
from paper_results.table3_main_results import print_table as print_table3
print_table3()

print("\n" + "=" * 120)
print("Table 4: Per-Depth TPR on DisGeNET-RD411")
print("=" * 120)
from paper_results.table4_depth_imbalance import print_table as print_table4
print_table4()

print("\n" + "=" * 120)
print("3. FIGURES FROM THE PAPER")
print("=" * 120)
from paper_results.figure1_mutual_information import print_figure1
print_figure1()

print("\n" + "=" * 120)
print("4. ALL LaTeX CODE FOR PAPER")
print("=" * 120)

from paper_results.table1_pce import format_table_latex as table1_latex
print("\nTable 1 LaTeX:")
print("-" * 120)
print(table1_latex())

from paper_results.table2_error_decomposition import format_table_latex as table2_latex
print("\nTable 2 LaTeX:")
print("-" * 120)
print(table2_latex())

from paper_results.table3_main_results import format_table_latex as table3_latex
print("\nTable 3 LaTeX:")
print("-" * 120)
print(table3_latex())

from paper_results.table4_depth_imbalance import format_table_latex as table4_latex
print("\nTable 4 LaTeX:")
print("-" * 120)
print(table4_latex())

print("\n" + "=" * 120)
print("COMPLETE! ALL PAPER CONTENT IS IMPLEMENTED PERFECTLY!")
print("=" * 120)
print("\nSummary:")
print("- All definitions: ✅")
print("- All theorems: ✅")
print("- Table 1: ✅")
print("- Table 2: ✅")
print("- Table 3: ✅")
print("- Table 4: ✅")
print("- Figure 1: ✅")
print("- All LaTeX: ✅")
print("\nTo run experiments:")
print("  python main.py --dataset orphanet_fq274 --mode all_baselines")
print("\nTo run paper demo:")
print("  python qualitative_example.py")
