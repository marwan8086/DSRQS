# =============================================================================
# DSRQS: Complete Paper Content - Show Everything!
# Perfect, exact implementation for scientific publication
# =============================================================================
import sys
import io

# Force UTF-8 for all console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 120)
print("DSRQS: Complete Paper Content Implementation")
print("=" * 120)
print("")
print("Marwan Dhifallah and Yu Liu")
print("Dalian University of Technology")
print("Code: https://github.com/marwan8086/DSRQS")
print("")

print("\n" + "=" * 120)
print("1. All Definitions, Assumptions, Theorems, Corollaries, Propositions")
print("=" * 120)
from paper_results.definitions import print_all as print_definitions
print_definitions()

print("\n" + "=" * 120)
print("2. Table 1: Depth-Conditional Relevance Shift (200 Annotated Queries)")
print("=" * 120)
from paper_results.table1 import print_table as print_table1
print_table1()

print("\n" + "=" * 120)
print("3. Table 2: Filter Error Decomposition on ≈14,000 Edges")
print("=" * 120)
from paper_results.table2 import print_table as print_table2
print_table2()

print("\n" + "=" * 120)
print("4. Table 3: Main Results (5-Fold CV)")
print("=" * 120)
from paper_results.table3 import print_table as print_table3
print_table3()

print("\n" + "=" * 120)
print("5. Table 4: Per-Depth TPR on DisGeNET-RD411")
print("=" * 120)
from paper_results.table4 import print_table as print_table4
print_table4()

print("\n" + "=" * 120)
print("6. Figure 1: Estimated Mutual Information I(Y;L|Q,R)")
print("=" * 120)
from paper_results.figure1 import print_figure
print_figure()

print("\n" + "=" * 120)
print("7. Appendix A: Complete Proofs")
print("=" * 120)
from paper_results.appendix_a import print_appendix_a
print_appendix_a()

print("\n" + "=" * 120)
print("8. Appendix B: OMIM-Hop3 Benchmark")
print("=" * 120)
from paper_results.appendix_b import print_appendix_b
print_appendix_b()

print("\n" + "=" * 120)
print("9. Appendix C: Reproducibility Checklist and Extended Implementation Details")
print("=" * 120)
from paper_results.appendix_c import print_appendix_c
print_appendix_c()

print("\n" + "=" * 120)
print("10. Appendix D: Additional Qualitative Example (Smith-Lemli-Opitz Syndrome)")
print("=" * 120)
from paper_results.appendix_d import print_appendix_d
print_appendix_d()

print("\n" + "=" * 120)
print("✅ PERFECT PAPER IMPLEMENTATION COMPLETE!")
print("=" * 120)
print("")
print("All content is exactly as in the paper, ready for publication!")
print("")
print("Run individual files for LaTeX code:")
print("  - python paper_results/table1.py      (Table 1 LaTeX)")
print("  - python paper_results/table2.py      (Table 2 LaTeX)")
print("  - python paper_results/table3.py      (Table 3 LaTeX)")
print("  - python paper_results/table4.py      (Table 4 LaTeX)")
print("  - python paper_results/appendix_a.py  (Appendix A LaTeX)")
print("  - python paper_results/appendix_b.py  (Appendix B LaTeX)")
print("  - python paper_results/appendix_c.py  (Appendix C LaTeX)")
print("  - python paper_results/appendix_d.py  (Appendix D LaTeX)")
print("")
print("All files are conference-ready (NeurIPS/ICML/ACL style)!")
print("=" * 120)
