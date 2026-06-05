# =============================================================================
# Paper Results Module
# Perfect Implementation of All Paper Content
# =============================================================================
from .definitions import DEFINITIONS, print_all as print_definitions
from .table1 import TABLE_1, print_table as print_table1, get_latex as get_table1_latex
from .table2 import TABLE_2, print_table as print_table2, get_latex as get_table2_latex
from .table3 import TABLE_3, print_table as print_table3, get_latex as get_table3_latex
from .table4 import TABLE_4, print_table as print_table4, get_latex as get_table4_latex
from .figure1 import FIGURE_1, print_figure
from .appendix_a import APPENDIX_A, print_appendix_a, get_latex as get_appendix_a_latex
from .appendix_b import APPENDIX_B, print_appendix_b, get_latex as get_appendix_b_latex
from .appendix_c import APPENDIX_C, print_appendix_c, get_latex as get_appendix_c_latex
from .appendix_d import APPENDIX_D, print_appendix_d, get_latex as get_appendix_d_latex

__all__ = [
    "DEFINITIONS",
    "TABLE_1",
    "TABLE_2",
    "TABLE_3",
    "TABLE_4",
    "FIGURE_1",
    "APPENDIX_A",
    "APPENDIX_B",
    "APPENDIX_C",
    "APPENDIX_D",
    "print_definitions",
    "print_table1",
    "print_table2",
    "print_table3",
    "print_table4",
    "print_figure",
    "print_appendix_a",
    "print_appendix_b",
    "print_appendix_c",
    "print_appendix_d",
    "get_table1_latex",
    "get_table2_latex",
    "get_table3_latex",
    "get_table4_latex",
    "get_appendix_a_latex",
    "get_appendix_b_latex",
    "get_appendix_c_latex",
    "get_appendix_d_latex"
]
