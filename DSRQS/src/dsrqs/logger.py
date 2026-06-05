# =============================================================================
# Author: Marwan Dhifallah*
# Email:  marwan@mail.dlut.edu.cn
# Affiliation: Dalian University of Technology 
#
# Description:
#   This file is part of the DSRQS framework for multi-hop reasoning over
#   biomedical knowledge graphs and retrieval-augmented generation (RAG).
#
# Copyright (c) 2026
# =============================================================================
import logging


def get_logger():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO
    )
    return logging.getLogger("DSRQS")