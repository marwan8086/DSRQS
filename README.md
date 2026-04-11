# DSRQS

**Depth-Stratified Relation-Query Scoring (DSRQS)**  
A method designed to eliminate the *position-conflation error* in multi-hop KG-RAG systems, with a focus on improving rare-disease diagnosis.

---

## 🔗 Real Data Integration

Real data connectors are already defined and ready to use.

To activate them:

1. Add your dataset credentials in the configuration section  
2. Uncomment the provided example calls  
3. Replace `DATASETS[ds_name]` with your actual knowledge graph data  
4. Run the model core implementation (Cells 6–18) works without modification  

---

## 📄 Dataset Licensing

Please ensure compliance with the following dataset licenses:

- **Orphanet**  CC BY 4.0  
- **DisGeNET**  CC BY-NC-SA 4.0  
- **OMIM** Requires an academic license  

---

## 💡 Notes

- The pipeline is designed to be modular and easily adaptable  
- Switching between synthetic and real data requires minimal changes  
- Ideal for research in medical KG-RAG and multi-hop reasoning  

---
