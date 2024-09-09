## Linux ubuntu 20.04 error
```
sudo apt-get install libxrender1
sudo apt-get install lzma liblzma-dev libbz2-dev

cd ${python_dir}
sudo ./configure --enable-optimizations
sudo make -j$(nproc)
```

## Data Augmentation
- ChEMBL 34 version 을 사용해 아래 쿼리 결과를 새로운 데이터로 활용 [다운로드 링크](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_sqlite.tar.gz)
- 기존 1,952 종의 Molecule 과 동일한 개수의 Molecule을 기준으로 추가된 실험결과 데이터가 존재
``` python
import pandas as pd
import sqlite3

DATABASE_URL = "../data/chembl_34.db"
db = sqlite3.connect(DATABASE_URL)

sql = f'''
SELECT m.chembl_id as 'Molecule ChEMBL ID'
     , a.standard_type as 'Standard Type'
     , a.standard_relation as 'Standard Relation'
     , a.standard_value as 'Standard Value'
     , a.standard_units as 'Standard Units'
     , a.pchembl_value as 'pChEMBL Value'
     , aa.chembl_id as 'Assay ChEMBL ID'
     , t.chembl_id as 'Target ChEMBL ID'
     , t.pref_name as 'Target Name'
     , t.organism as 'Target Organism'
     , t.target_type as 'Target Type'
     , d.chembl_id as 'Document ChEMBL ID'
     , a.standard_value as 'IC50_nM'
     , a.pchembl_value as 'pIC50'
     , c.canonical_smiles as 'Smiles'
  FROM compound_structures c
 INNER JOIN molecule_dictionary m ON m.molregno = c.molregno
 INNER JOIN compound_records cr ON cr.molregno = m.molregno
 INNER JOIN activities a ON a.molregno = m.molregno AND a.record_id = cr.record_id AND a.doc_id = cr.doc_id AND a.standard_type = 'IC50' AND a.standard_relation = '=' AND a.standard_units = 'nM'
 INNER JOIN docs d ON d.doc_id = a.doc_id AND d.doc_type IN ('PUBLICATION', 'PATENT')
 INNER JOIN assays aa ON aa.assay_id = a.assay_id AND aa.doc_id = d.doc_id
 INNER JOIN target_dictionary t ON aa.tid = t.tid AND t.target_type = 'SINGLE PROTEIN' AND t.organism = 'Homo sapiens' AND t.chembl_id = 'CHEMBL3778'
'''
cursor = db.execute(sql)
description = [desc[0] for desc in cursor.description]
df = pd.DataFrame(cursor.fetchall(), columns=description)
df = df[df['pIC50'].isna() == False]
```