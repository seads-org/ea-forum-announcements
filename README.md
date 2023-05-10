This repo is an initial attempt at automatic posts tagging in EA forum. It builds an initial, simple one head classifier on top of SBERT v1 embeddings trained on EA forum posts.


```

├── constants.py - Just some project constants
├── data - Some of EA posts manully labeled to "organisation", "project" or "other"
│   ├── labeled_posts.csv
│   ├── labelling_sheet.csv
│   └── labels.csv
├── models - Trained classifiers
│   ├── e5baseemb_post_classifier:2023-04-04.pth
│   └── eaembd_post_classifier:2023-04-04.pth
├── notebooks
│   ├── 01_labelling.ipynb
│   ├── 02_embeddings_viz.ipynb
│   ├── 03_error_analysis.ipynb
│   ├── 04_discover.ipynb - Use model on the unlabeled data to find new orgs and projects
│   ├── identified_orgs.csv - Output of 04 notebook
│   ├── identified_projects.csv - Output of 04 notebook
│   └── project_path.py
├── output
│   └── db.sqlite3 - Optuna studies
├── README.md
├── requirements.txt
└── src
    ├── classifier.py
    ├── data_utils.py
    ├── embed_posts.py
    ├── hyperparam_tuning.py
    ├── paths.py
    ├── text_split.py
    ├── training.py
    └── train_test_split.py
    
 ```

