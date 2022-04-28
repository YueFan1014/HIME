Please run 'python HIME.py' to generate the embedding model.
The parameters are as follows:
--dataset: 	    'dblp', 'protein_go' or 'gene_pathway', default 'dblp'.
--neg_num	    negative sampling number, default 5.
--emb_num	    branch vector number, default 8.
--emb_dim	    embedding dimension, default 32.
--epoch_num	    epoch number, default 50.
--batch_size	batch size, default 1000.
--LRU_period	LRU period, default 5.
The trained model will be saved into a directory called 'saved_model'.

To evaluate, please run 'python evaluate.py'. The parameters are as follows:
--dataset: 	    'dblp', 'protein_go' or 'gene_pathway', default 'dblp'.
--emb_num	    branch vector number, default 8.
--emb_dim	    embedding dimension, default 32.
--epoch_num	    epoch number, default 50.
The program will find the model specified by above parameters in 'saved_model', and the perform the evaluation.



