# EmELpp
EmEL++ is a geometric approach to generate embeddings for the description logic EL++
The implementation is done using Python and Tensorflow Library.

The code folder is organized as follows:
- src/experiments: This contains separate folder for each ontology the experiment is carried out upon.
- src/evaluation : This folder contains the implementation of evaluation metrics - Evaluating_HITS.py and Evaluating_accuracy.py
- sample_embeddings : This folder consists of trained embeddings obtained for GALEN dataset included as an example.

Since each of the ontologies have different characteristics based on normal forms 
thus inputs to the model defined using Keras change. Thus, we organise the implementation per ontology.

For Example:
-src/experiments/GALEN/GALEN_EmEL.py : This file represents the implementation of EmEL++ model 
for GALEN ontology.

-src/experiments/GALEN/GALEN_data/ : This folder consists of 4 processed files namely, normalized form of the ontology file
to be used for training, and training,validation & testing set obtained from subclass relations in ontology.

Similary, the corresponding files are organised for NCI,GO,FMA,ANATOMY and SNOMED ontology.

Implementation of the code is organised in Three Parts for classification task:

- First: Given an ontology OWL file we normalize it with Normalizer.groovy script using jcel jar.
	Command to Normalize: groovy -cp jcel.jar Normalizer.groovy -i <Input OWL ontology> -o <Output normalized-ontology>

	
- Second: Using the normalized-ontology we identify the subclass relations and generate training, testing and validation set using
		   split of 70%-20%-10%.

- Third: Performing training using the normalized-ontology file while removing the 30%(validation and testing) subclass relation axioms from it.
		Using validation data for hyper-parameter tuning and testing to evaluate the fine-tuned models.
		
Associated Files:
- Train_valid_test_set_generation.py : This file covers the second part mentioned above.
- <OntologyName>_EmEL.py : This file denotes the EmEL++ model implementation. for example: GALEN_EmEL.py

Executing the code:
- Before executing the code you need CUDA installed to use a GPU and list of python libraries as provided in requirements.txt.
- For execution of the code follow the directory structure as it is, further we demonstrate it using an example for GALEN dataset.
- Go to directory src/experiments/GALEN
- Run GALEN_EmEL.py using the command: python GALEN_EmEL.py, This will start the training and if you want to change the hyper-parameter
values or path of data one needs to specify it in the file, otherwise it will by default take the path as maintained by directory structure. 
- This will output corresponding embeddings for classes and relations in pkl files.
- For evaluating the embeddings run python scripts Evaluating_accuracy.py and Evaluating_accuracy.py provide the path of the pkl files.

Similary, you can do the process for other ontologies present.
