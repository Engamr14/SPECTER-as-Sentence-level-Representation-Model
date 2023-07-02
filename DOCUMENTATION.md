# SPECTER as a Sentence-level Representation Model

## Introduction
Representation Learning is very substantial part of Natural Language Processing. It is required for almost all applications in NLP. Therefore, many models have been developed to provide powerful representations for word-level, sentencelevel, and document-level. SPECTER [1] is a state-of-art model that has been developed to provide powerful documentlevel representation. It uses the title and/or the abstract of the input document to learn effective document representation that can be used for several downstream tasks. The citation information is used as naturally occurring inter-document incidental supervision signal indicating which documents are most related and formulate the signal into triplet-loss pretraining objective. In other words, the citation information is used as a ground truth that indicated the relatedness between two documents only for the training purpose. However, it is not required after deployment. Just the title and/or the abstract of a document are required to generate a representation for this document. This paper proposes the SPECTER [1] as a good candidate for sentences-level representation. This is because of two reasons. The first reason is the input title is
basically just a sentence. The second reason is that the citation information as a relatedness information between document can represent a semantic relationship between these sentences (the titles). Therefore, SPECTER [1] can be a competitive candidate for sentence-level representation, without any need for further training nor fine-tuning. The model is evaluated as sentence-representation model for several tasks to measure its performance and compare it to the well-known sentence representation models, and it is expected to generate an effective sentence representation.

## Methodology
For the SPECTER model itself, no change has been applied on it. For the input, the default input of the model is the title and the abstract of the input document separated by the [SEP] token: (title + [SEP] + abstract). However, the method to use SPECTER for our sentence representation, it is very easy. Therefore, this input sentence will be treated as the
document title and abstract will be empty: (sentence + [SEP] + ‘” ”). The output will be a fixed-length representation of the input sentence that can be used for any possible downstream task. This previously explained simple use of SPECTER as a sentence-level representation model is evaluated by the SentEval [2], which is an evaluation toolkit for universal sentence representations that provides evaluation over many types of classification tasks and semantic similarity tasks.

## Experiment and Results
Our experiment is to evaluate the SPECTER [1] performance as a sentence-representation model. This is done using the SentEval[2] evaluation toolkit. The tasks used to evaluate our model are several, but they are of two main groups: the classification tasks group, and the semantic similarity group. For the classification tasks group, the tasks are sentiment analysis tasks (MR, SST-2, and SST-5) [2], question-type task (TREC) [2], product reviews task (CR), Subjectivity/Objectivity task (SUBJ) [2], opinion polarity task (MPQA) [2], paraphrase equivalence detection (MRPC) [2], and entailment detection using SICK dataset (SICK-E) [2].However, the semantic textual similarity tasks group consists of the tasks STS from 12 to 16, which use the STS benchmark dataset [2] which consists of sentence pairs labeled with a semantic relatedness score between 0 and 5. Also, the semantic textual similarity group includes the SICK-R task [2] and the SSTBinary [2] which also measure semantic relatedness between two sentences. All previously mentioned tasks can have several subtasks, so for each main task, when evaluation we will only show the average result over the subtasks. Review table 1 and table 2 that show an example for task explained. We have applied all these tasks on the SPECTER as sentencelevel representation model and we compared these results with the results got from the most used sentence-level representation models like: GloVe [3], FastText [7], InferSent [8], SkipThought[4]. The results of the evaluation of SPECTER in semantic textual similarity tasks were promising. SPECTER achieved very high score outperforms GloVe and SkipThought models. However, the results of classification tasks evaluation were not better than the famous sentence-level representation models, except in the product review (CR) task, where the result of SPECTER [1] in this task was better than GloVe [3] and SkipThought [4].

![Alt text](image-3.png)
![Alt text](image-4.png)

## Future Work Plan
Although SPECTER achieved very good performance in semantic textual similarity tasks, it did not achieve the same good performance for other important classification tasks. Therefore, in this section we propose a plan to develop a new model inspired from SPECTER but dedicated for sentencelevel representation. The new model inspires the same triblet loss function used in training SPECTER, where the triblet consists of anchor, positive sample, and negative sample, and the model train to differentiate between negative and positive. Also, SPECTER uses both easy negative samples and hard negative samples, where easy negative sample is any document not cited by the anchor document, but the hard negative sample is a document not cited by the anchor document but cited by a citation paper. In out proposed model for sentence-level representation, we propose the same methodology, where for each anchor we will provide two triblets (anchor1, positive, easy negative) and (anchor1, positive, hard negative) typically as the SPECTER. To afford this type of data for training the model, we propose to use two powerful datasets: the Stanford Natural Language Inference (SNLI) dataset [5], and the Multi- Genre Natural Language Inference (MultiNLI) dataset [6]. We propose to use both datasets mixed. The two datasets provide for each sentence: a sentence represents an entailment, a sentence represents a contradiction, and a sentence represents a neutral. For our model, to generate the triblet, we will manipulate the datasets in the following way: the entailment sentence will represent a positive sample, the contradiction sentence will represent an easy negative sample, and the neutral sentence will represent a hard negative sample. In our project we have already done all the previously mentioned steps and the triblets have been already generated. This is an example that represents a couple of triblets shown below:

![Alt text](image-5.png)

The next step to be done is just using these generated triblet to train the model. When model training is done, the model will be evaluated the same way [2] we have evaluated SPECTER [1] in this report and the results will be compared to the most used sentence-level representation models results that have been shown in this report too.

## References
[1] A. Cohan, S. Feldman, and I. Beltagy, D. Downey, D. S. Weld,
“SPECTER: Document-level Representation Learning using Citationinformed
Transformers,” ACL, Jul 2020.

[2] A. Conneau, D. Kiela, ‘SentEval: An Evaluation Toolkit for Universal
Sentence Representations,” arXiv preprint arXiv:1803.05449, Mar 2018.

[3] J. Pennington, R. Socher, C. D. Manning, ‘GloVe: Global Vectors for
Word Representation,” EMNLP, Jan 2014.

[4] R. Kiros, Y. Zhu, R. Salakhutdinov, R. S. Zemel, A. Torralba, R. Urtasun,
S. Fidler, “Skip-Thought Vectors,” arXiv:1506.06726 , Jun 2015.

[5] S. R. Bowman, G. Angeli, C. Potts, C. D. Manning, “A large annotated
corpus for learning natural language inference” arXiv:1508.05326, Aug
2015.

[6] S. Kim, I. Kang, N. Kwak, “Semantic Sentence Matching with Denselyconnected
Recurrent and Co-attentive Information,” arXiv:1805.11360,
Nov 2018.

[7] S. Kim, I. Kang, N. Kwak, “Semantic Sentence Matching with Denselyconnected
Recurrent and Co-attentive Information,” arXiv:1805.11360,
Nov 2018.

[8] P. Bojanowski, E. Grave, A. Joulin, T. Mikolov, “EnrichingWord Vectors
with Subword Information,” arXiv:1607.04606, Jun 2017.
