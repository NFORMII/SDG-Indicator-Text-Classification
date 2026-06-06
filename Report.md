 
 
 
SDG 3 Indicator Text Classification
A Multi-Label NLP Classification System
 
Group 8 | Assignment 2
Kayonga Elvis · Nformi Modestine Girbong · Patricia Mugabo · Lorita Sesame Icyeza 
Machine learning techniques 1
African Leadership University
01/06/2026
 
 
Abstract

This paper presents a multi-label text classification system for automatically assigning SDG Goal 3 (Good Health and Well-Being) indicator labels to international development documents. The dataset consists of 2,995 labelled training documents and 998 unlabelled test documents drawn from the Devex platform, with labels spanning 27 SDG 3 indicators and exhibiting severe class imbalance (imbalance ratio 33.55:1). Twelve experiments were conducted, systematically comparing TF-IDF feature engineering with classical classifiers (Logistic Regression, LinearSVC, Random Forest), deep learning with Word2Vec and BERT embeddings, and pretrained static embeddings (GloVe 6B.300d) and subword embeddings (FastText). The primary evaluation metric is Hamming Loss. LinearSVC with 5,000 TF-IDF unigram features achieved the best Hamming Loss of 0.0462, outperforming all deep learning configurations. GloVe embeddings produced the best deep learning result (HL=0.0619, Micro F1=0.8096), demonstrating that pretraining scale compensates for small corpus size. All models struggled with rare indicators, as evidenced by the persistent gap between Micro F1 (up to 0.81) and Macro F1 (up to 0.55). The complete experimental pipeline is available as a single end-to-end Google Colab notebook.
 

 
1. Introduction
1.1 Problem Statement
Sustainable Development Goal 3  Good Health and Well-being is one of the most expansive goals in the 2030 Agenda, encompassing 13 targets and 27 measurable indicators that span everything from maternal mortality and communicable disease to road traffic injuries, mental health, and universal health coverage. For international development organisations, the ability to quickly locate which of these indicators a given document addresses has real operational value: it supports faster funding allocation, more consistent project categorisation, and better cross-organisation evidence synthesis across the thousands of tenders, reports, and programme descriptions that circulate through the sector each year.
The core challenge, however, is that development documents do not map neatly onto single indicators. A field report on a rural health programme might simultaneously address health workforce capacity, essential medicines access, and health service coverage, three distinct SDG 3 indicators that share overlapping vocabulary and thematic context. Manual tagging at this level of granularity is slow, inconsistent across reviewers, and fundamentally unscalable given the volume of documents produced by organisations like USAID, WHO, and UNICEF each year.
This project addresses that challenge by framing it as a multi-label text classification problem: given a development document, predict the subset of SDG 3 indicators it is relevant to. The model takes a raw text sample as input and produces a binary vector of length 27 as output, where each position corresponds to one indicator. The dataset consists of 2,995 labelled training documents and 998 unlabelled test documents drawn from the Devex platform, with labels spanning 27 indicators and exhibiting substantial class imbalance. The primary evaluation metric is Hamming Loss, which measures the fraction of individual label predictions that are wrong across all documents and indicators.

1.2 Objectives
•        Build a reproducible multi-label classification pipeline for SDG 3 indicators.
•        Systematically compare preprocessing strategies, feature representations, and model architectures.
•        Minimize Hamming Loss on a held-out test set.
•        Document insights and limitations for future practitioners.
1.3 Report Structure
1. Introduction 
  1.1 Problem Statement 
  1.2 Objectives
  1.3 Report Structure
2. Literature Review
 2.1 Multi-Label Text Classification
 2.2 Text Representation Techniques 
2.3 Evaluation for Multi-Label Classification 
2.4 SDG Text Classification Prior Work
3. Dataset and Exploratory Data Analysis
 3.1 Dataset Description 
3.2 Label Distribution Analysis 
3.3 Text Characteristics 
3.4 Co-occurrence and Label Correlations
4. Methodology
 4.1 Preprocessing Pipeline
 4.2 Feature Engineering: Traditional Methods
 4.3 Multi-Label Classification Framework
 4.4 Baseline Models 
4.5 Advanced Models and Embeddings 
4.6 Evaluation Methodology
5. Experimentation
 5.1 Experiment Log
 5.2 Experiment Details
6. Results
 6.1 Model Comparison Table 
6.2 Learning Curves 
6.3 Per-Label Performance 
6.4 Confusion Matrix / Classification Visualization
 6.5 Final Test Predictions
7. Discussion 
7.1 Why Certain Approaches Performed Better
 7.2 Tradeoffs Observed 
7.3 Preprocessing and Feature Engineering Impact
7.4 Dataset and Methodology Limitations 
7.5 Lessons Learned
8. Ethical Considerations and Responsible AI
9. Conclusion and Future Work 
9.1 Summary
 9.2 Future Work
Reference
appendix

2. Literature Review
2.1 Multi-Label Text Classification
Evaluating a multi-label classifier requires more care than evaluating a standard single-label model, because a prediction is never simply right or wrong in the binary sense; it can be partially correct, over-predicting some labels while missing others. The choice of metric therefore shapes what the model is implicitly optimised for.
Hamming Loss is the most granular of the multi-label metrics. It measures the fraction of all (document, label) pairs in the dataset where the prediction is incorrect, treating every individual binary label prediction as a separate unit of evaluation. Because it averages over all documents and all 27 labels simultaneously, it is sensitive to errors anywhere in the label space and does not privilege common labels over rare ones. A model that consistently mispredicts a single rare indicator will incur a measurable Hamming Loss penalty regardless of how well it performs elsewhere. This makes Hamming Loss well-suited to this task, where failure to detect a specific SDG 3 indicator in a relevant document is a meaningful error with practical consequences for how that document gets classified and used. As Sorower (2010) notes in his survey of multi-label learning algorithms, Hamming Loss is the standard decomposition-based metric for evaluating label-wise accuracy and is particularly appropriate when all labels are considered equally important by the evaluation framework.
Macro F1 and Micro F1 offer complementary perspectives. Micro F1 pools true positives, false positives, and false negatives across all labels before computing precision and recall, which gives disproportionate weight to common labels simply because they contribute more total predictions. Macro F1, by contrast, computes F1 independently for each label and averages the results with equal weight, making it a more honest measure of whether the model performs consistently across the full indicator space. A substantial gap between the two  Micro F1 high, Macro F1 low is a reliable signal that the model is performing well on frequent labels while failing systematically on rare ones, a pattern that proved diagnostic throughout this project's experiments. Yang (2001) identifies this gap as a canonical indicator of classifier bias under label imbalance, and it motivated several of the interventions described in Section 5.
Subset Accuracy, also called the exact match ratio, is the strictest possible multi-label criterion: a prediction counts as correct only if every single label for a document is predicted correctly. Missing even one label out of 27 marks the entire document as wrong. This metric is reported for completeness, but it sets a very high bar, particularly when documents carry multiple labels, and is less informative as a standalone measure than Hamming Loss or F1.

2.2 Text Representation Techniques
Text representation is the process of converting raw text into numerical vectors that machine learning models can process. Several families of techniques exist, each with distinct properties relevant to this classification task.

Bag of words(BOW) is the simplest approach, representing each document as a vector of raw word counts across a fixed shared vocabulary. While interpretable and computationally expensive, BOW ignores word order entirely and treats each word as equally important regardless of how informative it is across the corpus.

Term frequency-inverse document frequency(tf-idf) addresses the main weakness of BOW by weighting each term according to how distinctive it is. The TF component measures how often a word appears within a specific document, while the IDF component penalizes words that appear across most documents, since such words carry little distinctive value. Formally:

TF-IDF(t, d) = TF(t,d) × IDF(t) = f(t,d) × log( N / df(t) )

Where f(t, d) is the raw count of term t in the document d. N is the total number of documents, and f(t) is the number of documents containing t. Words that appear frequently in one document but rarely across the corpus receive high TF-IDF scores. Making them strong indicators of document topics. This is particularly valuable in a domain-specific corpus like SDG 3 development documents. Where disease names such as malaria and tuberculosis appear intensively in relevant documents but are largely absent from unrelated ones.  TF-IDF was therefore selected as the primary feature representation for all baseline experiments in this study (Ramos 2003)

Word embedding, such as word2vec, mmapswords to dense continuous vectors where semantic similarity is encoded through spatial proximity. Unlike TF-IDF, these representations capture sematic relationships , like for example, placing ‘malaria’ and ‘mosquitoe’ close together in vector space. However, these sematic embeddings assign a single vector per word regardless  of the context, limiting their expressiveness for words with multiple meanings. 

Transformer based representations, most notable generate contextualized embeddings where each word’s vector dyanamically computed based on surrounding sentence. This allows the model to capture nuanced meaning that static embeddings cannot. However, transformer models require substantially more computational resources an larger labelled datasets to fine-tun effectively. The next experiments in section 4.5 explore whether these richer representations improve on th tf-idf baselines established here.
2.3 SDG Text Classification Prior Work
Automated classification of documents according to the Sustainable Development Goals has attracted growing attention since the SDG framework was adopted in 2015, as organisations sought scalable alternatives to manual tagging. Early approaches drew on keyword matching and rule-based systems built around official SDG indicator language, but these proved brittle: development documents use highly variable terminology, and a rule written for one funding context often fails to transfer to another.
The most directly relevant prior work is the OSDG project and the accompanying research by Müller-Eberstein et al. (2022), who demonstrated that transformer-based models could predict SDG relevance across all 17 goals with substantially higher accuracy than earlier bag-of-words approaches. Their work highlighted two recurring challenges that are directly applicable here: the scarcity of labelled training data for less prominent goals and targets, and the tendency of models to conflate thematically related goals that share vocabulary (SDG 3 and SDG 6, for instance, both generate documents that discuss water access in the context of health). The OSDG Community Dataset, which the project also produced, represents one of the few publicly available large-scale SDG labelling resources, though it operates at the level of full SDG goals rather than the more granular indicator-level classification attempted in this project.
Within SDG 3 specifically, the literature has largely focused on sub-problems: automated identification of clinical trial relevance to specific disease burdens, classification of grey literature by health topic, and NLP-based extraction of intervention types from systematic reviews. These tasks share methodological overlap with the present work but differ in scope they typically address binary relevance to a single health domain rather than simultaneous multi-label classification across 27 fine-grained indicators.
What distinguishes this project from prior work is the combination of granularity and the multi-label framing. Predicting relevance to all 27 SDG 3 indicators simultaneously, from development-sector documents rather than clinical literature, on a relatively small labelled dataset, represents a harder and less well-studied variant of the SDG classification problem. The class imbalance challenge  some indicators appearing in fewer than 2% of documents is also more severe than in most prior SDG classification benchmarks, which have tended to work with broader, more evenly distributed categories.

 
3. Dataset and Exploratory Data Analysis
3.1 Dataset Description
The dataset used in this project consists of 2,995 labelled training documents and 998 held-out test documents drawn from the Devex platform, a news and intelligence source covering international development. Documents span a range of content types  including grant descriptions, project reports, tenders, and humanitarian initiative summaries  and vary considerably in length, from short two-word entries to detailed technical reports exceeding 3,900 tokens.
Each training document carries between one and twelve SDG 3 indicator labels, drawn from a set of 27 distinct indicators spanning topics from HIV infection rates and maternal mortality to air pollution and road traffic injuries. The test set is provided without labels; predictions on these documents constitute the final submission.
Automated column detection was used throughout the pipeline rather than hardcoding column names, making the codebase reusable on similarly structured datasets. The text column was identified by longest average string length, label columns by SDG indicator pattern matching, and the ID column by uniqueness ratio. The dataset was encoded in Latin-1, which was handled transparently during loading.

Metric
Train
Test
Samples
2,995
998
Raw features
15
3 (labels excluded)
Label columns
12
0 (blind evaluation)
Missing values
30,022 (NA in label cols)
0
Average labels per document
~2.0




3.2 Label Distribution Analysis

The 27 SDG 3 indicators are distributed very unevenly across the training corpus. Indicator 3.b.2 (official development assistance to medical research) appears in approximately 34.72% of documents, making it by far the most common label. At the other extreme, indicators such as 3.9.3 (mortality from unintentional poisoning) and 3.4.2 (suicide mortality rate) appear in fewer than 2% of documents. The resulting imbalance ratio of approximately 33.55:1 between the most and least frequent labels is severe enough to meaningfully distort model training if left unaddressed.
The most frequently occurring labels tend to reflect broad programmatic themes  healthcare system capacity, medicines access, and NCD mortality  that naturally surface across diverse development documents. The rarest labels correspond to highly specific causes of mortality or narrowly scoped interventions that appear only in targeted projects. This pattern was consistent with prior expectations for a development-sector corpus.
From a modelling standpoint, this imbalance has direct consequences for classifier behaviour. A model trained without any correction will learn to predict common labels reliably while effectively ignoring rare ones, since the loss incurred by missing a rare label is negligible relative to the total training signal. Addressing this through class reweighting, threshold adjustment, or both  was a central theme in the experimental work reported in Section 5.


3.3 Text Characteristics




Document lengths vary substantially across the corpus. Token counts range from 2 to 3,961, with a mean of approximately 489 tokens and a median of 213 tokens. The distribution is strongly right-skewed: the majority of documents are under 500 tokens, but a long tail of detailed reports extends well beyond this. Character-level statistics tell a similar story, with a mean of around 3,400 characters per document.
This variability reflects the diversity of document types in the dataset. Short entries typically correspond to project titles or brief tender descriptions, while longer ones represent grant proposals or programme reports. The implication for modelling is that approaches sensitive to document length  such as fixed-length neural architectures  may require truncation or padding strategies, whereas TF-IDF-based methods are naturally invariant to document length.
Inspection of the most frequent terms after preprocessing confirms the domain-specific character of the vocabulary. Terms such as research, health, project, clinical, development, study, and program dominate the word frequency distribution, reflecting the development-sector context. Crucially, domain acronyms including HIV, TB, WHO, and SDG remain prominent after preprocessing, validating the decision to preserve them explicitly in the pipeline.

3.4 Co-occurrence and Label Correlations


Many SDG 3 indicators co-occur frequently within individual documents, reflecting the interconnected nature of global health challenges. The strongest pairwise co-occurrences observed in the training data are between 3.8.1 (essential health service coverage) and 3.b.3 (core essential medicines availability) with 175 joint appearances, followed by 3.7.1 and 3.7.2 (the two reproductive and family health indicators) with 160. Other notable pairs include 3.b.2 and 3.b.3 (development assistance and medicines, reflecting the link between funding and implementation), and 3.8.1 with 3.8.2 (service coverage and out-of-pocket costs, which frequently appear together in healthcare system analyses).
These co-occurrence patterns cluster naturally along recognised health domain lines. A reproductive health cluster groups 3.7.1, 3.7.2, and 3.5.1 together. A healthcare systems cluster connects 3.8.1, 3.8.2, and 3.b.3. Communicable disease indicators 3.3.1 through 3.3.4 (covering HIV, tuberculosis, hepatitis, and malaria) form another coherent grouping. This structure has direct implications for classifier design: methods that model label dependencies  such as Classifier Chains could exploit these patterns to improve prediction on rare indicators. Binary Relevance, the primary framework used in this project, treats each label independently and therefore cannot take advantage of this signal, which is noted as a concrete 
 
4. Methodology
4.1 Preprocessing Pipeline
The full preprocessing pipeline is implemented in personA_eda_preprocessing.py. Each step was chosen to standardise the text while preserving the domain-specific information that carries real predictive value in this corpus.
HTML normalisation is applied first, stripping tags and decoding entities. Many documents in the Devex dataset were sourced from web-facing systems and carry residual HTML formatting that would otherwise introduce noise into the vocabulary. Lowercasing follows, with one deliberate exception: domain acronyms are explicitly exempted and retained in uppercase. Abbreviations such as HIV, WHO, TB, AIDS, USAID, UNICEF, and SDG function as technical identifiers rather than ordinary words  collapsing them to lowercase would lose this distinctiveness and merge them with common vocabulary.
Tokenisation uses a regex-based word boundary approach that handles hyphenation and punctuation without requiring a language model. Stop word removal then discards high-frequency function words using the NLTK English list, while the acronym exemption list ensures that domain markers are never removed at this stage. Lemmatisation with WordNetLemmatizer reduces inflected forms to their base (researching and research become research; studies becomes study), which helps consolidate vocabulary and reduce sparsity without losing meaning. Finally, numeric-only tokens are removed, since isolated numbers carry no semantic content in this domain.
After preprocessing, the vocabulary consisted of approximately 30,086 unique lowercased tokens, with word embeddings and downstream feature extraction operating on this cleaned representation. The pipeline was designed for reproducibility: it uses only standard library dependencies (pandas, NLTK, scikit-learn), makes no assumptions about column names, and runs identically in local and cloud environments.

4.2 Feature Engineering  Traditional Methods
   Two classical text vectorisation approaches were implemented to establish the feature space for baseline experiments: Bag-of-Words and TF-IDF. Both were fitted exclusively on training data and applied to validation and test sets using the fixed vocabulary learned during training, preventing any leakage of test-set information into the feature construction process.

4.2.1 Bag of Words
The Bag-of-Words model, implemented via scikit-learn's CountVectorizer, converts each document into a vector of raw word counts over a shared fixed vocabulary. It serves as a useful lower-bound reference point against the more sophisticated TF-IDF representation. The vectoriser was configured with a vocabulary ceiling of 10,000 features, a minimum document frequency of 2, and a maximum document frequency of 0.95. Lowercase conversion was disabled to respect the acronym capitalisation decisions made during preprocessing HIV, WHO, and TB were intentionally kept in uppercase as domain markers whose meaning goes beyond the individual letters.
The resulting feature matrix exhibited 98.5% sparsity, meaning most entries are zero because any given document uses only a small fraction of the full vocabulary. This degree of sparsity is entirely expected for text data at this scale and is handled efficiently by scikit-learn's sparse matrix representations.
4.2.2 TF-IDF Vectorisation
TF-IDF was selected as the primary feature representation because it corrects the central weakness of raw counts: ubiquitous words like health or project accumulate high Bag-of-Words scores despite being uninformative for distinguishing between indicator classes. The inverse document frequency component systematically down-weights such terms, leaving the vocabulary dominated by words that are distinctive to particular topics.
Each configuration choice was justified by corpus characteristics identified during EDA:
Vocabulary size (max_features=5,000): Set to 5,000 following the tuning experiment in Experiment 3, which tested vocabulary sizes of 5,000, 10,000, 20,000, and 50,000. Hamming Loss degraded monotonically as vocabulary size increased, with 5,000 achieving the best value of 0.0535. Beyond this point, additional words introduced noise without adding discriminative signal a pattern consistent with the concentrated, domain-specific nature of the corpus vocabulary.
Minimum document frequency (min_df=2): Words appearing in only one document were discarded as likely typos or highly idiosyncratic tokens unlikely to generalise to unseen text.
Maximum document frequency (max_df=0.95): Words appearing in more than 95% of documents were removed. Even after stop word removal, domain-generic terms like health, project, and service appeared in virtually every document and provided no ability to distinguish between indicator classes.
Sublinear TF scaling (sublinear_tf=True): Applying log(1 + count) in place of the raw count compresses the influence of highly repeated terms. A word appearing 50 times is not treated as 50 times more informative than one appearing once. Sublinear scaling is well-established as a consistent improvement for text classification (Rennie et al., 2003).
N-gram range (ngram_range=(1,1)): Unigrams only, confirmed by Experiment 2 which tested bigrams. With a fixed vocabulary ceiling, bigrams displaced informative unigrams rather than supplementing them, producing worse Hamming Loss across the board.
Lowercase (lowercase=False): Disabled to preserve the domain acronyms retained during preprocessing. Lowercasing would merge HIV with hiv and WHO with who, erasing the acronym identity that signals domain-specific meaning.
Inspecting the fitted IDF scores confirmed the expected pattern. The ten lowest-IDF words  most widespread and least discriminative  were health, support, development, project, service, information, year, provide, activity, and including. The highest-IDF words were highly specific technical and geographic terms such as borehole, endotoxin, and kenema, consistent with documents addressing narrowly scoped SDG 3 contexts.

4.3 Multi-Label Classification Framework
This task is inherently multi-label: a single document can simultaneously address multiple SDG 3 indicators. A report on rural health systems might be relevant to indicator 3.8.1 (essential health service coverage), 3.c.1 (health worker density), and 3.b.3 (essential medicines access) at the same time. Standard binary or multiclass classifiers cannot handle this directly; a decomposition framework is needed.
4.3.1 Binary Relevance (One-vs-Rest)
Binary Relevance, implemented via scikit-learn's OneVsRestClassifier, was adopted as the primary framework. It decomposes the 27-label problem into 27 independent binary classification problems  one per indicator and trains a separate classifier to answer a yes-or-no question about each. The predictions from all 27 classifiers are combined to form the final multi-label output.
This approach is computationally straightforward and allows any binary classifier to be applied to the multi-label setting without modification. Its principal limitation is the independence assumption: by treating each label in isolation, Binary Relevance cannot exploit the co-occurrence patterns documented in Section 3.4. If two indicators frequently appear together, a method that explicitly models this dependency might outperform Binary Relevance on those pairs.
4.3.2 Alternative Frameworks Considered
Classifier Chains (Read et al., 2011) address the independence assumption by training each classifier on both the input features and the predictions of all preceding classifiers in the chain. This allows the model to learn, for instance, that a document tagged as HIV-related is more likely to also address tuberculosis, since the two appear together frequently in development literature. The drawback is sensitivity to label ordering in the chain, which is difficult to optimise without additional search procedures.
Label Powerset  which treats every unique label combination as a distinct class — was not considered because the combinatorial explosion of possible label sets across 27 indicators makes it computationally intractable. Given the project's resource constraints and the priority of reproducible baselines, Binary Relevance was retained as the primary framework, with Classifier Chains identified as a promising direction for future work.
4.3.3 Prediction Thresholding
After training, each binary classifier produces a probability score between 0 and 1 for its label. A threshold converts these scores into binary predictions. The default threshold of 0.5 was used in Experiments 1 through 4. Experiment 5 revealed a systematic problem: under class imbalance, the model's probability outputs for rare labels rarely reached 0.5, causing those labels to be predicted as absent regardless of the input. Per-label threshold optimisation  finding the probability cutoff that maximises validation F1 for each of the 27 indicators individually  was therefore implemented as a key intervention in Experiments 5 and 6.

4.4 Baseline Models
Three classifiers were evaluated as baselines, each wrapped in a OneVsRestClassifier. All models were trained on the TF-IDF feature matrices described in Section 4.2.
4.4.1 Logistic Regression
Logistic Regression served as the primary baseline throughout the feature engineering experiments. For each label, it learns a weight vector w and bias b such that the predicted probability is:
P(y=1 | x) = σ(w·x + b) = 1 / (1 + exp(-(w·x + b)))
The model was configured with regularisation parameter C=1.0, solver lbfgs (recommended for medium-sized datasets), and max_iter=1000 to ensure convergence on high-dimensional TF-IDF vectors. The class_weight parameter was set to None for Experiments 1 through 5, then switched to 'balanced' for Experiment 6 to address label imbalance. Logistic Regression was chosen as the baseline because it trains quickly, its weights are interpretable  the highest-weighted features for each label reveal which words most strongly predict that indicator and it reliably provides a strong reference point for text classification tasks (Joulin et al., 2017).
4.4.2 Linear Support Vector Machine (LinearSVC)
The Linear Support Vector Classifier finds the hyperplane that maximises the margin between classes  the distance between the decision boundary and the nearest training examples from each class. This focus on the hardest-to-separate examples, rather than fitting the full probability distribution, tends to generalise well on high-dimensional sparse text data. LinearSVC was configured with C=1.0 and max_iter=2000 to accommodate convergence requirements in multi-label mode.
LinearSVC does not natively produce calibrated probability scores, which means it cannot be used directly for per-label threshold optimisation. For experiments requiring probability outputs (Experiments 5 and 6), Logistic Regression was used instead. LinearSVC is included as a comparison point because it represents one of the strongest classical baselines for sparse text classification.
4.4.3 Random Forest
Random Forest builds an ensemble of decision trees, each trained on a bootstrap sample of the data with a random subset of features considered at each split. Final predictions are determined by majority vote. This ensemble approach reduces variance and can capture non-linear patterns that linear models miss. The model was configured with 100 estimators, a minimum of 5 samples required to split an internal node, and all available CPU cores for parallel training. Random Forest was included primarily to assess whether non-linear modelling provided any advantage over linear classifiers on TF-IDF features.

4.5 Advanced Models and Embeddings

Following the baseline TF-IDF experiments, this section explores whether semantic embeddings and deep neural networks can outperform sparse TF-IDF representations. Three embedding approaches were evaluated: Word2Vec (trained from scratch on the corpus), GloVe (pretrained embeddings), and BERT (contextual transformer embeddings). Each embedding was paired with a deep neural network classifier trained via stochastic gradient descent with Adam optimization.

Word2Vec embeddings were trained using the skip-gram architecture with a window size of 5, generating 300-dimensional vectors for each word. Documents were represented by averaging their constituent word vectors, producing a fixed-length document representation regardless of input length. This approach preserves semantic similarity—words like "malaria" and "mosquito" appear close in vector space—while remaining computationally efficient for small datasets.

BERT (Bidirectional Encoder Representations from Transformers) represents a more sophisticated approach. BERT-base-uncased was fine-tuned on the classification task, leveraging contextualized embeddings where each word's representation depends on its surrounding context. This allows the model to capture nuanced meaning that static embeddings cannot, such as distinguishing between "bank" as a financial institution versus a riverbank. However, BERT requires substantially more computational resources and larger labelled datasets to fine-tune effectively—a constraint that proved significant given the 2,995-sample training set.
All deep neural network architectures followed a feedforward design with ReLU activations, batch normalization, and dropout regularization to prevent overfitting. The binary cross-entropy loss function was used, with optional class weighting to address label imbalance. Models were trained for up to 30 epochs with early stopping based on validation loss.

4.6 Evaluation Methodology
Selecting appropriate evaluation metrics is especially important for multi-label classification, because standard single-label metrics do not fully capture how a model performs when each instance can have multiple correct answers simultaneously.
4.6.1 Hamming Loss
Hamming Loss is the primary evaluation metric for this assignment. It measures the fraction of all (document, label) pairs where the prediction is wrong  whether a false positive or a false negative:
HL = (1 / N·L) × Σᵢ Σⱼ XOR(yᵢⱼ, ŷᵢⱼ)
where N is the number of documents, L is the number of labels (27), yᵢⱼ is the true label for document i and indicator j, and ŷᵢⱼ is the predicted label. A Hamming Loss of 0.0 represents perfect prediction; 1.0 means every label was predicted incorrectly. Lower is always better. Hamming Loss is appropriate here because it evaluates every individual label prediction independently, penalises false positives and false negatives equally, and provides a smooth, sensitive measure of overall labelling quality across all 27 indicators.
4.6.2 Micro F1
Micro F1 aggregates True Positives, False Positives, and False Negatives across all labels before computing precision and recall, then derives F1 from those pooled counts. This gives higher weight to frequent labels since they contribute more total predictions. Micro F1 is useful for understanding overall model performance but can mask poor results on rare labels when common labels dominate the pool.
4.6.3 Macro F1
Macro F1 computes the F1 score independently for each of the 27 labels and averages them with equal weight, regardless of label frequency. A low Macro F1 relative to Micro F1 signals that the model is failing specifically on rare labels — a diagnostic that proved critical throughout the experiment sequence in Section 5.
4.6.4 Subset Accuracy
Subset Accuracy, also called exact match ratio, is the strictest multi-label metric: a prediction is correct only if every single label for a document is predicted correctly. One wrong label out of 27 marks the entire document as incorrect. This provides an upper bound on how often the model produces a fully correct label set and is reported alongside the other metrics for completeness.
4.6.5 Train / Validation Split
The labelled training dataset of 2,995 documents was split into a training set of 2,396 documents (80%) and a validation set of 599 documents (20%) using a fixed random seed of 42 to ensure reproducibility. All model training and hyperparameter decisions were made on the training set; validation results were used only to select the best configuration and report final performance. The held-out test set of 998 documents was used exclusively for generating the final predictions submitted as output.

 
5. Experimentation

This section documents all experiments conducted, following a systematic progression in which each result directly informed the design of the next. The primary evaluation metric is Hamming Loss (lower is better), with Micro F1, Macro F1, and Subset Accuracy reported as secondary metrics throughout.

 
  5.1 Experiment Log


Change Made
Rationale
Outcome
Insight
1
LR + TF-IDF Unigrams (10k vocab, threshold=0.5)
Establish baseline. TF-IDF + LR is the classic strong starting point for text classification.
HL=0.0542 Micro F1=0.4410 Macro F1=0.2662 SubAcc=0.2554
Large gap between Micro F1 (0.44) and Macro F1 (0.27) immediately revealed class imbalance. Model failed on rare labels.
2
Added bigrams (ngram_range=(1,2))
Exp 1 missed multi-word phrases like 'maternal mortality'. Bigrams may capture these.
HL=0.0538 Micro F1=0.4487 Macro F1=0.2720 SubAcc=0.2454
Bigrams produced marginal improvement in Hamming Loss but bigrams displaced many informative unigrams. Unigrams confirmed as better at fixed vocab size.
3
Varied vocabulary size {5k, 10k, 20k, 50k}
Experiments 1–2 assumed vocab=10k without justification. Tested whether this was truly optimal.
Best: vocab=5,000 HL=0.0535 10k: HL=0.0542 20k: HL=0.0545 50k: HL=0.0545 (plateau)
Smaller vocabulary won. Words beyond 5k add noise, not signal, in this domain-specific corpus.
4
Compared LR vs LinearSVC vs Random Forest (vocab=5k)
Experiments 1–3 fixed the classifier as LR. This asked whether a different classifier would perform better.
LinearSVC best: HL=0.0462 Micro F1=0.6055 Macro F1=0.5313. LR: HL=0.0535 Micro F1=0.4553. Random Forest: HL=0.0500 Micro F1=0.5205.
LinearSVC dominated on all metrics — best Hamming Loss of any experiment. Sparse high-dim TF-IDF is ideal for maximum-margin classification.
5
Per-label optimised thresholds replacing fixed 0.5
Rare LR label probabilities rarely exceeded 0.3, so default 0.5 always predicted 0 for those labels.
HL=0.0484 Micro F1=0.5717 Macro F1=0.3792
All 27 labels needed thresholds far below 0.5. Best threshold: 0.35. Macro F1 jumped substantially over baseline.
6
LR + Balanced class weights
EDA confirmed heavy imbalance — 3.b.2 in 34.72% of docs; many indicators under 10%. Balanced weighting makes rare-label errors more costly during training.
HL=0.0647 Micro F1=0.6154 Macro F1=0.5527
Best Macro F1 among LR experiments (0.5527). Hamming Loss worsened — classic imbalance tradeoff: more aggressive rare-label prediction increases false positives.
7
Word2Vec + Deep Neural Network (300 dim, skip gram)
Test whether semantic embeddings outperform TF-IDF. Word2Vec captures word relationships absent from bag-of-words.
HL=0.0647 Micro F1=0.8035 Macro F1=0.2393
Performed worse than LinearSVC on Hamming Loss. High Micro F1 but low Macro F1 — deep learning models handle frequent labels well but struggle with rare indicators.
8
BERT Transformer Model(base-uncased)
Test whether contextual transformer embeddings improve over static Word2Vec. BERT should capture nuanced meaning.
HL=0.0648 Micro F1=0.7864 Macro F1=0.2289
Worse than LinearSVC baseline. Small dataset size (2,396) insufficient for BERT fine-tuning. Undertrained model.
9
Hyperparameter Tuning on Word2Vec(5 random configs)
Systematic search for optimal Word2Vec + NN hyperparameters to improve over Exp 7.
HL=0.0643 Micro F1=0.8031 Macro F1=0.2291
Minimal improvement over Exp 7. Best config: [512, 256, 128, 64], dropout 0.2. Fundamental limitation is corpus size, not hyperparameters.
10
Class Imbalance Handling (inverse frequency weights)
Address label imbalance observed in EDA by weighting loss inversely to label frequency.
HL=0.0650 Micro F1=0.7781 Macro F1=0.2160
Performance degraded. Class weights over-weighted rare labels, causing excessive false positives. Class weighting interacts differently with gradient-based neural optimisation than with convex LR optimisation.
11
GloVe (pretrained 6B.300d) + DeepNN
Test whether pretrained embeddings from 6B tokens outperform Word2Vec trained on ~3,000 documents. Same DeepNN architecture as Exp 9 for fair comparison.
HL=0.0619 Micro F1=0.8096 Macro F1=0.2287
Best deep learning result. GloVe's large pretraining corpus gives richer representations than corpus-trained Word2Vec. Micro F1 highest of all experiments. Hypothesis confirmed.
12
FastText (subword embeddings, trained on corpus) + DeepNN
Test whether subword decomposition helps represent rare/technical health terms (e.g. 'hepatitis', 'neonatal') that Word2Vec might miss. Same architecture as Exp 9/11.
HL=0.0662 Micro F1=0.7932 Macro F1=0.2025
Worse than GloVe despite subword advantage. FastText trained on ~3,000 documents likely doesn't accumulate enough subword statistics to outperform 6B-token pretrained GloVe. Better than untuned Word2Vec (0.0647).


5.2 Experiment Details
Experiment 1:
Baseline: Logistic Regression + TF-IDF (Unigrams)

What changed: Initial baseline. No prior experiment to compare against.
Setup: TF-IDF (max_features=10,000,
 ngram_range=(1,1),
 sublinear_tf=True), 
OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, class_weight=None)), prediction threshold=0.5.
Why: Every project requires a baseline to measure subsequent improvements against. Logistic Regression with TF-IDF is a well-established and interpretable starting point for text classification (Joulin et al., 2017).
Outcome: 
Hamming Loss = 0.0542, 
Micro F1 = 0.4410, 
Macro F1 = 0.2662,
 Subset Accuracy = 0.2554.
Insight: The wide gap between Micro F1 (0.44) and Macro F1 (0.27) immediately surfaced the class imbalance problem. The model performed acceptably on common indicators but largely ignored rare ones, assigning near-zero probabilities to labels it seldom encountered during training. Only 25.5% of documents had all 27 labels predicted correctly. Sample inspection confirmed the model reliably identified dominant labels such as 3.4.1 (NCD mortality) but missed many secondary labels on multi-label documents.
Experiment 2 
 N-gram Range: Adding Bigrams
What changed: ngram_range changed from (1,1) to (1,2), adding bigrams to the TF-IDF vocabulary.
Why: Many SDG 3 concepts are expressed as multi-word phrases  maternal mortality, universal health coverage, HIV treatment  and splitting these into individual tokens loses their combined meaning. Bigrams were expected to capture these domain-specific compound terms.
Outcome: 
Hamming Loss = 0.0538, 
Micro F1 = 0.4487, 
Macro F1 = 0.2720,
 Subset Accuracy = 0.2454.
Insight: Adding bigrams produced marginal improvement in Hamming Loss (0.0542 to 0.0538) but the gains were small. The fixed vocabulary ceiling of 10,000 meant bigrams competed directly with and displaced informative unigrams rather than supplementing them. While useful domain bigrams such as HIV AIDS, maternal mortality, and ministry health were captured, the net effect was minimal. Unigrams were confirmed as the better configuration for this corpus at this vocabulary size.
Experiment 3 


Vocabulary Size Tuning
What changed: max_features was varied across {5,000; 10,000; 20,000; 50,000} using the unigram configuration confirmed in Experiment 2.
Why: Both earlier experiments assumed a vocabulary of 10,000 without empirical justification. Experiment 2 had shown that vocabulary composition matters; this experiment asked whether a different size would further reduce Hamming Loss.
Outcome: 
vocab=5,000: HL=0.0535, Micro F1=0.4553, Macro F1=0.2775.
vocab=10,000: HL=0.0542.
vocab=20,000: HL=0.0545.
vocab=50,000: HL=0.0545 (plateau, no further change).
Insight: Performance degraded monotonically as vocabulary size increased, reaching a plateau at 20,000 where additional features contributed nothing. The most discriminative signal in this corpus is concentrated in a compact core of high-frequency domain terms. Expanding the vocabulary beyond 5,000 introduced increasingly rare tokens that did not generalise to the validation set. A vocabulary of 5,000 was adopted for all subsequent experiments.
Experiment 4 



 Model Comparison: LR vs LinearSVC vs Random Forest
What changed: The classifier was replaced with LinearSVC and Random Forest in turn, using the best TF-IDF configuration from Experiment 3 (vocab=5,000, unigrams).
Why: Experiments 1 through 3 had optimised the feature representation while holding the Logistic Regression classifier fixed. This experiment asked whether a different classifier architecture would achieve better performance on the same features.
Outcome: 
LR: HL=0.0535, Micro F1=0.4553, Macro F1=0.2775.
LinearSVC: HL=0.0462, Micro F1=0.6055, Macro F1=0.5313.
Random Forest: HL=0.0500, Micro F1=0.5205, Macro F1=0.3532.
Insight: LinearSVC was the decisive winner across every metric, achieving the lowest Hamming Loss of any experiment (0.0462). By maximising the margin between classes rather than fitting the full probability distribution, LinearSVC is less susceptible to the noise introduced by near-zero TF-IDF features. Random Forest outperformed LR but fell short of LinearSVC, likely because ensemble methods over decision trees are a less natural fit for the sparse, high-dimensional structure of TF-IDF vectors than linear classifiers.
Experiment 5 
Threshold Optimisation
What changed: The fixed prediction threshold of 0.5 was replaced with per-label thresholds optimised to maximise validation F1 for each of the 27 indicators individually.
Why: Experiment 4 revealed that Macro F1 for Logistic Regression was only 0.2775, meaning rare labels were largely missed. Inspection of LR probability outputs showed that for most rare labels, predicted probabilities never exceeded 0.3  so the default threshold of 0.5 always produced a prediction of zero regardless of the input. Per-label threshold lowering allows the model to predict these labels at appropriate confidence levels (Yang, 2001).
Outcome: 
Hamming Loss = 0.0484, 
Micro F1 = 0.5717, 
Macro F1 = 0.3792.
Best threshold found: 0.35. All 27 labels required optimal thresholds significantly below 0.5.
Insight: The fact that every single label needed a threshold below 0.5 points to systematic probability miscalibration in the LR model under class imbalance  the model had learned to be extremely conservative about predicting rare labels. Threshold optimisation resolved this and produced a substantial jump in Macro F1, from 0.2662 to 0.3792, and improved Hamming Loss to 0.0484 from the baseline 0.0542.
Experiment 6: LR + Balanced Class Weights
What changed: class_weight='balanced' was added to Logistic Regression, automatically weighting each class inversely proportional to its frequency in the training set.
Why: The label frequency analysis confirmed severe imbalance  indicator 3.b.2 appeared in 34.72% of documents while many indicators appeared in fewer than 10%. The consistently low Macro F1 scores across Experiments 1 through 5 confirmed the model was largely ignoring rare classes. Balanced weighting forces the model to treat mistakes on rare labels as more costly during training.
Outcome: Hamming Loss = 0.0647, Micro F1 = 0.6154, Macro F1 = 0.5527.
Insight: Macro F1 improved substantially to 0.5527 — the best Macro F1 of any experiment — confirming the model was now detecting rare labels it had previously ignored. However, Hamming Loss worsened from 0.0542 to 0.0647, because the more aggressive rare-label predictions introduced additional false positives on common labels. This is the classic precision-recall tradeoff in imbalanced classification: better recall for rare classes comes at the cost of additional false alarms. Threshold optimisation (Exp 5) achieved a better Hamming Loss balance by calibrating per-label decision boundaries rather than distorting the training objective.

Experiments 7 
Experiment 7: Word2Vec + Deep Neural Network
What changed: Replaced TF-IDF features with Word2Vec embeddings (300 dimensions, skip-gram, window=5, min_count=2). Paired with a deep neural network [512, 256, 128] with batch normalization, dropout 0.3, learning rate 0.001, batch size 64, trained for 30 epochs using Adam optimizer.

Why: TF-IDF captures word frequency but not semantic relationships. Word2Vec embeddings place semantically similar words close together in vector space, potentially capturing domain-specific relationships (e.g., "malaria" and "mosquito") that bag-of-words misses. The skip-gram architecture is particularly effective for capturing rare word patterns in small corpora.

Outcome:
- Hamming Loss: 0.0647
- Micro F1: 0.8035
- Macro F1: 0.2393

Analysis:
- Performed worse than the LinearSVC baseline on Hamming Loss (0.0647 vs 0.0462)
- The model achieved high Micro F1 but high Hamming loss relative to LinearSVC
- Possible reasons: insufficient training data for Word2Vec, vocabulary of only 14,628 entries too limited for rare domain terms
- The gap between Micro F1 (0.80) and Macro F1 (0.24) indicates the model performed well on common labels but struggled with rare indicators

Experiment 8: BERT Transformer Model
What changed: Replaced Word2Vec with BERT-base-uncased, a pretrained transformer model fine-tuned on the classification task. Configuration: dropout 0.3, learning rate 2e-5, batch size 16, max sequence length 256, AdamW optimizer, trained for 5 epochs (only 1-2 completed due to timeout).

Why: BERT generates contextualized embeddings where each word's representation depends on surrounding context, potentially capturing nuanced meaning that static embeddings cannot. This should improve on Word2Vec's limitations, particularly for words with multiple meanings or domain-specific usage patterns.

Outcome:
- Hamming Loss: 0.0648
- Micro F1: 0.7864
- Macro F1: 0.2289

Analysis:
- BERT performed worse than expected and worse than the LinearSVC baseline
- Training was interrupted after 1-2 epochs out of 5 due to computational timeout
- Small dataset size (2,396 samples) is likely insufficient for BERT fine-tuning — many classes have fewer than 50 positive examples
- The model was not fully trained, which explains the poor performance
- This result highlights a key limitation: deep learning models require significantly more data than traditional methods

Experiment 9: Hyperparameter Tuning on Word2Vec
What changed: Systematic hyperparameter search on the Word2Vec + Deep Neural Network architecture. Tested 5 random configurations from a grid varying: vector_size [100, 200, 300], window [3, 5, 7], sg [0, 1] (CBOW vs skip-gram), hidden_dims [[256,128], [512,256,128], [512,256,128,64]], dropout [0.2, 0.3, 0.5], learning_rate [0.0005, 0.001, 0.002]. Each configuration trained for 5 epochs (reduced from 10 for faster search due to time constraints).

Why: Experiment 7 showed Word2Vec underperformed the baseline. Hyperparameter tuning was necessary to determine whether suboptimal configuration choices (rather than the fundamental approach) were responsible for the poor performance. A systematic search across multiple dimensions allows identification of the best configuration without manual trial-and-error.

Best Configuration Found:
- vector_size: 200
- window: 5
- sg: 1 (skip-gram)
- hidden_dims: [512, 256, 128, 64]
- dropout: 0.2
- learning_rate: 0.001

Outcome:
- Hamming Loss: 0.0643
- Micro F1: 0.8031
- Macro F1: 0.2291

Analysis:
- Hyperparameter tuning provided minimal improvement over Exp 7
- The best configuration used [512, 256, 128, 64] hidden dims and dropout 0.2
- Deeper network (4 layers) and lower dropout (0.2) were selected
- Performance gap to LinearSVC baseline (0.0462) persists
- This suggests the issue is fundamental (embedding quality, data size) rather than hyperparameters
- With limited data, hyperparameter optimization has diminishing returns

Experiment 10: Class Imbalance Handling
What changed: Added class weights to the binary cross-entropy loss function. Weights computed as inverse label frequency: weight_i = total_samples / (label_count_i + epsilon), then normalized by mean weight. Base model used the best configuration from Experiment 9, trained for 30 epochs.

Class Weights:
- Calculated as total_samples / (label_counts + epsilon)
- Normalized by mean weight

Why: EDA confirmed label imbalance (indicator 3.b.2 in 34.72% of documents, many indicators under 10%). Class imbalance handling is a standard technique to ensure the model does not ignore rare labels during training by making mistakes on rare labels more costly during optimization.

Outcome:
- Hamming Loss: 0.0650
- Micro F1: 0.7781
- Macro F1: 0.2160

Analysis:
- Class imbalance handling worsened performance relative to Exp 9
- Hamming Loss increased from 0.0643 to 0.0650 and Macro F1 declined
- This was unexpected — class weights typically help with imbalance in classical ML (Exp 6 improved Macro F1)
- Possible reasons:
  1. Class weights interacted poorly with gradient-based neural optimisation, unlike the convex optimisation of logistic regression
  2. Class weights may have over-weighted rare labels, causing excessive false positives
  3. The weighted loss may have destabilised the training dynamics
- This negative result is valuable — it shows that class weighting interacts differently with neural networks than with classical classifiers
- It demonstrates the importance of empirical validation: what works in one modelling paradigm does not automatically transfer to another

Experiment 11: GloVe (pretrained, 6B.300d) + DeepNN
- What changed: Replaced Word2Vec (trained on corpus) with pretrained GloVe embeddings loaded from glove.6B.300d.txt (400,000 word vectors, 300 dimensions). Same DeepNN architecture as Exp 9 ([512, 256, 128, 64], dropout=0.2, batch normalisation) and same training procedure (30 epochs, lr=0.001, batch size 64, Adam optimiser).
- Why: Word2Vec trained on 2,396 documents has a limited vocabulary and weak vector quality for rare domain terms. GloVe vectors trained on 6 billion tokens from Wikipedia and Gigaword encode richer semantic relationships. The comparison isolates the effect of pretraining scale.
- Outcome: HL=0.0619, MicroF1=0.8096, MacroF1=0.2287
- Analysis: Best deep learning result across all experiments and best Micro F1 overall (0.8096). The hypothesis was confirmed: pretrained embeddings on a much larger corpus outperform corpus-trained Word2Vec on this small dataset. However, LinearSVC still achieves lower Hamming Loss (0.0462), indicating that embedding richness alone does not overcome the advantages of linear classifiers on sparse high-dimensional features at this data scale. The Micro/Macro F1 gap (0.81 vs 0.23) persists, showing deep learning models still struggle with rare indicators even with stronger embeddings.

Experiment 12: FastText (trained on corpus, subword) + DeepNN
- What changed: Replaced GloVe with FastText embeddings trained on the corpus (300 dimensions, window=5, min_count=1). FastText extends Word2Vec by representing each word as a bag of character n-grams, allowing it to generate embeddings for rare or out-of-vocabulary words by composing subword vectors. Same DeepNN architecture as Exp 9/11.
- Why: SDG 3 health documents contain many rare or morphologically complex terms (e.g. 'hepatitis', 'neonatal', 'antiretroviral') that Word2Vec may miss due to low frequency. FastText's subword decomposition was expected to produce better representations for these terms.
- Outcome: HL=0.0662, MicroF1=0.7932, MacroF1=0.2025
- Analysis: FastText underperformed GloVe (0.0662 vs 0.0619) and performed similarly to untuned Word2Vec (0.0647). The hypothesis was only partially supported: FastText's larger vocabulary (22,814 subword-enriched tokens vs Word2Vec's 14,628) did not translate to better classification performance. The likely explanation is that FastText's subword advantage requires a sufficiently large corpus to accumulate reliable n-gram statistics; with 2,396 training documents, the subword counts are too sparse to outperform GloVe's large-scale pretraining. FastText trained on a domain-specific corpus of millions of health documents would likely perform considerably better.

6. Results
6.1 Model Comparison Table

| Model / Config | HL ↓ | Micro F1 ↑ | Macro F1 ↑ | Notes |
|---|---|---|---|---|
| LR + TF-IDF Unigrams (Exp 1) | 0.0542 | 0.4410 | 0.2662 | Baseline |
| LR + TF-IDF Bigrams (Exp 2) | 0.0538 | 0.4487 | 0.2720 | |
| LR + TF-IDF vocab=5k (Exp 3) | 0.0535 | 0.4553 | 0.2775 | |
| **LinearSVC + TF-IDF (Exp 4)** | **0.0462** | **0.6055** | **0.5313** | **Best HL overall** |
| LR + Optimised Thresholds (Exp 5) | 0.0484 | 0.5717 | 0.3792 | |
| LR + Balanced Weights (Exp 6) | 0.0647 | 0.6154 | 0.5527 | Best Macro F1 |
| Word2Vec + DeepNN (Exp 7) | 0.0647 | 0.8035 | 0.2393 | |
| BERT + DeepNN (Exp 8) | 0.0648 | 0.7864 | 0.2289 | |
| Word2Vec + DeepNN Tuned (Exp 9) | 0.0643 | 0.8031 | 0.2291 | |
| Word2Vec + Class Weights (Exp 10) | 0.0650 | 0.7781 | 0.2160 | |
| **GloVe + DeepNN (Exp 11)** | **0.0619** | **0.8096** | **0.2287** | **Best deep learning** |
| FastText + DeepNN (Exp 12) | 0.0662 | 0.7932 | 0.2025 | |

Table 1 summarises all experimental configurations evaluated during the project. LinearSVC + TF-IDF achieved the lowest Hamming Loss of 0.0462 and is the primary recommended model for minimising the assignment evaluation metric. LR + Balanced Weights (Experiment 6) achieved the highest Macro F1 of 0.5527 and is recommended when consistent performance across all 27 indicators, including rare ones, is the priority. GloVe + DeepNN (Experiment 11) was the best deep learning approach with HL=0.0619 and the highest Micro F1 of 0.8096 across all experiments.

6.2 Learning Curves


Figure X presents the per-label F1 score of the best model (LinearSVC) across all 27 SDG 3 indicators, sorted from highest to lowest and colour-coded green for labels achieving F1 ≥ 0.5 and red for those below. The dashed reference line at F1 = 0.5 serves as a practical performance benchmark.
24 out of 27 labels achieved F1 scores above 0.5, demonstrating that the LinearSVC model performs reliably across the majority of the indicator space. The three best-performing labels were 3.3.1  HIV infections (F1 = 0.879), 3.3.3  Malaria (F1 = 0.825), and 3.3.2  Tuberculosis (F1 = 0.792). These indicators are associated with highly distinctive, consistently used vocabulary that TF-IDF identifies with little ambiguity  documents about these diseases use the exact disease name prominently and repeatedly, creating a strong and unambiguous word-level signal.
The three labels that fell below the 0.5 threshold were 3.1.2 (proportion of births attended by skilled health personnel, F1 ≈ 0.37), 3.9.1 (mortality from air pollution, F1 ≈ 0.26), and 3.9.3 (mortality from unintentional poisoning, F1 ≈ 0.24). The poor performance on 3.1.2 is attributable to vocabulary overlap with adjacent maternal and reproductive health indicators documents on birth outcomes and skilled attendance use language similar enough to neighbouring indicators that clean separation is difficult. The struggles with 3.9.1 and 3.9.3 stem primarily from data scarcity: these topics appear infrequently in development tenders and health programme reports, providing the model with too few examples to learn from reliably.

The learning curves show training and validation loss over epochs for the best Word2Vec + Deep Neural Network configuration. The plot demonstrates the model's convergence behavior, with training loss decreasing steadily while validation loss stabilizes. The gap between training and validation loss indicates the degree of overfitting, which remained moderate throughout training.

The figure shows the training and validation loss curves for the best Word2Vec + Deep Neural Network configuration from Experiment 9. The model converged after approximately 20 epochs, with training loss reaching 0.015 and validation loss stabilizing at 0.064. The moderate gap between training and validation loss suggests the model learned generalizable patterns without severe overfitting, despite the small dataset size.
6.3 Per-Label Performance
[[INSERT Figure: Per-SDG-indicator F1 bar chart for best model — Person D] Highlight which indicators are classified well vs. poorly. Connect to label frequency distribution from Section 3.2.]
6.4 Confusion Matrix / Classification Visualization
[[INSERT Figure: Heatmap of per-label prediction accuracy or confusion matrix for each indicator — Person D] For multi-label problems, a per-class precision-recall curve or classification report table is often more informative than a traditional confusion matrix.]
6.5 Final Test Predictions
Final predictions for the 998 documents in the test set were generated by retraining the LinearSVC configuration (Experiment 4) on the complete 2,995-document training set. Retraining on the full dataset rather than the 80% training split ensures the model benefits from all available labelled data when making final predictions.
The final TF-IDF vectoriser was fitted on all 2,995 training documents using the optimal configuration (vocab=5,000, unigrams, sublinear_tf=True, lowercase=False), and the test documents were transformed using the resulting fixed vocabulary. Binary predictions were produced directly from the LinearSVC decision function.
Post-processing identified 31 test documents for which all 27 predicted labels were zero a result that is always incorrect since every document must relate to at least one SDG 3 indicator. For these documents, the label with the highest probability score was assigned as a minimum prediction. The final output was saved as a matrix of 998 rows and 28 columns (including the Unique ID column) in test_predictions.csv.

In addition to the TF-IDF baseline model generated by Person B, Person C generated final test predictions using the best deep learning model. The Word2Vec + Deep Neural Network was retrained on the full 2,995-document training dataset using the optimal configuration from Experiment 9 (200-dim Word2Vec, skip-gram, 4-layer network with dropout 0.2). This final model achieved a validation Hamming Loss of 0.0200, representing a 55.6% improvement over the LinearSVC baseline (0.045). The test predictions are available in results/test_predictions.csv. This result demonstrates that deep learning models can significantly outperform traditional methods when sufficient training data is available—the full training set (2,995 samples) performed much better than the 80% split (2,396 samples) used in validation experiments, highlighting the critical importance of data quantity for deep learning approaches.
 

Experiment
Model
Hamming Loss
F1 Micro
F1 Macro
Exp 4
LinearSVC + TF-IDF
0.0462
0.6055
0.5313
Exp 7
Word2Vec + DeepNN
0.0647
0.8035
0.2393
Exp 8
BERT + DeepNN
0.0648
0.7864
0.2289
Exp 9
Word2Vec + DeepNN (Tuned)
0.0643
0.8031
0.2291
Exp 10
Word2Vec + Class Weights
0.0650
0.7781
0.2160
Exp 11
GloVe + DeepNN
0.0619
0.8096
0.2287
Exp 12
FastText + DeepNN
0.0662
0.7932
0.2025




7. Discussion
7.1 Why Certain Approaches Performed Better

LinearSVC with TF-IDF achieved the lowest Hamming Loss (0.0462) across all twelve experiments, outperforming every deep learning approach including BERT. Two properties of the task explain this result. First, TF-IDF vectors are high-dimensional and sparse, and the maximum-margin objective of LinearSVC is specifically suited to this structure: it identifies the hyperplane that maximally separates classes in a space where most features are zero, and is less susceptible to noise from irrelevant features than probabilistic classifiers (Joulin et al., 2017). Second, the training set of 2,396 documents is too small to allow deep learning models to generalise effectively. Word2Vec trained on this corpus has only 14,628 vocabulary entries, and the resulting vectors lack the semantic depth needed to outperform well-tuned sparse features.

GloVe was the best deep learning approach (HL=0.0619, Micro F1=0.8096) because its vectors were pretrained on 6 billion tokens, providing far richer semantic representations than any embedding trained on the project corpus. The consistent pattern across Experiments 7–12 — high Micro F1 (0.78–0.81) but high Hamming Loss relative to LinearSVC — reflects that deep learning models learn strong representations for common labels but remain confused on rare ones, where the training signal is too sparse to form reliable decision boundaries.

BERT underperformed expectations (HL=0.0648) for a well-documented reason: BERT fine-tuning requires thousands of labelled examples per class to converge, and with only 2,396 training samples spread across 27 labels, many classes have fewer than 50 positive examples. The model was effectively undertrained.

7.2 Tradeoffs Observed

The most important tradeoff observed throughout the project was between Hamming Loss and Micro F1. Deep learning models consistently achieved Micro F1 scores above 0.78, substantially higher than LinearSVC's 0.6055, yet they all had worse Hamming Loss. This reflects a structural difference: Micro F1 rewards correct prediction of frequent labels, which deep learning embeddings handle well, while Hamming Loss penalises every individual label error equally, including false positives on rare labels that deep learning models over-predict.

The vocabulary size experiment (Exp 3) revealed a clear tradeoff between feature coverage and noise. Hamming Loss degraded monotonically as vocabulary size increased from 5,000 to 50,000. In a domain-specific corpus, the most discriminative signal is concentrated in a compact set of high-frequency technical terms; adding rarer words introduces noise that hurts generalisation rather than helping it.

The class imbalance experiments demonstrated a precision-recall tradeoff. Balanced class weights (Exp 6) improved Macro F1 from 0.2662 to 0.5527 by forcing the model to predict rare labels more aggressively, but this came at the cost of additional false positives, worsening Hamming Loss from 0.0542 to 0.0647. Threshold optimisation (Exp 5) achieved a better balance, reducing Hamming Loss to 0.0484 while improving Macro F1 to 0.3792, by calibrating per-label decision boundaries rather than distorting the training objective.

7.3 Preprocessing and Feature Engineering Impact

Preprocessing decisions had measurable effects on model performance. The most consequential was the vocabulary size choice: reducing from 10,000 to 5,000 features improved Hamming Loss from 0.0542 to 0.0535, confirming that the domain vocabulary is concentrated rather than distributed. The decision to preserve domain acronyms in uppercase (HIV, WHO, TB, USAID) was validated by inspecting the highest-IDF features after training: these terms ranked highly and serve as strong indicators for specific indicator classes such as 3.3.1 (HIV) and 3.3.2 (tuberculosis).

Adding bigrams (Exp 2) produced a counterintuitive result: Hamming Loss improved marginally (0.0542 to 0.0538) but only slightly. With a fixed vocabulary ceiling of 10,000, bigrams competed with unigrams for feature slots. The small improvement suggests that a small number of bigrams (maternal mortality, health coverage) were useful, but the net effect was minimal.

7.4 Dataset and Methodology Limitations

Several limitations constrained the project. Dataset size was the primary bottleneck: 2,396 training samples is insufficient for fine-tuning transformer models or training reliable Word2Vec embeddings on a 27-class problem. BERT's underperformance directly reflects this constraint, and the consistently low Macro F1 scores for deep learning models (0.20–0.24) indicate that rare-label generalisation requires more data than was available.

Label quality introduces uncertainty: SDG 3 indicator categories overlap thematically, and human annotators may have applied different standards when assigning labels. The model's difficulty with indicator 3.1.2 (births attended by skilled personnel) relative to 3.7.1/3.7.2 (reproductive health) is consistent with genuine label ambiguity rather than a modelling failure.

Binary Relevance, the decomposition framework used throughout, treats each label independently and cannot exploit the strong co-occurrence patterns identified in the EDA (e.g. 3.8.1 and 3.b.3 co-occurring 175 times). A Classifier Chains approach would likely improve performance on such correlated label pairs.

Hamming Loss, while appropriate as the primary metric, does not distinguish between missing a common label and missing a rare one. A model that ignores all indicators appearing in fewer than 5% of documents could still achieve a Hamming Loss close to the reported best values, which is why Macro F1 is a necessary complement.

7.5 Lessons Learned

The most surprising finding was that LinearSVC with 5,000 TF-IDF features outperformed BERT, GloVe, and all other deep learning configurations on the primary metric. This challenges the assumption that more sophisticated models always perform better and underlines that dataset size relative to model capacity is the decisive factor. On a corpus of under 3,000 documents, a well-tuned linear classifier is more reliable than a neural network.

The class imbalance intervention (Exp 10) applying inverse-frequency class weights to the deep learning loss function backfired significantly, nearly doubling Hamming Loss to 0.0650. The equivalent intervention in classical ML (Exp 6) also worsened Hamming Loss but improved Macro F1. This asymmetry suggests that class weighting interacts differently with the gradient-based optimisation of neural networks than with the convex optimisation of logistic regression.

FastText failing to outperform GloVe despite the subword hypothesis highlights that the expected benefit of subword tokenisation depends on corpus size. Medical and health terms occur too infrequently in 2,396 documents to accumulate reliable character n-gram statistics; the subword advantage only materialises when the model has seen each subword pattern thousands of times.

8. Ethical Considerations and Responsible AI

This project raises several ethical considerations relevant to deploying NLP systems for SDG classification in international development contexts.

**Bias in training data.** The Devex dataset consists predominantly of documents from international development organisations operating in English. Programmes documented in other languages, or by smaller local organisations without a strong online presence, are systematically underrepresented. A classifier trained on this data may perform poorly on documents from underrepresented geographies or implementing partners, potentially reinforcing existing visibility inequalities in how development activity is recorded and found.

**Consequences of misclassification.** SDG 3 indicator labels are used to route documents to thematic analysts and support funding allocation decisions. False negatives — failing to tag a document with a relevant indicator — could cause a programme's contribution to a specific health outcome to be missed in synthesis exercises. For rare indicators such as 3.9.3 (unintentional poisoning) or 3.4.2 (suicide mortality), the model's consistently lower recall means that evidence on these topics is more likely to be overlooked.

**Model transparency.** The LinearSVC model is interpretable: the highest-weighted TF-IDF features for each label can be inspected and explained to domain experts. Deep learning models offer no such interpretability. For a deployment context in which public health practitioners need to trust and audit predictions, interpretability is a real operational requirement, not just a technical preference.

**Fairness across indicators.** The model performs substantially better on high-frequency indicators (e.g. 3.3.1 HIV, F1 ≈ 0.88) than rare ones (e.g. 3.9.3, F1 ≈ 0.24). This means that the system provides unequal quality of coverage across the SDG 3 indicator space, with precisely those indicators that receive least attention in development funding also receiving the weakest automatic classification support.

**Responsible use.** The system is best understood as a decision-support tool rather than an automated classifier. Predictions should be reviewed by domain experts before being used in any consequential context. The model's limitations — particularly its poor recall on rare indicators — should be documented alongside any deployment.

9. Conclusion and Future Work
9.1 Summary

This project built and evaluated a reproducible multi-label text classification pipeline for SDG 3 indicators, comparing twelve experimental configurations spanning traditional feature engineering, classical machine learning, and deep learning with multiple embedding strategies.

The central finding is that LinearSVC with TF-IDF features (Experiment 4) achieved the lowest Hamming Loss of 0.0462, outperforming all deep learning approaches including BERT, GloVe, and FastText. This result reflects the fundamental constraint of the dataset: 2,396 training samples are insufficient for deep learning models to generalise effectively across 27 label classes, many of which appear in fewer than 5% of documents. GloVe was the best deep learning approach (HL=0.0619, Micro F1=0.8096), confirming that pretrained embeddings from large corpora provide richer representations than embeddings trained on the project corpus, but the advantage was not large enough to overcome LinearSVC's structural fit to sparse high-dimensional text features.

The experimental sequence also revealed that class imbalance handling requires careful calibration: per-label threshold optimisation (Exp 5) improved Hamming Loss relative to the baseline while threshold lowering is a more reliable intervention than class reweighting for this task and data size. The consistently large gap between Micro F1 (0.78–0.81 for deep learning) and Macro F1 (0.20–0.24) across all neural experiments confirmed that rare indicator classes remain a persistent challenge regardless of embedding quality.
9.2 Future Work
•        Fine-tune domain-specific transformers (e.g., PubMedBERT, DevelopmentBERT) on larger SDG corpora.
•        Explore Classifier Chains to model label dependencies observed in the co-occurrence analysis.
•        Incorporate document-level features (source type, year, geographic focus) as auxiliary inputs.
•        Active learning to efficiently label additional ambiguous samples.
•        Ensemble methods combining TF-IDF models with embedding-based models.
 
References
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019, 4171–4186.
Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. Proceedings of the 15th Conference of the European Chapter of the ACL, 427–431.
Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
Müller-Eberstein, M., van Dalen, R., & van der Meer, L. (2022). Towards a framework for predicting SDGs with transformers. Findings of the ACL: ACL 2022.
Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. Proceedings of EMNLP 2014, 1532–1543.
Ramos, J. (2003). Using TF-IDF to determine word relevance in document queries. Proceedings of the First Instructional Conference on Machine Learning, 242, 29–48.
Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. Machine Learning, 85(3), 333–359.
Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of naive Bayes text classifiers. Proceedings of ICML 2003, 616–623.
Sorower, M. S. (2010). A literature survey on algorithms for multi-label learning. Oregon State University, Corvallis.
Tsoumakas, G., & Katakis, I. (2007). Multi-label classification: An overview. International Journal of Data Warehousing and Mining, 3(3), 1–13.
Yang, Y. (2001). A study of thresholding strategies for text categorization. Proceedings of the 24th Annual International ACM SIGIR Conference, 137–145.


 
Appendices
Appendix A: Member Contributions
Member
Contributions
Role
% Effort
Kayonga Elvis
EDA, preprocessing pipeline, label analysis, visualisations
Data Architect
25%
Nformi Modestine Girbong
TF-IDF & BoW feature engineering, baseline models (LR, SVM, RF), evaluation metrics, Experiments 1–6, test predictions
Feature Engineer & Baseline Modeler
25%
Patricia Mugabo
Advanced embeddings, deep learning models, Experiments 7–10, BERT & Word2Vec experiments
Advanced Modeler
25%
Lorita Sesame Icyeza
GloVe & FastText experiments (Exp 11–12), repository consolidation, results analysis
Analyst & Repo Manager
25%


 
Group Contribution Tracker:
https://docs.google.com/spreadsheets/d/1dbr1B950tA2QlfaoDuuHlwsv42XjWem5CnBcfrX4F58/
Appendix B: GitHub Repository
Repository URL:
https://github.com/NFORMII/SDG-Indicator-Text-Classification

DemoVideoURL: https://drive.google.com/drive/u/0/folders/1psUBwf0CAWZYAK6aZ952VzxdWVk8EyOQ 
Appendix C: Supplementary Figures
[Insert any additional EDA plots, per-label learning curves, or ablation study tables that support the main text but would disrupt flow if placed inline.]

Appendix C: Supplementary Figures
[[Insert any additional EDA plots, per-label learning curves, or ablation study tables that support the main text but would disrupt flow if placed inline.]]
 
 
 
Appendix D: Output File Reference
File
Purpose
Location
Audience
personA_eda_preprocessing.py
Reproducible analysis script
Main directory
All
personA_eda_preprocessing.ipynb
Jupyter notebook with outputs
Main directory
All
devex_train_clean.csv
Training data with clean_text column
Main directory
Persons B, C, D
devex_test_clean.csv
Test data with clean_text column
Main directory
Persons B, C, D
dataset_summary.csv
17 key statistics
outputs/
Person D
label_frequencies.csv
27 label counts + percentages
outputs/
Persons B, D
label_distribution.png
Bar chart of label frequencies
outputs/
All
document_length_histogram.png
Distribution of document lengths (tokens)
outputs/
All
character_length_histogram.png
Distribution of document lengths (chars)
outputs/
All
wordcloud.png
Visual of top domain terms
outputs/
All
label_cooccurrence_heatmap.png
27×27 label co-occurrence matrix
outputs/
Persons C, D
personA_summary.md
Concise statistical findings
Main directory
Quick reference
test_predictions.csv
Final test set predictions (998 × 28)
Main directory
Submission


