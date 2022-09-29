from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                      num_labels=10, 
                                                      problem_type="multi_label_classification")