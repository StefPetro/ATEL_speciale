from atel.data import BookCollection
from data_clean import set_seed
from lstm_model import lstm_data, lstm_text
from pytorch_lightning import Trainer
import fasttext

set_seed(42)

book_col = BookCollection(data_file="./data/book_col_271120.pkl")

print('Loading fastText model...')
ft = fasttext.load_model('fasttext_model/cc.da.300.bin')  # Download from fastTexts website
print('Loading complete!')

settings = {
    'multi_label': True,
    'n_features': 300, 
    "hidden_size": 256, 
    "num_layers": 1, 
    "dropout": 0.2, 
    "batch_size": 32, 
    "learning_rate" : 1e-4,
    "output_size": 15
}

model = lstm_text(**settings)
data = lstm_data(book_col, 'Genre', ft)
trainer = Trainer(max_epochs=1)

trainer.fit(model, data)
