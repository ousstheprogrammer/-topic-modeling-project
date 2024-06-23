from gensim import corpora
from gensim.models import LdaModel


dictionary = corpora.Dictionary(df['processed'])
corpus = [dictionary.doc2bow(text) for text in df['processed']]


lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)


for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}")
