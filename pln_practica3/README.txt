En el ejercicio 1, el modelo Word2Vec preentrenado que usamos para hacer el embedding pesa poco más de 1 GB comprimido,
por lo que no es imposible adjuntarlo en el Moodle. Lo hemos descargado de este enlace:

https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz


Y luego lo hemos usado en el Colab subiéndolo manualmente. En el Colab se usa en la siguiente línea:


w2v = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz",
    binary=True
)



Disculpa las molestias,
Equipo 01.



P.D: En el ejercicio 3 los resultados salen con el formato de la memoria, solo que en el .txt no se ha guardado el formato de los bullet points.