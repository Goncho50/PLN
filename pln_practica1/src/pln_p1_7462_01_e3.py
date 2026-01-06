from CorpusBGG import CorpusBGG  # importa tu clase del archivo donde la tengas
import nltk
nltk.download("stopwords")
nltk.download("punkt")  # por si no lo tienes, sirve para tokenización
nltk.download('punkt_tab')
import pln_p1_7462_01_e1 as p1

# 1. Se van a descargar y a guardar en el corpus 15 juegos
# (5 juegos con buenas reseñas, 5 con reseñas medias, 5 con bajas reseñas)
game_ids = [278292,245932,149915,310175,181621,
          123885,4864,206175,222,533,
          12205,5130,246701,3728,16398]


# PASO 1: LLAMAR AL CRAWLER
p1.obtener_reviews(game_ids)

# PASO 2: CARGAR EL CORPUS COMPLETO
corpus_total = CorpusBGG.load_json_from_ids(game_ids)

print("Número total de reseñas:", len(corpus_total))
print("Estadísticas:", corpus_total.stats())

# Preprocesar y guardar
corpus_total.preprocess_all(sentence_language_code="english")
corpus_total.save_json("corpus_total.json")

print("Corpus combinado guardado en corpus_total.json")


