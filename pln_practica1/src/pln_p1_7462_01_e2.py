#-----------------------------------------------------------
# ------- LIBRERÍAS EXTERNAS -------
import requests
import xml.etree.ElementTree as ET
import json
import time



# ------- FUNCIONES AUXILIARES -------
def get_text(game, tag):
    """Función auxiliar para obtener
        texto de etiquetas.
    """
    element = game.find(tag)
    return element.text if element is not None else "N/A"


def get_name(game):
    """Función auxiliar para obtener el
        nombre principal del juego.
    """
    element = game.find("name[@primary='true']")
    return element.text if element is not None else "N/A"


def guardar_metadatos(game_id):
    # Descargar metadatos del juego (sin comentarios)
    url_game = f"https://boardgamegeek.com/xmlapi/boardgame/{game_id}"
    response = requests.get(url_game)

    if response.status_code != 200:
        raise Exception(f"Error descargando los datos del juego: {response.status_code}")

    root = ET.fromstring(response.content)
    game = root.find('boardgame')

    # Guardar metadatos del juego
    game_data = {
        "id": game_id,
        "name": get_name(game),
        "yearpublished": get_text(game, 'yearpublished'),
        "family": get_text(game, 'boardgamefamily'),
        "category": [c.text for c in game.findall('boardgamecategory')],
        "mechanic": [m.text for m in game.findall('boardgamemechanic')],
        "type": [t.text for t in game.findall('boardgamesubdomain')],
        "minplayers": get_text(game, 'minplayers'),
        "maxplayers": get_text(game, 'maxplayers'),
        "minplaytime": get_text(game, 'minplaytime'),
        "maxplaytime": get_text(game, 'maxplaytime'),
        "age": get_text(game, 'age'),
        "description": get_text(game, 'description'),
        "image": get_text(game, 'image'),
        "thumbnail": get_text(game, 'thumbnail'),
        "publishers": [p.text for p in game.findall('boardgamepublisher')],
        "designers": [d.text for d in game.findall('boardgamedesigner')],
        "artists": [a.text for a in game.findall('boardgameartist')],
        "comments": []
    }
    return game_data


# ------- FUNCIÓN PRINCIPAL -------
def descarga_reseñas(game_id, meta_data):

    # Descargar reseñas
    page = 1
    seen = set()
    while True:
        print(f"Página {page}")
        new_in_page = 0
        url_comments = f"https://boardgamegeek.com/xmlapi/boardgame/{game_id}?comments=1&page={page}"
        response = requests.get(url_comments)
        root = ET.fromstring(response.content)
        game_xml = root.find('boardgame')
        comments = game_xml.findall('comment')

        for comment in comments:
            comment_data = {
                "username": comment.attrib.get('username', 'N/A'),
                "rating": comment.attrib.get('rating', 'N/A'),
                "comment": comment.text.strip() if comment.text else ''
            }

            key = (comment_data["username"] or "", comment_data["comment"] or "")
            if key not in seen:
                seen.add(key)
                meta_data["comments"].append(comment_data)
                new_in_page += 1

        if new_in_page == 0:
            print("Página sin novedades (contenido repetido). Fin de la paginación.")
            break

        page+=1

        time.sleep(1)

    return meta_data


# ------- EJECUCIÓN -------
def obtener_reviews_api(game_ids):
    for index in game_ids:
        print(f"Procesando reseñas con ID: {index}")
        OUTPUT_FILE = f"bgg_reviews_{index}_api.json"

        # PASO 1: Obtención de metadatos
        game_data = guardar_metadatos(index)

        # PASO 2: Obtener reseñas (con comentarios)
        game_data = descarga_reseñas(index,game_data)

        # Guardar reseñas en un archivo JSON
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(game_data, f, ensure_ascii=False, indent=2)
        reseñas = game_data["comments"]
        print(f"Se guardaron {len(reseñas)} reseñas en {OUTPUT_FILE}")


# ------- EJEMPLO -------
if __name__ == "__main__":
    game_ids = [246701, 533]
    obtener_reviews_api(game_ids)