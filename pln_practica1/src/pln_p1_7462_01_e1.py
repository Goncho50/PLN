#-----------------------------------------------------------
# ------- LIBRERÍAS EXTERNAS -------
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
from datetime import datetime
import time


# ------- FUNCIONES AUXILIARES -------
def month_conversion(cadena:str)-> str:
    """ Convierte los meses de entrada en formato español
        a formato utilizado en datetime.

    Args:
        cadena (str): Un mes en formato incorrecto.

    Returns:
        str: El mes en el formato correcto.
    """
    # cadena = 21 ago 2025 -> 21 aug 2025
    meses = {
        "ene": "jan",
        "abr": "apr",
        "ago": "aug",
        "sept": "sep",
        "dic": "dec",
    }
    for mes in meses.keys():
        if mes in cadena:
            cadena = cadena.replace(mes,meses[mes])
    return cadena


def create_driver():
    """Crea un driver web de Selenium.

    Returns:
        _type_: El objeto deriver.
    """
    # Configurar Selenium (headless para que no abra ventana)
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    return driver


def obtain_url(base_url:str, driver)-> str|None:
    """ Obtiene la URL completa de las reseñas
    de un juego en BGG a partir de un ID numérico.

    Args:
        base_url (str): La URL base de BGG.
        driver (_type_): El driver web.

    Returns:
        str|None: La URL completa de las reseñas de
                  un juego en BGG. Si no encuentra la
                  URL en el CSS devuelve None.
    """
    try:
        driver.get(base_url)
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR,
                                            "meta[property='og:url']"))
        )
        game_url = driver.find_element(By.CSS_SELECTOR,
                                       "meta[property='og:url']").get_attribute("content")
        ratings_url = game_url + "/ratings"
        return ratings_url
    except:
        ratings_url = None
        return ratings_url


# ------- FUNCIÓN PRINCIPAL (SCRAPING) -------
def download_reviews(driver, ratings_url):
    current_page = 1
    reviews = []
    # control de duplicados para detectar páginas repetidas
    seen = set()

    while True:
        print(f"Descargando reseñas (página {current_page})...")
        url = f"{ratings_url}?pageid={current_page}"
        driver.get(url)

        try:
            # Esperar a que aparezcan reviews
            WebDriverWait(driver, 3).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR,
                                                     "li.summary-rating-item"))
            )
        except:
            print("Se han encontrado reseñas sin rating.")
            continue

        items = driver.find_elements(By.CSS_SELECTOR,
                                     "li.summary-rating-item")
        if not items:
            print("No hay reviews")
            break

        # contador de nuevas reseñas en una página
        new_in_page = 0

        for item in items:
            # username
            try:
                username_elem = item.find_element(By.CSS_SELECTOR,
                                                  "a.comment-header-user")
                username = username_elem.get_attribute("innerText").strip()
            except:
                username = None

            # rating
            try:
                rating_elem = item.find_element(By.CSS_SELECTOR,
                                                "div.rating-angular")
                rating = rating_elem.get_attribute("innerText").strip()
            except:
                rating = None

            # comment
            try:
                comment_elem = item.find_element(By.CSS_SELECTOR,
                                                 "div.comment-body p")
                comment = comment_elem.get_attribute("innerText").strip()
            except:
                comment = None

            # timestamp
            try:
                timestamp_elem = item.find_element(By.CSS_SELECTOR,
                                                   "span.comment-header-timestamp span")

                # Aparecen como: "Last updated: 21 Mar 2025"
                raw_timestamp = timestamp_elem.get_attribute("title")
                timestamp = raw_timestamp.replace("Last updated:", "").strip()

                timestamp = month_conversion(timestamp)
                timestamp = datetime.strptime(timestamp, '%d %b %Y') # a datetime
                timestamp = int(datetime.timestamp(timestamp)) # a timestamp
            except:
                timestamp = None

            # clave para evitar duplicados
            key = (username or "", comment or "")
            if key not in seen:
                seen.add(key)
                reviews.append({
                    "username": username,
                    "rating": rating,
                    "comment": comment,
                    "timestamp": timestamp,
                })
                new_in_page += 1

        # si esta página no aporta nada nuevo, cortamos el bucle
        if new_in_page == 0:
            print("Página sin novedades. Fin de la paginación.")
            break

        current_page += 1
        time.sleep(1)

    return reviews


# ------- EJECUCIÓN -------
def obtener_reviews(game_ids):
    selenium_driver = create_driver()
    for index in game_ids:
        print(f"Procesando reseñas con ID: {index}")
        BASE_URL = f"https://boardgamegeek.com/boardgame/{index}"
        OUTPUT_FILE = f"bgg_reviews_{index}_crawler.json"

        # PASO 1: Crear driver

        print("Paso 1 completado...")

        # PASO 2: Obtener URL de reseñas
        ratings_url = obtain_url(BASE_URL, selenium_driver)
        print("Paso 2 completado...")

        # PASO 3: Descargar reseñas
        data = download_reviews(selenium_driver,ratings_url)

        # Guardar reseñas en un archivo JSON
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Se guardaron {len(data)} reseñas en {OUTPUT_FILE}")
    selenium_driver.quit()


# ------- EJEMPLO -------
if __name__ == "__main__":
    game_ids = [246701, 533]
    obtener_reviews(game_ids)
