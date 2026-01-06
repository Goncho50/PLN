from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Iterable, Callable
import json
import re
import datetime as dt

# Dependencias recomendadas
# pip install nltk langdetect
import nltk
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# ---------- Utilidades de limpieza ----------

def default_clean_text(text: str) -> str:
    """Limpieza básica: etiquetas HTML, emojis/símbolos raros, repeticiones, espacios."""
    text = text or ""
    text = re.sub(r"<.*?>", " ", text)  # quitar HTML
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # normalizar repeticiones
    text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\'\-]", " ", text)  # quitar símbolos raros
    text = re.sub(r"\s+", " ", text).strip()  # colapsar espacios
    return text

# ---------- Helpers robustos para tipos/fechas ----------

def _safe_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(str(x).strip())
    except (ValueError, TypeError):
        return None

def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def _epoch_to_iso8601(x) -> Optional[str]:
    """Convierte segundos epoch a ISO-8601 (UTC). Si ya parece ISO/texto, lo devuelve tal cual."""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.isdigit():
        try:
            return dt.datetime.utcfromtimestamp(int(s)).isoformat() + "Z"
        except (OverflowError, ValueError):
            return None
    return s  # asumimos que ya es ISO-8601 o una fecha legible

# ---------- Clase Review ----------

@dataclass
class ReviewBGG:
    # Opcionales
    game_id: Optional[int] = None
    rating: Optional[float] = None
    text_raw: Optional[str] = None

    # Obligatorios
    user: str = ""
    timestamp: str = ""

    # Campos derivados
    lang: Optional[str] = None
    text_clean: Optional[str] = None
    sentences: Optional[List[str]] = None
    tokens_by_sentence: Optional[List[List[str]]] = None

    def __post_init__(self):
        if isinstance(self.rating, str):
            try:
                self.rating = float(self.rating.strip())
            except ValueError:
                self.rating = None

    def preprocess(
        self,
        *,
        detect_language: bool = True,
        target_lang: Optional[str] = "en",
        cleaner: Callable[[str], str] = default_clean_text,
        filter_stopwords: bool = True,
        sentence_language_code: str = "english"
    ) -> None:
        """Aplica: detección de idioma, limpieza, segmentación en oraciones, tokenización."""
        # 1) Detección de idioma (solo langdetect)
        lang = None
        if detect_language and self.text_raw and self.text_raw.strip():
            try:
                lang = detect(self.text_raw)
            except LangDetectException:
                lang = None
        self.lang = lang

        # 2) Limpieza
        text_clean = cleaner(self.text_raw or "")
        self.text_clean = text_clean

        # 3) Segmentación en oraciones
        sentences = sent_tokenize(text_clean, language=sentence_language_code) if text_clean else []
        self.sentences = sentences

        # 4) Tokenización + stopwords
        toks_sent: List[List[str]] = []
        sw = set(stopwords.words(sentence_language_code)) if filter_stopwords else set()

        for s in sentences:
            toks = [t for t in word_tokenize(s) if t.strip()]
            if filter_stopwords:
                toks = [t.lower() for t in toks if t.isalpha() and t.lower() not in sw]
            else:
                toks = [t.lower() for t in toks if t.isalpha()]
            toks_sent.append(toks)

        self.tokens_by_sentence = toks_sent

# ---------- Clase Corpus ----------

class CorpusBGG:
    def __init__(self):
        self._reviews: List[ReviewBGG] = []

    def add_review(self, review: ReviewBGG) -> None:
        self._reviews.append(review)

    def add_from_bgg_single_game(self, items: Iterable[Dict[str, Any]], *, game_id: Optional[int] = None) -> None:
        """Ingiere reseñas del formato BGG (username, rating, comment, timestamp)."""
        for d in items:
            if not isinstance(d, dict):
                continue
            user = d.get("username") or d.get("user")
            if not user:
                continue  # obligatorio
            rating = _safe_float(d.get("rating"))
            text_raw = d.get("comment") or d.get("text") or ""
            ts_iso = _epoch_to_iso8601(d.get("timestamp"))
            if not ts_iso:
                continue  # obligatorio
            self._reviews.append(
                ReviewBGG(
                    game_id=game_id,
                    user=str(user),
                    rating=rating,
                    timestamp=ts_iso,
                    text_raw=str(text_raw)
                )
            )

    @classmethod
    def load_json_single_game(cls, path: str, *, game_id: Optional[int] = None) -> "CorpusBGG":
        """Carga un JSON con una lista de reseñas de un solo juego."""
        corpus = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get("reviews") or data.get("items") or [data]
            if not isinstance(data, list):
                data = [data]
            corpus.add_from_bgg_single_game(data, game_id=game_id)
        return corpus
    @classmethod
    def load_json_from_ids(cls, game_ids, *, base_pattern: str = "bgg_reviews_{}_crawler.json") -> "CorpusBGG":
        """
        Carga múltiples juegos en un único corpus usando una lista de IDs.
        Los archivos deben seguir el formato base_pattern (por defecto: 'bgg_reviews_{id}_crawler.json').

        Ejemplo:
            corpus = CorpusBGG.load_json_from_ids([8749, 106753, 142527])
        """
        combined = cls()
        for gid in game_ids:
            path = base_pattern.format(gid)
            try:
                print(f"Cargando juego {gid} desde {path}...")
                corpus_part = cls.load_json_single_game(path, game_id=gid)
                for r in corpus_part.reviews():
                    combined.add_review(r)
            except FileNotFoundError:
                print(f"Archivo no encontrado: {path}")
            except json.JSONDecodeError:
                print(f"Error leyendo JSON: {path}")
        print(f"Combinadas {len(combined)} reseñas de {len(game_ids)} juegos.")
        return combined


    def preprocess_all(
        self,
        *,
        detect_language: bool = True,
        target_lang: Optional[str] = "en",
        cleaner: Callable[[str], str] = default_clean_text,
        filter_stopwords: bool = True,
        sentence_language_code: str = "english"
    ) -> None:
        for r in self._reviews:
            r.preprocess(
                detect_language=detect_language,
                target_lang=target_lang,
                cleaner=cleaner,
                filter_stopwords=filter_stopwords,
                sentence_language_code=sentence_language_code
            )

    def __len__(self) -> int:
        return len(self._reviews)

    def reviews(self) -> List[ReviewBGG]:
        return self._reviews

    def reviews_by_game(self, game_id: int) -> List[ReviewBGG]:
        return [r for r in self._reviews if r.game_id == game_id]

    def reviews_with_text(self) -> List[ReviewBGG]:
        return [r for r in self._reviews if r.text_raw and r.text_raw.strip()]

    def reviews_with_rating(self) -> List[ReviewBGG]:
        return [r for r in self._reviews if r.rating is not None]

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self._reviews], f, ensure_ascii=False, indent=2)

    def stats(self) -> Dict[str, Any]:
        n_total = len(self._reviews)
        n_with_rating = sum(1 for r in self._reviews if r.rating is not None)
        n_with_text = sum(1 for r in self._reviews if r.text_raw and r.text_raw.strip())
        n_with_both = sum(1 for r in self._reviews if (r.rating is not None and r.text_raw and r.text_raw.strip()))
        games = {r.game_id for r in self._reviews if r.game_id is not None}
        users = {r.user for r in self._reviews if r.user}

        dist: Dict[str, int] = {}
        for r in self._reviews:
            if r.rating is not None:
                key = str(int(r.rating)) if float(r.rating).is_integer() else str(r.rating)
                dist[key] = dist.get(key, 0) + 1

        return {
            "n_reviews": n_total,
            "n_with_rating": n_with_rating,
            "n_with_text": n_with_text,
            "n_with_rating_and_text": n_with_both,
            "n_games": len(games),
            "n_users": len(users),
            "rating_distribution": dict(sorted(dist.items(), key=lambda kv: float(kv[0])))
        }
