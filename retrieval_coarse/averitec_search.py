#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google Search Retriever (versão robusta)
----------------------------------------
Busca documentos na web com base em claims do Averitec,
usando Google Custom Search API, multiprocessamento seguro
e salvamento limpo de resultados.
"""

import argparse
import json
import os
import gc
import tqdm
import pandas as pd
from time import sleep
from urllib.parse import urlparse
from googleapiclient.discovery import build
from html2lines import url2lines
from nltk import pos_tag, word_tokenize
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# Configuração de argumentos
# ============================================================
parser = argparse.ArgumentParser(description="Download and store search pages for FCB files.")
parser.add_argument("--averitec_file", default="data/dev.generated_questions.json", help="Arquivo JSON com claims.")
parser.add_argument("--misinfo_file", default="data/misinfo_list.txt", help="Lista de sites desinformativos.")
parser.add_argument("--n_pages", default=3, type=int, help="Número de páginas de resultados por busca.")
parser.add_argument("--store_folder", default="store/retrieved_docs", help="Diretório para armazenar os arquivos.")
parser.add_argument("--start_idx", default=0, type=int, help="Índice inicial (para grandes corpora).")
parser.add_argument("--n_to_compute", default=-1, type=int, help="Número de claims a processar (-1 = todos).")
parser.add_argument("--resume", default="", help="Arquivo de progresso para retomar execução.")
args = parser.parse_args()

# ============================================================
# Pastas e arquivos
# ============================================================
if not os.path.exists(args.store_folder):
    os.makedirs(args.store_folder)

misinfo_list = []
if os.path.exists(args.misinfo_file):
    with open(args.misinfo_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                misinfo_list.append(line.strip().lower())
else:
    print(f"[WARN] Arquivo {args.misinfo_file} não encontrado. Continuando sem blacklist adicional.")

# ============================================================
# Configuração de API
# ============================================================
api_key = os.getenv("GOOGLE_CSE_API_KEY", "AIzaSyAMgvRmIRDjXPsDiIELAeqwdX7zEOHpoIg")
search_engine_id = os.getenv("SEARCH_ENGINE_ID", "6207c4cbcfc394937")

# ============================================================
# Blacklists
# ============================================================
blacklist_domains = [
    "jstor.org",
    "facebook.com",
    "ftp.cs.princeton.edu",
    "nlp.cs.princeton.edu",
    "huggingface.co",
]

blacklist_files = [
    "/glove.",
    "ftp://ftp.cs.princeton.edu/pub/cs226/autocomplete/words-333333.txt",
    "https://web.mit.edu/adamrose/Public/googlelist",
]

# ============================================================
# Utilitários
# ============================================================
def get_domain_name(url: str) -> str:
    if "://" not in url:
        url = "http://" + url
    domain = urlparse(url).netloc
    return domain[4:] if domain.startswith("www.") else domain

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res.get("items", [])

def string_to_search_query(text, author):
    parts = word_tokenize(text.strip())
    tags = pos_tag(parts)
    keep_tags = ["CD", "JJ", "NN", "VB"]

    search_string = author.split() if author else []
    for token, tag in zip(parts, tags):
        if any(tag[1].startswith(k) for k in keep_tags):
            search_string.append(token)
    return " ".join(search_string)

def get_google_search_results(api_key, search_engine_id, google_search, sort_date, search_string, page=0):
    for _ in range(3):
        try:
            return google_search(
                search_string,
                api_key,
                search_engine_id,
                num=10,
                start=0 + 10 * page,
                sort=f"date:r:19000101:{sort_date}",
                dateRestrict=None,
                gl="US",
            )
        except Exception:
            sleep(3)
    return []

# ============================================================
# Função executada em subprocessos
# ============================================================
def get_and_store(url_link, fp):
    """Roda em subprocesso isolado; salva texto extraído de uma URL."""
    try:
        page_lines = url2lines(url_link)
        if not isinstance(page_lines, list):
            page_lines = [str(page_lines)]
        with open(fp, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join([url_link] + page_lines))
        return (fp, url_link, True, "ok")
    except Exception as e:
        return (fp, url_link, False, f"{type(e).__name__}: {e}")

# ============================================================
# Carrega arquivo principal
# ============================================================
with open(args.averitec_file, "r", encoding="utf-8") as f:
    examples = json.load(f)

existing = {}
if args.resume:
    first = True
    next_claim = {"claim": None}
    for line in open(args.resume, "r", encoding="utf-8"):
        if first:
            first = False
            continue
        parts = line.strip().split("\t")
        claim = parts[1]
        if claim != next_claim["claim"]:
            if next_claim["claim"]:
                existing[next_claim["claim"]] = next_claim
            next_claim = {"claim": claim, "lines": []}
        next_claim["lines"].append(line.strip())

# ============================================================
# Loop principal
# ============================================================
print("\t".join(["index", "claim", "link", "page", "search_string", "search_type", "store_file"]))

start_idx = args.start_idx
end_idx = None if args.n_to_compute == -1 else args.start_idx + args.n_to_compute
MAX_WORKERS = 8  # ajuste conforme sua CPU/memória

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_meta = {}

    for idx, example in tqdm.tqdm(enumerate(examples[start_idx:end_idx], start=start_idx)):
        claim = example["claim"]
        if claim in existing:
            for line in existing[claim]["lines"]:
                print(line)
            continue

        speaker = (example.get("speaker") or "").strip() or None
        questions = [q["question"] for q in example.get("questions", [])]

        # Normaliza data
        try:
            year, month, date = example["check_date"].split("-")
        except Exception:
            month, date, year = "01", "01", "2022"

        if len(year) == 2:
            year = ("20" if int(year) <= 30 else "19") + year
        elif len(year) == 1:
            year = "200" + year
        if len(month) == 1:
            month = "0" + month
        if len(date) == 1:
            date = "0" + date

        sort_date = year + month + date

        # Gera termos de busca
        search_strings = []
        search_types = []

        if speaker:
            search_strings.append(string_to_search_query(claim, speaker))
            search_types.append("claim+author")

        search_strings += [string_to_search_query(claim, None), claim]
        search_types += ["claim", "claim-noformat"]

        search_strings += questions
        search_types += ["question"] * len(questions)

        visited = {}
        store_counter = 0

        for this_search_string, this_search_type in zip(search_strings, search_types):
            for page_num in range(args.n_pages):
                results = get_google_search_results(
                    api_key, search_engine_id, google_search, sort_date, this_search_string, page=page_num
                )
                for result in results:
                    link = str(result.get("link", ""))
                    if not link:
                        continue

                    domain = get_domain_name(link)
                    if domain in blacklist_domains:
                        continue
                    if any(bf in link for bf in blacklist_files):
                        continue
                    if link.lower().endswith((".pdf", ".doc", ".docx")):
                        continue

                    if link in visited:
                        store_file_path = visited[link]
                    else:
                        store_counter += 1
                        store_file_path = os.path.join(args.store_folder, f"search_result_{idx}_{store_counter}.store")
                        visited[link] = store_file_path

                        fut = executor.submit(get_and_store, link, store_file_path)
                        future_to_meta[fut] = (idx, claim, page_num, this_search_string, this_search_type, store_file_path)

                    # imprime resultado
                    print("\t".join([str(idx), claim, link, str(page_num), this_search_string, this_search_type, visited[link]]))

    # Espera terminar todos os downloads
    for fut in as_completed(future_to_meta):
        idx, claim, pnum, sstr, stype, store_fp = future_to_meta[fut]
        try:
            fp, url, ok, msg = fut.result()
            if ok:
                print(f"[OK] {idx} | {url} -> salvo em {fp}")
            else:
                print(f"[FAIL] {idx} | {url} -> {msg}")
        except Exception as e:
            print(f"[EXC] {idx} | {store_fp} -> {e}")

gc.collect()
print("✅ Finalizado com sucesso.")
