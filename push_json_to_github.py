import os
import json
import base64
import requests
import hashlib
import datetime
import time  # Dodane dla funkcji retry

# --- Konfiguracja ---
# Token GitHub - POBIERANY ZE ZMIENNEJ ŚRODOWISKOWEJ DLA BEZPIECZEŃSTWA
# Upewnij się, że masz ustawioną zmienną środowiskową GITHUB_TOKEN
# np. export GITHUB_TOKEN="twój_klucz_pat_tutaj"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Dane repozytorium
REPO_OWNER = "ApexStrikeLive"
REPO_NAME = "my-web"
TARGET_BRANCH = "main"  # Gałąź, do której chcesz pchać zmiany
FILE_PATH = "traced_signals.json"  # Nazwa pliku do śledzenia

# Ustawienia komunikatów
COMMIT_MESSAGE_PREFIX = "Automatyczna aktualizacja JSON: "

# Ustawienia ponownych prób (retry) dla API GitHub
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5

# Adresy URL API GitHub
BASE_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
CONTENTS_URL = f"{BASE_URL}/contents/{FILE_PATH}"
REFS_URL = f"{BASE_URL}/git/refs"
COMMITS_URL = f"{BASE_URL}/git/commits"
TREES_URL = f"{BASE_URL}/git/trees"

# Nagłówki HTTP do autoryzacji
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}


def make_github_request(method, url, data=None, params=None):
    """Wykonuje bezpieczne zapytanie do API GitHub z obsługą ponownych prób."""
    for i in range(MAX_RETRIES):
        try:
            if method == "GET":
                response = requests.get(url, headers=HEADERS, params=params)
            elif method == "POST":
                response = requests.post(url, headers=HEADERS, data=json.dumps(data))
            elif method == "PUT":
                response = requests.put(url, headers=HEADERS, data=json.dumps(data))
            elif method == "PATCH":
                response = requests.patch(url, headers=HEADERS, data=json.dumps(data))

            response.raise_for_status()  # Rzuca wyjątek dla kodów statusu 4xx/5xx
            return response
        except requests.exceptions.RequestException as e:
            print(f"Błąd API GitHub ({method} {url}): {e}")
            if e.response:
                print(f"Status odpowiedzi: {e.response.status_code}")
                print(f"Treść odpowiedzi: {e.response.text}")
            if i < MAX_RETRIES - 1:
                print(f"Ponawiam próbę za {RETRY_DELAY_SECONDS} sekund... ({i + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise  # Rzuca wyjątek po wyczerpaniu ponownych prób
    return None  # Powinno być nieosiągalne, ale dla pewności


def get_current_file_sha(branch_name):
    """Pobiera SHA pliku na docelowej gałęzi."""
    try:
        response = make_github_request("GET", CONTENTS_URL, params={"ref": branch_name})
        return response.json().get("sha")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Plik {FILE_PATH} nie znaleziony na gałęzi {branch_name}. Może to być pierwsze wysłanie.")
            return None
        print(f"Błąd podczas pobierania bieżącego SHA pliku: {e}")
        return None


def get_latest_commit_sha(branch_name):
    """Pobiera SHA ostatniego commita na docelowej gałęzi."""
    response = make_github_request("GET", f"{REFS_URL}/heads/{branch_name}")
    return response.json()["object"]["sha"]


def get_tree_sha(commit_sha):
    """Pobiera SHA drzewa z SHA commita."""
    response = make_github_request("GET", f"{BASE_URL}/git/commits/{commit_sha}")
    return response.json()["tree"]["sha"]


def create_blob(content):
    """Tworzy Git blob z zawartością pliku."""
    payload = {
        "content": content,
        "encoding": "base64"
    }
    response = make_github_request("POST", f"{BASE_URL}/git/blobs", data=payload)
    return response.json()["sha"]


def create_tree(base_tree_sha, blob_sha, file_path):
    """Tworzy nowe drzewo Git ze zaktualizowanym plikiem."""
    payload = {
        "base_tree": base_tree_sha,
        "tree": [
            {
                "path": file_path,
                "mode": "100644",  # Standardowe uprawnienia dla plików
                "type": "blob",
                "sha": blob_sha
            }
        ]
    }
    response = make_github_request("POST", TREES_URL, data=payload)
    return response.json()["sha"]


def create_commit(tree_sha, parent_commit_sha, message):
    """Tworzy nowy commit Git."""
    payload = {
        "message": message,
        "parents": [parent_commit_sha],
        "tree": tree_sha
    }
    response = make_github_request("POST", COMMITS_URL, data=payload)
    return response.json()["sha"]


def update_ref(branch_name, commit_sha):
    """Aktualizuje istniejącą referencję Git (gałąź) do nowego commita."""
    payload = {
        "sha": commit_sha,
        "force": False  # Zostawiamy False, aby uniknąć force push, chyba że jest to konieczne
    }
    make_github_request("PATCH", f"{REFS_URL}/heads/{branch_name}", data=payload)


def get_local_file_hash(file_path):
    """Oblicza hash SHA-1 zawartości lokalnego pliku."""
    with open(file_path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def main():
    if not GITHUB_TOKEN:
        print("Błąd: Zmienna środowiskowa GITHUB_TOKEN nie jest ustawiona.")
        return

    if not os.path.exists(FILE_PATH):
        print(f"Błąd: Plik '{FILE_PATH}' nie znaleziony.")
        return

    try:
        with open(FILE_PATH, 'rb') as f:
            local_content_bytes = f.read()
            local_content_base64 = base64.b64encode(local_content_bytes).decode('utf-8')
            local_file_hash = hashlib.sha1(local_content_bytes).hexdigest()
    except Exception as e:
        print(f"Błąd odczytu lokalnego pliku {FILE_PATH}: {e}")
        return

    print(f"Lokalny hash pliku '{FILE_PATH}': {local_file_hash}")

    # Pobierz bieżący SHA pliku z GitHub
    github_file_sha_on_target = get_current_file_sha(TARGET_BRANCH)
    print(f"Hash pliku '{FILE_PATH}' na GitHubie (gałąź '{TARGET_BRANCH}'): {github_file_sha_on_target}")

    # Jeśli plik istnieje na GitHubie, pobierz jego zawartość i porównaj hash
    if github_file_sha_on_target:
        try:
            github_content_response = make_github_request("GET", CONTENTS_URL, params={"ref": TARGET_BRANCH})
            github_content_base64 = github_content_response.json()["content"]
            github_content_bytes = base64.b64decode(github_content_base64)
            github_file_hash = hashlib.sha1(github_content_bytes).hexdigest()
            print(f"Hash zawartości pliku '{FILE_PATH}' na GitHubie (gałąź '{TARGET_BRANCH}'): {github_file_hash}")

            if local_file_hash == github_file_hash:
                print(f"'{FILE_PATH}' nie zmienił się. Pchanie nie jest potrzebne.")
                return
        except Exception as e:
            print(
                f"Błąd podczas pobierania zawartości pliku z GitHub do porównania: {e}. Kontynuuję, zakładając, że plik się zmienił.")
    else:
        # Plik nie istnieje na zdalnej gałęzi, więc uznajemy go za nowy i wymagający pchnięcia
        print(f"Plik '{FILE_PATH}' nie istnieje na gałęzi '{TARGET_BRANCH}'. Traktuję jako nową zmianę.")

    print(f"'{FILE_PATH}' uległ zmianie. Inicjuję proces pchania...")

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    commit_message = f"{COMMIT_MESSAGE_PREFIX}{timestamp}"

    try:
        # 1. Pobierz SHA ostatniego commita z gałęzi docelowej
        latest_commit_sha = get_latest_commit_sha(TARGET_BRANCH)
        print(f"Ostatni commit SHA na '{TARGET_BRANCH}': {latest_commit_sha}")

        # 2. Pobierz SHA drzewa z ostatniego commita
        base_tree_sha = get_tree_sha(latest_commit_sha)
        print(f"Bazowe drzewo SHA: {base_tree_sha}")

        # 3. Utwórz blob dla nowej zawartości pliku
        new_blob_sha = create_blob(local_content_base64)
        print(f"Nowy blob SHA: {new_blob_sha}")

        # 4. Utwórz nowe drzewo ze zaktualizowanym plikiem
        new_tree_sha = create_tree(base_tree_sha, new_blob_sha, FILE_PATH)
        print(f"Nowe drzewo SHA: {new_tree_sha}")

        # 5. Utwórz nowy commit
        new_commit_sha = create_commit(new_tree_sha, latest_commit_sha, commit_message)
        print(f"Nowy commit SHA: {new_commit_sha}")

        # 6. Bezpośrednio zaktualizuj referencję gałęzi docelowej
        print(f"Bezpośrednia aktualizacja gałęzi '{TARGET_BRANCH}' z commitem {new_commit_sha}...")
        update_ref(TARGET_BRANCH, new_commit_sha)
        print(f"Plik '{FILE_PATH}' został pomyślnie pchnięty bezpośrednio do gałęzi '{TARGET_BRANCH}'.")

    except requests.exceptions.RequestException as e:
        print(f"Błąd sieciowy/API GitHub: {e}")
        if e.response:
            print(f"Status odpowiedzi: {e.response.status_code}")
            print(f"Treść odpowiedzi: {e.response.text}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")


if __name__ == "__main__":
    main()