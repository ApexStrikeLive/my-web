import time
import subprocess
import os

# --- Konfiguracja monitoringu ---
# Nazwa pliku, który ma być monitorowany (do sprawdzenia istnienia)
FILE_TO_WATCH = "traced_signals.json"
# Nazwa skryptu, który ma być uruchomiony
PUSH_SCRIPT_NAME = "push_json_to_github.py"
# Interwał w sekundach (5 minut = 300 sekund)
PUSH_INTERVAL_SECONDS = 100


def main():
    print(f"--- Rozpoczęto automatyczne pchanie pliku '{FILE_TO_WATCH}' co {PUSH_INTERVAL_SECONDS} sekund ---")
    print("Naciśnij Ctrl+C, aby zatrzymać.")

    while True:
        # Sprawdzamy, czy plik istnieje, zanim spróbujemy go pchnąć
        if not os.path.exists(FILE_TO_WATCH):
            print(
                f"Błąd: Plik '{FILE_TO_WATCH}' nie znaleziony w katalogu '{os.path.abspath(os.getcwd())}'. Sprawdzę ponownie za {PUSH_INTERVAL_SECONDS} sekund.")
        else:
            print(f"\n--- Uruchamiam skrypt push_json_to_github.py o {time.ctime()} ---")
            try:
                # Uruchamia skrypt push_json_to_github.py
                # Ważne: zmienne środowiskowe (jak GITHUB_TOKEN) są dziedziczone
                result = subprocess.run(
                    ["python", PUSH_SCRIPT_NAME],
                    capture_output=True,  # Zbieraj wyjście (stdout i stderr)
                    text=True,  # Zwracaj wyjście jako tekst
                    check=False,  # Nie rzucaj wyjątku dla niezerowego kodu wyjścia, obsłużymy to ręcznie
                    env=os.environ  # Przekaż wszystkie bieżące zmienne środowiskowe (w tym GITHUB_TOKEN)
                )
                print("\n--- Wynik działania skryptu push_json_to_github.py: ---")
                print(result.stdout)
                if result.stderr:
                    print("--- Błędy skryptu push_json_to_github.py: ---")
                    print(result.stderr)

                if result.returncode != 0:
                    print(
                        f"--- Skrypt push_json_to_github.py zakończył się z błędem (kod wyjścia: {result.returncode}) ---\n")
                else:
                    print("--- Skrypt push_json_to_github.py zakończył działanie pomyślnie ---\n")

            except FileNotFoundError:
                print(
                    f"Błąd: Nie można znaleźć skryptu '{PUSH_SCRIPT_NAME}'. Upewnij się, że jest w tym samym katalogu.")
            except Exception as e:
                print(f"Nieoczekiwany błąd podczas wywoływania skryptu: {e}")

        print(f"--- Czekam {PUSH_INTERVAL_SECONDS} sekund do następnego pchnięcia... ---")
        time.sleep(PUSH_INTERVAL_SECONDS)  # Czekaj określony czas


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n--- Automatyczne pchanie zatrzymane przez użytkownika. ---")