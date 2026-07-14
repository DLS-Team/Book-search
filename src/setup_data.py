import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# Вставьте сюда прямую ссылку с Hugging Face (замените ВАШ_НИК и имя репозитория)
DOWNLOAD_URL = "https://huggingface.co/datasets/fortunoalib/processed_data/resolve/main/processed_data.zip"

ZIP_FILENAME = "processed_data.zip"

# Получаем корневую директорию проекта (на уровень выше, чем папка src)
ROOT_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT_DIR / ZIP_FILENAME

# Файл-маркер, по которому мы понимаем, что данные уже распакованы
MARKER_FILE = ROOT_DIR / "data" / "processed" / "processed_chapters.jsonl"


def download_file(url: str, dest_path: Path):
    """Потоковая загрузка больших файлов."""
    print(f"Начинаем скачивание...")

    session = requests.Session()
    response = session.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    # Защита от скачивания HTML-страницы с ошибкой вместо архива
    if 0 < total_size < 10 * 1024 * 1024:
        raise ValueError("Сервер вернул файл слишком малого размера. Проверьте ссылку.")

    with open(dest_path, 'wb') as file, tqdm(
            desc=ZIP_FILENAME,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))


def main():
    if MARKER_FILE.exists():
        print("Предподготовленные данные уже существуют. Скачивание пропущено.")
        return

    try:
        # 1. Скачивание
        download_file(DOWNLOAD_URL, ZIP_PATH)
        print(f"Файл {ZIP_FILENAME} успешно скачан.")

        # 2. Распаковка напрямую в корень (сохраняя структуру папок)
        print("Распаковка архива (это займет несколько минут)...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(ROOT_DIR)

        # 3. Удаление архива
        os.remove(ZIP_PATH)
        print("Временный ZIP-архив удален.")

        # Финальная проверка
        if MARKER_FILE.exists():
            print(f"Данные успешно распакованы в {ROOT_DIR}")
        else:
            print("ОШИБКА: После распаковки маркерный файл не найден. Проверь структуру внутри ZIP-архива.")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка сети при скачивании файла: {e}")
        if ZIP_PATH.exists():
            os.remove(ZIP_PATH)

    except zipfile.BadZipFile:
        print("ОШИБКА: Скачанный файл не является ZIP-архивом.")
        if ZIP_PATH.exists():
            os.remove(ZIP_PATH)

    except ValueError as e:
        print(f"Ошибка: {e}")
        if ZIP_PATH.exists():
            os.remove(ZIP_PATH)


if __name__ == "__main__":
    main()