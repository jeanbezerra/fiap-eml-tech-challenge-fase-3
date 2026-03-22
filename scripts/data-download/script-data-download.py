from pathlib import Path

import requests


FILES_TO_DOWNLOAD = {
    "airlines.csv": "https://fiap-eml-tech-challenge-static-files-177862772785-sa-east-1-an.s3.sa-east-1.amazonaws.com/fase-3/airlines.csv",
    "airports.csv": "https://fiap-eml-tech-challenge-static-files-177862772785-sa-east-1-an.s3.sa-east-1.amazonaws.com/fase-3/airports.csv",
    "flights.csv": "https://fiap-eml-tech-challenge-static-files-177862772785-sa-east-1-an.s3.sa-east-1.amazonaws.com/fase-3/flights.csv",
}


def download_file(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES_TO_DOWNLOAD.items():
        destination = raw_data_dir / filename
        print(f"Baixando {filename}...")
        download_file(url, destination)
        print(f"Arquivo salvo em: {destination}")


if __name__ == "__main__":
    main()
