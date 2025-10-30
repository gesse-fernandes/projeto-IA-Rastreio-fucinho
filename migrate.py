"""Simple CLI to run database migrations (table creation)."""
from __future__ import annotations

import argparse

from database import init_db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Criar tabelas necessárias no banco configurado."
    )
    parser.parse_args()
    init_db()
    print("Migração concluída: tabelas verificadas/criadas com sucesso.")


if __name__ == "__main__":
    main()
