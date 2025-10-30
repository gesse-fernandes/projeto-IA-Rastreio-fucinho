# Rastreio de Focinho Bovino

Este projeto fornece uma API Flask para cadastro e identificação de bovinos a partir do focinho.

## Pré-requisitos

1. Python 3.10+
2. Servidor MySQL acessível (por padrão `localhost:3306`, banco `bovine_nose`).
3. Variáveis de ambiente opcionais:
   - `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`
   - `DATABASE_URL` (para sobrescrever a URL completa do SQLAlchemy)
   - `NOSE_DETECTOR_WEIGHTS` (caminho dos pesos YOLO, se desejar recorte automático)

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Migrando o banco de dados

Antes de iniciar a API, crie as tabelas executando:

```bash
python migrate.py
```

Isso usa a conexão configurada para garantir que as tabelas `animals` e `nose_embeddings` existam.

## Executando o servidor Flask

Inicie a aplicação com:

```bash
flask --app app:app run --host 0.0.0.0 --port 5000
```

ou, se preferir debug automático:

```bash
export FLASK_APP=app:app
flask run --debug
```

Também é possível iniciar diretamente pelo módulo:

```bash
python app.py
```

## Fluxo geral

1. Cadastre animais via `POST /animals` enviando os campos necessários e uma imagem em `image`.
2. Consulte ou atualize registros com os endpoints CRUD (`GET/PUT/DELETE /animals/<id>`).
3. Adicione mais focinhos com `POST /animals/<id>/embeddings`.
4. Identifique um animal enviando uma foto para `POST /identify`.

Para popular automaticamente usando imagens locais, organize arquivos em `data/enroll/<ID_ANIMAL>/` e execute:

```bash
python build_index.py
```

Esse script criará/atualizará os animais e embeddings no banco de dados.
