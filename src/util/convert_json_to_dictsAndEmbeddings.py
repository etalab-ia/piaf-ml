def convert_json_to_dictsAndEmbeddings(dir_path: str, split_paragraphs: bool = False):
    """
    Convert all Json in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.
    expected format for Json is :
    {
        text : string,
        link : string
    }
    :param dir_path: path for the documents to be written to the database
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.
    :return: None
    """

    file_paths = [p for p in Path(dir_path).glob("**/*")]

    documents = []
    for path in file_paths:
        if path.suffix.lower() == ".json":
            with open(path) as doc:
                jsonDoc = json.load(doc)
                text = jsonDoc["text"]
                question_emb = retriever.embed(texts=text)[0]
        else:
            raise Exception(f"Indexing of {path.suffix} files is not currently supported.")


        if split_paragraphs:
            raise Exception(f"Splitting paragraph not currently supported.")
        else:
            documents.append({"text": text, "question": text, "question_emb": question_emb, "meta": {"name": path.name, "link": f"https://www.service-public.fr/particuliers/vosdroits/{path.name.split('--', 1)[0]}" }})

    return documents
