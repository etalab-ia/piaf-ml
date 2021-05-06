# Naming of files :
### Raw dataset
We keep the same names for the file given by our clients. If the client gives an update with the same name, we add a version at the end of the filename like so : -Vx
### Python treatment
We usually need to process our raw dataset with python scripts. We try to name them exactly after the names of the dataset file (replacing the extension with .py)

# Versioning :
We must add the name of the raw file inside the SQuAD formatted dataset. It helps to know how was created the SQuAD dataset.  

```py
  "version": "name of the raw file"
```
