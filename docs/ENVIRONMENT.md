# Environnement technique

> À remplir avec les versions **de votre machine** (le barème demande les numéros de version).

| Élément | Version |
|---|---|
| OS | … |
| GPU / pilote / CUDA | … |
| Python | … |
| PyTorch | … |
| Ultralytics | … |
| OpenCV | … |
| NumPy / pandas | … |
| Streamlit | … |

Commandes utiles :

```bash
python --version
python -c "import torch, ultralytics, cv2; print(torch.__version__, torch.version.cuda, ultralytics.__version__, cv2.__version__)"
pip freeze > docs/requirements.lock.txt
nvidia-smi
git rev-parse --short HEAD     # commit à citer dans la présentation
```
