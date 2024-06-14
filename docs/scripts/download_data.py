"""run script from docs directory"""
from sed.dataset import dataset

root_dir = "./tutorial"
dataset.get("WSe2", remove_zip=True, root_dir=root_dir)
dataset.get("Gd_W110", remove_zip=True, root_dir=root_dir)
dataset.get("TaS2", remove_zip=True, root_dir=root_dir)