from fastai.vision.all import *
from fastai.data.transforms import _get_files


def from_grandparent(path, remap:dict=None):
    if remap is not None and path.parent.name in remap.keys():
        return remap[path.parent.name]
    return path.parent.parent.name


def _greatgrandparent_idxs_mskd(items, name, excl:list=[]):
    def _inner(items, name, excl): return mask2idxs(Path(o).parent.parent.parent.name == name and Path(o).parent.name not in excl for o in items)
    return [i for n in L(name) for i in _inner(items,n,excl)]

def GreatGrandparentSplitter(train_name='train', valid_name='valid',excl_train:list=[], excl_test:list=[]):
    """
    Split `items` from the great grand parent folder names (`train_name` and `valid_name`).
    Exclude any parent folders given in `excl_*`.
    """
    def _inner(o):
        return _greatgrandparent_idxs_mskd(o, train_name, excl_train),_greatgrandparent_idxs_mskd(o, valid_name, excl_test)
    return _inner


def RandomSplitter(valid_pct:float=0.2, generator: Optional[Generator] = None):
    def _inner(o):
        indices = torch.randperm(
                                    len(o), 
                                    generator=generator if generator is not None else torch.Generator()
                                ).tolist()

        cut = int(math.floor((1-valid_pct) * len(o)))
        train_idxs = indices[:cut]
        val_idxs = indices[cut:]

        return train_idxs, val_idxs
    return _inner


def RandomSplitterMaskedParent(valid_pct:float=0.2, excl:list=[], generator: Optional[Generator] = None):
    """Create function that splits `items` between train/val with `valid_pct` randomly
     and masks any objects in folders listed in `excl` for the train set."""
    def _inner(o):
        train_idxs, val_idxs = RandomSplitter(valid_pct=valid_pct, generator=generator)(o)

        # mask after permutation is complete 
        train_idxs = [ti for ti in train_idxs if Path(o[ti]).parent.name not in excl]

        return train_idxs, val_idxs
    return _inner


def mask_mode(df, idx_list, excl_mode, label_mode:str="mode"):
    return [ti for ti in idx_list if df.loc[ti,label_mode] != excl_mode] if excl_mode is not None else idx_list

def RandomSplitterMaskedMode(valid_pct=0.2, masked_mode:str=None, generator: Optional[Generator] = None, label_mode:str="mode"):
    def _inner(df):
        splits = RandomSplitter(valid_pct=valid_pct, generator=generator)(range_of(df))
        return L(mask_mode(df, splits[0], masked_mode, label_mode)), splits[1]
    return _inner


def get_image_files_filtered(path, recurse=True, rec_excl_folders: dict = {}, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, excluding folders (vals) in parent folder (keys), if specified."
    path = Path(path)
    extensions = setify(image_extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(d) > 0:
                relpath = os.path.relpath(Path(p), path)
                excl_parents = [parent for parent in rec_excl_folders.keys() if parent in relpath]

                excl_folders_nested = [rec_excl_folders[parent] for parent in excl_parents]
                excl_folders_vals = []
                for item in excl_folders_nested:
                    if isinstance(item, list):
                        excl_folders_vals.extend(item)
                    else:
                        excl_folders_vals.append(item)
                excl_folders = [folder for folder in d if folder in excl_folders_vals] if len(excl_parents) > 0 else []

                if len(excl_folders) > 0:
                    d[:] = [o for o in d if o not in excl_folders]
                elif len(excl_parents) > 1:
                    raise ValueError(f"Detected multiple exclusion parents in current relpath. This should not happen! ({excl_parents})")
                else:
                    d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return L(res)


def get_df_item(path, item_hdl:str='X', lbl_hdl:str='y', data_coord_x:str='x', data_coord_y:str='y', data_label_class:str='class', data_label_mode:str='mode'):
    npzfile = np.load(path)
    return pd.DataFrame(np.hstack([npzfile[item_hdl],npzfile[lbl_hdl]]), columns=[data_coord_x,data_coord_y,data_label_class,data_label_mode])


def get_col(tpl_idx, dtype=None):
    def _inner(o):
        _item = o[tpl_idx]
        return _item.to_numpy(dtype=dtype) if isinstance(_item,pd.Series) else _item.astype(dtype=dtype)
    return _inner
