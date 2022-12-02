"""Data utils functions for pre-processing and data loading."""
import json, gzip, os
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
import torch
from sklearn.preprocessing import PowerTransformer


from data_prep import ProteinDrugNetwork
import logging

try:
    from ..config import parser
except ImportError:
    from gcn.config import parser


class EdgeBlock:
    def __init__(self, proteins, chems, links_df):
        err_msg = "proteins in links must be a subset of the supplied proteins"
        assert set(links_df["protein"]).issubset(proteins), err_msg

        err_msg = "chemicals in links must be a subset of the supplied chems"
        assert set(links_df["chemical"]).issubset(chems), err_msg

        self.proteins_ = set(proteins)
        self.chems_ = set(chems)
        self.links_ = links_df.reset_index(drop=True)
        self._chem_degrees_ = None

    def __repr__(self):
        return self.links_.__repr__()

    def INDUCED(links_df):
        return EdgeBlock(links_df["protein"].unique(), links_df["chemical"].unique(), links_df)

    def COMPLETE(proteins, chems):
        all_links = pd.DataFrame(
            np.array(np.meshgrid(list(proteins), list(chems))
        ).T.reshape(-1,2), columns=["protein","chemical"])
        return EdgeBlock(proteins, chems, all_links)

    def EMPTY(proteins, chems):
        empty_links = pd.DataFrame(np.zeros((0,2), dtype=int), columns=["protein", "chemical"])
        return EdgeBlock(proteins, chems, empty_links)

    def union(self, other_block):
        proteins = self.proteins_ | other_block.proteins_
        chems = self.chems_ | other_block.chems_
        links = pd.concat([self.links_, other_block.links_], axis=0).drop_duplicates(subset=["protein", "chemical"])
        return EdgeBlock(proteins, chems, links)

    def get_chem_degrees(self):
        if self._chem_degrees_ is None:
            self._chem_degrees_ = self.links_.groupby("chemical")["protein"].count().rename("degree")
        return self._chem_degrees_

    def filt_chems(self, shuffle_degrees=None, chem_filt=None,
                                             max_chem_degree=None, num_edges=None):
        filt = True
        chem_degrees = self.get_chem_degrees()

        if shuffle_degrees is not None:
            chem_degrees = chem_degrees.sample(frac=1, random_state=shuffle_degrees)

        if chem_filt is not None:
            filt &= chem_degrees.index.isin(chem_filt)

        if max_chem_degree is not None:
            filt &= chem_degrees <= max_chem_degree

        if num_edges is not None:
            filt &= chem_degrees.cumsum() <= num_edges

        keep_chems = set(chem_degrees.index[filt])
        rem_chems = set(chem_degrees.index[~filt])

        # Create a filtered EdgeBlock
        filt_chem_degrees = chem_degrees[filt].copy()
        keep = self.links_["chemical"].isin(filt_chem_degrees.index)
        filt_block = EdgeBlock(self.proteins_, keep_chems, self.links_[keep].copy())
        filt_block._chem_degrees_ = filt_chem_degrees

        if keep.all():              # no remaining EdgeBlock
            return filt_block, None
        else:                       # everything excluded by filt
            rem_block = EdgeBlock(self.proteins_, rem_chems, self.links_[~keep].copy())
            rem_block._chem_degrees_ = chem_degrees[~filt].copy()
            return filt_block, rem_block

    def get_edges(self, shuffle=None):
        edges = self.links_[["protein","chemical"]].to_numpy()

        if shuffle is not None:
            np.random.seed(shuffle)
            np.random.shuffle(edges)
        return torch.LongTensor(edges)

    def get_neg_edges(self, shuffle=None):
        all_links = EdgeBlock.COMPLETE(self.proteins_, self.chems_).links_
        pos_links = self.links_.copy()
        pos_links["positive"] = True

        df = pd.merge(all_links, pos_links, how="left")
        is_neg = df["positive"].isnull()
        neg_edges = ( df[is_neg][["protein","chemical"]]
                                        .sort_values(by=["protein","chemical"])
                                        .to_numpy() )

        if shuffle is not None:
            np.random.seed(shuffle)
            np.random.shuffle(neg_edges)
        return torch.LongTensor(neg_edges)


class Data(object):
    def __init__(self, args=None):
        self.args = parser.parse_args() if args is None else args

    def build(self):
        args = self.args
        self.dataset = args.dataset
        self.target_organism = args.target_organism
        self.other_organisms = args.other_organisms
        self.task = args.task
        self.normalize_adj = args.normalize_adj

        logging.getLogger().setLevel(logging.INFO)

        logging.info("\n----------  Loading data -----------")
        logging.info(f"Target Organism {self.target_organism}, "
                                 f"Other Organisms {self.other_organisms}")

        self.data = self.load_multi_org_data(self.target_organism, self.other_organisms)

        self.num_proteins = len(self.data["protein_to_idx"])
        self.num_chemicals = len(self.data["chem_to_idx"])
        logging.info(f"Num proteins {self.num_proteins},"
                                 f"Num chemicals {self.num_chemicals}")

        self.split_data_folds(self.data, args.test_prop, args.fold_prop,
                                                    args.split_seed, args.max_chem_degree)

        self.features = None if args.skip_feats else self.load_features()
        return self


    def load_multi_org_data(self, target_organism, other_organisms):
        args = self.args

        def load_network(org):
            return ProteinDrugNetwork(data_directory=args.data_prep_path, organism=org).from_json()

        networks = {target_organism: load_network(target_organism)}

        for organism in other_organisms:
            networks[organism] = load_network(organism)

        rm_flagged_chems = args.flagged_chems_path is not None
        if rm_flagged_chems:
            flags = pd.read_csv(args.flagged_chems_path, sep="\t")
            rm_chems = flags["chemical"][flags["flagged_for_removal"]==1].unique()

        def get_protein_chemical_links(network, test_drug_links=False):
            if test_drug_links:
                links = network.test_drug_pc_links
                if links is None:
                    return set(), set(), []
            else:
                links = network.pc_links

            links = links[["protein", "chemical"]].drop_duplicates()

            if rm_flagged_chems and not test_drug_links:    # test_drugs don't get removed
                links = links[~links["chemical"].isin(rm_chems)]

            return (set(links.protein),
                            set(links.chemical),
                            list(links.to_records(index=False)))

        def log_protein_chemical_links(proteins, chemicals, links, desc):
            logging.info(f"Loaded {len(proteins)} {desc} proteins, "
                                     f"{len(chemicals)} {desc} chemicals, "
                                     f"{len(links)} {desc} links")

        target_proteins, target_chemicals, target_links = \
                get_protein_chemical_links(networks[target_organism])

        target_known_proteins, target_known_chemicals, target_known_links = \
                get_protein_chemical_links(networks[target_organism], test_drug_links=True)

        log_protein_chemical_links(target_proteins, target_chemicals, target_links, "target")
        log_protein_chemical_links(target_known_proteins, target_known_chemicals, target_known_links, "target known")
        log_protein_chemical_links(target_proteins | target_known_proteins,
                                                             target_chemicals | target_known_chemicals,
                                                             target_links + target_known_links, "total target")

        other_proteins, other_chemicals, other_links = set(), set(), []
        other_known_proteins, other_known_chemicals, other_known_links = set(), set(), []
        for organism in other_organisms:
            proteins, chemicals, links = get_protein_chemical_links(networks[organism])
            other_proteins.update(proteins)
            other_chemicals.update(chemicals)
            other_links.extend(links)

            proteins, chemicals, links = get_protein_chemical_links(networks[organism], test_drug_links=True)
            other_known_proteins.update(proteins)
            other_known_chemicals.update(chemicals)
            other_known_links.extend(links)

        log_protein_chemical_links(other_proteins, other_chemicals, other_links, "other")
        log_protein_chemical_links(other_known_proteins, other_known_chemicals, other_known_links, "other test")
        log_protein_chemical_links(other_proteins | other_known_proteins,
                                                             other_chemicals | other_known_chemicals,
                                                             other_links + other_known_links, "total other")

        #inference_chemicals = other_chemicals - target_chemicals
        cold_chemicals = other_chemicals.intersection(target_chemicals)
        total_chemicals = target_chemicals | other_chemicals | target_known_chemicals | other_known_chemicals
        total_proteins = target_proteins | other_proteins | target_known_proteins | other_known_proteins

        # sort for reproducibility when generating index mapping
        total_proteins = sorted(list(total_proteins))
        total_chemicals = sorted(list(total_chemicals))

        def get_idx_maps(items):
            items_to_idx = {item: i for i, item in enumerate(items)}
            idx_to_items = {i: item for i, item in enumerate(items)}
            return items_to_idx, idx_to_items

        protein_to_idx, idx_to_protein = get_idx_maps(total_proteins)
        chem_to_idx, idx_to_chem = get_idx_maps(total_chemicals)

        def get_protein_protein_links(network, total_proteins, protein_to_idx):
            links = network.protein_links
            links.columns = ["protein1_name", "protein2_name"]
            links_filtered = links[(links.protein1_name.isin(total_proteins)) & (links.protein2_name.isin(total_proteins))].copy()

            links_filtered.loc[:,"protein1"] = links_filtered.apply(lambda x: protein_to_idx[x.protein1_name],axis=1)
            links_filtered.loc[:,"protein2"] = links_filtered.apply(lambda x: protein_to_idx[x.protein2_name], axis=1)
            links_final = list(links_filtered[["protein1","protein2"]].to_numpy())
            return links_final

        return {
                "protein_to_idx": protein_to_idx,
                "idx_to_protein": idx_to_protein,
                "chem_to_idx": chem_to_idx,
                "idx_to_chem": idx_to_chem,
                "target_proteins": target_proteins,
                "target_chemicals": target_chemicals,
                "target_links": target_links,
                "target_known_proteins": target_known_proteins,
                "target_known_chemicals": target_known_chemicals,
                "target_known_links": target_known_links,
                "other_proteins": other_proteins,
                "other_chemicals": other_chemicals,
                "other_links":other_links,
                "other_known_proteins": other_known_proteins,
                "other_known_chemicals": other_known_chemicals,
                "other_known_links": other_known_links,
                "cold_chemicals": cold_chemicals
            }


# ############### DATA SPLITS #####################################################
    def split_data_folds(self, data, test_prop, fold_prop, seed, max_chem_degree=None):
        # Create target / other blocks, and target subblocks
        target_block = self.get_block(data, "target")

        other_block = self.get_block(data, "other")
        other_known_block = self.get_block(data, "other_known")
        other_block = other_block.union(other_known_block)

        total_edges = len(target_block.links_) + len(other_block.links_)
        num_test_pos_edges = test_prop*total_edges
        num_fold_pos_edges = fold_prop*total_edges

        cold_chemicals = [data["chem_to_idx"][c] for c in data["cold_chemicals"]]
        cold_block, warm_block = target_block.filt_chems(shuffle_degrees=seed,
                                                                                                         chem_filt=cold_chemicals,
                                                                                                         max_chem_degree=max_chem_degree)

        test_block, rem_block = cold_block.filt_chems(num_edges=num_test_pos_edges)
        target_known_block = self.get_block(data, "target_known")
        test_block = test_block.union(target_known_block)

        fold_blocks, fold_chems, n_fold_degrees = [], [], []
        while rem_block is not None:
            fold_block, rem_block = rem_block.filt_chems(num_edges=num_fold_pos_edges)
            fold_blocks.append(fold_block)
            fold_chems.append(fold_block.chems_)
            n_fold_degrees.append(fold_block.get_chem_degrees().sum())

        num_folds = len(fold_blocks)

        data['test_chemicals'] = test_block.chems_
        data['fold_chemicals'] = fold_chems
        logging.info(f"Test chemicals (total degree): {len(data['test_chemicals'])} "
                                 f"({test_block.get_chem_degrees().sum()}).")
        logging.info(f"{num_folds}-fold chemicals (total degrees): "
                                 f"{[len(c) for c in data['fold_chemicals']]} {n_fold_degrees}")

        # Edges
        test_edges_pos = test_block.get_edges(shuffle=seed)
        warm_edges_pos = warm_block.get_edges(shuffle=seed)
        other_edges_pos = other_block.get_edges(shuffle=seed)

        fold_edges_pos, n_fold_pos = [], []
        for fold_block in fold_blocks:
            fold_edges_pos.append(fold_block.get_edges(shuffle=seed))
            n_fold_pos.append(len(fold_edges_pos[-1]))

        maplist = lambda f, it: list(map(f, it))
        frac = lambda n: round(n/total_edges, 2)
        logging.info(f"Num (frac) of {num_folds}-fold links: {n_fold_pos} "
                                 f"({maplist(frac, n_fold_pos)})")

        test_edges_false = test_block.get_neg_edges(shuffle=seed)
        warm_edges_false = warm_block.get_neg_edges(shuffle=seed)
        other_edges_false = other_block.get_neg_edges(shuffle=seed)

        fold_edges_false = []
        for fold_block in fold_blocks:
            fold_edges_false.append(fold_block.get_neg_edges(shuffle=seed))

        def edges_msg(text, edges, islist=False):
            n = maplist(len, edges) if islist else len(edges)
            f = maplist(frac, n) if islist else frac(n)
            return f"{text} Edges: {n} ({f})"

        logging.info(", ".join([edges_msg("Test Pos", test_edges_pos),
                                                        edges_msg("Test Neg", test_edges_false)]))
        logging.info(", ".join([edges_msg(f"{num_folds}-fold Pos", fold_edges_pos, True),
                                                        edges_msg(f"{num_folds}-fold Neg", fold_edges_false, True)]))
        logging.info(", ".join([edges_msg("Other Pos", other_edges_pos),
                                                        edges_msg("Other Neg", other_edges_false)]))
        logging.info(", ".join([edges_msg("Warm Pos", warm_edges_pos),
                                                        edges_msg("Warm Neg", warm_edges_false)]))

        # Save the edges
        data['other_edges'] = other_edges_pos
        data['other_edges_false'] = other_edges_false
        data['warm_edges'] = warm_edges_pos
        data['warm_edges_false'] = warm_edges_false
        data['fold_edges'] = fold_edges_pos
        data['fold_edges_false'] = fold_edges_false
        data['test_edges'] = test_edges_pos
        data['test_edges_false'] = test_edges_false

        # Save inference edges
        inf_chems = other_block.chems_ - target_block.chems_ - other_known_block.chems_
        data['inference_chemicals'] = inf_chems
        data['inference_edges'] = (
            EdgeBlock.EMPTY(target_block.proteins_, inf_chems).get_edges())
        data['inference_edges_false'] = (
            EdgeBlock.COMPLETE(target_block.proteins_, inf_chems).get_edges(shuffle=seed))

    def get_block(self, data, blocktype):
        proteins = {data["protein_to_idx"][p] for p in data[f"{blocktype}_proteins"]}
        chemicals = {data["chem_to_idx"][c] for c in data[f"{blocktype}_chemicals"]}
        recs = data[f"{blocktype}_links"]

        links_df = pd.DataFrame.from_records(recs, columns=["protein_name", "chemical_name"]).drop_duplicates()

        if len(links_df) > 0:
            links_df["protein"] = links_df.apply(lambda x: data["protein_to_idx"][x.protein_name], axis=1)
            links_df["chemical"] = links_df.apply(lambda x: data["chem_to_idx"][x.chemical_name], axis=1)
            return EdgeBlock(proteins, chemicals, links_df)
        else:
            return EdgeBlock.EMPTY(proteins, chemicals)

    def set_adj_train(self, args, fold=None, normalized=None ):
        data = self.data

        if normalized is None:
            normalized=self.normalize_adj

        def create_train_edges(edges, fold):
            folds_for_train = (
                data[f"fold_{edges}"][:fold] + data[f"fold_{edges}"][fold+1:]
                if fold is not None else data[f"fold_{edges}"]
            )
            if not args.include_other_organisms:
                train_edges = [data[f"warm_{edges}"]]
            elif set(args.include_other_organisms) == set(args.other_organisms):
                # include all other organisms
                train_edges = [data[f"other_{edges}"], data[f"warm_{edges}"]]
            else:
                other_org_protein_idx = []
                other_filtered_edges = []
                for organism in args.include_other_organisms:
                    other_org_protein_idx.extend([data["protein_to_idx"][p] for p in data["protein_to_idx"].keys() if p.startswith(str(organism))])
                for edge in data[f"other_{edges}"].cpu().detach().numpy():
                    if edge[0] in set(other_org_protein_idx):
                        other_filtered_edges.append(edge)
                other_filtered_edges = torch.LongTensor(other_filtered_edges)
                train_edges = [other_filtered_edges, data[f"warm_{edges}"]]

            train_edges += folds_for_train
            return torch.cat(train_edges)

        # No need to save train_edges* / val_edges* to JSON since they can easily be reconstructed
        data['train_edges'] = create_train_edges("edges", fold)
        data['train_edges_false'] = create_train_edges("edges_false", fold)

        data['val_edges'] = None if fold is None else data["fold_edges"][fold]
        data['val_edges_false'] = None if fold is None else data["fold_edges_false"][fold]

        logging.info(f"Train (excl. fold {fold}). "
                                 f"Pos Edges: {len(data['train_edges'])}, "
                                 f"Neg Edges: {len(data['train_edges_false'])}")

        if fold is not None:
            logging.info(f"Val (fold {fold}). "
                                     f"Pos Edges: {len(data['val_edges'])}, "
                                     f"Neg Edges: {len(data['val_edges_false'])}")

        logging.info('\nGenerating adj csr... ')
        num_nodes = self.num_proteins+self.num_chemicals

        # Here we only have two components: R_train (PxC) by R_train' (CxP)
        # So row-wise we will first see the proteins, then the chemicals
        # And col-wise we will first see the chemicals, then the proteins
        adj_data = np.ones(data['train_edges'].shape[0] * 2)
        rows = np.concatenate((
            data['train_edges'][:, 0],
            data['train_edges'][:, 1] + self.num_proteins
        ))
        cols = np.concatenate((
            data['train_edges'][:, 1] + self.num_proteins,
            data['train_edges'][:, 0]
        ))

        self.adj_train = sp.coo_matrix(
            (adj_data, (rows, cols)), shape=(num_nodes, num_nodes)
        ).tocsr().astype(np.float32)

        self.num_train_edges_false = len(data['train_edges_false'])
        self.num_train_edges = len(data['train_edges'])
        self.num_nodes = self.adj_train.shape[0]

        if normalized:
            self.adj_train_norm = normalize(self.adj_train + sp.eye(self.adj_train.shape[0]))
            self.adj_train_norm = sparse_mx_to_torch_sparse_tensor(self.adj_train_norm)

        logging.info('Adjacency matrix shape: {}'.format(self.adj_train.shape))


    def load_features(self, filepath=None, normalize=None):
        args = self.args

        if filepath is None:
            filepath = args.features_path

        if normalize is None:
            normalize = args.normalize_feats

        with open(filepath, "rb") as f:
            features = pickle.load(f)

        def create_feature_matrix(feature_key, is_chem, use_pca, median_imputation=True):
            ref_matrix = features[feature_key][
                f"feature_matrix{'_pca' if use_pca else ''}"]
            obj_to_refidx = features[feature_key][
                f"{'chemical' if is_chem else 'protein'}_to_index"]
            idx_to_obj = self.data[f"idx_to_{'chem' if is_chem else 'protein'}"]

            out_matrix = np.zeros((len(idx_to_obj), ref_matrix.shape[1]))
            median_features = np.median(ref_matrix, axis=0)
            for rowidx, obj in idx_to_obj.items():
                try:
                    out_matrix[rowidx] = ref_matrix[obj_to_refidx[obj]]
                except KeyError:
                    if median_imputation:
                        logging.info(f"Imputing {obj} not available in {feature_key} features")
                        out_matrix[rowidx] = median_features
                    else:
                        pass  # by default out_matrix values are zero

            if normalize:
                pt = PowerTransformer()
                out_matrix = pt.fit_transform(out_matrix)
            return out_matrix

        Xp = torch.tensor(
            create_feature_matrix("protein", False, True), dtype=torch.float)
        args.protein_feat_dim = Xp.shape[1]

        Xc = torch.tensor(np.concatenate([
            create_feature_matrix("rdkit", True, False),
            create_feature_matrix("mordred", True, True)
        ], axis=1), dtype=torch.float)
        args.chem_feat_dim = Xc.shape[1]

        return torch.block_diag(Xp, Xc)

# ############### FEATURES PROCESSING ####################################

def normalize(mx):
    """left Row-normalize sparse matrix."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    print("Left L1 normalizing")
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ############### UTILS ####################################

def write_json_gz(json_dict, outname):
    # More memory efficient way of writing to gzip
    json_name = f"{outname}.json"
    gz_name = f"{json_name}.gz"
    logging.info(f"Writing to {json_name}")

    with open(json_name, "w") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)

    logging.info(f"Compressing to {gz_name}")
    with open(json_name, "rb") as f:
        with gzip.GzipFile(gz_name, "wb") as fout:
            fout.write(f.read())

    os.remove(json_name)
