from tqdm import tqdm
from collections import defaultdict
import numpy as np
import tempfile
from typing import DefaultDict, List, Tuple, Dict, Set
from scipy.sparse import coo_matrix


def augment_kb_with_inv_edges(file_name: str) -> None:
    # Create temporary file read/write
    t = tempfile.NamedTemporaryFile(mode="r+")
    # Open input file read-only
    i = open(file_name, 'r')

    # Copy input file to temporary file, modifying as we go
    temp_list = []
    for line in i:
        t.write(line.strip() + "\n")
        e1, r, e2 = line.strip().split("\t")
        temp_list.append((e1, r, e2))
        temp_list.append((e2, "_" + r, e1))

    i.close()  # Close input file
    o = open(file_name, "w")  # Reopen input file writable
    # Overwriting original file with temporary file contents
    for (e1, r, e2) in temp_list:
        o.write("{}\t{}\t{}\n".format(e1, r, e2))
    t.close()  # Close temporary file, will cause it to be deleted
    o.close()


def create_adj_list(file_name: str) -> DefaultDict[str, List[Tuple[str, str]]]:
    out_map = defaultdict(list)
    fin = open(file_name)
    for line_ctr, line in tqdm(enumerate(fin)):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        out_map[e1].append((r, e2))
    return out_map


def create_adj_list_from_triples(triples: List[Tuple[str, str, str]]) -> DefaultDict[str, List[Tuple[str, str]]]:
    out_map = defaultdict(list)
    for edge in triples:
        e1, r, e2 = edge
        out_map[e1].append((r, e2))
    return out_map


def load_data(file_name: str) -> DefaultDict[Tuple[str, str], list]:
    out_map = defaultdict(list)
    fin = open(file_name)

    for line in tqdm(fin):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        out_map[(e1, r)].append(e2)

    return out_map


def load_data_from_triples(triples: List[Tuple[str, str, str]]) -> DefaultDict[Tuple[str, str], list]:
    out_map = defaultdict(list)
    for edge in tqdm(triples):
        e1, r, e2 = edge
        out_map[(e1, r)].append(e2)
    return out_map


def load_data_all_triples(train_file: str, dev_file: str, test_file: str) -> DefaultDict[Tuple[str, str], list]:
    """
    Returns a map of all triples in the knowledge graph. Use this map only for filtering in evaluation.
    :param train_file:
    :param dev_file:
    :param test_file:
    :return:
    """
    out_map = defaultdict(list)
    for file_name in [train_file, dev_file, test_file]:
        fin = open(file_name)
        for line in tqdm(fin):
            line = line.strip()
            e1, r, e2 = line.split("\t")
            out_map[(e1, r)].append(e2)
    return out_map


def create_vocab(kg_file: str) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    entity_vocab, rev_entity_vocab = {}, {}
    rel_vocab, rev_rel_vocab = {}, {}
    fin = open(kg_file)
    entity_ctr, rel_ctr = 0, 0
    for line in tqdm(fin):
        line = line.strip()
        e1, r, e2 = line.split("\t")
        if e1 not in entity_vocab:
            entity_vocab[e1] = entity_ctr
            rev_entity_vocab[entity_ctr] = e1
            entity_ctr += 1
        if e2 not in entity_vocab:
            entity_vocab[e2] = entity_ctr
            rev_entity_vocab[entity_ctr] = e2
            entity_ctr += 1
        if r not in rel_vocab:
            rel_vocab[r] = rel_ctr
            rev_rel_vocab[rel_ctr] = r
            rel_ctr += 1
    return entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab


def create_vocab_wikidata(file_name: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    vocab, rev_vocab = {}, {}
    with open(file_name) as fin:
        for line_counter, line in tqdm(enumerate(fin)):
            if line_counter == 0:
                continue
            line = line.strip()
            qid, num = line.split("\t")
            vocab[qid] = int(num)
            rev_vocab[int(num)] = qid
    return vocab, rev_vocab


def read_graph(file_name: str, entity_vocab: Dict[str, int], rel_vocab: Dict[str, int]) -> np.ndarray:
    adj_mat = np.zeros((len(entity_vocab), len(rel_vocab)))
    fin = open(file_name)
    for line in tqdm(fin):
        line = line.strip()
        e1, r, _ = line.split("\t")
        adj_mat[entity_vocab[e1], rel_vocab[r]] = 1

    return adj_mat


def read_graph_from_triples(triples: List[Tuple[str, str, str]], entity_vocab: Dict[str, int],
                            rel_vocab: Dict[str, int]) -> np.ndarray:
    adj_mat = np.zeros((len(entity_vocab), len(rel_vocab)))
    for edge in tqdm(triples):
        e1, r, _ = edge
        adj_mat[entity_vocab[e1], rel_vocab[r]] = 1

    return adj_mat


def read_graph_wikidata(file_name: str) -> coo_matrix:
    fin = open(file_name)
    data, row, col = [], [], []
    for line_counter, line in tqdm(enumerate(fin)):
        line = line.strip()
        e1, r, _ = line.split("\t")
        row.append(int(e1))
        col.append(int(r))
        data.append(1)

    adj_mat = coo_matrix((data, (row, col)))
    fin.close()
    return adj_mat


def load_mid2str(mid2str_file: str) -> DefaultDict[str, str]:
    mid2str = defaultdict(str)
    with open(mid2str_file) as fin:
        for line in tqdm(fin):
            line = line.strip()
            try:
                mid, ent_name = line.split("\t")
            except ValueError:
                continue
            if mid not in mid2str:
                mid2str[mid] = ent_name
    return mid2str


def get_unique_entities(kg_file: str) -> Set[str]:
    unique_entities = set()
    fin = open(kg_file)
    for line in fin:
        e1, r, e2 = line.strip().split()
        unique_entities.add(e1)
        unique_entities.add(e2)
    fin.close()
    return unique_entities


def get_unique_entities_from_triples(triples: List[Tuple[str, str, str]]) -> Set[str]:
    unique_entities = set()
    for edge in triples:
        e1, r, e2 = edge
        unique_entities.add(e1)
        unique_entities.add(e2)
    return unique_entities


def get_entities_group_by_relation(file_name: str) -> DefaultDict[str, List[str]]:
    rel_to_ent_map = defaultdict(list)
    fin = open(file_name)
    for line in fin:
        e1, r, e2 = line.strip().split('\t')
        rel_to_ent_map[r].append(e1)
    return rel_to_ent_map


def get_entities_group_by_relation_from_triples(triples: List[Tuple[str, str, str]]) -> DefaultDict[str, List[str]]:
    rel_to_ent_map = defaultdict(list)
    for edge in triples:
        e1, r, e2 = edge
        rel_to_ent_map[r].append(e1)
    return rel_to_ent_map


def load_rules_for_FB122(file_name: str) -> DefaultDict[str, List[str]]:
    fin = open(file_name)
    rules_map = defaultdict(list)
    for line in fin:
        line = line.strip()
        # parse the rule
        hypothesis, premise = line.split(":-")
        hypothesis, premise = hypothesis.strip(), premise.strip()
        premise = premise.split(",")
        premise = tuple([p.strip() for p in premise])
        rules_map[hypothesis].append(premise)
    fin.close()

    return rules_map


def is_inv_relation(r: str, dataset_name="nell") -> bool:
    if dataset_name == "nell":
        if r[-4:] == "_inv":
            return True
        else:
            return False
    else:
        if r[:2] == "__" or r[:2] == "_/":
            return True
        else:
            return False


def get_inv_relation(r: str, dataset_name="nell") -> str:
    if dataset_name == "nell":
        if r[-4:] == "_inv":
            return r[:-4]
        else:
            return r + "_inv"
    else:
        if r[:2] == "__" or r[:2] == "_/":
            return r[1:]
        else:
            return "_" + r


def return_nearest_relation_str(sim_sorted_ind, rev_rel_vocab, rel, k=5):
    """
    helper method to print nearest relations
    :param sim_sorted_ind: sim matrix sorted wrt index
    :param rev_rel_vocab:
    :param rel: relation we want sim for
    :return:
    """
    print("====Query rel: {}====".format(rev_rel_vocab[rel]))
    nearest_rel_inds = sim_sorted_ind[rel, :k]
    return [rev_rel_vocab[i] for i in nearest_rel_inds]
