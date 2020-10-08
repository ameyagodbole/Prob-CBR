from collections import Counter
import logging
import numpy as np
import os

from prob_cbr.data.data_utils import get_inv_relation, is_inv_relation
logger = logging.getLogger('stream_utils')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


def read_triple_raw(file_path, dataset_name):
    """
    Read triples and map them into ids.
    """
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            if not is_inv_relation(r, dataset_name):
                triples.append((h, r, t))
    return triples


class KBStream:
    def __init__(self, dataset_name, data_path, test_file_name=None,
                 stream_init_proportion=0.5, n_stream_updates=10, seed=42):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.stream_init_proportion = stream_init_proportion
        self.n_stream_updates = n_stream_updates
        self.stream_rng = np.random.default_rng(seed)
        self.train_rng = np.random.default_rng(seed)

        self.entity_set, self.relation_set = set(), set()

        with open(os.path.join(self.data_path, 'entities.dict')) as fin:
            for line in fin:
                eid, entity = line.strip().split('\t')
                self.entity_set.add(entity)

        with open(os.path.join(self.data_path, 'relations.dict')) as fin:
            for line in fin:
                rid, relation = line.strip().split('\t')
                self.relation_set.add(relation)

        if test_file_name is None or test_file_name == '':
            test_file_name = 'test.txt'
        if dataset_name == 'nell':
            graph_file = 'full_graph.txt'
        else:
            graph_file = 'graph.txt'
        self.train_triples = read_triple_raw(os.path.join(self.data_path, graph_file), self.dataset_name)
        self.valid_triples = read_triple_raw(os.path.join(self.data_path, 'dev.txt'), self.dataset_name)
        self.test_triples = read_triple_raw(os.path.join(self.data_path, test_file_name), self.dataset_name)
        self.kb_state = {'entity2id': {}, 'relation2id': {},
                         'train_triples': [], 'valid_triples': [], 'test_triples': []}

    def get_max_num_entities(self):
        return len(self.entity_set)

    def get_max_num_relations(self):
        return 2*len(self.relation_set)

    def get_init_kb(self):
        # INIT
        # Sample 10% of the most common nodes (hubs)
        # Sample (stream_init_proportion - 10)% of the remaining nodes randomly
        node_usage_train = Counter([e for (e, _, _) in self.train_triples] + [e for (_, _, e) in self.train_triples])
        init_entities = [_ent for _ent, _ in node_usage_train.most_common(len(node_usage_train) // 10)]
        for _ent in init_entities:
            del node_usage_train[_ent]
        permutation = self.stream_rng.permutation(len(node_usage_train))
        usage_list = list(node_usage_train.most_common())
        sample_size = int(np.ceil(max(self.stream_init_proportion - 0.1, 0.0)*len(self.entity_set)))
        init_entities.extend([usage_list[j][0] for j in permutation[:sample_size]])
        assert len(init_entities) == len(set(init_entities))
        init_entities = set(init_entities)

        entity2id, relation2id = {}, {}
        id2entity, id2relation = {}, {}
        for eid, entity in enumerate(sorted(init_entities)):
            entity2id[entity] = eid
            id2entity[eid] = entity

        edge_coverage = {'train': 0, 'valid': 0, 'test': 0}
        init_train_triples, init_valid_triples, init_test_triples = [], [], []
        for edge in self.train_triples:
            e1, r, e2 = edge
            if e1 in init_entities and e2 in init_entities:
                if r not in relation2id:
                    new_id = len(relation2id)
                    relation2id[r] = new_id
                    id2relation[new_id] = r
                    new_id = len(relation2id)
                    r_inv = get_inv_relation(r, self.dataset_name)
                    relation2id[r_inv] = new_id
                    id2relation[new_id] = r_inv
                init_train_triples.append((e1, r, e2))
                edge_coverage['train'] += 1

        for edge in self.valid_triples:
            e1, r, e2 = edge
            if e1 in init_entities and e2 in init_entities:
                if r not in relation2id:
                    new_id = len(relation2id)
                    relation2id[r] = new_id
                    id2relation[new_id] = r
                    new_id = len(relation2id)
                    r_inv = get_inv_relation(r, self.dataset_name)
                    relation2id[r_inv] = new_id
                    id2relation[new_id] = r_inv
                init_valid_triples.append((e1, r, e2))
                edge_coverage['valid'] += 1

        for edge in self.test_triples:
            e1, r, e2 = edge
            if e1 in init_entities and e2 in init_entities:
                if r not in relation2id:
                    new_id = len(relation2id)
                    relation2id[r] = new_id
                    id2relation[new_id] = r
                    new_id = len(relation2id)
                    r_inv = get_inv_relation(r, self.dataset_name)
                    relation2id[r_inv] = new_id
                    id2relation[new_id] = r_inv
                init_test_triples.append((e1, r, e2))
                edge_coverage['test'] += 1

        logger.info(f"[STREAM] Init edge_coverage: "
              f"train: {edge_coverage['train']} ({edge_coverage['train'] / len(self.train_triples) * 100:0.2f}%) "
              f"valid: {edge_coverage['valid']} ({edge_coverage['valid'] / len(self.valid_triples) * 100:0.2f}%) "
              f"test: {edge_coverage['test']} ({edge_coverage['test'] / len(self.test_triples) * 100:0.2f}%)")
        logger.info(f'[STREAM] Init entity_coverage:'
              f' {len(init_entities)} ({len(init_entities) / (len(self.entity_set)) * 100:0.2f}%)')

        self.kb_state['entity2id'] = entity2id.copy()
        self.kb_state['relation2id'] = relation2id.copy()
        self.kb_state['id2entity'] = id2entity.copy()
        self.kb_state['id2relation'] = id2relation.copy()
        self.kb_state['train_triples'] = init_train_triples.copy()
        self.kb_state['valid_triples'] = init_valid_triples.copy()
        self.kb_state['test_triples'] = init_test_triples.copy()

        # RotatE explicitly adds them in model
        rev_train_triples = [(e2, get_inv_relation(r, self.dataset_name), e1) for (e1, r, e2) in init_train_triples]
        rev_valid_triples = [(e2, get_inv_relation(r, self.dataset_name), e1) for (e1, r, e2) in init_valid_triples]
        rev_test_triples = [(e2, get_inv_relation(r, self.dataset_name), e1) for (e1, r, e2) in init_test_triples]
        init_train_triples = init_train_triples + rev_train_triples
        init_valid_triples = init_valid_triples + rev_valid_triples
        init_test_triples = init_test_triples + rev_test_triples

        return entity2id, id2entity, relation2id, id2relation, \
               init_train_triples + init_valid_triples + init_test_triples,\
               init_train_triples, init_valid_triples, init_test_triples

    def batch_generator(self):
        for step in range(self.n_stream_updates):
            logger.info(f'[STREAM] Generating batch {step + 1}...')
            entity2id, relation2id = self.kb_state['entity2id'], self.kb_state['relation2id']
            id2entity, id2relation = self.kb_state['id2entity'], self.kb_state['id2relation']
            curr_train_triples, curr_valid_triples, curr_test_triples = \
                self.kb_state['train_triples'], self.kb_state['valid_triples'], self.kb_state['test_triples']
            new_train_triples, new_valid_triples, new_test_triples = [], [], []

            seen_entities = set(entity2id.keys())
            unseen_entities = sorted(self.entity_set.difference(seen_entities))
            permutation = self.stream_rng.permutation(len(unseen_entities))
            sample_size = int(np.ceil((1 - self.stream_init_proportion) / self.n_stream_updates * len(self.entity_set)))
            if step == self.n_stream_updates - 1:
                sample_size = len(unseen_entities)
            new_entities = [unseen_entities[j] for j in permutation[:sample_size]]
            new_entities = set(new_entities)

            for entity in sorted(new_entities):
                if entity not in entity2id:
                    new_id = len(entity2id)
                    entity2id[entity] = new_id
                    id2entity[new_id] = entity

            for edge in self.train_triples:
                e1, r, e2 = edge
                if e1 in seen_entities and e2 in seen_entities:
                    continue
                if (e1 in new_entities or e1 in seen_entities) and (e2 in new_entities or e2 in seen_entities):
                    if r not in relation2id:
                        new_id = len(relation2id)
                        relation2id[r] = new_id
                        id2relation[new_id] = r
                        new_id = len(relation2id)
                        r_inv = get_inv_relation(r, self.dataset_name)
                        relation2id[r_inv] = new_id
                        id2relation[new_id] = r_inv
                    new_train_triples.append((e1, r, e2))

            for edge in self.valid_triples:
                e1, r, e2 = edge
                if e1 in seen_entities and e2 in seen_entities:
                    continue
                if (e1 in new_entities or e1 in seen_entities) and (e2 in new_entities or e2 in seen_entities):
                    if r not in relation2id:
                        new_id = len(relation2id)
                        relation2id[r] = new_id
                        id2relation[new_id] = r
                        new_id = len(relation2id)
                        r_inv = get_inv_relation(r, self.dataset_name)
                        relation2id[r_inv] = new_id
                        id2relation[new_id] = r_inv
                    new_valid_triples.append((e1, r, e2))

            for edge in self.test_triples:
                e1, r, e2 = edge
                if e1 in seen_entities and e2 in seen_entities:
                    continue
                if (e1 in new_entities or e1 in seen_entities) and (e2 in new_entities or e2 in seen_entities):
                    if r not in relation2id:
                        new_id = len(relation2id)
                        relation2id[r] = new_id
                        id2relation[new_id] = r
                        new_id = len(relation2id)
                        r_inv = get_inv_relation(r, self.dataset_name)
                        relation2id[r_inv] = new_id
                        id2relation[new_id] = r_inv
                    new_test_triples.append((e1, r, e2))

            all_train_triples = new_train_triples + curr_train_triples
            all_valid_triples = new_valid_triples + curr_valid_triples
            all_test_triples = new_test_triples + curr_test_triples
            logger.info(f"[STREAM] Batch edge_coverage: "
                  f"train: {len(new_train_triples)} ({len(new_train_triples) / len(self.train_triples) * 100:0.2f}%) "
                  f"valid: {len(new_valid_triples)} ({len(new_valid_triples) / len(self.valid_triples) * 100:0.2f}%) "
                  f"test: {len(new_test_triples)} ({len(new_test_triples) / len(self.test_triples) * 100:0.2f}%)")
            logger.info(f"[STREAM] Total edge_coverage: "
                  f"train: {len(all_train_triples)} ({len(all_train_triples) / len(self.train_triples) * 100:0.2f}%) "
                  f"valid: {len(all_valid_triples)} ({len(all_valid_triples) / len(self.valid_triples) * 100:0.2f}%) "
                  f"test: {len(all_test_triples)} ({len(all_test_triples) / len(self.test_triples) * 100:0.2f}%)")
            logger.info(f'[STREAM] Total entity_coverage:'
                  f' {len(entity2id)} ({len(entity2id) / (len(self.entity_set)) * 100:0.2f}%)')

            self.kb_state['entity2id'] = entity2id.copy()
            self.kb_state['relation2id'] = relation2id.copy()
            self.kb_state['id2entity'] = id2entity.copy()
            self.kb_state['id2relation'] = id2relation.copy()
            self.kb_state['train_triples'] = all_train_triples.copy()
            self.kb_state['valid_triples'] = all_valid_triples.copy()
            self.kb_state['test_triples'] = all_test_triples.copy()

            # RotatE explicitly adds them in model
            rev_train_triples = [(e2, get_inv_relation(r, self.dataset_name), e1) for (e1, r, e2) in all_train_triples]
            rev_valid_triples = [(e2, get_inv_relation(r, self.dataset_name), e1) for (e1, r, e2) in all_valid_triples]
            rev_test_triples = [(e2, get_inv_relation(r, self.dataset_name), e1) for (e1, r, e2) in all_test_triples]
            all_train_triples = all_train_triples + rev_train_triples
            all_valid_triples = all_valid_triples + rev_valid_triples
            all_test_triples = all_test_triples + rev_test_triples

            rev_valid_triples = [(e2, get_inv_relation(r, self.dataset_name), e1) for (e1, r, e2) in new_valid_triples]
            rev_test_triples = [(e2, get_inv_relation(r, self.dataset_name), e1) for (e1, r, e2) in new_test_triples]
            new_valid_triples = new_valid_triples + rev_valid_triples
            new_test_triples = new_test_triples + rev_test_triples

            yield entity2id, id2entity, relation2id, id2relation, \
                  all_train_triples + all_valid_triples + all_test_triples, \
                  all_train_triples, all_valid_triples, new_valid_triples, all_test_triples, new_test_triples
