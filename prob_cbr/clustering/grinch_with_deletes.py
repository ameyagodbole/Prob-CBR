import time
import numpy as np
import wandb
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger('grinch')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


class GrinchWithDeletes(object):

    def __init__(self, points, rotate_cap=100, graft_cap=100, norm='l2', sim='dot'):
        logger.info('[GrinchWithDeletes] points %s', str(points.shape))
        self.point_counter = 0  # largest index of any point
        self.points = points
        self.deleted_points = []
        self.deleted_points_set = set()
        self.dim = points.shape[1]
        self.max_nodes = 1000000
        self.ancs = [[] for _ in range(self.max_nodes)]
        self.sibs = None
        self.children = [[] for _ in range(self.max_nodes)]
        self.descendants = [[] for _ in range(self.max_nodes)]
        self.scores = -np.inf * np.ones(self.max_nodes, dtype=np.float32)
        self.needs_update_model = np.zeros(self.max_nodes, dtype=np.bool_)
        self.new_node = np.ones(self.max_nodes, dtype=np.bool_)
        # self.new_node[0:self.points.shape[0]] = 0
        self.needs_update_desc = np.zeros(self.max_nodes, dtype=np.bool_)
        self.parent = -1 * np.ones(self.max_nodes, dtype=np.int32)
        self.next_node_id = len(points)
        self.centroids = np.zeros((self.max_nodes, self.dim), dtype=np.float32)
        self.sums = np.zeros((self.max_nodes, self.dim), dtype=np.float32)
        self.num_descendants = -1 * np.ones(self.max_nodes, dtype=np.int32)
        self.rotate_cap = rotate_cap
        self.graft_cap = graft_cap

        self.graft_percentages = np.zeros(self.points.shape[0])
        self.inst_number_of_graft_allowable = np.zeros(self.points.shape[0])

        if norm == 'l2':
            logger.info('Using centroid = l2')
            self.compute_centroid = self.compute_centroid_l2_norm
        elif norm == 'l_inf':
            logger.info('Using centroid = l_inf')
            self.compute_centroid = self.compute_centroid_l_inf_norm
        elif norm == 'none':
            logger.info('Using centroid = none')
            self.compute_centroid = self.compute_centroid_no_norm

        if sim == 'dot':
            logger.info('Using csim = dot')
            self.csim = self.csim_dot
        elif sim == 'l2':
            logger.info('Using csim = l2')
            self.csim = self.csim_l2
        elif sim == 'sql2':
            logger.info('Using csim = sql2')
            self.csim = self.csim_sql2
        elif sim == 'ward':
            logger.info('Using csim = sql2')
            self.csim = self.csim_ward

        self.k = 1

        # Timing and stats
        self.time_in_search = 0
        self.time_in_rotate = 0
        self.time_in_update = 0
        self.time_in_update_walk = 0
        self.time_in_update_from_children = 0
        self.time_in_graft_score_only = 0
        self.time_in_graft = 0
        self.time_in_lca = 0
        self.time_in_descendants = 0
        self.time_in_centroid = 0
        self.time_in_graft_search = 0
        self.time_in_graft_get_scores = 0
        self.this_time_in_graft_get_scores = 0

        self.number_of_rotates = 0
        self.number_of_grafts = 0
        self.number_of_grafts_considered = 0
        self.number_of_graft_allowable = 0
        self.number_of_rotates_considered = 0

        self.this_number_of_rotates = 0
        self.this_number_of_grafts = 0
        self.this_number_of_grafts_considered = 0
        self.this_number_of_rotates_considered = 0
        self.this_time_in_search = 0
        self.this_time_in_rotate = 0
        self.this_time_in_update = 0
        self.this_time_in_graft = 0
        self.this_time_in_graft_search = 0
        self.cached_nns = None

    def clear_stats(self):
        self.this_number_of_rotates = 0
        self.this_number_of_grafts = 0
        self.this_number_of_grafts_considered = 0
        self.this_number_of_rotates_considered = 0
        self.this_time_in_search = 0
        self.this_time_in_rotate = 0
        self.this_time_in_update = 0
        self.this_time_in_graft = 0
        self.this_time_in_graft_search = 0
        self.this_time_in_graft_get_scores = 0

    def stats_string(self):
        r = 'search_time=%s\trotate_time=%s\tgraft_time=%s\tupdate_time=%s\tdescendant_time=%s\tcentroid_time=%s\tlca_time=%s\tnum_rotate=%s\tnum_graft=%s\tnum_rotate_considered=%s\tnum_graft_considered=%s\t%s\n' % (
            self.time_in_search, self.time_in_rotate, self.time_in_graft, self.time_in_update, self.time_in_descendants,
            self.time_in_centroid, self.time_in_lca, self.number_of_rotates, self.number_of_grafts,
            self.number_of_rotates_considered, self.number_of_grafts_considered, self.this_stats_string()
        )
        wandb.log({'time/search_time': self.time_in_search,
                   'time/rotate_time': self.time_in_rotate,
                   'time/graft_time': self.time_in_graft,
                   'time/update_time': self.time_in_update,
                   'time/graft_score_only_time': self.time_in_graft_score_only,
                   'time/graft_search_time': self.time_in_graft_search,
                   'time/graft_get_comparison_scores': self.time_in_graft_get_scores,
                   'time/find_dependent_update_nodes_time': self.time_in_update_walk,
                   'time/mark_for_lazy_update_time': self.time_in_update_from_children,
                   'time/time_getting_descendants': self.time_in_descendants,
                   'time/lca_time': self.time_in_lca,
                   'time/centroid_time': self.time_in_centroid,
                   'rotate/total_performed': self.number_of_rotates,
                   'rotate/total_considered': self.number_of_rotates_considered,
                   'rotate/inst_performed': self.this_number_of_rotates,
                   'rotate/inst_considered': self.this_number_of_rotates_considered,
                   'graft/total_performed': self.number_of_grafts,
                   'graft/total_considered': self.number_of_grafts_considered,
                   'graft/total_allowable': self.number_of_graft_allowable,
                   'graft/percent_allowable': float(self.number_of_graft_allowable) / max(1.0, float(
                       self.number_of_grafts_considered)),
                   'graft/inst_performed': self.this_number_of_grafts,
                   'graft/inst_allowable': self.inst_number_of_graft_allowable[self.point_counter - 2],
                   'graft/inst_allowable_percent': self.graft_percentages[self.point_counter - 2],
                   'graft/inst_considered': self.this_number_of_grafts_considered}, step=self.point_counter)
        if self.point_counter > 0:
            wandb.log({"graft_allowable_percent": wandb.Histogram(self.graft_percentages[0:self.point_counter])},
                      step=self.point_counter)
            wandb.log(
                {"graft_allowable_count": wandb.Histogram(self.inst_number_of_graft_allowable[0:self.point_counter])},
                step=self.point_counter)
            wandb.run.summary.update(
                {"graft_allowable_percent": wandb.Histogram(self.graft_percentages[0:self.point_counter])})
            wandb.run.summary.update(
                {"graft_allowable_count": wandb.Histogram(self.inst_number_of_graft_allowable[0:self.point_counter])})
        self.clear_stats()
        return r

    def this_stats_string(self):
        r = 'last_stats\tnum_rotate=%s\tnum_graft=%s\tnum_rotate_considered=%s\tnum_graft_considered=%s\n' % (
            self.this_number_of_rotates, self.this_number_of_grafts,
            self.this_number_of_rotates_considered, self.this_number_of_grafts_considered
        )
        return r

    def lca_and_ancestors(self, i, j):
        s = time.time()
        if i == j:
            return (i, [], [])
        if self.parent[i] == -1:
            logger.debug('lca_and_ancestors i = root %s', i)
            return (i, [], [])
        curr_node = j
        thisAnclist = self.get_ancs_with_self(i)
        thisAnc = dict([(nid, idx) for idx, nid in enumerate(thisAnclist)])
        other2lca = []
        while curr_node not in thisAnc:
            other2lca.append(curr_node)
            curr_node = self.get_parent(curr_node)
        this2lca = thisAnclist[:thisAnc[curr_node]]
        self.time_in_lca += time.time() - s
        return (curr_node, [x for x in this2lca if self.num_descendants[x] < self.graft_cap],
                [x for x in other2lca if self.num_descendants[x] < self.graft_cap])

    def find_rotate(self, gnode, sib):
        s = time.time()
        logger.debug('[rotate] find_rotate(%s, %s)', gnode, sib)
        curr = sib
        score = self.e_score(gnode, sib)
        curr_parent = self.get_parent(curr)
        curr_parent_score = -np.Inf if curr_parent == -1 else self.get_score(curr_parent)
        while curr_parent != -1 \
                and score < curr_parent_score \
                and self.num_descendants[curr_parent] < self.rotate_cap:
            logger.debug('[rotate] curr %s curr_parent %s gnode %s score %s curr_parent_score %s', curr, curr_parent,
                         gnode, score, curr_parent_score)
            curr = curr_parent
            curr_parent = self.get_parent(curr)
            curr_parent_score = -np.Inf if curr_parent == -1 else self.get_score(curr_parent)
            score = self.e_score(gnode, sib)
            self.number_of_rotates += 1
            self.this_number_of_rotates += 1
            self.number_of_rotates_considered += 1
            self.this_number_of_rotates_considered += 1
        logger.debug('[rotate] find_rotate(%s, %s) = %s', gnode, sib, curr)
        self.time_in_rotate += time.time() - s
        self.this_time_in_rotate += time.time() - s
        return curr

    def get_centroid_batch(self, i):
        return self.centroids[i]

    def get_centroid(self, i):
        return np.expand_dims(self.centroids[i], 0)

    def compute_centroid_l2_norm(self, i):
        # if the node is new we don't need to zero
        self.centroids[i] *= 0
        self.centroids[i] += self.sums[i]
        self.centroids[i] /= self.num_descendants[i]
        centroid_norm = np.linalg.norm(self.centroids[i])
        if centroid_norm > 0.0:
            self.centroids[i] /= centroid_norm

    def compute_centroid_no_norm(self, i):
        # if the node is new we don't need to zero
        self.centroids[i] *= 0
        self.centroids[i] += self.sums[i]
        self.centroids[i] /= self.num_descendants[i]

    def compute_centroid_l_inf_norm(self, i):
        # if the node is new we don't need to zero
        self.centroids[i] *= 0
        self.centroids[i] += self.sums[i]
        self.centroids[i] /= self.num_descendants[i]
        self.centroids[i] /= np.linalg.norm(self.centroids[i], np.inf)

    def get_sum(self, i):
        return np.expand_dims(self.sums[i], 0)

    def set_sum(self, i, v):
        self.sums[i] = v

    def update_desc(self, i):
        s = time.time()
        needs_update = []
        to_check = [i]
        while to_check:
            curr = to_check.pop(0)
            if self.needs_update_desc[curr]:
                needs_update.append(curr)
                for c in self.get_children(curr):
                    to_check.append(c)
        self.time_in_update_walk += time.time() - s
        for j in range(len(needs_update) - 1, -1, -1):
            self.single_update_desc(needs_update[j])
        self.time_in_update += time.time() - s

    def single_update_desc(self, i):
        s = time.time()
        logger.debug('updating node %s', i)
        assert self.needs_update_desc[i]
        kids = self.get_children(i)
        self.descendants[i].clear()
        self.descendants[i].extend(self.descendants[kids[0]])
        self.descendants[i].extend(self.descendants[kids[1]])
        self.time_in_descendants += time.time() - s
        self.new_node[i] = False
        self.needs_update_desc[i] = False

    def get_descendants(self, i):
        if self.needs_update_desc[i]:
            logger.debug('Updating because of get_descendants!')
            self.update_desc(i)
        return self.descendants[i]

    def graft(self, gnode):
        s = time.time()
        logger.debug('[graft] graft(%s)', gnode)
        curr = gnode

        # Find offlimits
        offlimits1 = self.get_descendants(curr)
        if self.get_parent(curr) != -1 and len(self.get_children(self.get_sibling(curr))) == 0:
            offlimits2 = [self.get_sibling(curr)]
        else:
            offlimits2 = []
        logger.debug('[graft] offlimits1 %s offlimits2 %s', len(offlimits1), len(offlimits2))

        # Find updates
        self.update(curr)
        curr_v = self.get_centroid(curr)

        # Do search
        search_st = time.time()
        _, nns = self.cknn(curr_v, self.k, offlimits1, offlimits2)
        self.time_in_graft_search += time.time() - search_st
        self.this_time_in_graft_search += time.time() - search_st
        if len(nns) == 0:
            logger.debug('No nearest neighbors after nns....')
            return

        oneNN = nns[0]
        logger.debug('Nearest neighbor is %s', oneNN)
        lca, this2anc, other2anc = self.lca_and_ancestors(gnode, oneNN)
        logger.debug('lca %s len(this2anc) %s len(other2anc) %s', lca, len(this2anc), len(other2anc))
        if this2anc and other2anc:
            # find all pairwise distances
            # this_vecs = self.get_centroid(this2anc)
            # anc_vecs = self.get_centroid(other2anc)

            # M by N
            M = len(this2anc)
            N = len(other2anc)
            score_if_grafted = self.e_score_batch(this2anc, other2anc)
            assert score_if_grafted.shape[0] == M
            assert score_if_grafted.shape[1] == N
            # 1 by N
            nn_parent_score = np.expand_dims(self.get_score_batch(self.get_parent(other2anc)), 0)
            assert nn_parent_score.shape[0] == 1
            assert nn_parent_score.shape[1] == N
            # M by 1
            curr_parent_score = np.expand_dims(self.get_score_batch(self.get_parent(this2anc)), 1)
            assert curr_parent_score.shape[0] == M
            assert curr_parent_score.shape[1] == 1

            not_i_like_you = score_if_grafted <= curr_parent_score
            not_you_like_me = score_if_grafted <= nn_parent_score
            assert not_i_like_you.shape[0] == M
            assert not_i_like_you.shape[1] == N
            assert not_you_like_me.shape[0] == M
            assert not_you_like_me.shape[1] == N

            graft_condition = not_i_like_you | not_you_like_me
            # num_meeting_condition = graft_condition.sum()
            # self.number_of_graft_allowable += num_meeting_condition
            # total_candidate_grafts = max(1.0, len(this2anc) * len(other2anc))
            # self.graft_percentages[self.point_counter-1] = 1.0 - (float(num_meeting_condition) / total_candidate_grafts)
            # self.inst_number_of_graft_allowable[self.point_counter - 1] = total_candidate_grafts - float(num_meeting_condition)

            score_if_grafted[graft_condition] = 0
            argmax = np.argmax(score_if_grafted)
            argmax_row = int(argmax / score_if_grafted.shape[1])
            argmax_col = argmax % score_if_grafted.shape[1]
            best_1 = this2anc[argmax_row]
            best_2 = other2anc[argmax_col]
            if not not_i_like_you[argmax_row, argmax_col] and not not_you_like_me[argmax_row, argmax_col]:
                self.number_of_grafts += 1
                self.this_number_of_grafts += 1
                bestPair2gp = self.get_parent(self.get_parent(best_2))
                parent = self.node_from_nodes(best_1, best_2)
                self.make_sibling(best_1, best_2, parent)
                logger.debug('[graft] node %s grafts node %s, scores %s > max(%s, %s)' % (best_1, best_2,
                                                                                          score_if_grafted[argmax_row,
                                                                                                           argmax_col],
                                                                                          curr_parent_score[
                                                                                              argmax_row, 0],
                                                                                          nn_parent_score[
                                                                                              0, argmax_col]))
                for start in [bestPair2gp, self.get_parent(curr)]:
                    curr_update = start
                    while curr_update != -1:
                        self.updated_from_children(curr_update)
                        curr_update = self.get_parent(curr_update)
            else:
                logger.debug('[graft] There was no case where we wanted to graft.')
        self.number_of_grafts_considered += len(this2anc) * len(other2anc)
        self.time_in_graft += time.time() - s

        self.this_number_of_grafts_considered = len(this2anc) * len(other2anc)
        self.this_time_in_graft += time.time() - s

    def updated_from_children(self, i):
        s = time.time()
        self.num_descendants[i] = self.num_descendants[self.children[i][0]] + \
                                  self.num_descendants[self.children[i][1]]
        self.scores[i] = -float('inf')
        self.needs_update_model[i] = True
        self.needs_update_desc[i] = True
        self.time_in_update += time.time() - s
        self.time_in_update_from_children += time.time() - s

    def update(self, i):
        s = time.time()
        needs_update = []
        to_check = [i]
        while to_check:
            curr = to_check.pop(0)
            if self.needs_update_model[curr]:
                needs_update.append(curr)
                for c in self.get_children(curr):
                    to_check.append(c)
        self.time_in_update_walk += time.time() - s
        for j in range(len(needs_update) - 1, -1, -1):
            self.single_update(needs_update[j])
        self.time_in_update += time.time() - s

    def single_update(self, i):
        logger.debug('updating node %s', i)
        assert self.needs_update_model[i]
        kids = self.get_children(i)
        s = time.time()
        c1 = self.get_sum(kids[0])
        c2 = self.get_sum(kids[1])
        self.num_descendants[i] = self.num_descendants[kids[0]] + self.num_descendants[kids[1]]
        self.set_sum(i, c1 + c2)
        self.compute_centroid(i)
        self.time_in_centroid += time.time() - s
        self.new_node[i] = False
        self.needs_update_model[i] = False

    def node_from_nodes(self, n1, n2):
        logger.debug('creating new node from nodes %s and %s', n1, n2)
        new_node_id = self.next_node_id
        logger.debug('new node is %s', new_node_id)
        assert self.next_node_id < self.max_nodes
        self.next_node_id += 1
        self.needs_update_model[new_node_id] = True
        self.needs_update_desc[new_node_id] = True
        self.num_descendants[new_node_id] = self.num_descendants[n1] + self.num_descendants[n2]
        return new_node_id

    def add_pt(self, i):
        self.sums[i] += self.points[i]
        self.num_descendants[i] = 1
        if i not in self.deleted_points_set:
            self.point_counter += 1
        self.descendants[i].append(i)
        self.compute_centroid(i)
        self.new_node[i] = False

    def insert(self, i, i_vec=None):
        if i_vec is not None:
            self.points[i] = i_vec
        s = time.time()
        logger.debug('[insert] insert(%s)', i)
        # first point
        if self.point_counter == 0:
            self.add_pt(i)
        else:
            i_vec = np.expand_dims(self.points[i], 0)
            dists, nns = self.cknn(i_vec, self.k, None, None)
            self.add_pt(i)
            sib = self.find_rotate(i, nns[0])
            parent = self.node_from_nodes(sib, i)
            self.make_sibling(sib, i, parent)
            curr_update = parent
            while curr_update != -1:
                self.updated_from_children(curr_update)
                curr_update = self.get_parent(curr_update)
            self.graft(parent)
        t = time.time()
        logger.debug('[insert] finished insert(%s) in %s seconds', i, t - s)
        # check to see if i has previously been deleted pc would have been incremented..:
        if i in self.deleted_points_set:
            self.deleted_points.remove(i)
            self.deleted_points_set.remove(i)

    def csim_dot(self, x, y):
        sims = np.matmul(x, y.transpose(1, 0))
        return sims

    def csim_l2(self, x, y):
        dists = cdist(x, y)
        return 1.0 / (1 + dists)

    def csim_sql2(self, x, y):
        dists = cdist(x, y, 'sqeuclidean')
        return 1.0 / (1 + dists)

    def csim_ward(self, x, y, ndx, ndy):
        dists = (ndx * ndy)(ndx + ndy) * cdist(x, y, 'sqeuclidean')
        return 1.0 / (1 + dists)

    def cknn(self, i_vec, k, offlimits1, offlimits2, pc=None):
        k = min(self.point_counter, k)
        s = time.time()
        # TODO: Allow offlimits to 2D
        # import pdb; pdb.set_trace()
        if pc is None:
            pc = self.point_counter
        point_vecs = self.points[0:pc]
        # sims = np.matmul(i_vec, point_vecs.transpose(1, 0))
        sims = self.csim(i_vec, point_vecs)
        sims[:, self.deleted_points] = -np.inf
        if offlimits1 is not None:
            sims[:, offlimits1] = -np.inf
        if offlimits2 is not None:
            sims[:, offlimits2] = -np.inf
        indices = np.argmax(sims, axis=1)
        distances = sims[0, indices]
        indices = indices[distances != -np.Inf]
        distances = distances[distances != -np.Inf]
        self.time_in_search += time.time() - s
        return distances, indices

    def e_score_batch(self, i, j):
        if np.any(self.needs_update_model[i]):
            for ii in i:
                if self.needs_update_model[ii]:
                    self.update(ii)
        if np.any(self.needs_update_model[j]):
            for jj in j:
                if self.needs_update_model[jj]:
                    self.update(jj)

        i_vec = self.get_centroid_batch(i)
        j_vec = self.get_centroid_batch(j)

        s1 = time.time()
        # sims = np.matmul(i_vec, j_vec.transpose(1, 0))
        sims = self.csim(i_vec, j_vec)
        self.time_in_graft_score_only += time.time() - s1
        return sims

    def e_score(self, i, j):
        if self.needs_update_model[i]:
            self.update(i)

        if self.needs_update_model[j]:
            self.update(j)

        i_vec = self.get_centroid(i)
        j_vec = self.get_centroid(j)
        # sims = np.matmul(i_vec, j_vec.transpose(1, 0))
        sims = self.csim(i_vec, j_vec)

        return sims[0][0]

    def get_score_batch(self, i):
        # todo vectorize
        s = time.time()
        if not np.all(np.isfinite(self.scores[i])):
            for ii in i:
                if not np.isfinite(self.scores[ii]):
                    kids = self.get_children(ii)
                    res = self.e_score(kids[0], kids[1])
                    self.scores[ii] = res
        self.time_in_graft_get_scores += time.time() - s
        self.this_time_in_graft_get_scores += time.time() - s
        return self.scores[i]

    def get_score(self, i):
        """Get the linkage score at node with index i."""
        if not np.all(np.isfinite(self.scores[i])):
            kids = self.get_children(i)
            res = self.e_score(kids[0], kids[1])
            self.scores[i] = res
        return self.scores[i]

    def make_sibling(self, node, new_sib, parent):
        sib_parent = self.get_parent(new_sib)
        if sib_parent != -1:
            sib_gp = self.get_parent(sib_parent)
            old_sib = self.get_sibling(new_sib)
            self.set_parent(old_sib, sib_gp)
            if sib_gp != -1:
                self.remove_child(sib_gp, sib_parent)
                self.add_child(sib_gp, old_sib)
            self.clear_children(sib_parent)
            self.parent[sib_parent] = -2  # Code for deletion
        else:
            assert self.is_leaf(new_sib)

        self.set_parent(parent, self.get_parent(node))
        parentparent = self.get_parent(parent)
        if parentparent != -1:
            self.remove_child(parentparent, node)
            self.add_child(parentparent, parent)
        self.add_child(parent, node)
        self.add_child(parent, new_sib)
        self.set_parent(node, parent)
        self.set_parent(new_sib, parent)

    def is_leaf(self, i):
        return len(self.get_children(i)) == 0

    def get_parent(self, i):
        return self.parent[i]

    def set_parent(self, i, p):
        self.parent[i] = p

    def get_sibling(self, i):
        p = self.get_parent(i)
        return [x for x in self.get_children(p) if x != i][0]

    def get_ancs_with_self(self, i):
        needs_anc = [i]
        # walk up until we find someone who has known ancestors
        curr = self.get_parent(i)
        while curr != -1:
            needs_anc.append(curr)
            curr = self.get_parent(curr)
        return needs_anc

    def get_ancs(self, i):
        # TODO speed up w/ cache.
        needs_anc = []
        # walk up until we find someone who has known ancestors
        curr = self.get_parent(i)
        while curr != -1:
            needs_anc.append(curr)
            curr = self.get_parent(curr)
        return needs_anc

    def get_children(self, i):
        return self.children[i]

    def add_child(self, p, c):
        self.children[p].append(c)

    def remove_child(self, p, c):
        self.children[p].remove(c)

    def clear_children(self, i):
        self.children[i].clear()

    def write_tree(self, filename, lbls):
        with open(filename, 'w') as fin:
            for i in range(self.points.shape[0]):
                fin.write('%s\t%s\t%s\n' % (i, self.get_parent(i), lbls[i]))
            for j in range(self.points.shape[0], self.next_node_id):
                if self.parent[j] != -2:
                    fin.write('%s\t%s\tNone\n' % (j, self.parent[j]))
            r = self.root()
            fin.write('-1\tNone\tNone\n' % r)

    def root(self):
        r = 0
        while self.get_parent(r) != -1:
            r = self.get_parent(r)
        return r

    def flat_clustering(self, threshold):
        frontier = [self.root()]
        clusters = []
        while frontier:
            n = frontier.pop(0)
            if len(self.children[n]) != 0 and self.get_score(n) < threshold:
                frontier.extend(self.children[n])
            else:
                clusters.append(n)
        assignments = -1 * np.ones(self.points.shape[0])
        for c_idx, c in enumerate(clusters):
            for d in self.get_descendants(c):
                assignments[d] = c_idx
        return assignments

    def delete_point(self, i):
        sib = self.get_sibling(i)
        # first remove the parent:
        p = self.get_parent(i)
        gp = self.get_parent(p)
        if gp != -1:
            self.add_child(gp, sib)
            self.remove_child(gp, p)
        self.parent[sib] = gp
        self.parent[p] = -2
        self.parent[i] = -1
        self.children[p] = []
        # remove the node i
        self.deleted_points.append(i)
        self.deleted_points_set.add(i)
        curr = gp
        while curr != -1:
            self.updated_from_children(curr)
            curr = self.get_parent(curr)
        self.sums[i] *= 0
        self.centroids[i] *= 0
        self.num_descendants[i] = 1
        self.descendants[i] = []
