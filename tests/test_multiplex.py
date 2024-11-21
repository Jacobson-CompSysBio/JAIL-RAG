from utils.multiplex import Multiplex
import pytest
import networkx as nx
import numpy as np
from scipy import sparse

class TestMultiplex:
    @pytest.fixture
    def bad_filenae_flist(self):
        return 'badfilename.flist'
    
    @pytest.fixture
    def bad_format_flist(self):
        return 'tests/data/bad_format.flist'
    
    @pytest.fixture
    def bad_graph_path_flist(self):
        return 'tests/data/bad_graph_path.flist'
    
    @pytest.fixture
    def monoplex_flist(self):
        return 'tests/data/monoplex.flist'
    
    @pytest.fixture
    def multiplex_flist(self):
        return'tests/data/multiplex.flist'
    
    @pytest.fixture
    def expected_src_0(self):
        return [0,0,0,0,1,1,1,2,2,3,3,4,6]
    
    @pytest.fixture
    def expected_src_1(self):
        return [0,1,2,2,2,2,2,3,3,4,5,6]
    
    @pytest.fixture
    def expected_src_2(self):
        return [0,0,1,2,2,2,3,3,3,4,4,5,6,7,7]
    
    @pytest.fixture
    def expected_src_3(self):
        return [0,0,0,0,1,2,2,3,3,3,6,6,7]
    
    @pytest.fixture
    def expected_src_4(self):
        return [0,0,1,2,2,3,5,5,6,6,7,8]
    
    @pytest.fixture
    def expected_src_all(self, expected_src_0, expected_src_1, expected_src_2, expected_src_3, expected_src_4):
        return [expected_src_0, expected_src_1, expected_src_2, expected_src_3, expected_src_4]

    @pytest.fixture
    def expected_src_comb(self):
        return [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,8]
    
    @pytest.fixture
    def expected_dst_0(self):
        return [2,3,4,5,3,4,9,6,8,4,7,7,8]
    
    @pytest.fixture
    def expected_dst_1(self):
        return [2,3,4,5,6,7,8,4,5,5,9,7]
    
    @pytest.fixture
    def expected_dst_2(self):
        return [2,7,7,3,4,6,4,5,6,7,8,7,7,8,9]
    
    @pytest.fixture
    def expected_dst_3(self):
        return [2,4,5,7,9,3,9,4,6,8,7,8,8]
    
    @pytest.fixture
    def expected_dst_4(self):
        return [2,9,3,4,7,4,6,7,8,9,8,9]
    
    @pytest.fixture
    def expected_dist_comb(self):
        return[2,2,2,2,2,3,4,4,5,5,7,7,9,3,3,3,4,7,9,9,3,3,4,4,4,5,6,6,6,7,7,8,8,9,4,4,4,4,4,5,5,6,6,7,8,5,7,7,8,6,7,7,9,7,7,7,8,8,8,9,8,8,8,9,9]
    
    @pytest.fixture
    def expected_dst_all(self, expected_dst_0, expected_dst_1, expected_dst_2, expected_dst_3, expected_dst_4):
        return [expected_dst_0, expected_dst_1, expected_dst_2, expected_dst_3, expected_dst_4]
    
    @pytest.fixture
    def adj_mono0(self, expected_src_0, expected_dst_0):
        adj = sparse.lil_matrix((10,10))

        for i, src in enumerate(expected_src_0):
            dst = expected_dst_0[i]
            adj[src, dst] = 1
            adj[dst, src] = 1

        return adj.tocsr()

    @pytest.fixture
    def adj_mono1(self, expected_src_1, expected_dst_1):
        adj = sparse.lil_matrix((10,10))

        for i, src in enumerate(expected_src_1):
            dst = expected_dst_1[i]
            adj[src, dst] = 1
            adj[dst, src] = 1

        return adj.tocsr()
    
    @pytest.fixture
    def adj_mono2(self, expected_src_2, expected_dst_2):
        adj = sparse.lil_matrix((10,10))

        for i, src in enumerate(expected_src_2):
            dst = expected_dst_2[i]
            adj[src, dst] = 1
            adj[dst, src] = 1

        return adj.tocsr()
    
    @pytest.fixture
    def adj_mono3(self, expected_src_3, expected_dst_3):
        adj = sparse.lil_matrix((10,10))

        for i, src in enumerate(expected_src_3):
            dst = expected_dst_3[i]
            adj[src, dst] = 1
            adj[dst, src] = 1

        return adj.tocsr()
    
    @pytest.fixture
    def adj_mono4(self, expected_src_4, expected_dst_4):
        adj = sparse.lil_matrix((10,10))

        for i, src in enumerate(expected_src_4):
            dst = expected_dst_4[i]
            adj[src, dst] = 1
            adj[dst, src] = 1

        return adj.tocsr()
    
    @pytest.fixture
    def adj_multi(self, adj_mono0, adj_mono1, adj_mono2, adj_mono3, adj_mono4):
        def _adj_multi(delta):
            L = 5
            N = 10
            intra_layer_scale = 1 - delta
            inter_layer_scale = delta / (L-1)

            eye = inter_layer_scale * sparse.identity(N, format='csr')

            diag = [adj_mono0, adj_mono1, adj_mono2, adj_mono3, adj_mono4]

            blocks = [[intra_layer_scale * diag[l] if l == i else eye for l in range(L)] for i in range(L)]

            adj = sparse.block_array(blocks, format='csr')

            return {'delta': delta, 'adj': adj}

        return _adj_multi

    # Test multiplex object when no flist is provided
    def test_no_flist(self):
        mp = Multiplex()
        adj = mp.adj_matrix()

        assert mp.layers == []
        assert mp._nodes == []
        assert len(mp) == 0
        assert adj.size == 0
        assert mp.num_nodes == 0
        assert mp.nodes == []
        assert mp.src() == []
        assert mp.dst() == []

        with pytest.raises(IndexError):
            assert mp[0]

    # Test that the correct error is raised when a bad flist path is provided
    def test_bad_flist_path(self, bad_filenae_flist):
        with pytest.raises(FileNotFoundError):
            mp = Multiplex(bad_filenae_flist)

    # Test that the correct error is raised when a flist with bad format is provided
    def test_flist_bad_format(self, bad_format_flist):
        with pytest.raises(ValueError):
            mp = Multiplex(bad_format_flist)

    # Test that the correct error is raised when a graph path does not exist
    def test_bad_graph_path(self, bad_graph_path_flist):
        with pytest.raises(FileNotFoundError):
            mp = Multiplex(bad_graph_path_flist)

    # Test that the correct error is raised when delta < 0
    def test_adj_matrix_low_delta(self):
        mp = Multiplex()
        with pytest.raises(ValueError):
            mp.adj_matrix(delta=-1)

    # Test that the correct error is raised when delta > 1
    def test_adj_matrix_high_delta(self):
        mp = Multiplex()
        with pytest.raises(ValueError):
            mp.adj_matrix(delta=2)

    # Test that correct error is raised when layer_idx < -1 for monoplex
    def test_src_low_layer_idx_monoplex(self, monoplex_flist):
        mp = Multiplex(monoplex_flist)
        with pytest.raises(ValueError):
            mp.src(-2)

    # Test that correct error is raised when layer_idx < -1 for multiplex
    def test_src_low_layer_idx_multiplex(self, multiplex_flist):
        mp = Multiplex(multiplex_flist)
        with pytest.raises(ValueError):
            mp.src(-2)

    # Test that correct error is raised when layer_idx in not integer
    def test_src_layer_not_int(self, monoplex_flist):
        mp = Multiplex(monoplex_flist)
        with pytest.raises(TypeError):
            mp.src(0.5)

    # Test that correct error is raised when layer_idx > len(self) for mono_plex
    def test_src_high_layer_idx_monoplex(self, monoplex_flist):
        mp = Multiplex(monoplex_flist)
        with pytest.raises(ValueError):
            mp.src(1)

    # Test that correct error is raised when layer_idx > len(self) for multiplex
    def test_src_high_layer_idx_multiplex(self, multiplex_flist):
        mp = Multiplex(multiplex_flist)
        with pytest.raises(ValueError):
            mp.src(10)

    # Test that correct list is returned when layer_idx is specified for monoplex
    def test_src_specific_layer_monoplex(self, monoplex_flist, expected_src_0):
        mp = Multiplex(monoplex_flist)
        src = mp.src(0)
        assert src == expected_src_0

    # Test that correct list is returned when layer_idx == -1 for monoplex
    def test_src_all_layers_monoplex(self, monoplex_flist, expected_src_0):
        mp = Multiplex(monoplex_flist)
        src = mp.src()
        assert src == expected_src_0

    # Test that correct list is returned when layer_idx is specified for multiplex
    def test_src_specific_layer_multiplex(self, multiplex_flist, expected_src_all):
        mp = Multiplex(multiplex_flist)
        
        assert mp.src(0) == expected_src_all[0]
        assert mp.src(1) == expected_src_all[1]
        assert mp.src(2) == expected_src_all[2]
        assert mp.src(3) == expected_src_all[3]
        assert mp.src(4) == expected_src_all[4]

    # Test that correct list is returned when layer_idx == -1 for multiplex
    def test_src_all_layers_multiplex(self, multiplex_flist, expected_src_comb):
        mp = Multiplex(multiplex_flist)
        
        assert mp.src() == expected_src_comb
        assert mp.src(-1) == expected_src_comb

    # Test that correct list is returned when layer_idx is specified for monoplex
    def test_dst_specific_layer_monoplex(self, monoplex_flist, expected_dst_0):
        mp = Multiplex(monoplex_flist)
        dst = mp.dst(0)
        assert dst == expected_dst_0

    # Test that correct list is returned when layer_idx == -1 for monoplex
    def test_dst_all_layers_monoplex(self, monoplex_flist, expected_dst_0):
        mp = Multiplex(monoplex_flist)
        dst = mp.dst()
        assert dst == expected_dst_0

    # Test that correct list is returned when layer_idx is specified for multiplex
    def test_dst_specific_layer_multiplex(self, multiplex_flist, expected_dst_all):
        mp = Multiplex(multiplex_flist)

        assert mp.dst(0) == expected_dst_all[0]
        assert mp.dst(1) == expected_dst_all[1]
        assert mp.dst(2) == expected_dst_all[2]
        assert mp.dst(3) == expected_dst_all[3]
        assert mp.dst(4) == expected_dst_all[4]

    # Test that correct list is returned when layer_idx == -1 for monoplex
    def test_dst_all_layers_mulyplex(self, multiplex_flist, expected_dist_comb):
        mp = Multiplex(multiplex_flist)

        assert mp.dst() == expected_dist_comb
        assert mp.dst(-1) == expected_dist_comb

    def test_monoplex_flist(self, monoplex_flist, adj_mono0):
        mp = Multiplex(monoplex_flist)

        assert len(mp) == 1
        assert isinstance(mp[0]['graph'], nx.Graph)
        assert mp[0]['layer_name'] == 'type_1'

        assert mp.nodes == ['G1','G10','G2','G3','G4','G5','G6','G7','G8','G9']
        assert mp.nodes == mp._nodes
        assert mp.num_nodes == 10

        adj = mp.adj_matrix()
        assert (np.all(adj.indptr == adj_mono0.indptr)
                and np.all(adj.indices == adj_mono0.indices)
                and np.allclose(adj.data, adj_mono0.data))
        
    def test_multiplex_flist(self, multiplex_flist, adj_multi):
        mp = Multiplex(multiplex_flist)

        assert len(mp) == 5
        assert mp[0]['layer_name'] == 'type_1'
        assert mp[1]['layer_name'] == 'type_2'
        assert mp[2]['layer_name'] == 'type_3'
        assert mp[3]['layer_name'] == 'type_4'
        assert mp[4]['layer_name'] == 'type_5'

        assert mp.nodes == ['G1','G10','G2','G3','G4','G5','G6','G7','G8','G9']
        assert mp.nodes == mp._nodes
        assert mp.num_nodes == 10

        delta = 0.0
        adj = mp.adj_matrix(delta)
        expected_adj = adj_multi(delta)['adj']
        assert (np.all(adj.indptr == expected_adj.indptr)
                and np.all(adj.indices == expected_adj.indices)
                and np.allclose(adj.data, expected_adj.data))
        
        delta = 0.2
        adj = mp.adj_matrix(delta)
        expected_adj = adj_multi(delta)['adj']
        assert (np.all(adj.indptr == expected_adj.indptr)
                and np.all(adj.indices == expected_adj.indices)
                and np.allclose(adj.data, expected_adj.data))
        
        delta = 0.5
        adj = mp.adj_matrix(delta)
        expected_adj = adj_multi(delta)['adj']
        assert (np.all(adj.indptr == expected_adj.indptr)
                and np.all(adj.indices == expected_adj.indices)
                and np.allclose(adj.data, expected_adj.data))
        
        delta = 0.7
        adj = mp.adj_matrix(delta)
        expected_adj = adj_multi(delta)['adj']
        assert (np.all(adj.indptr == expected_adj.indptr)
                and np.all(adj.indices == expected_adj.indices)
                and np.allclose(adj.data, expected_adj.data))

        delta = 1.0
        adj = mp.adj_matrix(delta)
        expected_adj = adj_multi(delta)['adj']
        assert (np.all(adj.indptr == expected_adj.indptr)
                and np.all(adj.indices == expected_adj.indices)
                and np.allclose(adj.data, expected_adj.data))

    # def test_add_layer(self):
    #     pass
