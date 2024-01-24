import networkx as nx

def mean_dijkstra_subset(graph, P):
    '''
    Computa a média das menores distâncias entre pares de pontos
    '''

    # Calcular as distâncias mais curtas entre os vértices do subconjunto
    distances = []

    for src in P:
        for target in P:
            if src != target:
                try:
                    distance = nx.shortest_path_length(graph, source=src, target=target)
                    distances.append(distance)
                except:
                    continue

    return sum(distances) / len(distances)
    
def rewiring(graph, L, P, ro):
    '''
    Para o grafo G_l, computa os caminhos mínimos entre pares positivos, selecione o maior deles (k). Crie uma matriz mi_k
    onde mi_i,j representem a maior quantidade de caminhos de tamanho pelo menos k entre os nós i e j pertencentes ao conjunto
    P, ligo os mi_i,j maiores valores gerando o grafo G_l+1

    G = G_0
    '''

    return_graphs = []

    for l in range(2, L+1):
        # print(f'Quantidade de vértices no início da iteração {l}: {len(graph.edges)}')
        # print(f'l = {l}')
        # print('calculando o valor de mi inicial')
        mi = mean_dijkstra_subset(graph, P)
        mi_matrix = dict()
        # print(f'mi = {mi}')
        for u in P:
            # print(f' \033[K computando os valores dos vizinhos de {u}')
            l_hop = nx.descendants_at_distance(graph, u, l)
            # print(l_hop)
            for v in l_hop:
                # print(f'\033[K verificando o valor de mi_u_v para os vértices {u} e {v}', end = '\r')
                graph.add_edge(u,v)
                mi_u_v = mean_dijkstra_subset(graph, P)
                mi_matrix[(u,v)] = mi - mi_u_v
                graph.remove_edge(u,v)
        
        mi_matrix = dict(sorted(mi_matrix.items(), key=lambda item: item[1], reverse=True))
        mi_matrix = {chave : valor for chave, valor in mi_matrix.items() if valor > 0}
        # print('\n mi_matrix', mi_matrix)

        ro_abs = int(ro * len(mi_matrix))

        if ro_abs == 0:
            ro_abs = 1

        # print('ro_abs', ro_abs)
        # print('add nodes')
        edges_to_add = list(mi_matrix.keys())[:ro_abs]  
        # print([x for x in edges_to_add if not x in list(graph.edges())])
        graph.add_edges_from(edges_to_add)
        # print(f'Quantidade de vértices no final da iteração {l}: {len(graph.edges)}')

        return_graphs.append(graph.copy())

    return return_graphs