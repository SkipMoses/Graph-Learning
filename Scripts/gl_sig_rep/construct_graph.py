import NetworkX as nx
"""
function [G, XCoords, YCoords] = construct_graph(N,opt,varargin1,varargin2)
% Graph construction
"""
def construct_graph(N,opt,varagin1):
    
    """
    %% construct the graph
    switch opt
        case 'er', % Erdos-Renyi random graph
            p = varargin1;
            G = erdos_reyni(N,p);

        case 'pa', % scale-free graph with preferential attachment
            m = varargin1;
            G = preferential_attachment_graph(N,m);
    end
    """
    if opt == 'er':
        p = varargin1
        nx.erdos_reyni_graph(N,p)
    else:
        m = varagin1
        G = nx.barabasi_albert_graph(n,m)
    return(G)

