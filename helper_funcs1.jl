using Oscar
using DataStructures

function symm_edge_polytope_h_star_polynomial_coeff(adj)
    G = Oscar.graph_from_adjacency_matrix(Undirected,adj)
    edgs = collect(Oscar.edges(G))
    num_vert = Int(Oscar.nv(G))
    vecs = []
    
    for edge in edgs
        zero_vec1 = zeros(Int,num_vert)
        zero_vec2 = zeros(Int,num_vert)
        s = Int(Oscar.src(edge))
        d = Int(Oscar.dst(edge))
        zero_vec1[s]=1
        zero_vec1[d]=-1
        zero_vec2[s]=-1
        zero_vec2[d]=1
        push!(vecs,zero_vec1)
        push!(vecs,zero_vec2)
    end

    conv_h = Oscar.convex_hull([x for x in vecs])

    h_star = Oscar.h_star_polynomial(conv_h)
    d = Int(degree(h_star))
    coeffs = []
    for i in range(0,d)
        c = Oscar.coeff(h_star,Int(i))
        push!(coeffs,c)
    end

    return coeffs

end

# adj = [0 0 0 0 0 0 0 0 0 1;
#         0 0 1 0 0 0 0 0 0 0;
#         0 1 0 0 0 0 1 0 0 0;
#         0 0 0 0 0 1 1 0 1 0;
#         0 0 0 0 0 1 0 1 0 0;
#         0 0 0 1 1 0 1 0 0 0;
#         0 0 1 1 0 1 0 0 0 1;
#         0 0 0 0 1 0 0 0 0 0;
#         0 0 0 1 0 0 0 0 0 0;
#         1 0 0 0 0 0 1 0 0 0]
# @time symm_edge_polytope_h_star_polynomial_coeff(adj)


