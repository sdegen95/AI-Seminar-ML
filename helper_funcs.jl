using Oscar
using Graphs
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
    d = Int(Oscar.degree(h_star))
    coeffs = []
    for i in range(0,d)
        c = Oscar.coeff(h_star,Int(i))
        push!(coeffs,c)
    end

    return coeffs

end

function cosmological_polytope_h_star_polynomial_coeff(adj)
    G = Oscar.graph_from_adjacency_matrix(Undirected,adj)
    edgs = collect(Oscar.edges(G))
    num_vert = Int(Oscar.nv(G))
    num_egdes = Int(Oscar.ne(G))

    # Create a dictionary of the edges
    EdgeDict = OrderedDict([])
    for (id,edge) in enumerate(edgs)
        merge!(EdgeDict,Dict(id+num_vert=>edge))
    end

    # Create the vertex-vectors
    vecs = []
    for pair in EdgeDict
        zero_vec1 = zeros(Int,num_vert + num_egdes)
        zero_vec2 = zeros(Int,num_vert + num_egdes)
        zero_vec3 = zeros(Int,num_vert + num_egdes)
        s = Int(Oscar.src(pair[2]))
        d = Int(Oscar.dst(pair[2]))
        e_id = Int(pair[1])
        zero_vec1[e_id]=1
        zero_vec1[s]=1
        zero_vec1[d]=-1
        zero_vec2[e_id]=1
        zero_vec2[s]=-1
        zero_vec2[d]=1
        zero_vec3[e_id]=-1
        zero_vec3[s]=1
        zero_vec3[d]=1
        push!(vecs,zero_vec1)
        push!(vecs,zero_vec2)
        push!(vecs,zero_vec3)
    end

    # for i in range(1,num_vert)
    #     zero_vec = zeros(Int,num_vert + num_egdes)
    #     zero_vec[i] = 1
    #     push!(vecs,zero_vec)
    # end

    # Compute the convex-hull of the vecs
    conv_h = Oscar.convex_hull([x for x in vecs])

    # Compute the h_star polynomial and return it's coefficients
    h_star = Oscar.h_star_polynomial(conv_h)
    d = Int(Oscar.degree(h_star))
    coeffs = []
    for i in range(0,d)
        c = Oscar.coeff(h_star,Int(i))
        push!(coeffs,c)
    end

    return coeffs

end

function check_connected(adj)
    G = Graphs.SimpleGraph(adj)
    conn = Graphs.is_connected(G)

    return conn
    
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
# @time cosmological_polytope_h_star_polynomial_coeff(adj)
# check_connected(adj)


