def update_Z(moead, P):
    Z = []
    a = P.min(axis=0)

    Z1, Z2 = a[moead.gene_num + 2], a[moead.gene_num + 3]
    moead.Z = [Z1, Z2]
    return [Z1, Z2]
