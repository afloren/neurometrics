
RAS = (('R','L'),('A','P'),('S','I'))

def axcodes2ornt(labels1,labels2):
    ornt = []
    for l1 in labels1:
        found = False
        for i,l2 in enumerate(labels2):
            if set(l1) == set(l2):
                flip = 1 if (l1[0] == l2[0]) else -1
                ornt.append((i,flip))
                found = True
                break
        if not found:
            raise Error('no matching label sets for %s'.format(l1))
    return ornt
                
        
