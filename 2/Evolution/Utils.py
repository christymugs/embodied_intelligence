# Evolution/Utils.py
def GetAllLimbs(Root):
    Limbs = [Root]
    for Conn in Root.Connections:
        Limbs.extend(GetAllLimbs(Conn.ChildLimb))
    return Limbs