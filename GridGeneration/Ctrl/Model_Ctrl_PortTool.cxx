#include <Model_Ctrl.hxx>


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ComputeBaryCenterNormalDirOfFace(const Standard_Integer theFaceIndex, 
						  gp_Pnt& theBaryCenter, 
						  GridLineDir& theLineDir, 
						  Standard_Integer& theRelativeDir) const
{
  gp_Vec V(0,0,0);
  ComputeBaryCenterNormalVectorOfFace(theFaceIndex, theBaryCenter, V);

  gp_Vec xdir(1,0,0);
  gp_Vec ydir(0,1,0);
  gp_Vec zdir(0,0,1);

  theLineDir = DIRX;
  theRelativeDir = 1;

  Standard_Real xcomp = V.X();
  Standard_Real ycomp = V.Y();
  Standard_Real zcomp = V.Z();

  if(V.IsOpposite(xdir,Precision::Angular())){
    theLineDir = DIRX;
    theRelativeDir = -1;
  }else if(V.IsParallel(xdir,Precision::Angular())){
    theLineDir = DIRX;
    theRelativeDir = 1;
  }else if(V.IsOpposite(ydir,Precision::Angular())){
    theLineDir = DIRY;
    theRelativeDir = -1;
  }else if(V.IsParallel(ydir,Precision::Angular())){
    theLineDir = DIRY;
    theRelativeDir = 1;
  }else if(V.IsOpposite(zdir,Precision::Angular())){
    theLineDir = DIRZ;
    theRelativeDir = -1;
  }else if(V.IsParallel(zdir,Precision::Angular())){
    theLineDir = DIRZ;
    theRelativeDir = 1;
  }else{
    cout<<"Model_Ctrl::ComputeBaryCenterNormalDirOfFace--------Error------Face Do not have correct dir"<<endl;
  }
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ComputeBaryCenterNormalVectorOfFace(const Standard_Integer theFaceIndex, 
						     gp_Pnt& theBaryCenter,
						     gp_Vec& V) const
{
    //===========>>> Surface
  if(!HasFaceIndex(theFaceIndex)) return;

  const TopoDS_Face& aFace = GetFaceWithIndex(theFaceIndex);
  if (aFace.IsNull()) {
    cout<<"Model_Ctrl::ComputePortNormalVector--------Error--------The input face is NULL"<<endl;
    return;
  }

  //===========<<< Surface
  Handle(Geom_Surface) aSurf = BRep_Tool::Surface(aFace);

  // Normal direction
  gp_Vec Vec1,Vec2;
  BRepAdaptor_Surface SF (aFace);
  
  Standard_Real U1,U2,V1,V2;
  Standard_Real p_u,p_v;
  
  ShapeAnalysis::GetFaceUVBounds(aFace,U1,U2,V1,V2);
  p_u = U1 + (U2-U1) * 0.5;
  p_v = V1 + (V2-V1) * 0.5;
  
  SF.D1( p_u, p_v, theBaryCenter, Vec1, Vec2);
  V = Vec1.Crossed(Vec2);

  Standard_Real mod = V.Magnitude();
  if (mod < Precision::Confusion()){
    return;
  }
  
  V.Normalize();

  // consider the face orientation
  if (aFace.Orientation() == TopAbs_REVERSED || aFace.Orientation() == TopAbs_INTERNAL) {
    V = - V;
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ComputeNormalVectorOfFaceWithPnt(const Standard_Integer theFaceIndex, 
						  const gp_Pnt& the3DPnt,
						  gp_Pnt& thePntOnFace,
						  gp_Vec& V) const
{
    //===========>>> Surface
  if(!HasFaceIndex(theFaceIndex)){
    cout<<"Model_Ctrl::ComputeNormalVectorOfFaceWithPnt--------Error--------The input face is NULL"<<endl;
    return;
  }

  const TopoDS_Face& aFace = GetFaceWithIndex(theFaceIndex);
  if (aFace.IsNull()) {
    cout<<"Model_Ctrl::ComputeNormalVectorOfFaceWithPnt--------Error--------The input face is NULL"<<endl;
    return;
  }

  // Point parameters on surface
  Handle(Geom_Surface) aSurf = BRep_Tool::Surface(aFace);
  Handle(ShapeAnalysis_Surface) aSurfAna = new ShapeAnalysis_Surface (aSurf);
  gp_Pnt2d pUV = aSurfAna->ValueOfUV(the3DPnt, Precision::Confusion());
  
  // Normal direction
  gp_Vec Vec1,Vec2;
  BRepAdaptor_Surface SF (aFace);
  SF.D1(pUV.X(), pUV.Y(), thePntOnFace, Vec1, Vec2);
  V = Vec1.Crossed(Vec2);
  Standard_Real mod = V.Magnitude();

  if (mod < Precision::Confusion()){
    return;
  }

  V.Normalize();

  // consider the face orientation
  if (aFace.Orientation() == TopAbs_REVERSED || aFace.Orientation() == TopAbs_INTERNAL) {
    V = - V;
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::IsPlaneWithSpecialDir(const Standard_Integer theFaceIndex,
				       const GridLineDir theDir) const
{
  bool result = false;
  gp_Pnt theBaryCenter;
  GridLineDir theLineDir;
  Standard_Integer theRelativeDir;

  if(IsPlaneWithFaceIndex(theFaceIndex)){
    ComputeBaryCenterNormalDirOfFace(theFaceIndex, theBaryCenter, theLineDir, theRelativeDir);
    if(theLineDir==theDir) result = true;
  }
  return result;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::IsPlaneWithFaceIndex(const Standard_Integer theFaceIndex) const
{
  const TopoDS_Face& aFace = GetFaceWithIndex(theFaceIndex);
  BRepAdaptor_Surface surface(aFace,true);
  GeomAbs_SurfaceType SurfaceType =  surface.GetType();

  bool result = true;
  if(SurfaceType != GeomAbs_Plane) result = false;
  return result;
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::IsFaceAsPlaneWithFaceIndex(const Standard_Integer theFaceIndex) const
{
  const TopoDS_Face& aFace = GetFaceWithIndex(theFaceIndex);
  BRepAdaptor_Surface surface(aFace,true);
  GeomAbs_SurfaceType SurfaceType =  surface.GetType();

  bool result = true;
  if(SurfaceType != GeomAbs_Plane) result = false;
  return result;
}
