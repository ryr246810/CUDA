#include <Model_Ctrl.hxx>
#include <BaseDataDefine.hxx>
#include <PortDataFunc.hxx>

/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::SetupPorts()
{
  cout<<"Model_Ctrl::SetupPorts()================>>"<<endl;
  ClearPortsDefine();

  TColStd_DataMapIteratorOfDataMapOfIntegerInteger Iter;
  for(Iter.Initialize(m_FacesWithTypeTool); Iter.More(); Iter.Next()){
    Standard_Integer theFaceIndex = Iter.Key();
    Standard_Integer theFaceMaterialMaterial = Iter.Value();

    Standard_Integer thePortType;

    if( IsOnePortType(theFaceMaterialMaterial, thePortType) ){
      cout<<"Model_Ctrl::SetupPorts---------theFaceIndex of the Port\t=\t"<<theFaceIndex<<endl;
      SetPortToFace(theFaceIndex, thePortType);
    }
  }
  cout<<"Model_Ctrl::SetupPorts()================<<"<<endl;
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ClearPortsDefine()
{
  m_PortsWithTypeTool.Clear();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::IsPort(const TopoDS_Face& theFace) const
{
  Standard_Integer theFaceIndex = GetFaceIndex(theFace);
  return IsPort(theFaceIndex);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::IsPort(const Standard_Integer aFaceIndex) const
{
  bool result = false;

  if( m_PortsWithTypeTool.IsBound(aFaceIndex) ){
    result = true;
  }

  return result;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ComputeNormalVectorOfPort(const Standard_Integer theFaceIndexOfPort,
					   gp_Pnt& theBaryCenter, 
					   gp_Vec& V) const
{
  ComputeBaryCenterNormalVectorOfFace(theFaceIndexOfPort, theBaryCenter, V);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ComputePortDirWithFaceIndexOfPort(const Standard_Integer theFaceIndexOfPort,
						   gp_Pnt& theBaryCenter,
						   GridLineDir& theLineDir,
						   Standard_Integer& theRelativeDir) const
{
  ComputeBaryCenterNormalDirOfFace(theFaceIndexOfPort, theBaryCenter, theLineDir, theRelativeDir);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::CanBeDefinedAsPort(const TopoDS_Face& theFace) const
{
  bool result = false;
  Standard_Integer theFaceIndex = GetFaceIndex(theFace);
  return CanBeDefinedAsPort(theFaceIndex);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::CanBeDefinedAsPort(const Standard_Integer theFaceIndex) const
{
  bool result = false;
  if(HasCorrectGeomTypeForDefiningFaceAsPort(theFaceIndex)){
    if(HasCorrectDirForDefiningFaceAsPort(theFaceIndex)){
      result = true;
    }
  }
  return result;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::IsPortDefinedExactly(const TopoDS_Face& theFace) const
{
  return  CanBeDefinedAsPort(theFace);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::IsPortDefinedExactly(const Standard_Integer theFaceIndex) const
{
  return CanBeDefinedAsPort(theFaceIndex);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::HasCorrectDirForDefiningFaceAsPort(const TopoDS_Face& theFace) const
{
  Standard_Integer theFaceIndex = GetFaceIndex(theFace);
  if(theFaceIndex==0) return false;
  return HasCorrectGeomTypeForDefiningFaceAsPort(theFaceIndex);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::HasCorrectDirForDefiningFaceAsPort(const Standard_Integer theFaceIndexOfPort) const
{
  gp_Vec V(0,0,0);
  gp_Pnt theBaryCenter(0,0,0);
  ComputeNormalVectorOfPort(theFaceIndexOfPort, theBaryCenter, V);

  bool result = false;

  gp_Vec xdir(1,0,0);
  gp_Vec ydir(0,1,0);
  gp_Vec zdir(0,0,1);

  if(V.IsOpposite(xdir,Precision::Angular())||V.IsParallel(xdir,Precision::Angular())|| 
     V.IsOpposite(ydir,Precision::Angular())||V.IsParallel(ydir,Precision::Angular())|| 
     V.IsOpposite(zdir,Precision::Angular())||V.IsParallel(zdir,Precision::Angular())){
    result = true;
  }
  return result;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::HasCorrectGeomTypeForDefiningFaceAsPort(const TopoDS_Face& Face) const
{
  BRepAdaptor_Surface surface(Face,true);
  GeomAbs_SurfaceType SurfaceType =  surface.GetType();

  bool result = true;
  //*
  if(SurfaceType != GeomAbs_Plane){
    cout<<" Model_Ctrl::HasCorrectGeomTypeForDefiningFaceAsPort------------------------------->>error"<<endl;
    // needed to be modified--------------------------------------2014.08.26
    result = false;
  }
  //*/
  return result;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::HasCorrectGeomTypeForDefiningFaceAsPort(const Standard_Integer theFaceIndexOfPort) const
{
  const TopoDS_Face& aFace = GetFaceWithIndex(theFaceIndexOfPort);
  return HasCorrectGeomTypeForDefiningFaceAsPort(aFace);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::SetPortToFace(const TopoDS_Face& theFace, const Standard_Integer thePortType)
{
  if(theFace.IsNull()){
    return;
  }
  Standard_Integer theFaceIndex = GetFaceIndex(theFace);
  SetPortToFace(theFaceIndex, thePortType);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::SetPortToFace(const Standard_Integer theFaceIndexOfPort,
			       const Standard_Integer thePortType)
{
  if( !CanBeDefinedAsPort(theFaceIndexOfPort) ){
    return;
  }

  if( !(m_PortsWithTypeTool.IsBound(theFaceIndexOfPort)) ){
    m_PortsWithTypeTool.Bind(theFaceIndexOfPort, thePortType);
  }else{
    m_PortsWithTypeTool.ChangeFind(theFaceIndexOfPort) = thePortType;
  }
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::UnsetPortOfFace(const TopoDS_Face & theFace)
{
  if(theFace.IsNull()){
    return;
  }
  Standard_Integer theFaceIndex = GetFaceIndex(theFace);
  UnsetPortOfFace(theFaceIndex);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::UnsetPortOfFace(const Standard_Integer theFaceIndexOfPort)
{
  if( m_PortsWithTypeTool.IsBound(theFaceIndexOfPort) ){
    m_PortsWithTypeTool.UnBind(theFaceIndexOfPort);
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer Model_Ctrl::GetPortTypeWithPortIndex(const Standard_Integer theIndex) const
{
  Standard_Integer thePortType = 0;
  if( m_PortsWithTypeTool.IsBound(theIndex) ){
    thePortType = m_PortsWithTypeTool.Find(theIndex);
  }
  return thePortType;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer Model_Ctrl::GetPortIndex(const TopoDS_Face& theFace) const
{
  Standard_Integer theIndex = m_FacesWithIndexTool.Find(theFace);
  if( !(m_PortsWithTypeTool.IsBound(theIndex)) ){
    theIndex = 0;
  }
  return theIndex;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  :
 */
/****************************************************************/
void Model_Ctrl::ComputeBndBoxOfPort(const Standard_Integer thePortIndex, 
				     TxSlab<Standard_Real>& rgn) const
{
  Standard_Real xmin,ymin,zmin;
  Standard_Real xmax,ymax,zmax;
  const TopoDS_Face& thePortFace = GetFaceWithIndex(thePortIndex);
  ComputeProperBndOfShape( thePortFace, xmin, ymin, zmin, xmax, ymax, zmax);

  rgn.setBounds(xmin,ymin,zmin, xmax,ymax,zmax);
}

