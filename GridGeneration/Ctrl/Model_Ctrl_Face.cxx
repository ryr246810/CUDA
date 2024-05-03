#include <Model_Ctrl.hxx>


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::InitTypeToFace()
{
  ClearFacesTypeDefine();

  TColStd_DataMapIteratorOfDataMapOfIntegerInteger Iter;
  for(Iter.Initialize(m_FacesWithShapeTool); Iter.More(); Iter.Next()){
    Standard_Integer theFaceIndex = Iter.Key();
    Standard_Integer theShapeIndex = Iter.Value();

    //cout<<"InitTypeToFace--------theFaceIndex\t=\t"<<theFaceIndex<<endl;

    Standard_Integer theMaterialType = GetMaterialTypeWithShapeIndex(theShapeIndex);

    m_FacesWithTypeTool.Bind(theFaceIndex, theMaterialType);
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::AppendSpecialFace(const TopoDS_Face& theFace,
				   const Standard_Integer theMaterialType)
{
  if(theFace.IsNull()){
    return;
  }
  if( !(m_SpecialFacesWithTypeTool.IsBound(theFace)) ){
    m_SpecialFacesWithTypeTool.Bind(theFace, theMaterialType);
  }else{
    m_SpecialFacesWithTypeTool.ChangeFind(theFace) = theMaterialType;
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::EraseSpecialFace(const TopoDS_Face & theFace)
{
  if(theFace.IsNull()){
    return;
  }
  if( m_SpecialFacesWithTypeTool.IsBound(theFace) ){
    m_SpecialFacesWithTypeTool.UnBind(theFace);
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
/*
void Model_Ctrl::SetSpecialTypeToFace()
{
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_SpecialFacesWithTypeTool); Iter.More(); Iter.Next()){

    const TopoDS_Shape & theFace = Iter.Key();
    Standard_Integer theMaterialType = Iter.Value();

    cout<<"Model_Ctrl::SetSpecialTypeToFace===============theMaterialType\t=\t"<<theMaterialType<<endl;

    SetSpecialTypeToFace(TopoDS::Face(theFace), theMaterialType);
  }
}
//*/


void Model_Ctrl::SetSpecialTypeToFace()
{
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_SpecialFacesWithTypeTool); Iter.More(); Iter.Next()){
    const TopoDS_Face & theFace = TopoDS::Face(Iter.Key());
    Standard_Integer theMaterialType = Iter.Value();

    cout<<"Model_Ctrl::SetSpecialTypeToFace===============theMaterialType\t=\t"<<theMaterialType<<endl;

    SetSpecialTypeToFace(theFace, theMaterialType);
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::SetSpecialTypeToFace(const Standard_Integer theFaceIndex,
				      const Standard_Integer theMaterialType)
{
  if( m_FacesWithTypeTool.IsBound(theFaceIndex) ){
    Standard_Integer theOldMaterialType =  m_FacesWithTypeTool.Find(theFaceIndex);
    m_FacesWithTypeTool.ChangeFind(theFaceIndex) = theMaterialType | theOldMaterialType;
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::SetSpecialTypeToFace(const TopoDS_Face& theFace,
				      const Standard_Integer theMaterialType)
{
  if(theFace.IsNull()) return;
  Standard_Integer theFaceIndex = GetFaceIndex(theFace);
  if(theFaceIndex==0){
    cout<<"Model_Ctrl::SetSpecialTypeToFace---------------error"<<endl;
    return;
  }
  SetSpecialTypeToFace(theFaceIndex, theMaterialType);
}






/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ClearFacesTypeDefine()
{
  m_FacesWithTypeTool.Clear();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer Model_Ctrl::GetMaterialTypeWithFaceIndex(const Standard_Integer theIndex) const
{
  Standard_Integer theMaterialType = 0;

  if( m_FacesWithTypeTool.IsBound(theIndex) ){
    theMaterialType = m_FacesWithTypeTool.Find(theIndex);
  }

  return theMaterialType;
}

