#include <Model_Ctrl.hxx>

/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::AppendShape(const TopoDS_Shape& theShape,
			     const Standard_Integer theMaterialType)
{
  if(theShape.IsNull()){
    return;
  }
  if( !(m_ShapesWithTypeTool.IsBound(theShape)) ){
    m_ShapesWithTypeTool.Bind(theShape, theMaterialType);
  }else{
    m_ShapesWithTypeTool.ChangeFind(theShape) = theMaterialType;
  }
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::EraseShape(const TopoDS_Shape & theShape)
{
  if(theShape.IsNull()){
    return;
  }
  if( m_ShapesWithTypeTool.IsBound(theShape) ){
    m_ShapesWithTypeTool.UnBind(theShape);
  }
}





/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::SetupShapeIndex()
{
  ResetShapesIndex();
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  Standard_Integer CurrentIndx=1;

  for(Iter.Initialize(m_ShapesWithTypeTool); Iter.More(); Iter.Next() ){
    const TopoDS_Shape & theShape = Iter.Key();
    m_ShapesWithIndexTool.Bind(theShape,CurrentIndx);
    m_IndexWithShapesTool.Bind(CurrentIndx,theShape);
    CurrentIndx++;
  }
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ResetShapesIndex()
{
  m_ShapesWithIndexTool.Clear();
  m_IndexWithShapesTool.Clear();
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ClearShapesTypeDefine()
{
  m_ShapesWithTypeTool.Clear();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::SetupFaceIndex()
{
  ResetFacesIndex();

  TopExp_Explorer  Ex;
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  Standard_Integer CurrentIndx=1;
  for(Iter.Initialize(m_ShapesWithIndexTool); Iter.More(); Iter.Next() ){
    const TopoDS_Shape & theShape = Iter.Key();
    const Standard_Integer theShapeIndex = Iter.Value();

    for( Ex.Init(theShape,TopAbs_FACE);Ex.More(); Ex.Next() ){
      m_FacesWithIndexTool.Bind(Ex.Current(),CurrentIndx);
      m_IndexWithFacesTool.Bind(CurrentIndx,Ex.Current());
      m_FacesWithShapeTool.Bind(CurrentIndx,theShapeIndex);
      CurrentIndx++;
    }
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ResetFacesIndex()
{
  m_FacesWithIndexTool.Clear();
  m_IndexWithFacesTool.Clear();
  m_FacesWithShapeTool.Clear();
}



/****************************************************************/
/****************************************************************/
/***********************    Tools    ****************************/
/****************************************************************/
/****************************************************************/


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::HasShapeIndex(const Standard_Integer theIndex) const
{
  bool result=false;
  if(m_IndexWithShapesTool.IsBound(theIndex)){
    result=true;
  }
  return result;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
const TopoDS_Shape& Model_Ctrl::GetShapeWithIndex(const Standard_Integer theIndex) const
{
  return m_IndexWithShapesTool.Find(theIndex);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer Model_Ctrl::GetShapeIndex(const TopoDS_Shape& theShape) const
{
  Standard_Integer theShapeIndex = 0;
  if(m_ShapesWithIndexTool.IsBound(theShape)){
    theShapeIndex = m_ShapesWithIndexTool.Find(theShape);
  }
  return theShapeIndex;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::HasFaceIndex(const Standard_Integer theIndex) const
{
  bool result=false;
  if(m_IndexWithFacesTool.IsBound(theIndex)){
    result=true;
  }
  return result;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
//*
Standard_Integer Model_Ctrl::GetFaceIndex(const TopoDS_Face& theFace) const
{
  Standard_Integer theFaceIndex = 0;
  if(m_FacesWithIndexTool.IsBound(theFace)){
    theFaceIndex = m_FacesWithIndexTool.Find(theFace);
  }
  return theFaceIndex;
}
//*/

/*
Standard_Integer Model_Ctrl::GetFaceIndex(const TopoDS_Face& theFace) const
{
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;
  Standard_Integer theFaceIndex = 0;
  for(Iter.Initialize(m_FacesWithIndexTool); Iter.More(); Iter.Next()){
    const TopoDS_Face & currFace = TopoDS::Face(Iter.Key());
    if(currFace.IsSame(theFace)){
      theFaceIndex = Iter.Value();
    }
  }
  return theFaceIndex;
}
//*/

/*
Standard_Integer Model_Ctrl::GetFaceIndex(const TopoDS_Face& theFace) const
{
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;
  Standard_Integer theFaceIndex = 0;
  for(Iter.Initialize(m_FacesWithIndexTool); Iter.More(); Iter.Next()){
    const TopoDS_Face & currFace = TopoDS::Face(Iter.Key());
    if( (theFace.TShape()==currFace.TShape()) && 
	(theFace.Orientation()==currFace.Orientation()) ){
      theFaceIndex = Iter.Value();
    }
  }
  return theFaceIndex;
}
//*/



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
const TopoDS_Face& Model_Ctrl::GetFaceWithIndex(const Standard_Integer theIndex) const
{
  return TopoDS::Face(m_IndexWithFacesTool.Find(theIndex));
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer Model_Ctrl::GetMaterialTypeWithShapeIndex(const Standard_Integer theIndex) const
{
  Standard_Integer theMaterialType = 0;
  if(m_IndexWithShapesTool.IsBound(theIndex)){
    const TopoDS_Shape& theShape = m_IndexWithShapesTool.Find(theIndex); 
    if( m_ShapesWithTypeTool.IsBound(theShape) ){
      theMaterialType = m_ShapesWithTypeTool.Find(theShape);
    }
  }
  return theMaterialType;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::IsFaceBelongToOneShape(const Standard_Integer theFaceIndex,
					Standard_Integer& theShapeIndex) const
{
  bool result=false;
  theShapeIndex=0;
  if(m_FacesWithShapeTool.IsBound(theFaceIndex)){
    theShapeIndex = m_FacesWithShapeTool.Find(theFaceIndex);
    result=true;
  }
  return result;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Model_Ctrl::DoesShapeHasFace(const Standard_Integer theShapeIndex,
				  const Standard_Integer theFaceIndex) const
{
  bool result=false;
  Standard_Integer tmpShapeIndex=0;

  if(IsFaceBelongToOneShape(theFaceIndex, tmpShapeIndex)){
    if(theShapeIndex==tmpShapeIndex){
      result=true;
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
void Model_Ctrl::CheckAndGetShapeIndexFromFaceIndices(const vector<Standard_Integer>& theFaceIndices, 
						      Standard_Integer& theShapeIndex, bool& isFacesBelongOneShape) const
{
  isFacesBelongOneShape = true;

  if( theFaceIndices.size()<1 ){
    theShapeIndex = 0;
    isFacesBelongOneShape = false;
    return ;
  }

  Standard_Integer theFirstFaceIndex = theFaceIndices[0];
  IsFaceBelongToOneShape(theFirstFaceIndex, theShapeIndex);

  for(Standard_Size i=1; i<theFaceIndices.size(); i++){
    Standard_Integer tmpShapeIndex = 0;
    IsFaceBelongToOneShape(theFaceIndices[i], tmpShapeIndex);
    if(tmpShapeIndex != theShapeIndex){
      isFacesBelongOneShape = false;
      break;
    }
  }

}






/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::CheckAndGetShapeIndexFromFaceIndices(const TColStd_ListOfInteger & theFaceIndices, 
						      Standard_Integer& theShapeIndex, bool& isFacesBelongOneShape) const
{
  isFacesBelongOneShape = true;

  if( theFaceIndices.Extent()<1 ){
    theShapeIndex = 0;
    isFacesBelongOneShape = false;
    return ;
  }

  IsFaceBelongToOneShape(theFaceIndices.First(), theShapeIndex);

  TColStd_ListIteratorOfListOfInteger theListIter;
  for( theListIter.Initialize(theFaceIndices); theListIter.More(); theListIter.Next() ){
    Standard_Integer tmpShapeIndex = 0;
    IsFaceBelongToOneShape(theListIter.Value(), tmpShapeIndex);
    if(tmpShapeIndex != theShapeIndex){
      isFacesBelongOneShape = false;
      break;
    }
  }
}








/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::CheckAndGetShapeIndexFromVertexIndex(const Standard_Integer theVertexIndex, 
						      Standard_Integer& theShapeIndex, 
						      bool& isVertexBelongOneShape) const
{
  isVertexBelongOneShape = true;
  theShapeIndex = 0;

  const TColStd_ListOfInteger& theFaceIndices = m_VertexWithFaceTool.Find(theVertexIndex);
  if( theFaceIndices.Extent()<1 ){
    theShapeIndex = 0;
    isVertexBelongOneShape = false;
    return ;
  }

  Standard_Integer theFirstFaceIndex = theFaceIndices.First();
  IsFaceBelongToOneShape(theFirstFaceIndex, theShapeIndex);

  TColStd_ListIteratorOfListOfInteger iter;
  for(iter.Initialize(theFaceIndices); iter.More();  iter.Next() ){
    Standard_Integer currFaceIndex = iter.Value();
    Standard_Integer tmpShapeIndex = 0;
    IsFaceBelongToOneShape(currFaceIndex, tmpShapeIndex);
    if(tmpShapeIndex != theShapeIndex){
      isVertexBelongOneShape = false;
      break;
    }
  }
}






void Model_Ctrl::Write_Faces()
{


  const TopTools_DataMapOfShapeInteger& theAllFaces = GetFacesWithIndex();

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(theAllFaces); Iter.More(); Iter.Next() ){
    const TopoDS_Shape & theShape = Iter.Key();
    const Standard_Integer theIndex = Iter.Value();

      ostringstream sstr;
      sstr<<"SubFace_";
      sstr<<theIndex;
      sstr<<".brep";
      string s=sstr.str();

      BRepTools::Write(theShape,s.c_str()); 
  }

}
